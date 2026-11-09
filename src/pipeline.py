"""사진 4장(A~D) → 탑뷰 변환 → 스팟 탐지 → 특징량 → 물질 판별 → TU 등급 → Grad-CAM
Streamlit 화면(src/app.py)과 다른 코드에서 함께 쓰는 분석 단계 모음

앞 단계에서 정한 것을 그대로 따름
  - 스팟 탐지: LOSO fold 모델(runs/spot/foldXX). 36 epoch 에서 멈춘 최종 모델은 쓰지 않음
      · 보통: fold 모델 13개가 각자 찾은 상자를 모아 과반수(7개 이상)가 찾으면 '스팟 있음', 상자는 중앙값
      · 데모 모드(데이터셋 시료 k 의 사진): 시료 k 를 본 적 없는 fold k 모델만 사용
  - 물질 판별: 교차검증에서 가장 좋았던 'ResNet-50 특징 + 랜덤포레스트, 스팟 A만' (LOSO 8/13)
  - Grad-CAM: 같은 ResNet-50 특징(A~D) 위의 설명용 로지스틱 회귀 판정층 (gradcam_spots.py 와 같은 방식)
  - TU: toxic_unit.py 의 문헌 EC50 과 저/중/고 기준. 농도는 사용자가 입력 (모델은 농도를 예측하지 않음)
  - 데모 모드에서는 판별·Grad-CAM 모델도 시료 k 를 빼고 학습 → 처음 보는 사진처럼 정직하게 평가
"""
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extract_features import spot_features  # noqa: E402  (특징량 4가지: 앞 단계와 같은 계산)
from toxic_unit import SAMPLES, assess  # noqa: E402

CLASSES = ['Ctrl', 'Cu', 'NaCl', 'SDS']
CHANNELS = 'ABCD'
CHANNEL_NAME = {'A': '증류수', 'B': '탄닌', 'C': '셀룰로오스', 'D': '탄닌+셀룰로오스'}
SAMPLE_NAME = {1: 'SDS10', 2: 'Cu1', 3: 'NaCl3000', 4: 'Ctrl', 5: 'Cu50', 6: 'SDS1', 7: 'NaCl100',
               8: 'Cu0.1', 9: 'SDS100', 10: 'NaCl10000', 11: 'Cu10', 12: 'SDS30', 13: 'NaCl1000'}
RUNS = ROOT / 'runs' / 'spot'
FEAT_CACHE = ROOT / 'data' / 'features' / 'resnet50_features.npz'
DIM = 2049  # 채널 하나당: 링 있음 표시 1 + ResNet 특징 2048
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------ 0. 파일 이름으로 시료 알아내기
def parse_name(filename):
    """'A5.jpg', 'A5-1.jpg' → ('A', 5). 데이터셋 이름 규칙이 아니면 (None, None)"""
    m = re.match(r'^([A-Da-d])(\d{1,2})(?:-\d{1,2})?(?:\.\w+)?$', Path(filename).name)
    if m and int(m.group(2)) in SAMPLES:
        return m.group(1).upper(), int(m.group(2))
    return None, None


def decode_image(data):
    """업로드된 바이트 → BGR 이미지"""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ 1. 탑뷰 변환
def topview(img, camera_angle_deg=57.0, hfov_deg=70.0):
    """비스듬히(지면과 camera_angle_deg) 찍은 사진을 위에서 본 모습으로 바꿈.
    src/topview_convert.py 가 있으면 그 함수를 우선 사용하고, 없으면 촬영 각도로 근사 변환.
    근사 방법: 카메라를 (90° - 촬영각)만큼 돌려 바로 아래를 보게 하는 '회전 호모그래피'
               (휴대폰 카메라 가로 화각을 약 70°로 가정)"""
    custom = _custom_topview()
    if custom is not None:
        return custom(img), 'src/topview_convert.py'
    h, w = img.shape[:2]
    f = (max(w, h) / 2) / math.tan(math.radians(hfov_deg / 2))  # 휴대폰 화각은 보통 긴 변 기준
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    t = math.radians(90 - camera_angle_deg)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    for sign in (1, -1):  # 먼 쪽(사진 위쪽)이 넓어지는 방향을 고름
        R = np.array([[1, 0, 0], [0, math.cos(sign * t), -math.sin(sign * t)],
                      [0, math.sin(sign * t), math.cos(sign * t)]])
        H = K @ R @ np.linalg.inv(K)
        c = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        if (c[1, 0] - c[0, 0]) > (c[2, 0] - c[3, 0]):
            break
    x0, y0 = c.min(axis=0)
    x1, y1 = c.max(axis=0)
    scale = min(1.0, 2400 / max(x1 - x0, y1 - y0))  # 데이터셋 사진과 비슷한 크기로
    T = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale], [0, 0, 1]])
    size = (int((x1 - x0) * scale), int((y1 - y0) * scale))
    out = cv2.warpPerspective(img, T @ H, size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return out, f'근사 변환 (촬영각 {camera_angle_deg:g}°, 화각 {hfov_deg:g}° 가정)'


def _custom_topview():
    """프로젝트의 src/topview_convert.py 에 topview/convert/to_topview 함수가 있으면 사용"""
    p = SRC / 'topview_convert.py'
    if not p.exists():
        return None
    import importlib
    mod = importlib.import_module('topview_convert')
    for name in ('topview', 'convert', 'to_topview'):
        fn = getattr(mod, name, None)
        if callable(fn):
            def wrapped(im, fn=fn):
                out = fn(im)
                return out[0] if isinstance(out, tuple) else out  # (사진, 기타) 를 돌려줘도 사진만 사용
            return wrapped
    return None


# ------------------------------------------------------------------ 2. 스팟 탐지
def load_detectors(sample=None):
    """sample 이 있으면 fold{sample} 모델만(데모 모드), 없으면 fold 모델 13개 전부"""
    from ultralytics import YOLO
    folds = [sample] if sample else list(SAMPLES)
    models = []
    for k in folds:
        p = RUNS / f'fold{k:02d}' / 'weights' / 'best.pt'
        if p.exists():
            models.append(YOLO(str(p)))
    if not models:
        raise FileNotFoundError(f'{RUNS} 에 fold 모델(best.pt)이 없습니다.')
    return models


def detect(img, models, conf=0.25, imgsz=640):
    """모델마다 확신도가 가장 높은 상자 1개 → 과반수가 찾으면 스팟 있음, 상자는 중앙값"""
    boxes, confs = [], []
    for m in models:
        r = m.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        if len(r.boxes):
            k = int(r.boxes.conf.argmax())
            boxes.append(r.boxes.xyxy[k].tolist())
            confs.append(float(r.boxes.conf[k]))
    votes, need = len(boxes), len(models) // 2 + 1
    if votes < need:
        return {'detected': False, 'votes': votes, 'n_models': len(models), 'box': None, 'conf': None}
    return {'detected': True, 'votes': votes, 'n_models': len(models),
            'box': [float(v) for v in np.median(np.array(boxes), axis=0)], 'conf': float(np.mean(confs))}


def crop_spot(img, box, pad=0.10):
    """상자를 10% 넓혀 잘라냄 (crop_spots.py 와 같은 방식)"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return img[max(0, int(y0 - py)):min(h, int(y1 + py)), max(0, int(x0 - px)):min(w, int(x1 + px))]


def draw_box(img, det):
    out = img.copy()
    if det['detected']:
        x0, y0, x1, y1 = (int(v) for v in det['box'])
        th = max(3, int(max(img.shape[:2]) / 300))
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), th)
    return out


# ------------------------------------------------------------------ 3~5. ResNet 특징, 판별, Grad-CAM
class Analyzer:
    """ResNet-50 + 판별 랜덤포레스트(A만) + 설명용 로지스틱 회귀(A~D).
    exclude_sample 이 있으면 그 시료를 빼고 학습 (데모 모드)"""

    def __init__(self, exclude_sample=None, seed=0):
        import torch
        import torchvision
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if not FEAT_CACHE.exists():
            raise FileNotFoundError(f'{FEAT_CACHE} 가 없습니다. python src/classify_toxicant.py 를 먼저 실행하세요.')
        self.torch = torch
        net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval()
        self.body = torch.nn.Sequential(*list(net.children())[:-2])  # layer4 까지

        z = np.load(FEAT_CACHE)
        vec = {n: x for n, x in zip(z['names'], z['X'])}
        XA, X4, y = [], [], []
        for s, (chem, _) in SAMPLES.items():
            if s == exclude_sample:
                continue
            for a in range(1, 21):
                XA.append(vec[f'A{s}-{a}'])
                X4.append(np.concatenate([vec[f'{c}{s}-{a}'] for c in CHANNELS]))
                y.append(CLASSES.index(chem))
        XA, X4, y = np.array(XA), np.array(X4), np.array(y)
        self.rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1).fit(XA, y)
        self.scaler = StandardScaler().fit(X4)
        self.logit = LogisticRegression(C=1.0, max_iter=5000).fit(self.scaler.transform(X4), y)
        W = np.zeros((len(CLASSES), X4.shape[1]), np.float32)
        W[self.logit.classes_] = self.logit.coef_ / self.scaler.scale_
        self.W = W - W[self.logit.classes_].mean(axis=0)
        self.exclude_sample = exclude_sample

    def _tensor(self, crop):
        rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return self.torch.from_numpy((((rgb.astype(np.float32) / 255) - MEAN) / STD)[None]).permute(0, 3, 1, 2)

    def feature(self, crop):
        """잘라낸 스팟 → [링 있음 1, ResNet 특징 2048]. 스팟이 없으면 전부 0"""
        if crop is None:
            return np.zeros(DIM, np.float32)
        with self.torch.no_grad():
            f = self.body(self._tensor(crop)).mean(dim=(2, 3))[0].numpy()
        return np.concatenate([[1.0], f]).astype(np.float32)

    def classify(self, feats):
        """feats: {'A': 벡터, ...} → 판별 랜덤포레스트(A만) 확률, 설명용 판정층(A~D) 확률"""
        p_rf = np.zeros(len(CLASSES))
        p_rf[self.rf.classes_] = self.rf.predict_proba(feats['A'][None])[0]
        x4 = np.concatenate([feats[c] for c in CHANNELS])[None]
        p_lr = np.zeros(len(CLASSES))
        p_lr[self.logit.classes_] = self.logit.predict_proba(self.scaler.transform(x4))[0]
        return p_rf, p_lr

    def gradcam(self, crop, channel, class_idx, alpha=0.45):
        """잘라낸 스팟 위에 class_idx 쪽으로 판단을 끌어당긴 부분을 히트맵으로 겹침"""
        torch = self.torch
        ci = CHANNELS.index(channel)
        fmap = self.body(self._tensor(crop))
        fmap.retain_grad()
        wc = torch.from_numpy(self.W[class_idx, ci * DIM + 1:(ci + 1) * DIM])
        (wc * fmap.mean(dim=(2, 3))[0]).sum().backward()
        a = fmap.grad.mean(dim=(2, 3))[0]
        cam = torch.relu((a[:, None, None] * fmap[0]).sum(0)).detach().numpy()
        cam = cam / (cam.max() + 1e-8)
        h, w = crop.shape[:2]
        heat = cv2.applyColorMap((cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1) * 255)
                                 .astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(crop, 1 - alpha, heat, alpha, 0)


# ------------------------------------------------------------------ 전체 실행
def run_pipeline(images, conc_mg_L, analyzer, detectors, do_topview=True, conf=0.25,
                 camera_angle_deg=57.0, hfov_deg=70.0):
    """images: {'A': BGR, 'B': ..., 'C': ..., 'D': ...} → 단계별 결과 사전"""
    res = {'channels': {}}
    feats = {}
    for ch in CHANNELS:
        img = images[ch]
        tv, how = topview(img, camera_angle_deg, hfov_deg) if do_topview else (img, '변환 생략 (이미 탑뷰)')
        det = detect(tv, detectors, conf=conf)
        crop = crop_spot(tv, det['box']) if det['detected'] else None
        hand = spot_features(tv, det['box']) if det['detected'] else None
        feats[ch] = analyzer.feature(crop)
        res['channels'][ch] = {'topview': tv, 'topview_how': how, 'det': det, 'boxed': draw_box(tv, det),
                               'crop': crop, 'features': hand}
    p_rf, p_lr = analyzer.classify(feats)
    pred = int(p_rf.argmax())
    res['proba'], res['proba_explainer'] = p_rf, p_lr
    res['pred'] = CLASSES[pred]
    res['tu'] = assess(CLASSES[pred], conc_mg_L)
    for ch in CHANNELS:
        c = res['channels'][ch]
        c['gradcam'] = analyzer.gradcam(c['crop'], ch, pred) if c['crop'] is not None else None
    return res
