"""비스듬히 찍은 커피링 사진을 위에서 내려다본 사진(탑뷰)으로 바꾸기

원래 코드: 바탕 화면 '커피링_탑뷰변환.zip' 안의 topview_convert.py (데이터셋 원본 52장을 만든 코드)
  - 변환 계산은 원래 코드와 똑같음. 바꾼 점:
    1) 다른 코드(Streamlit 화면)에서 부를 수 있게 topview(사진) 함수로 묶음
    2) 한글 폴더 이름에서도 사진을 읽고 쓸 수 있게 함 (cv2.imread 는 한글 경로에서 실패함)
    3) 주석을 한국어로
  - 이 함수로 바꾼 결과가 프로젝트의 원본 사진(X-1.jpg)과 같은지 확인 완료

촬영 조건 (이 조건으로 찍은 사진에만 맞음)
  - 휴대폰과 지면이 이루는 각도 57°, 스팟까지 약 11 cm
  - Galaxy S24 Ultra 메인 렌즈(35 mm 환산 23 mm) → 초점거리 약 3,795 픽셀 (전체 해상도 5712×4284 기준)
  - 잘라낸(크롭한) 사진은 사진 중앙을 렌즈 중심으로 가정 → 스팟이 중앙에서 멀수록 오차가 조금 커짐

방법
  카메라를 57° 돌려 바로 아래를 내려다보는 '가상 카메라'로 바꾸는 원근 변환(호모그래피).
  세로가 약 1.84배(= 1/sin 33°) 늘어나고, 먼 쪽(위)과 가까운 쪽(아래)의 크기 차이도 보정됨.
  가로 크기는 사진 중앙 기준으로 원본과 같게 유지하고, 변환 후 생긴 검은 여백은 잘라냄.

사용법
  python src/topview_convert.py <원본폴더> <저장폴더>
  (다른 코드에서)  from topview_convert import topview;  탑뷰사진 = topview(사진)
"""
import glob
import math
import os
import sys

import cv2
import numpy as np

TILT_DEG = 57.0  # 휴대폰 화면과 지면이 이루는 각도
F_PX = 3795.0    # 초점거리(픽셀): 23 mm 환산 렌즈, 24MP(5712×4284) 센서 기준


def homography(w, h, sign):
    """가상 카메라를 sign 방향으로 57° 돌려 바로 아래를 보게 하는 변환 행렬"""
    phi = math.radians(TILT_DEG) * sign
    K = np.array([[F_PX, 0, w / 2], [0, F_PX, h / 2], [0, 0, 1]])
    R = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]])
    return K @ R @ np.linalg.inv(K)


def apply(H, pts):
    """점들에 변환 행렬 적용"""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]


def local_scale(H, x, y, e=2.0):
    """(x, y) 근처에서 가로·세로가 몇 배로 늘어나는지"""
    a = apply(H, np.array([[x - e, y], [x + e, y], [x, y - e], [x, y + e]]))
    return np.linalg.norm(a[1] - a[0]) / (2 * e), np.linalg.norm(a[3] - a[2]) / (2 * e)


def topview(img):
    """비스듬히 찍은 사진(BGR) → 탑뷰 사진(BGR)"""
    h, w = img.shape[:2]
    # 두 회전 방향 중 '세로로 늘어나고, 먼 쪽(위)이 확대되는' 쪽을 고름 (비스듬히 본 바닥면)
    best = None
    for s in (1, -1):
        H = homography(w, h, s)
        sx, sy = local_scale(H, w / 2, h / 2)
        top = local_scale(H, w / 2, h * 0.1)[0]
        bot = local_scale(H, w / 2, h * 0.9)[0]
        if sy / sx > 1 and top > bot:
            best = H
            break
    H = best
    sx, _ = local_scale(H, w / 2, h / 2)
    H = np.diag([1 / sx, 1 / sx, 1]) @ H  # 사진 중앙의 가로 배율을 1로 유지
    c = apply(H, np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    x0, y0 = c.min(0)
    x1, y1 = c.max(0)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    H = T @ H
    W, Hh = int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0))
    out = cv2.warpPerspective(img, H, (W, Hh), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (W, Hh), flags=cv2.INTER_NEAREST)
    cols = np.where((mask[4:-4] > 0).all(axis=0))[0]  # 모든 줄에서 유효한 열만 → 검은 사다리꼴 여백 제거
    return out[:, cols.min():cols.max() + 1]


def main():
    if len(sys.argv) != 3:
        sys.exit('사용법: python src/topview_convert.py <원본폴더> <저장폴더>')
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src_dir, '*.jpg'))):
        img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)  # 한글 경로도 읽기
        h, w = img.shape[:2]
        out = topview(img)
        cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(
            os.path.join(out_dir, os.path.basename(f)))
        print(os.path.basename(f), w, h, out.shape[1], out.shape[0])


if __name__ == '__main__':
    main()
