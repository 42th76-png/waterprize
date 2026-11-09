"""잘라낸 커피링 스팟에서 특징량 4가지를 뽑고, 화학물질별 농도-특징량 그래프 그리기

먼저 할 일: python src/crop_spots.py  (data/crops/spots.csv 가 있어야 함)

특징량 (모두 '픽셀' 단위. 원본 사진은 같은 카메라 거리·탑뷰 변환이라 축척이 같다고 가정)
  1) 침착 지름   : 탐지 상자의 가로·세로 평균 (px)
  2) 고리 세기   : 가장자리 띠(반지름 75~100%) 평균 밝기 − 안쪽(0~60%) 평균 밝기
                   (배경 밝기를 뺀 회색조 값. 양수가 클수록 가장자리에 물질이 몰린 '커피링')
  3) 잔류물 면적 : 스팟 타원 안에서 배경보다 확실히 밝은(배경 잡음의 3배 이상) 픽셀 수 (px²)
                   + 타원 넓이 대비 비율(%)
  4) 결정 개수   : 타원 안의 작고 밝은 점(결정 알갱이) 개수

배경: 스팟 타원 바깥 고리(반지름 115~150%) 영역의 중앙값과 잡음(MAD)으로 계산

그래프: 원본 사진(-1) 기준, 시료(조건)마다 채널 A~D 4개 값
  - 증강본은 같은 사진을 돌리고 크기·밝기를 바꾼 것이라 반복 실험이 아니므로 그래프에 넣지 않음
  - 스팟이 없거나(커피링 미관찰) 탐지 못한 사진은 아래쪽에 × 로 표시

사용법 (프로젝트 폴더에서)
  python src/extract_features.py
결과
  data/features/spot_features.csv          사진 852장(스팟 찾은 것 전부)의 특징량
  data/features/original_features.csv      원본 52장 (스팟 없음 포함) — 그래프에 쓴 값
  docs/figures/concentration_features.png  농도-특징량 그래프
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / 'data' / 'crops'
IMAGES = ROOT / 'data' / 'yolo' / 'images'
OUT = ROOT / 'data' / 'features'
FIG = ROOT / 'docs' / 'figures'

# 시료 번호 → (물질, 농도 mg/L)
SAMPLE_INFO = {1: ('SDS', 10), 2: ('Cu', 1), 3: ('NaCl', 3000), 4: ('Ctrl', 0), 5: ('Cu', 50),
               6: ('SDS', 1), 7: ('NaCl', 100), 8: ('Cu', 0.1), 9: ('SDS', 100), 10: ('NaCl', 10000),
               11: ('Cu', 10), 12: ('SDS', 30), 13: ('NaCl', 1000)}
CHANNEL_NAME = {'A': '증류수', 'B': '탄닌', 'C': '셀룰로오스', 'D': '탄닌+셀룰로오스'}
FEATURES = [('diameter_px', '침착 지름 (px)'),
            ('ring_strength', '고리 세기 (가장자리-안쪽 밝기)'),
            ('residue_area_px', '잔류물 면적 (px²)'),
            ('crystal_count', '결정 개수 (개)')]


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def mad_sigma(v):
    """잡음 크기: 중앙값에서 얼마나 흩어져 있는지 (튀는 값에 강한 표준편차 대용)"""
    med = np.median(v)
    return 1.4826 * np.median(np.abs(v - med)) + 1e-6


def spot_features(im, box):
    """사진 한 장 + 탐지 상자 → 특징량 사전"""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    x0, y0, x1, y1 = box
    cx, cy, a, b = (x0 + x1) / 2, (y0 + y1) / 2, max(2.0, (x1 - x0) / 2), max(2.0, (y1 - y0) / 2)

    # 계산을 빠르게: 스팟 주변(반지름 1.6배)만 잘라서 다룸
    rx0, rx1 = max(0, int(cx - 1.6 * a)), min(w, int(cx + 1.6 * a) + 1)
    ry0, ry1 = max(0, int(cy - 1.6 * b)), min(h, int(cy + 1.6 * b) + 1)
    sub = g[ry0:ry1, rx0:rx1]
    yy, xx = np.mgrid[ry0:ry1, rx0:rx1]
    r = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)  # 타원 기준 반지름 (1 = 가장자리)

    inside = r <= 1.0
    bgmask = (r >= 1.15) & (r <= 1.5)
    if bgmask.sum() < 50:  # 사진 끝에 붙어 배경이 부족하면 상자 바깥 전체로
        bgmask = r > 1.05
    bg = float(np.median(sub[bgmask]))
    noise = mad_sigma(sub[bgmask])
    sig = sub - bg  # 배경 밝기를 뺀 값

    # 2) 고리 세기
    ring_band = (r > 0.75) & inside
    center = r <= 0.6
    ring_mean = float(sig[ring_band].mean())
    center_mean = float(sig[center].mean())

    # 3) 잔류물 면적: 살짝 흐리게 해 잡음을 줄인 뒤, 배경 잡음의 3배보다 밝은 픽셀
    smooth = cv2.GaussianBlur(sub, (0, 0), 1.5) - bg
    noise_s = mad_sigma(smooth[bgmask])
    residue = inside & (smooth > 3 * noise_s)
    ellipse_area = math.pi * a * b

    # 4) 결정 개수: 작고 밝은 점만 남기는 '탑햇' 필터 → 잡음의 4배 이상 밝은 덩어리 세기
    th = cv2.morphologyEx(sub, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    th_noise = mad_sigma(th[bgmask])
    spots = ((th > max(4 * th_noise, 8)) & inside).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(spots, connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    crystals = int(((areas >= 4) & (areas <= 400)).sum())  # 너무 작으면 잡음, 너무 크면 결정이 아님

    return {'diameter_px': round((a + b), 1),                 # 지름 = (가로+세로)/2 = a+b
            'ring_strength': round(ring_mean - center_mean, 2),
            'ring_mean': round(ring_mean, 2), 'center_mean': round(center_mean, 2),
            'residue_area_px': int(residue.sum()),
            'residue_cover_pct': round(100 * residue.sum() / ellipse_area, 1),
            'crystal_count': crystals,
            'bg': round(bg, 1), 'noise': round(noise, 2)}


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 9})
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    # 검증된 기본 팔레트 앞 4색 (채널 순서 고정) + 모양을 달리해 색만으로 구분하지 않게
    STYLE = {'A': ('#2a78d6', 'o'), 'B': ('#eb6834', 's'), 'C': ('#1baf7a', '^'), 'D': ('#eda100', 'D')}
    chems = ['Cu', 'NaCl', 'SDS']
    concs = {c: sorted({v[1] for v in SAMPLE_INFO.values() if v[0] == c}) for c in chems}

    fig, axes = plt.subplots(len(FEATURES), 3, figsize=(11, 12), facecolor=SURF)
    for i, (feat, flabel) in enumerate(FEATURES):
        vals = [r[feat] for r in rows if r.get(feat) not in (None, '')]
        ymax = max(vals) if vals else 1
        ymin = min(0, min(vals)) if vals else 0
        pad = (ymax - ymin) * 0.08
        for j, chem in enumerate(chems):
            ax = axes[i, j]
            ax.set_facecolor(SURF)
            xs = concs[chem]
            x_ctrl = xs[0] / 10  # 대조군(농도 0)은 로그축에 못 그리므로 왼쪽에 따로 표시
            for ch, (color, mk) in STYLE.items():
                pts = []
                for x in xs:
                    r = next(r for r in rows if r['chemical'] == chem and r['conc'] == x and r['channel'] == ch)
                    pts.append((x, r.get(feat)))
                good = [(x, y) for x, y in pts if y not in (None, '')]
                if good:
                    ax.plot(*zip(*good), color=color, lw=1.6, marker=mk, ms=6,
                            markeredgecolor=SURF, markeredgewidth=1.0, zorder=3)
                for x, y in pts:  # 스팟 없음/미검출
                    if y in (None, ''):
                        ax.plot(x, ymin - pad * 0.5, marker='x', color=color, ms=6, mew=1.6, zorder=3)
                rc = next(r for r in rows if r['chemical'] == 'Ctrl' and r['channel'] == ch)
                if rc.get(feat) not in (None, ''):
                    ax.plot(x_ctrl, rc[feat], marker=mk, ms=6, mfc='none', mec=color, mew=1.4, zorder=3)
                else:
                    ax.plot(x_ctrl, ymin - pad * 0.5, marker='x', color=color, ms=6, mew=1.6, zorder=3)
            ax.set_xscale('log')
            ax.set_xlim(x_ctrl / 2, xs[-1] * 2)
            ax.set_xticks([x_ctrl] + xs)
            ax.set_xticklabels(['Ctrl'] + [f'{x:g}' for x in xs])
            ax.minorticks_off()
            ax.axvline(math.sqrt(x_ctrl * xs[0]), color=GRID, lw=1, ls=(0, (2, 2)), zorder=1)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.grid(axis='y', color=GRID, lw=0.8)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            for s in ('left', 'bottom'):
                ax.spines[s].set_color(INK2)
            ax.tick_params(colors=INK2, labelsize=8)
            if i == 0:
                ax.set_title(chem, color=INK, fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(flabel, color=INK, fontsize=9)
            if i == len(FEATURES) - 1:
                ax.set_xlabel('농도 (mg/L, 로그 눈금)', color=INK2, fontsize=8)

    handles = [Line2D([], [], color=c, marker=m, lw=1.6, ms=6, label=f'{ch} {CHANNEL_NAME[ch]}')
               for ch, (c, m) in STYLE.items()]
    handles += [Line2D([], [], color=INK2, marker='o', mfc='none', lw=0, ms=6, label='Ctrl (빈 기호)'),
                Line2D([], [], color=INK2, marker='x', lw=0, ms=6, mew=1.6, label='스팟 없음·미검출')]
    fig.legend(handles=handles, loc='upper center', ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle('화학물질 농도에 따른 커피링 특징량 (원본 사진, 시료당 채널별 1장)',
                 color=INK, fontsize=13, fontweight='bold', y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURF)
    print(f'그래프 저장: {path}')


def main():
    sys.stdout.reconfigure(line_buffering=True)
    spots_csv = CROPS / 'spots.csv'
    if not spots_csv.exists():
        sys.exit('[오류] data/crops/spots.csv 가 없습니다. 먼저 python src/crop_spots.py 를 실행하세요.')
    spots = list(csv.DictReader(open(spots_csv, encoding='utf-8-sig')))
    OUT.mkdir(parents=True, exist_ok=True)

    feats_all = []
    for i, s in enumerate(spots):
        sample = int(s['sample'])
        chem, conc = SAMPLE_INFO[sample]
        row = {'image': s['image'], 'sample': sample, 'chemical': chem, 'conc': conc,
               'channel': s['channel'], 'aug': int(s['aug']), 'original': int(s['original']),
               'detected': int(s['detected']), 'gt_ring': int(s['gt_ring']), 'conf': s['conf']}
        if row['detected']:
            im = imread(IMAGES / f"{s['image']}.jpg")
            row.update(spot_features(im, [float(s[k]) for k in ('x0', 'y0', 'x1', 'y1')]))
        feats_all.append(row)
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(spots)}장 처리')

    keys = ['image', 'sample', 'chemical', 'conc', 'channel', 'aug', 'original', 'detected', 'gt_ring', 'conf',
            'diameter_px', 'ring_strength', 'ring_mean', 'center_mean', 'residue_area_px',
            'residue_cover_pct', 'crystal_count', 'bg', 'noise']
    with open(OUT / 'spot_features.csv', 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows([r for r in feats_all if r['detected']])
    origs = sorted([r for r in feats_all if r['original']], key=lambda r: (r['sample'], r['channel']))
    with open(OUT / 'original_features.csv', 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(origs)
    print(f'특징량 저장: {OUT / "spot_features.csv"} ({sum(r["detected"] for r in feats_all)}장), '
          f'{OUT / "original_features.csv"} (원본 {len(origs)}장)')

    print('\n[원본 52장 특징량]  ( - = 스팟 없음/미검출)')
    print(f"{'사진':<6}{'조건':<14}{'지름':>8}{'고리세기':>9}{'잔류물면적':>11}{'덮임%':>7}{'결정':>6}")
    for r in origs:
        cond = f"{r['chemical']} {r['conc']:g}"
        if r['detected']:
            print(f"{r['channel']}{r['sample']:<5}{cond:<14}{r['diameter_px']:>8}{r['ring_strength']:>9}"
                  f"{r['residue_area_px']:>11}{r['residue_cover_pct']:>7}{r['crystal_count']:>6}")
        else:
            print(f"{r['channel']}{r['sample']:<5}{cond:<14}{'-':>8}{'-':>9}{'-':>11}{'-':>7}{'-':>6}"
                  + ('   (정답엔 링 있음: 미검출)' if r['gt_ring'] else '   (커피링 미관찰)'))
    plot(origs, FIG / 'concentration_features.png')


if __name__ == '__main__':
    main()
