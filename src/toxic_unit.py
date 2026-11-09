"""이론 독성 단위(TU) 계산과 저/중/고 등급 매기기

TU(Toxic Unit, 독성 단위) = 물질 농도 ÷ 그 물질의 EC50
  - EC50: 물벼룩의 50%가 영향을 받는 농도 (문헌값)
  - TU = 1 이면 농도가 딱 EC50 만큼, TU = 10 이면 EC50 의 10배 → 독성이 강함
  - 이 프로젝트의 TU 는 물벼룩 실측이 아니라 '문헌 EC50 으로 계산한 이론값'

등급 (기본값, Persoone et al. 2003 의 5단계 분류를 3단계로 묶음, 원문 기준값 확인 완료)
  저: TU < 1          (급성 독성 없음 ~ 약간)
  중: 1 ≤ TU < 10     (급성 독성)
  고: TU ≥ 10         (높은 급성 독성)
  → 기준을 바꾸려면 tu_grade(tu, low_max=..., mid_max=...)

사용 예 (다른 코드에서)
  from toxic_unit import assess
  assess('Cu', 10)   → {'chemical': 'Cu', 'conc_mg_L': 10, 'ec50_mg_L': 0.521, 'tu': 19.19, 'grade': '고'}

직접 실행하면 (프로젝트 폴더에서)
  python src/toxic_unit.py
  python src/toxic_unit.py --method ResNet50+RF --spots A만
  1) 13개 시료의 정답 TU·등급 표
  2) 분류 모델이 예측한 물질 종류 + 실험 농도로 계산한 TU·등급, 정답 등급과 비교
  → results/tu/tu_table.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 문헌 EC50 (mg/L). 시료 농도와 같은 기준(물질 형태)이어야 TU 가 맞음
EC50 = {
    'Cu': {'ec50_mg_L': 0.521, 'basis': '황산구리', 'organism': 'Daphnia magna',
           'source': 'Kang et al. (2022), Ecotoxicology and Environmental Safety 241: 113817',
           'note': '구리 EC50 은 수질(경도·유기물 등)에 따라 크게 달라짐. 논문 시험수 조건의 값을 사용. '
                   '시료 농도(Cu 0.1~50 mg/L)도 황산구리 기준이라 단위 변환 없이 그대로 나눔 (2026-11-09 확인)'},
    'NaCl': {'ec50_mg_L': 4770.0, 'basis': 'NaCl', 'organism': 'Daphnia magna',
             'source': 'Mount et al. (1997), Environmental Toxicology and Chemistry 16: 2009-2019',
             'note': ''},
    'SDS': {'ec50_mg_L': 5.5, 'basis': 'SDS', 'organism': 'Ceriodaphnia dubia (물벼룩류)',
            'source': 'Dyer et al. (1996), ECHA 등록자료 인용',
            'note': 'D. magna 가 아닌 C. dubia 값'},
}
CONTROL = 'Ctrl'  # 대조군: 독성 물질 없음 → TU = 0

# 시료 번호 → (물질, 실험 농도 mg/L)
SAMPLES = {1: ('SDS', 10), 2: ('Cu', 1), 3: ('NaCl', 3000), 4: ('Ctrl', 0), 5: ('Cu', 50),
           6: ('SDS', 1), 7: ('NaCl', 100), 8: ('Cu', 0.1), 9: ('SDS', 100), 10: ('NaCl', 10000),
           11: ('Cu', 10), 12: ('SDS', 30), 13: ('NaCl', 1000)}


def theoretical_tu(chemical, conc_mg_L):
    """물질 종류와 농도(mg/L)로 이론 TU 계산. 대조군은 0"""
    if conc_mg_L < 0:
        raise ValueError(f'농도는 0 이상이어야 합니다: {conc_mg_L}')
    if chemical == CONTROL:
        return 0.0
    if chemical not in EC50:
        raise ValueError(f'EC50 이 없는 물질입니다: {chemical} (가능: {CONTROL}, {", ".join(EC50)})')
    return conc_mg_L / EC50[chemical]['ec50_mg_L']


def tu_grade(tu, low_max=1.0, mid_max=10.0):
    """TU → '저' / '중' / '고'. 경계값은 위 등급으로 (TU 가 정확히 1 이면 '중')"""
    if tu < low_max:
        return '저'
    if tu < mid_max:
        return '중'
    return '고'


def assess(chemical, conc_mg_L, low_max=1.0, mid_max=10.0):
    """물질·농도 → TU 와 등급을 한 번에"""
    tu = theoretical_tu(chemical, conc_mg_L)
    return {'chemical': chemical, 'conc_mg_L': conc_mg_L,
            'ec50_mg_L': EC50[chemical]['ec50_mg_L'] if chemical in EC50 else None,
            'tu': round(tu, 3), 'grade': tu_grade(tu, low_max, mid_max)}


def main():
    ap = argparse.ArgumentParser(description='이론 TU 계산과 저/중/고 등급')
    ap.add_argument('--method', default='ResNet50+RF', help='predictions.csv 의 분류 방법 (RF, SVM, ResNet50+RF)')
    ap.add_argument('--spots', default='A만', help='A만 또는 A~D')
    ap.add_argument('--low-max', type=float, default=1.0, help='이 값 미만이면 저')
    ap.add_argument('--mid-max', type=float, default=10.0, help='이 값 미만이면 중, 이상이면 고')
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    print('[사용한 EC50]')
    for c, e in EC50.items():
        print(f"  {c:<5} {e['ec50_mg_L']:>8g} mg/L ({e['basis']}, {e['organism']}) - {e['source']}")
    print(f'[등급] 저: TU < {args.low_max:g} / 중: {args.low_max:g} ≤ TU < {args.mid_max:g} / 고: TU ≥ {args.mid_max:g}\n')

    pred_csv = ROOT / 'results' / 'classification' / 'predictions.csv'
    preds = {}
    if pred_csv.exists():
        for r in csv.DictReader(open(pred_csv, encoding='utf-8-sig')):
            if r['method'] == args.method and r['spots'] == args.spots:
                preds[int(r['sample'])] = r['pred']
    if not preds:
        print(f'[주의] {pred_csv} 에서 {args.method}/{args.spots} 예측을 찾지 못해 정답 TU 만 계산합니다.')

    rows = []
    print(f"{'시료':<12}{'농도':>8}{'정답TU':>9}{'등급':>4}   {'예측종류':<6}{'예측TU':>9}{'등급':>4}  일치")
    for s, (chem, conc) in SAMPLES.items():
        t = assess(chem, conc, args.low_max, args.mid_max)
        row = {'sample': s, 'chemical': chem, 'conc_mg_L': conc, 'tu_true': t['tu'], 'grade_true': t['grade']}
        line = f"{s:>2} {chem + str(conc if chem != CONTROL else ''):<9}{conc:>8g}{t['tu']:>9.3f}{t['grade']:>4}"
        if s in preds:
            # 예측한 물질 종류 + 실험 농도 (현재 모델은 농도를 예측하지 않음)
            p = assess(preds[s], conc, args.low_max, args.mid_max)
            row.update(pred_chemical=preds[s], tu_pred=p['tu'], grade_pred=p['grade'],
                       grade_match=int(p['grade'] == t['grade']))
            line += f"   {preds[s]:<6}{p['tu']:>9.3f}{p['grade']:>4}  {'O' if row['grade_match'] else 'x'}"
        rows.append(row)
        print(line)

    if preds:
        n_ok = sum(r['grade_match'] for r in rows)
        n_type = sum(r['pred_chemical'] == r['chemical'] for r in rows)
        print(f'\n물질 종류 정답 {n_type}/13, TU 등급 일치 {n_ok}/13 ({args.method}, 스팟 {args.spots})')
        print('※ 농도는 실험 농도를 그대로 사용 (모델이 예측한 값 아님)')
    out = ROOT / 'results' / 'tu'
    out.mkdir(parents=True, exist_ok=True)
    keys = ['sample', 'chemical', 'conc_mg_L', 'tu_true', 'grade_true', 'pred_chemical', 'tu_pred',
            'grade_pred', 'grade_match']
    with open(out / 'tu_table.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'저장: {out / "tu_table.csv"}')


if __name__ == '__main__':
    main()
