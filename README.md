---
title: Coffee Ring Ecotox Screening
emoji: 💧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 이중채널 건조패턴 생태독성 스크리닝 (KJWP 2026)

시료 한 개의 커피링 스팟 사진 4장(A 증류수 · B 탄닌 · C 셀룰로오스 · D 탄닌+셀룰로오스)을 올리면
**탑뷰 변환 → 스팟 탐지 → 독성 물질 판별 → 이론 TU 등급 → Grad-CAM** 을 차례로 보여 주는 화면입니다.
사진이 없으면 화면에서 데이터셋 예시 시료를 골라 해 볼 수 있습니다.

> ⚠️ 고등학생 연구용 시연입니다. 시료 13개로 학습한 모델이라 정확도에 한계가 있고,
> TU 는 물벼룩 실측이 아닌 문헌 EC50 으로 계산한 이론값입니다. 실제 수질 판정에 쓰지 마세요.

## 구성
| 단계 | 방법 |
|---|---|
| 탑뷰 변환 | 휴대폰-지면 57°, 약 11 cm, Galaxy S24 Ultra 메인 렌즈 기준 원근 변환 (`src/topview_convert.py`) |
| 스팟 탐지 | YOLOv8n, 시료 단위 leave-one-sample-out fold 모델 13개의 과반수 투표 (`runs/spot/foldXX`) |
| 물질 판별 | ResNet-50(ImageNet) 특징 + 랜덤포레스트, 스팟 A만 (교차검증 13개 시료 중 8개 정답) |
| TU 등급 | TU = 농도 ÷ 문헌 EC50 (Cu 0.521, NaCl 4,770, SDS 5.5 mg/L), 저 < 1 ≤ 중 < 10 ≤ 고 (Persoone et al. 2003) |
| Grad-CAM | 같은 ResNet-50 특징(A~D) 위의 설명용 로지스틱 회귀 판정층 |

EC50 출처: Cu — Kang et al. (2022) Ecotoxicol. Environ. Saf. 241:113817 / NaCl — Mount et al. (1997)
Environ. Toxicol. Chem. 16:2009–2019 / SDS — Dyer et al. (1996), ECHA 등록자료 인용

## 직접 실행
```
pip install -r requirements.txt
streamlit run src/app.py
```
