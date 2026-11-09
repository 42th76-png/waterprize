"""커피링 생태독성 스크리닝 화면 (Streamlit)

실행 (프로젝트 폴더에서)
  .venv\\Scripts\\streamlit run src/app.py
  → 브라우저에서 http://localhost:8501 이 열림

순서: 사진 4장(A~D) 올리기 → 탑뷰 변환 → 스팟 탐지 → 물질 판별 → TU 등급 → Grad-CAM
분석 단계는 src/pipeline.py 에 있음
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402

st.set_page_config(page_title='커피링 생태독성 스크리닝', page_icon='💧', layout='wide')

GRADE_COLOR = {'저': '#0ca30c', '중': '#c98500', '고': '#d03b3b'}
CLASS_KO = {'Ctrl': '대조군(독성 물질 없음)', 'Cu': '구리(Cu)', 'NaCl': '염화나트륨(NaCl)', 'SDS': '계면활성제(SDS)'}


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read(path):
    """한글 경로도 읽기 (cv2.imread 는 한글 경로에서 실패함)"""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


@st.cache_resource(show_spinner='모델 불러오는 중 (처음 한 번만, 1분 정도)...')
def get_models(exclude_sample):
    return P.Analyzer(exclude_sample=exclude_sample), P.load_detectors(exclude_sample)


st.title('💧 이중채널 건조패턴 생태독성 스크리닝')
st.caption('시료 한 개의 스팟 사진 4장(A 증류수 · B 탄닌 · C 셀룰로오스 · D 탄닌+셀룰로오스)을 올리면 '
           '탑뷰 변환 → 스팟 탐지 → 독성 물질 판별 → 이론 TU 등급 → Grad-CAM 을 차례로 보여 줍니다.')
st.caption('⚠️ KJWP 2026 고등학생 연구용 시연입니다. 시료 13개로 학습한 모델이라 정확도에 한계가 있고, '
           'TU 는 물벼룩 실측이 아닌 문헌 EC50 으로 계산한 이론값입니다. 실제 수질 판정에 쓰지 마세요.')

# ---------------------------------------------------------------- 옆 메뉴: 설정
with st.sidebar:
    st.header('설정')
    mode = st.radio('올리는 사진 종류', ['원본 촬영 사진 (57°) → 탑뷰 변환', '이미 탑뷰로 변환된 사진 (데이터셋 사진 등)'])
    do_topview = mode.startswith('원본')
    angle, fov = 57.0, 70.0
    if do_topview and not (P.SRC / 'topview_convert.py').exists():
        st.info('src/topview_convert.py 가 없어 촬영 각도로 근사 변환합니다. '
                '파일을 넣으면 자동으로 그 변환을 씁니다.')
        angle = st.slider('촬영 각도 (지면과 이루는 각, °)', 30.0, 90.0, 57.0, 1.0)
        fov = st.slider('카메라 화각 (긴 변 기준, °)', 40.0, 100.0, 70.0, 1.0,
                        help='휴대폰 기본 카메라는 보통 65~80°')
    conf = st.slider('스팟 탐지 확신도 기준', 0.05, 0.9, 0.25, 0.05,
                     help='이보다 확신이 낮은 상자는 스팟이 아니라고 봅니다.')
    st.divider()
    st.markdown('**사용하는 모델** (앞 단계 결정 그대로)\n'
                '- 스팟 탐지: LOSO fold 모델 13개의 과반수 투표\n'
                '- 판별: ResNet-50 특징 + 랜덤포레스트, 스팟 A만 (교차검증 13개 중 8개 정답)\n'
                '- Grad-CAM: 설명용 로지스틱 회귀 판정층 (A~D)\n'
                '- TU: 문헌 EC50 (Cu 0.521, NaCl 4,770, SDS 5.5 mg/L)')

# ---------------------------------------------------------------- 사진 올리기
st.subheader('① 사진 올리기')
cols = st.columns(4)
files = {}
for col, ch in zip(cols, P.CHANNELS):
    with col:
        files[ch] = st.file_uploader(f'{ch} · {P.CHANNEL_NAME[ch]}', type=['jpg', 'jpeg', 'png'], key=f'up_{ch}')

# 예시 사진: 사진이 없는 사람도 해 볼 수 있게 데이터셋 원본(이미 탑뷰)을 골라 쓰기
EXAMPLES = P.ROOT / 'data' / 'examples'
example = None
if EXAMPLES.exists():
    opts = ['사용 안 함'] + [f'시료 {s} ({P.SAMPLE_NAME[s]})' for s in P.SAMPLES
                             if all((EXAMPLES / f'{c}{s}-1.jpg').exists() for c in P.CHANNELS)]
    choice = st.selectbox('또는 예시 사진으로 해 보기 (데이터셋 원본 사진, 이미 탑뷰)', opts)
    if choice != '사용 안 함':
        example = int(choice.split()[1])
        st.image([rgb(read(EXAMPLES / f'{c}{example}-1.jpg')) for c in P.CHANNELS],
                 caption=[f'{c} · {P.CHANNEL_NAME[c]}' for c in P.CHANNELS], width=150)

# 파일 이름이 데이터셋 규칙(A5.jpg, A5-1.jpg)이면 시료 번호를 알아냄 → 데모 모드
samples = {P.parse_name(f.name)[1] for f in files.values() if f is not None}
known = example or (samples.pop() if len(samples) == 1 and None not in samples else None)
demo = False
if known:
    demo = st.checkbox(f'데모 모드: 데이터셋 시료 {known} ({P.SAMPLE_NAME[known]}) 사진으로 보입니다. '
                       f'시료 {known} 을 학습에 쓰지 않은 모델만 사용', value=True,
                       help='학습에 쓴 사진을 학습한 모델로 보면 결과가 실제보다 좋게 나옵니다. '
                            '체크하면 이 시료를 뺀 모델(fold 모델, 판별 모델)로 처음 보는 사진처럼 평가합니다.')

default_conc = float(P.SAMPLES[known][1]) if known else 1.0
conc = st.number_input('시료 농도 (mg/L) — TU 계산용, 모델은 농도를 예측하지 않습니다',
                       min_value=0.0, value=default_conc, step=1.0, format='%g')

ready = bool(example) or all(files.values())
if not ready:
    st.info('A~D 사진 4장을 모두 올리거나, 예시 사진을 고르면 분석을 시작할 수 있습니다.')
if st.button('분석 시작', type='primary', disabled=not ready):
    exclude = known if demo else None
    analyzer, detectors = get_models(exclude)
    if example:  # 예시 사진은 이미 탑뷰라 변환하지 않음
        images = {ch: read(EXAMPLES / f'{ch}{example}-1.jpg') for ch in P.CHANNELS}
        use_topview = False
    else:
        images = {ch: P.decode_image(f.getvalue()) for ch, f in files.items()}
        use_topview = do_topview
    if any(im is None for im in images.values()):
        st.error('사진을 읽지 못했습니다. jpg/png 파일인지 확인하세요.')
        st.stop()
    with st.spinner('분석 중...'):
        st.session_state['res'] = P.run_pipeline(images, conc, analyzer, detectors, use_topview, conf,
                                                 camera_angle_deg=angle, hfov_deg=fov)
        st.session_state['res_meta'] = {'demo': demo, 'known': known}

res = st.session_state.get('res')
if res:
    meta = st.session_state['res_meta']
    if meta['demo']:
        st.success(f"데모 모드: 시료 {meta['known']} 을 뺀 모델로 분석했습니다 (정답: {P.SAMPLES[meta['known']][0]}).")

    # ② 탑뷰 + ③ 스팟 탐지
    st.subheader('② 탑뷰 변환 → ③ 스팟 탐지')
    st.caption(f"탑뷰: {res['channels']['A']['topview_how']}")
    cols = st.columns(4)
    for col, ch in zip(cols, P.CHANNELS):
        c = res['channels'][ch]
        d = c['det']
        with col:
            st.image(rgb(c['boxed']), caption=f'{ch} · {P.CHANNEL_NAME[ch]}', width='stretch')
            if d['detected']:
                st.markdown(f"✅ 스팟 찾음 · 확신도 {d['conf']:.2f} · 모델 {d['votes']}/{d['n_models']}개 동의")
                st.image(rgb(c['crop']), caption='잘라낸 스팟', width='stretch')
            else:
                st.markdown(f"⬜ 스팟 없음 (모델 {d['votes']}/{d['n_models']}개만 찾음) → '커피링 미관찰'로 처리")

    # 특징량 표
    st.markdown('**스팟 특징량**')
    table = []
    for ch in P.CHANNELS:
        f = res['channels'][ch]['features']
        table.append({'채널': f'{ch} {P.CHANNEL_NAME[ch]}',
                      '침착 지름 (px)': f['diameter_px'] if f else '-',
                      '고리 세기': f['ring_strength'] if f else '-',
                      '잔류물 면적 (px²)': f['residue_area_px'] if f else '-',
                      '결정 개수': f['crystal_count'] if f else '-'})
    st.dataframe(table, hide_index=True, width='stretch')

    # ④ 판별 + ⑤ TU
    left, right = st.columns([3, 2])
    with left:
        st.subheader('④ 독성 물질 판별')
        st.markdown(f"### {CLASS_KO[res['pred']]}")
        for k, cls in enumerate(P.CLASSES):
            st.progress(float(res['proba'][k]), text=f'{CLASS_KO[cls]}: {res["proba"][k]:.2f}')
        st.caption('판별에는 A(증류수) 스팟만 사용합니다 (교차검증에서 가장 좋았던 방식, 13개 중 8개 정답). '
                   '대조군 시료가 1개뿐이라 대조군 판별은 특히 믿기 어렵습니다.')
    with right:
        st.subheader('⑤ 이론 TU 등급')
        tu = res['tu']
        color = GRADE_COLOR[tu['grade']]
        st.markdown(f"<div style='font-size:3rem;font-weight:700;color:{color}'>{tu['grade']}</div>",
                    unsafe_allow_html=True)
        st.metric('TU (농도 ÷ EC50)', f"{tu['tu']:.3g}")
        ec50 = f"{tu['ec50_mg_L']:g} mg/L" if tu['ec50_mg_L'] else '없음 (대조군 TU = 0)'
        st.caption(f"판별 결과 {res['pred']} · 입력 농도 {tu['conc_mg_L']:g} mg/L · EC50 {ec50}\n\n"
                   '등급: 저 TU < 1 / 중 1~10 / 고 ≥ 10. 판별이 틀리면 TU 도 틀립니다 '
                   '(특히 Cu 로 잘못 판별하면 EC50 이 작아 TU 가 크게 부풀려짐).')

    # ⑥ Grad-CAM
    st.subheader('⑥ Grad-CAM: 어느 부분을 보고 판단했나')
    pe = res['proba_explainer']
    st.caption(f"빨간색일수록 판별 결과({res['pred']}) 쪽으로 판단을 끌어당긴 부분입니다. "
               f"설명용 판정층(A~D)의 자체 예측: {P.CLASSES[int(pe.argmax())]} ({pe.max():.2f}). "
               '히트맵이 스팟이 아닌 배경에 몰려 있으면, 패턴보다 배경·밝기에 끌린 판단일 수 있습니다.')
    cols = st.columns(4)
    for col, ch in zip(cols, P.CHANNELS):
        c = res['channels'][ch]
        with col:
            if c['gradcam'] is not None:
                st.image(rgb(c['gradcam']), caption=f'{ch} · {P.CHANNEL_NAME[ch]}', width='stretch')
            else:
                st.markdown(f'{ch} · 스팟 없음')
