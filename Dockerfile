# Hugging Face Spaces (Docker) 용 설정
FROM python:3.13-slim

# OpenCV 가 쓰는 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face 는 사용자 번호 1000 으로 실행하는 것을 권장
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ResNet-50 사전학습 가중치를 미리 받아 둠 (앱 첫 실행이 빨라짐)
RUN python -c "import torchvision; torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)"

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
