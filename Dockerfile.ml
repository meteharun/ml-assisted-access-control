# ============================================================
# Dockerfile.ml  —  Detection & Classification container
# Contains: PaddleOCR, YOLO, DeBERTa, PyTorch
# ============================================================
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements_ml.txt .
RUN pip install --upgrade pip && \
    pip install matplotlib==3.10.3 \
                opencv-python-headless==4.6.0.66 \
                transformers==4.39.3 \
                sentencepiece==0.2.0 \
                ultralytics==8.3.148 \
                paddlepaddle==2.6.2 && \
    pip install --only-binary=:all: \
                "PyMuPDF==1.24.10" \
                "pdf2docx==0.5.8" && \
    pip install shapely==2.1.1 \
                imgaug==0.4.0 \
                scikit-image==0.25.2 \
                lmdb==1.7.3 \
                pyclipper==1.3.0.post6 \
                protobuf==3.20.3 \
                visualdl==2.5.3 \
                bce-python-sdk==0.9.46 && \
    pip install --no-deps paddleocr==2.6.1.3 && \
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 && \
    pip install --force-reinstall numpy==1.26.4 imgaug==0.4.0

# Copy source code
COPY src/ /app/src/

# Data and model directories are mounted at runtime via docker-compose
# /app/data/samples     — input images
# /app/data/detections  — output JSON written here (shared volume)
# /app/models           — model weights (mounted read-only)

CMD ["python", "src/detect.py", "--help"]
