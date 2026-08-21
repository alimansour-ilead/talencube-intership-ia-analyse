FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ← Casse le cache Docker à partir d'ici — change la valeur ci-dessous
# à chaque fois qu'un rebuild complet est nécessaire (ex: nouvelles
# dépendances système, nouveau fichier .task, etc.), même si
# requirements.txt lui-même n'a pas changé.
ARG CACHEBUST=1

COPY requirements.txt .

RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir "moviepy<2.0" && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "import mediapipe; print('mediapipe version:', mediapipe.__version__); print('solutions disponible:', hasattr(mediapipe, 'solutions'))"

RUN CV2_DATA_DIR=$(python -c "import cv2; print(cv2.data.haarcascades)") && \
    mkdir -p "$CV2_DATA_DIR" && \
    curl -L -o "${CV2_DATA_DIR}haarcascade_frontalface_default.xml" \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml && \
    curl -L -o "${CV2_DATA_DIR}haarcascade_eye.xml" \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml

RUN mkdir -p models/mediapipe && \
    curl -L -o models/mediapipe/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task && \
    curl -L -o models/mediapipe/pose_landmarker_lite.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]