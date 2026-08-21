FROM python:3.11-slim
WORKDIR /app

# Librairies système nécessaires à OpenCV, MediaPipe et à la compilation d'insightface
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Torch en version CPU uniquement — évite plusieurs Go inutiles (pas de GPU sur Railway standard)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir "moviepy<2.0" && \
    pip install --no-cache-dir -r requirements.txt

# Modèles MediaPipe Tasks pour face_analyzer.py (mode professionnel : gaze, posture précis)
# Sans eux, fallback OpenCV automatique — fonctionnel mais moins riche.
RUN mkdir -p models/mediapipe && \
    curl -L -o models/mediapipe/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task && \
    curl -L -o models/mediapipe/pose_landmarker_lite.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

COPY . .

# ← CRITIQUE : Railway fournit le port via $PORT, jamais 8000 en dur
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]