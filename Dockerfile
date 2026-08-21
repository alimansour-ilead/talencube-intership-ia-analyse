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

# ← Diagnostic build-time : confirme la version de mediapipe réellement
# installée et si mediapipe.solutions est bien exposé. Visible dans les
# logs de build Railway — utile pour vérifier sans attendre le runtime.
RUN python -c "import mediapipe; print('mediapipe version:', mediapipe.__version__); print('solutions disponible:', hasattr(mediapipe, 'solutions'))"

# ← Contourne le conflit opencv-python / opencv-contrib-python (mediapipe
# dépend de ce dernier en interne, qui peut écraser les fichiers de
# données lors de l'installation selon l'ordre de résolution pip) en
# récupérant les XML Haar cascade manuellement, peu importe quelle
# variante de cv2 a "gagné" l'installation.
RUN CV2_DATA_DIR=$(python -c "import cv2; print(cv2.data.haarcascades)") && \
    mkdir -p "$CV2_DATA_DIR" && \
    curl -L -o "${CV2_DATA_DIR}haarcascade_frontalface_default.xml" \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml && \
    curl -L -o "${CV2_DATA_DIR}haarcascade_eye.xml" \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml

# Modèles MediaPipe Tasks pour face_analyzer.py (mode professionnel :
# gaze/posture précis). Sans eux, fallback OpenCV automatique — le
# code reste fonctionnel mais moins riche.
RUN mkdir -p models/mediapipe && \
    curl -L -o models/mediapipe/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task && \
    curl -L -o models/mediapipe/pose_landmarker_lite.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

COPY . .

# ← CRITIQUE : Railway fournit le port via $PORT, jamais 8000 en dur
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]