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
    unzip \
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

# ← AJOUT : préchargement des modèles InsightFace À LA CONSTRUCTION de
# l'image, au lieu de les laisser se télécharger au démarrage du
# conteneur (comportement par défaut d'insightface, sans ligne
# explicite ici auparavant — contrairement aux modèles MediaPipe et
# aux cascades OpenCV ci-dessus, déjà préchargés de cette façon).
#
# Confirmé en production : buffalo_m échouait systématiquement à se
# charger sur Railway (probablement un timeout ou un souci réseau au
# démarrage — plus probable pour ce modèle, plus volumineux que
# buffalo_sc, qui lui réussissait à charger). Précharger ici élimine
# complètement cette dépendance réseau au runtime : le modèle est
# déjà présent sur le disque de l'image avant même que le conteneur
# ne démarre, quelle que soit la condition réseau de Railway à ce
# moment précis.
#
# buffalo_sc reste aussi préchargé ici par cohérence (déjà fonctionnel
# via le téléchargement au runtime, mais autant fiabiliser les deux
# de la même façon et gagner quelques secondes au démarrage).
RUN mkdir -p /root/.insightface/models && \
    curl -L -o /tmp/buffalo_sc.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip && \
    mkdir -p /tmp/extract_sc && \
    unzip -q /tmp/buffalo_sc.zip -d /tmp/extract_sc && \
    mkdir -p /root/.insightface/models/buffalo_sc && \
    find /tmp/extract_sc -name "*.onnx" -exec mv {} /root/.insightface/models/buffalo_sc/ \; && \
    rm -rf /tmp/buffalo_sc.zip /tmp/extract_sc && \
    curl -L -o /tmp/buffalo_m.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip && \
    mkdir -p /tmp/extract_m && \
    unzip -q /tmp/buffalo_m.zip -d /tmp/extract_m && \
    mkdir -p /root/.insightface/models/buffalo_m && \
    find /tmp/extract_m -name "*.onnx" -exec mv {} /root/.insightface/models/buffalo_m/ \; && \
    rm -rf /tmp/buffalo_m.zip /tmp/extract_m && \
    echo "Modèles InsightFace préchargés avec succès" && \
    echo "--- buffalo_sc ---" && ls -la /root/.insightface/models/buffalo_sc/ && \
    echo "--- buffalo_m ---" && ls -la /root/.insightface/models/buffalo_m/

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]