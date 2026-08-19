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
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Torch en version CPU uniquement — évite plusieurs Go inutiles (pas de GPU sur Railway standard)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir "moviepy<2.0" && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# ← CRITIQUE : Railway fournit le port via $PORT, jamais 8000 en dur
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]