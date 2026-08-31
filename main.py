from fastapi import (FastAPI, File, UploadFile, BackgroundTasks,
                     Form, WebSocket, WebSocketDisconnect)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ultralytics import YOLO
from transformers import (AutoImageProcessor, AutoModelForImageClassification)
import warnings
import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import json
from datetime import datetime
from collections import deque
from pathlib import Path
from urllib.parse import urlparse
import moviepy.editor as mp
import soundfile as sf
from models.hubert_model import SpeechEmotionHuBERT
import tempfile
import imageio_ffmpeg as ffmpeg_pkg
from fastapi import Form as FastAPIForm
import subprocess
import asyncio
import functools
import base64
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import (create_engine, Column, Integer, String,
                        DateTime, LargeBinary, Text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tracking_manager import TrackingManager, State, CFG
from face_analyzer import FaceAnalyzer  
from patches_v7 import (
    calculate_behavioral_tension_signals, analyze_speech_patterns,
    compute_metric_uncertainty, diagnose_absence_reason,
    ABSENCE_REASON_LABELS_FR,
)
from au_analyzer import (
    load_au_detector, correct_emotion_probs, MODEL_STATUS_AU,
)
import builtins
import time as _time_module
from datetime import datetime
import re
# ═══════════════════════════════════════════════════════════════════
# HORODATAGE AUTOMATIQUE DE TOUS LES LOGS (pour synchronisation debug)
# ═══════════════════════════════════════════════════════════════════
import builtins
import time as _time_module
_original_print = builtins.print
def _timestamped_print(*args, **kwargs):
    ts = _time_module.strftime("%H:%M:%S.") + f"{int(_time_module.time()*1000)%1000:03d}"
    _original_print(f"[{ts}]", *args, **kwargs)

builtins.print = _timestamped_print
# ═══════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):          # ← CORRECTION numpy bool
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    return obj


warnings.filterwarnings('ignore')

# ── FFmpeg ───────────────────────────────────────────────────────
FFMPEG_PATH = ffmpeg_pkg.get_ffmpeg_exe()

# ═══════════════════════════════════════════════════════════════════
# APPLICATION FASTAPI
# ═══════════════════════════════════════════════════════════════════
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Emotion Analysis API - TalenCube")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_url_video_cache = {}

# ═══════════════════════════════════════════════════════════════════
# BASE DE DONNÉES
# ═══════════════════════════════════════════════════════════════════
DATABASE_URL = "sqlite:///./videos.db"
engine       = create_engine(DATABASE_URL,
                              connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class VideoRecord(Base):
    __tablename__ = "videos"
    id              = Column(Integer, primary_key=True, index=True)
    url             = Column(String, nullable=False)
    filename        = Column(String, nullable=False)
    video_data      = Column(LargeBinary, nullable=True)
    created_at      = Column(DateTime, default=datetime.now)
    analysis_result = Column(Text, nullable=True)


Base.metadata.create_all(bind=engine)


class VideoURLRequest(BaseModel):
    url:                 str
    filename:            Optional[str]  = None
    store_in_db:         Optional[bool] = True
    skip_face_detection: Optional[bool] = False
    public_id:           Optional[str]  = None


# ═══════════════════════════════════════════════════════════════════
# INITIALISATION MODÈLES
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("INITIALISATION DES MODELES D'ANALYSE EMOTIONNELLE")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class Config:
    MODEL_PATH         = "models/emotion_model.pth"
    HISTORY_PATH       = "data/analysis_history.json"
    TRAINING_DATA_PATH = "data/training_data"
    BATCH_SIZE         = 32
    LEARNING_RATE      = 2e-5
    NUM_EPOCHS         = 10
    CONFIDENCE_THRESHOLD = 0.6


Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)


class EnhancedEmotionModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, pixel_values):
        return self.base_model(pixel_values).logits

    def extract_features(self, pixel_values):
        outputs = self.base_model(pixel_values, output_hidden_states=True)
        return outputs.hidden_states[-1][:, 0, :]


class EmotionDataset(Dataset):
    def __init__(self, data_dir, processor, transform=None):
        self.data           = []
        self.processor      = processor
        self.transform      = transform
        self.emotion_labels = ['sad', 'disgust', 'angry', 'neutral',
                               'fear', 'surprise', 'happy']
        if os.path.exists(data_dir):
            self._load_data(data_dir)

    def _load_data(self, data_dir):
        for emotion_idx, emotion in enumerate(self.emotion_labels):
            emotion_dir = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_dir):
                for img_file in os.listdir(emotion_dir):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        self.data.append({
                            'path':  os.path.join(emotion_dir, img_file),
                            'label': emotion_idx
                        })
        print(f"Dataset chargé: {len(self.data)} images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        image = cv2.imread(item['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        inputs       = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze()
        return {'pixel_values': pixel_values,
                'labels':       torch.tensor(item['label'], dtype=torch.long)}

MODEL_STATUS: Dict[str, bool] = {}

# ── Modèle ViT ───────────────────────────────────────────────────
print("Chargement du modèle ViT...")
base_processor = None
base_model     = None
model          = None
try:
    BASE_MODEL     = "dima806/facial_emotions_image_detection"
    base_processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    base_model     = AutoModelForImageClassification.from_pretrained(BASE_MODEL)
    model          = EnhancedEmotionModel(base_model).to(device)

    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=device))
        print("Modèle chargé depuis l'entraînement précédent")
    else:
        print("Utilisation du modèle de base")
    model.eval()
    MODEL_STATUS["vit"] = True
    print("[ViT] ✅ Chargé")
except Exception as e:
    MODEL_STATUS["vit"] = False
    print(f"[ViT] ❌ Échec chargement: {e} — analyse d'émotion désactivée")

# ── YOLO ─────────────────────────────────────────────────────────
yolo_model = None
try:
    for yolo_name, yolo_url in [
        ("yolov8s-face.pt",
         "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt"),
        ("yolov8n-face.pt",
         "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"),
    ]:
        if not os.path.exists(yolo_name):
            print(f"Téléchargement {yolo_name}...")
            try:
                import urllib.request
                req = urllib.request.Request(
                    yolo_url, headers={'User-Agent': 'Mozilla/5.0'})
                with (urllib.request.urlopen(req) as resp,
                      open(yolo_name, 'wb') as f):
                    f.write(resp.read())
                print(f"✅ {yolo_name} téléchargé")
            except Exception as e:
                print(f"Échec téléchargement {yolo_name}: {e}")

    if os.path.exists("yolov8s-face.pt"):
        yolo_model = YOLO("yolov8s-face.pt")
        print("[YOLO] ✅ v8s-face chargé (précision améliorée)")
    elif os.path.exists("yolov8n-face.pt"):
        yolo_model = YOLO("yolov8n-face.pt")
        print("[YOLO] ✅ v8n-face chargé (standard)")
    else:
        yolo_model = YOLO("yolov8n.pt")
        print("[YOLO] ⚠️ v8n généraliste (fallback)")
    # ← OPTIMISATION PERF : calculé une seule fois ici, au lieu de faire
    # deux appels os.path.exists() (I/O disque) à CHAQUE frame dans
    # detect_faces() — ces fichiers ne changent jamais après le
    # démarrage du serveur.
    IS_FACE_MODEL = (os.path.exists("yolov8s-face.pt") or
                      os.path.exists("yolov8n-face.pt"))
    MODEL_STATUS["yolo"] = True
except Exception as e:
    MODEL_STATUS["yolo"] = False
    IS_FACE_MODEL = False
    print(f"[YOLO] ❌ Échec chargement: {e} — détection de visage désactivée")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
print("YOLO chargé" if MODEL_STATUS.get("yolo") else "YOLO indisponible")

print("Chargement HuBERT...")
audio_model     = None
audio_processor = None
try:
    audio_model     = SpeechEmotionHuBERT(num_classes=7).to(device)
    audio_processor = audio_model.feature_extractor
    MODEL_STATUS["hubert"] = True
    print("[HuBERT] ✅ Chargé")
except Exception as e:
    MODEL_STATUS["hubert"] = False
    print(f"[HuBERT] ❌ Échec chargement: {e} — analyse audio désactivée")

MODEL_STATUS["au_pyfeat"] = load_au_detector()
# ── ONNX Runtime ─────────────────────────────────────────────────
import onnxruntime as ort

print("Chargement ONNX Runtime...")
try:
    vit_session    = ort.InferenceSession(
        "models/vit_emotion.onnx",
        providers=['CPUExecutionProvider']
    )
    hubert_session = ort.InferenceSession(
        "models/hubert_audio.onnx",
        providers=['CPUExecutionProvider']
    )
    USE_ONNX = True
    print("[ONNX] Activé — vitesse x6")
except Exception as e:
    vit_session = hubert_session = None
    USE_ONNX    = False
    print(f"[ONNX] Non disponible ({e}) — fallback PyTorch")

# ── Labels & Couleurs ─────────────────────────────────────────────
EMOTION_LABELS   = ['sad', 'disgust', 'angry', 'neutral',
                    'fear', 'surprise', 'happy']

EMOTION_NAMES_FR = {
    'sad':      'Mélancolie',
    'disgust':  'Inconfort',
    'angry':    'Tension',
    'neutral':  'Neutre',
    'fear':     'Appréhension',
    'surprise': 'Surprise',
    'happy':    'Joie'
}

EMOTION_EMOJIS = {
    'sad': '[SAD]', 'disgust': '[DISGUST]', 'angry': '[ANGRY]',
    'neutral': '[NEUTRAL]', 'fear': '[FEAR]',
    'surprise': '[SURPRISE]', 'happy': '[HAPPY]'
}

EMOTION_COLORS = {
    'sad': '#2196f3', 'disgust': '#795548', 'angry': '#f44336',
    'neutral': '#9e9e9e', 'fear': '#9c27b0',
    'surprise': '#ff9800', 'happy': '#4caf50'
}

EMOTION_THRESHOLDS = {
    'happy':    0.16,
    'surprise': 0.16,
    'sad':      0.35,
    'angry':    0.45,
    'fear':     0.42,
    'disgust':  0.45,
    'neutral':  0.0
}

# ── ArcFace PARTAGÉ ───────────────────────────────────────────────
print("Chargement ArcFace (instance partagée)...")
try:
    from insightface.app import FaceAnalysis as _SharedFA
    import torch as _torch

    _has_gpu = _torch.cuda.is_available()

    if _has_gpu:
        _arc_model = 'buffalo_l'
        _providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print(f"[ArcFace] GPU → buffalo_l (99% précision, ~15ms)")
    else:
        _arc_model = 'buffalo_sc'
        _providers = ['CPUExecutionProvider']
        print(f"[ArcFace] CPU → buffalo_sc (88% précision, ~50ms) "
              f"[buffalo_m disponible mais trop lent sur CPU]")

    try:
        shared_arcface = _SharedFA(name=_arc_model, providers=_providers)
        shared_arcface.prepare(
            ctx_id=0 if _has_gpu else -1, det_size=(640, 640))
        print(f"[ArcFace] ✅ {_arc_model} chargé")
    except Exception as _e1:
        print(f"[ArcFace] ⚠️ {_arc_model} indisponible ({_e1}) → fallback buffalo_sc")
        shared_arcface = _SharedFA(name='buffalo_sc',
                                   providers=['CPUExecutionProvider'])
        shared_arcface.prepare(ctx_id=0, det_size=(640, 640))
        print("[ArcFace] ✅ buffalo_sc chargé")
except Exception as e:
    shared_arcface = None
    MODEL_STATUS["arcface"] = False
    print(f"[ArcFace] ⚠️ Indisponible ({e})")
else:
    MODEL_STATUS["arcface"] = shared_arcface is not None

preview_arcface = shared_arcface

# ── ArcFace RAPIDE (PATCH v6.4) ────────────────────────────────────
try:
    if shared_arcface is not None:
        fast_arcface = _SharedFA(name=_arc_model, providers=_providers)
        fast_arcface.prepare(
            ctx_id=0 if _has_gpu else -1, det_size=(320, 320))
        print(f"[ArcFace] ✅ Instance rapide {_arc_model} chargée "
              f"(det_size=320, verify() temps réel)")
    else:
        fast_arcface = None
except Exception as e:
    fast_arcface = None
    print(f"[ArcFace] ⚠️ Instance rapide indisponible ({e}) — "
          f"fallback sur l'instance standard pour verify()")

# ── FaceAnalyzer MediaPipe ────────────────────────────────────────
print("Chargement FaceAnalyzer (MediaPipe)...")
try:
    face_analyzer_global = FaceAnalyzer()
    MODEL_STATUS["mediapipe"] = face_analyzer_global.enabled
except Exception as e:
    MODEL_STATUS["mediapipe"] = False
    print(f"[FaceAnalyzer] ❌ Échec chargement: {e} — "
          f"analyse comportementale (regard/posture) désactivée")
   
    class _DisabledFaceAnalyzer:
        enabled = False
        def analyze(self, *a, **k): return None
        def reset(self): pass
        def get_boost_params(self, *a, **k): return {}
        def get_result_dict(self, *a, **k): return {}
    face_analyzer_global = _DisabledFaceAnalyzer()

# ← SUPPRIMÉ : diarisation (pyannote) — n'était utile que pour annoter
# la transcription par locuteur. Sans transcription, cette étape ne
# sert plus à rien.

# ── Historique ────────────────────────────────────────────────────
class AnalysisHistory:
    def __init__(self, max_size=1000):
        self.history = deque(maxlen=max_size)
        self.load()

    def add(self, data):
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'data':      data
        })
        self.save()

    def save(self):
        with open(Config.HISTORY_PATH, 'w') as f:
            json.dump(list(self.history), f, indent=2)

    def load(self):
        if os.path.exists(Config.HISTORY_PATH):
            with open(Config.HISTORY_PATH, 'r') as f:
                self.history.extend(json.load(f))

    def get_stats(self):
        return {
            'total_analyses': len(self.history),
            'last_analysis':  (self.history[-1]['timestamp']
                               if self.history else None)
        }


history_manager = AnalysisHistory()
executor        = ThreadPoolExecutor(max_workers=4)

# ← CORRECTION : pool DÉDIÉ, séparé, pour le traitement vidéo lourd
# (extract_candidates_preview, analyze_video). Avant cette séparation,
# ces deux endpoints partageaient le même pool `executor` que
# ws_analyze_realtime (décodage de frame + traitement audio par
# connexion temps réel). Un traitement vidéo long (30s à plusieurs
# minutes) pouvait alors monopoliser les 4 threads du pool partagé,
# mettant en file d'attente — donc "bloquant" en pratique — le
# décodage des frames et l'audio des sessions WebSocket actives.
# Avec un pool séparé, le traitement vidéo lourd ne peut plus jamais
# affamer le temps réel, quelle que soit sa durée.
video_processing_executor =ThreadPoolExecutor(max_workers=2)

print("=" * 60)
print("TOUS LES MODELES SONT PRETS!")
print("=" * 60)



def compute_dynamic_threshold(brightness: float, blur: float) -> float:
    """
    Calcule le seuil ArcFace adaptatif selon la qualité de l'image
    (luminosité + netteté). Plus l'image est sombre/surexposée/floue,
    plus le seuil de similarité exigé est abaissé — pour ne pas
    rejeter à tort le bon candidat sur une frame de mauvaise qualité
    passagère.
    """
    if brightness < 60:
        light_factor = 0.10
    elif brightness > 190:
        light_factor = 0.10
    elif brightness > 150:
        light_factor = 0.05
    else:
        light_factor = 0.0

    if blur < 50:
        blur_factor = 0.08
    elif blur < 100:
        blur_factor = 0.04
    else:
        blur_factor = 0.0

    return max(0.25, 0.38 - light_factor - blur_factor)


def detect_jump(cx: float, cy: float,
                locked_cx: Optional[float], locked_cy: Optional[float],
                frame_width: float) -> bool:
    """
    Détecte un "saut spatial suspect" — un déplacement du candidat
    trop important pour être un mouvement normal, plus probablement
    un changement de plan / bascule de locuteur. Déclenche une
    vérification ArcFace stricte immédiate côté appelant.
    Même règle, auparavant dupliquée entre ws_analyze_realtime et
    _analyze_video_sync (JUMP_DISTANCE_RATIO vérifié séparément aux
    deux endroits).
    """
    if locked_cx is None or locked_cy is None:
        return False
    jump_dist = np.hypot(cx - locked_cx, cy - locked_cy)
    return jump_dist > (CFG.JUMP_DISTANCE_RATIO * frame_width)


def detect_faces(frame):
    """
    Détection YOLO 320px (rapide) + Haar cascade fallback.
    """
    faces    = []
    img_h, img_w = frame.shape[:2]
    is_face_model = IS_FACE_MODEL

    results = yolo_model(frame, imgsz=320, conf=0.22, verbose=False)
    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.22:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_w, x2); y2 = min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            if is_face_model:
                w, h   = x2 - x1, y2 - y1
                pad_w  = int(w * 0.40)
                pad_h  = int(h * 0.40)
                x1_p   = max(0, x1 - pad_w)
                y1_p   = max(0, y1 - int(pad_h * 1.2))
                x2_p   = min(img_w, x2 + pad_w)
                y2_p   = min(img_h, y2 + int(pad_h * 1.5))
                face_img = frame[y1_p:y2_p, x1_p:x2_p]
                if face_img.size > 0:
                    faces.append((face_img, conf,
                                  (x1, y1, x2, y2),
                                  (x1_p, y1_p, x2_p, y2_p)))
            else:
                if cls == 0:
                    person_roi = frame[y1:y2, x1:x2]
                    if person_roi.size > 0:
                        gray = cv2.cvtColor(
                            person_roi, cv2.COLOR_BGR2GRAY)
                        dfs  = face_cascade.detectMultiScale(
                            gray, 1.05, 3, minSize=(30, 30))
                        if len(dfs) > 0:
                            dfs = sorted(dfs,
                                key=lambda x: x[2]*x[3], reverse=True)
                            fx, fy, fw, fh = dfs[0]
                            pad_w = int(fw * 0.40)
                            pad_h = int(fh * 0.40)
                            x1_p  = max(0, x1+fx-pad_w)
                            y1_p  = max(0, y1+fy-int(pad_h*1.2))
                            x2_p  = min(img_w, x1+fx+fw+pad_w)
                            y2_p  = min(img_h, y1+fy+fh+int(pad_h*1.5))
                            tight = (x1+fx, y1+fy, x1+fx+fw, y1+fy+fh)
                            faces.append((frame[y1_p:y2_p, x1_p:x2_p],
                                          conf, tight,
                                          (x1_p, y1_p, x2_p, y2_p)))

    if not faces:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dfs  = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(35, 35))
        for (x, y, w, h) in dfs:
            pad_w  = int(w * 0.40)
            pad_h  = int(h * 0.40)
            x1_p   = max(0, x - pad_w)
            y1_p   = max(0, y - int(pad_h * 1.2))
            x2_p   = min(img_w, x + w + pad_w)
            y2_p   = min(img_h, y + h + int(pad_h * 1.5))
            tight  = (x, y, x+w, y+h)
            faces.append((frame[y1_p:y2_p, x1_p:x2_p],
                          0.65, tight, (x1_p, y1_p, x2_p, y2_p)))
    return faces


def preprocess_face(face):
    return face


def calibrate_and_smooth_probs(probs, prev_probs=None):
    probs = np.where(probs < 0.05, 0.0, probs)
    probs = probs / np.sum(probs) if np.sum(probs) > 0 \
        else np.array([0., 0., 0., 1., 0., 0., 0.])

    temperature = 0.85
    probs       = np.exp(np.log(probs + 1e-9) / temperature)
    probs      /= np.sum(probs)

    if prev_probs is None:
        smoothed = probs
    else:
        diff     = np.sum(np.abs(probs - prev_probs))
        alpha    = float(np.clip(0.30 + 0.45 * (diff ** 2), 0.30, 0.75))
        smoothed = alpha * probs + (1 - alpha) * prev_probs

    cal_w = np.array([0.55, 0.50, 0.50, 1.50, 0.45, 1.05, 1.25])
    cal   = smoothed * cal_w
    cal  /= np.sum(cal)

    idx       = np.argmax(cal)
    candidate = EMOTION_LABELS[idx]
    conf      = float(cal[idx])
    threshold = EMOTION_THRESHOLDS.get(candidate, 0.22)
    if conf < threshold:
     print(f"[Calibration] {candidate}→neutral (conf={conf:.2f} < seuil={threshold})")
     emotion = 'neutral'
     conf    = max(conf, 0.55)
    else:
        emotion = candidate

    return emotion, conf, cal, smoothed


def calibrate_single_frame(probs):
    probs = np.where(probs < 0.05, 0.0, probs)
    probs = probs / np.sum(probs) if np.sum(probs) > 0 \
        else np.array([0., 0., 0., 1., 0., 0., 0.])

    temperature = 0.85
    probs       = np.exp(np.log(probs + 1e-9) / temperature)
    probs      /= np.sum(probs)

    cal_w = np.array([0.55, 0.50, 0.50, 1.50, 0.45, 1.05, 1.25])
    cal   = probs * cal_w
    cal  /= np.sum(cal)

    idx       = np.argmax(cal)
    candidate = EMOTION_LABELS[idx]
    conf      = float(cal[idx])
    threshold = EMOTION_THRESHOLDS.get(candidate, 0.22)

    
    if conf < threshold:
     print(f"[Calibration] {candidate}→neutral (conf={conf:.2f} < seuil={threshold})")
     emotion = 'neutral'
     conf    = max(conf, 0.55)
    else:
        emotion = candidate

    return emotion, conf, cal


def predict_emotion_enhanced(face, reset_session=False):
    try:
        face_rgb = cv2.cvtColor(preprocess_face(face), cv2.COLOR_BGR2RGB)
        inputs   = base_processor(images=face_rgb, return_tensors="pt").to(device)
        if USE_ONNX:
            pv_np     = inputs['pixel_values'].cpu().numpy()
            logits, _ = vit_session.run(None, {"pixel_values": pv_np})
            probs     = F.softmax(torch.tensor(logits), dim=-1).numpy()[0]
        else:
            with torch.no_grad():
                probs = F.softmax(
                    model(inputs['pixel_values']), dim=-1
                ).cpu().numpy()[0]

        probs = correct_emotion_probs(face, probs)

        emotion, conf, cal = calibrate_single_frame(probs)
        top3_idx = np.argsort(cal)[-3:][::-1]
        top3     = [(EMOTION_LABELS[i], float(cal[i])) for i in top3_idx]
        return emotion, conf, top3, probs   # ← AJOUT : probs en 4e retour
    except Exception as e:
        print(f"Erreur prédiction: {e}")
        return "neutral", 0.5, [("neutral", 0.5)], np.array([0.,0.,0.,1.,0.,0.,0.])  # ← AJOUT
def _vit_raw_probs_sync(face_img):
    face_rgb = cv2.cvtColor(preprocess_face(face_img), cv2.COLOR_BGR2RGB)
    inp_v    = base_processor(images=face_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = F.softmax(
            model(inp_v['pixel_values']), dim=-1).cpu().numpy()[0]
    # ← PATCH v7.2 : cette fonction alimente calculate_candidate_metrics
    # (stress/assurance/communication) — la corriger aussi, sinon les
    # métriques resteraient biaisées même si l'émotion affichée est
    # corrigée par ailleurs.
    return correct_emotion_probs(face_img, probs)


def calculate_deception_risk(emotion_history, confidence_history, frame_times):
    if len(emotion_history) < 5:
        return 0, "Analyse insuffisante", {}

    total = max(1, len(emotion_history))

    dec_emotions = ['fear', 'surprise', 'disgust', 'sad']
    dec_count    = sum(1 for e in emotion_history if e in dec_emotions)

    for i in range(len(emotion_history) - 1):
        if (emotion_history[i] == 'angry' and
                emotion_history[i+1] == 'fear'):
            dec_count += 1

    emotion_score = min(100.0, (dec_count / total) * 333.3)

    changes   = sum(1 for i in range(1, total)
                    if emotion_history[i] != emotion_history[i-1])
    var_score = min(100.0, (changes / total) * 500.0)

    micro = 0
    for i in range(2, total):
        if (emotion_history[i] == emotion_history[i-2] and
                emotion_history[i] != emotion_history[i-1]):
            micro += 1
    micro_score = min(100.0, (micro / max(1, total)) * 1000.0)

    avg_conf   = np.mean(confidence_history) if confidence_history else 0.8
    conf_score = max(0.0, (1.0 - avg_conf) * 200.0)

    stress_periods = []
    current_stress = False
    for e in emotion_history:
        is_stress = e in ['fear', 'sad', 'disgust']
        if is_stress != current_stress:
            stress_periods.append(1)
            current_stress = is_stress
    pattern_score = min(100.0, len(stress_periods) * 15.0)

    total_score = (emotion_score * 0.30 + var_score   * 0.20 +
                   micro_score  * 0.25 + conf_score   * 0.15 +
                   pattern_score * 0.10)

    details = {
        'emotion_score':     emotion_score,
        'variability_score': var_score,
        'micro_expressions': micro,
        'confidence_score':  conf_score,
        'pattern_score':     pattern_score,
        'total_score':       total_score
    }

    level = ("Faible - Discours probablement authentique"
             if total_score < 30 else
             "Modéré - Situation à surveiller"
             if total_score < 60 else
             "Élevé - Forte probabilité de tromperie")

    return total_score, level, details


def calculate_candidate_metrics(visual_probs, audio_probs=None,
                                 audio_energy=None, history=None,
                                 face_analysis=None,
                                 use_full_history: bool = False):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if visual_probs is None:
        visual_probs = np.array([0., 0., 0., 1., 0., 0., 0.])

    v_sad, v_dis, v_ang, v_neu, v_fea, v_sur, v_hap = range(7)
    negative_labels = ['fear', 'angry', 'sad', 'disgust']

    # ── STRESS ───────────────────────────────────────────────────
    # ← CORRECTION LOGIQUE : ne compte comme "instabilité stressante"
    # que les bascules ENTRE DEUX ÉTATS NÉGATIFS (ex: colère→tristesse).
    # Avant : une bascule négatif→positif (ex: colère→joie, une
    # récupération) était comptée à tort comme un signal de stress
    # au même titre qu'une bascule entre deux états négatifs.
    if history and len(history) >= 5:
        recent       = history if use_full_history else history[-10:]
        n            = len(recent)
        stress_count = sum(1 for e in recent if e in negative_labels)
        volatile_negative_changes = sum(
            1 for i in range(1, n)
            if recent[i] != recent[i-1]
            and recent[i]   in negative_labels
            and recent[i-1] in negative_labels
        )
        raw_stress = (stress_count / n) * 0.70 + \
                     (volatile_negative_changes / max(1, n-1)) * 0.30
        stress_management = (1 - raw_stress) * 100
    else:
        r_stress = (visual_probs[v_fea] * 0.8 +
                    visual_probs[v_ang] * 0.4 +
                    visual_probs[v_sad] * 0.8)
        stress_management = (1 - sigmoid(r_stress * 3 - 1.5)) * 100

    if audio_probs is not None and len(audio_probs) == 4:
        audio_stress_raw  = audio_probs[3] * 1.2 + audio_probs[2] * 0.8
        audio_stress_mgmt = (1 - sigmoid(audio_stress_raw * 3 - 1.5)) * 100
        stress_management = stress_management * 0.70 + \
                            audio_stress_mgmt * 0.30

    # ── COMMUNICATION ────────────────────────────────────────────
    # ← CORRECTION LOGIQUE : la colère et le dégoût NUISENT à la
    # communication perçue au lieu de la booster. Avant : angry*0.2
    # contribuait POSITIVEMENT au score, ce qui n'a pas de sens —
    # un candidat en colère obtenait un score proche du neutre.
    r_comm = (visual_probs[v_neu] * 0.5 +
              visual_probs[v_hap] * 1.0 +
              visual_probs[v_sur] * 0.8 -
              visual_probs[v_ang] * 0.3 -
              visual_probs[v_dis] * 0.2)

    if audio_probs is not None and len(audio_probs) == 4:
        r_comm += audio_probs[0] * 0.5 + audio_probs[1] * 1.0

    communication = sigmoid(r_comm * 2.0 - 0.8) * 100

    if history and len(history) >= 5:
        fluent_emotions = ['neutral', 'happy', 'surprise']
        recent_comm     = history if use_full_history else history[-10:]
        fluent_ratio    = sum(1 for e in recent_comm
                              if e in fluent_emotions) / len(recent_comm)
        communication = communication * 0.70 + fluent_ratio * 100 * 0.30

    # ── EXPRESSIVITÉ ─────────────────────────────────────────────
    # ← CORRECTION LOGIQUE : ne pénalise plus un candidat sincèrement
    # expressif et stable (ex: sourire soutenu). Avant : la formule
    # basée sur l'entropie pure confondait un signal stable positif
    # (entropie basse) avec un visage inexpressif/figé (entropie basse
    # aussi) — les deux obtenaient le même score minimal.
    # Nouvelle logique : la base est l'éloignement du neutre (visage
    # figé = score bas), avec un bonus modéré — jamais une pénalité —
    # pour la variété d'expressions si elle existe.
    entropy      = -np.sum(visual_probs * np.log(visual_probs + 1e-9))
    entropy_norm = entropy / np.log(7)

    dominant_strength = 1.0 - visual_probs[v_neu]
    expr_raw = dominant_strength * (0.80 + 0.20 * entropy_norm)
    expressivity = float(np.clip(expr_raw * 100, 10, 95))

    if history and len(history) >= 5:
        recent_expr = history if use_full_history else history[-10:]
        pos_count   = sum(1 for e in recent_expr
                          if e in ['neutral', 'happy', 'surprise'])
        rhythm      = pos_count / len(recent_expr)
        expressivity = min(95, expressivity + rhythm * 10)

    # ── FLUIDITÉ VERBALE (renommé depuis speech_rate) ─────────────
    # ← RENOMMAGE P1 : ce champ n'a jamais été un débit de parole réel
    # (mots/minute) — c'est un score 20-80 dérivé de l'activité audio
    # (HuBERT) ou, en repli, du taux de changement d'émotions. Le vrai
    # débit de parole (WPM) est calculé séparément côté Angular sous
    # le nom `whisperWpm`, à partir de la transcription Whisper.
    # L'ancien nom `speech_rate` prêtait à confusion avec cette valeur.
    if audio_probs is not None and len(audio_probs) == 4:
        audio_activity = (audio_probs[1] + audio_probs[3]) * 100
        verbal_fluidity_score = float(np.clip(40 + audio_activity * 0.6, 20, 80))
    elif history and len(history) >= 5:
        recent_sr    = history if use_full_history else history[-8:]
        changes_sr   = sum(1 for i in range(1, len(recent_sr))
                           if recent_sr[i] != recent_sr[i-1])
        change_ratio = changes_sr / max(1, len(recent_sr) - 1)
        verbal_fluidity_score = float(np.clip(35 + change_ratio * 45, 20, 80))
    else:
        verbal_fluidity_score = 50.0

    # ── ASSURANCE ────────────────────────────────────────────────
    # ← CORRECTION LOGIQUE : la colère et le dégoût NUISENT à
    # l'assurance perçue au lieu de la booster. Avant : angry*0.3
    # contribuait POSITIVEMENT, et disgust n'était jamais pris en
    # compte — un candidat en colère semblait presque aussi "sûr de
    # lui" qu'un candidat neutre.
    raw_assur = (visual_probs[v_neu] * 0.8 +
                 visual_probs[v_hap] * 1.2 -
                 visual_probs[v_fea] * 0.6 -
                 visual_probs[v_sad] * 0.5 -
                 visual_probs[v_ang] * 0.2 -
                 visual_probs[v_dis] * 0.2)

    if audio_probs is not None and len(audio_probs) == 4:
        raw_assur += (audio_probs[0] * 0.8 + audio_probs[1] * 1.2 -
                      audio_probs[2] * 0.5)

    assurance = sigmoid(raw_assur * 2.0) * 100

    # ── STABILITÉ DE PRÉDICTION (renommé depuis confidence_score) ──
    # ← RENOMMAGE P3 : ancien nom `confidence_score` trop proche de
    # `confidence` (confiance du modèle ViT sur l'émotion détectée)
    # et de `assurance_level` (assurance comportementale perçue) —
    # source de confusion. Ce champ mesure en réalité la stabilité
    # des prédictions émotionnelles sur l'historique récent.
    if history and len(history) >= 5:
        recent_conf  = history if use_full_history else history[-10:]
        stable_count = sum(1 for e in recent_conf
                           if e in ['neutral', 'happy'])
        prediction_stability = (stable_count / len(recent_conf)) * 100
    else:
        prediction_stability = float(np.max(visual_probs) * 60 +
                           (40 if audio_probs is not None else 0))

    global_score = max(0, min(100,
        stress_management     * 0.25 +
        communication         * 0.28 +
        assurance             * 0.25 +
        expressivity          * 0.12 +
        prediction_stability  * 0.10
    ))

    if face_analysis is not None:
        bp = face_analysis if isinstance(face_analysis, dict) and \
             'stability_score' in face_analysis else {}

        quality_score   = float(bp.get('quality_score',   0.5))
        movement_score  = float(bp.get('movement_score',  0.5))
        stability_score = float(bp.get('stability_score', 0.5))
        blink_score     = float(bp.get('blink_score',     0.5))
        gaze_score      = float(bp.get('gaze_score',      0.5))
        contact_ratio   = float(bp.get('contact_ratio',   0.5))
        posture_score   = float(bp.get('posture_score',   0.5))
        tension_score   = float(bp.get('tension_score',   0.5))

        assurance_boost = (gaze_score + posture_score) / 2 \
                          if gaze_score != 0.5 else stability_score
        if assurance_boost > 0.7 and quality_score > 0.7:
            assurance = min(100, assurance * 1.10)
        elif assurance_boost < 0.4 or movement_score < 0.4:
            assurance = max(0, assurance * 0.90)

        if blink_score > 0.8:
            stress_management = min(100, stress_management * 1.05)

        comm_boost = contact_ratio if gaze_score != 0.5 else stability_score
        communication = communication * 0.85 + comm_boost * 100 * 0.15

        global_score = max(0, min(100,
            stress_management     * 0.25 +
            communication         * 0.28 +
            assurance             * 0.25 +
            expressivity          * 0.12 +
            prediction_stability  * 0.10
        ))

    return {
        'stress_management':      float(max(0, min(100, stress_management))),
        'communication':          float(max(0, min(100, communication))),
        'expressivity':           float(max(0, min(100, expressivity))),
        'verbal_fluidity_score':  float(max(20, min(80, verbal_fluidity_score))),
        'assurance_level':        float(max(0, min(100, assurance))),
        'prediction_stability':   float(max(0, min(100, prediction_stability))),
        'global_score':           float(global_score)
    }


# ← SUPPRIMÉ : analyze_soft_skills() — plus appelée nulle part
# (Leadership / Empathie / Adaptabilité / Communication
# Interpersonnelle). soft_skills est désormais toujours un dict vide
# dans la réponse ; le PDF gère déjà nativement ce cas
# ("Données non disponibles pour cet entretien").


def analyze_speech_deception(transcript):
    if not transcript:
        return 0.0, []
    t_lower        = transcript.lower()
    hesitation_w   = ["euh", "bah", "en fait", "je crois", "peut-être",
                       "genre", "comment dire", "je ne sais pas", "enfin"]
    justif_w       = ["honnêtement", "pour être franc", "à vrai dire",
                       "croyez-moi", "sincèrement", "je vous jure",
                       "en toute franchise", "absolument"]
    hesitations    = sum(t_lower.count(w) for w in hesitation_w)
    justifications = sum(t_lower.count(w) for w in justif_w)
    word_count     = max(1, len(t_lower.split()))
    risk_score     = 0.0
    flags          = []

    h_ratio = hesitations / word_count
    if h_ratio > 0.015:
        risk_score += min(50.0, (h_ratio / 0.05) * 50.0)
        flags.append(f"Hésitations fréquentes ({hesitations} détectées).")

    j_ratio = justifications / word_count
    if j_ratio > 0.005:
        risk_score += min(50.0, (j_ratio / 0.02) * 50.0)
        flags.append(f"Sur-justification ({justifications} détectées).")

    return min(100.0, risk_score), flags


def detect_inconsistencies(transcript, history):
    inconsistencies = []
    positive_words  = ["content", "heureux", "ravi",
                       "enthousiaste", "super", "génial"]
    t_lower = transcript.lower()
    if history and any(w in t_lower for w in positive_words):
        avg_fear = np.mean([h[4] for h in history])
        avg_sad  = np.mean([h[0] for h in history])
        if avg_fear > 0.3 or avg_sad > 0.3:
            inconsistencies.append(
                "Décalage : Discours positif mais expressions anxieuses."
            )
    return inconsistencies


# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS BASIQUES
# ═══════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {
        "name":    "Nexum IA - Emotion Analysis API",
        "version": "2.0.0",
        "status":  "online"
    }


@app.get("/model_info")
async def model_info():
    return {
        "model_type": "ViT-Base fine-tuned",
        "num_classes": 7,
        "emotions":    EMOTION_LABELS,
        "emotions_fr": EMOTION_NAMES_FR,
        "is_trained":  os.path.exists(Config.MODEL_PATH)
    }


# ═══════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════
class TrainingRequest(BaseModel):
    dataset_path:  str
    num_epochs:    Optional[int]   = 10
    batch_size:    Optional[int]   = 32
    learning_rate: Optional[float] = 2e-5


@app.post("/train")
async def train_model(request: TrainingRequest,
                      background_tasks: BackgroundTasks):
    background_tasks.add_task(
        perform_training, request.dataset_path,
        request.num_epochs, request.batch_size, request.learning_rate
    )
    return {"message": "Entraînement démarré", "status": "training"}


async def perform_training(dataset_path, num_epochs,
                           batch_size, learning_rate):
    try:
        print(f"\nEntraînement sur {dataset_path}")
        train_dataset = EmotionDataset(dataset_path, base_processor)
        train_loader  = DataLoader(train_dataset,
                                   batch_size=batch_size, shuffle=True)
        optimizer     = AdamW(model.parameters(),
                              lr=learning_rate, weight_decay=0.01)
        scheduler     = CosineAnnealingLR(optimizer, T_max=num_epochs)
        class_weights = torch.tensor(
            [1.0, 1.2, 1.3, 0.8, 1.0, 1.1, 1.2]
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        model.train()
        for epoch in range(num_epochs):
            total_loss = correct = total = 0
            for batch in train_loader:
                pv     = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                optimizer.zero_grad()
                out    = model(pv)
                loss   = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                _, pred = out.max(1)
                total   += labels.size(0)
                correct += pred.eq(labels).sum().item()
            scheduler.step()
            acc = 100. * correct / total
            print(f"Epoch {epoch+1}/{num_epochs} — "
                  f"Loss: {total_loss/len(train_loader):.4f} "
                  f"Acc: {acc:.2f}%")
        torch.save(model.state_dict(), Config.MODEL_PATH)
        print(f"Modèle sauvegardé → {Config.MODEL_PATH}")
        model.eval()
    except Exception as e:
        print(f"Erreur entraînement: {e}")


# ═══════════════════════════════════════════════════════════════════
# TÉLÉCHARGEMENT VIDÉO
# ═══════════════════════════════════════════════════════════════════
async def download_video_from_url(url: str, custom_filename: str = None,
                                   max_size_mb: int = 500):
    import urllib.request, uuid
    uid        = str(uuid.uuid4())[:8]
    filename   = f"candidate_{uid}.mp4"
    temp_path  = os.path.join(tempfile.gettempdir(), filename)
    fixed_path = os.path.join(tempfile.gettempdir(), f"fixed_{filename}")
    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0'}
        )
        urllib.request.urlretrieve(url, temp_path)
        size = os.path.getsize(temp_path)
        if size == 0:
            raise ValueError("Fichier vide")
        if size > max_size_mb * 1024 * 1024:
            os.remove(temp_path)
            raise ValueError(f"Vidéo trop grande (max {max_size_mb}MB)")
        result = subprocess.run(
            [FFMPEG_PATH, '-i', temp_path, '-c', 'copy',
             '-movflags', '+faststart', '-y', fixed_path],
            capture_output=True, text=True, timeout=120
        )
        if (result.returncode == 0 and os.path.exists(fixed_path) and
                os.path.getsize(fixed_path) > 0):
            os.remove(temp_path)
            return fixed_path, filename
        if os.path.exists(fixed_path):
            os.remove(fixed_path)
        return temp_path, filename
    except Exception as e:
        for p in [temp_path, fixed_path]:
            if os.path.exists(p):
                os.remove(p)
        raise Exception(f"Erreur téléchargement: {e}")


# ═══════════════════════════════════════════════════════════════════
# extract_candidates_preview
# ═══════════════════════════════════════════════════════════════════
def _arcface_embed_preview(face_img):
    if preview_arcface is None or face_img is None or face_img.size == 0:
        return None
    h, w = face_img.shape[:2]
    if w < 30 or h < 30:
        return None

    strategies = []
    try:
        c = np.zeros((640, 640, 3), dtype=np.uint8)
        c[170:470, 170:470] = cv2.resize(face_img, (300, 300),
                                          interpolation=cv2.INTER_LINEAR)
        strategies.append(("S3_c300", c))
    except Exception:
        pass
    try:
        c = np.zeros((640, 640, 3), dtype=np.uint8)
        c[70:570, 70:570] = cv2.resize(face_img, (500, 500),
                                        interpolation=cv2.INTER_LINEAR)
        strategies.append(("S4_c500", c))
    except Exception:
        pass
    try:
        strategies.append(("S2_raw",
            cv2.resize(face_img, (112, 112),
                       interpolation=cv2.INTER_LINEAR)))
    except Exception:
        pass

    for name, frame in strategies:
        try:
            faces = preview_arcface.get(frame)
            if not faces:
                continue
            best = max(faces, key=lambda f: f.det_score)
            if best.det_score < 0.35:
                continue
            emb  = best.embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception:
            continue
    return None


def _extract_candidates_preview_sync(file_bytes: bytes, filename: str):
    """
    ← PATCH ANTI-BLOCAGE (Windows WinError 64) : tout le traitement lourd
    et bloquant (écriture disque, FFmpeg, décodage vidéo frame par frame,
    YOLO, ArcFace) est isolé ici et exécuté dans un thread séparé via
    run_in_executor(). Avant ce patch, tout ceci tournait DANS la boucle
    asyncio principale — sur une vidéo de ~6-7 minutes, ce traitement
    bloquait le serveur pendant plus d'une minute d'affilée, empêchant
    toute nouvelle connexion WebSocket d'être acceptée. Sous Windows,
    ce blocage prolongé provoquait le crash du socket d'écoute lui-même
    (OSError WinError 64 sur l'accept() asyncio), rendant le serveur
    injoignable même après la fin du traitement.
    Retourne un tuple (payload_dict, status_code) — jamais de JSONResponse
    ici, puisqu'on est hors du contexte async/FastAPI dans ce thread.
    """
    import time
    tmp_path = fixed_path = None
    try:
        tmp_path   = os.path.join(tempfile.gettempdir(),
                                  f"preview_{filename}")
        fixed_path = os.path.join(tempfile.gettempdir(),
                                  f"fixed_preview_{filename}")
        with open(tmp_path, "wb") as buf:
            buf.write(file_bytes)

        print(f"[Preview] Réparation: {filename}")
        # ← OPTIMISATION MAJEURE : -t 35 limite le réencodage FFmpeg aux
        # 35 premières secondes (30s de scan + marge), au lieu de toute
        # la vidéo. C'était le vrai goulot d'étranglement restant :
        # même avec le scan Python limité à 30s (SCAN_DURATION_CAP),
        # cette étape de "réparation" réencodait l'INTÉGRALITÉ de la
        # vidéo avant même que le scan ne démarre — pour une vidéo de
        # 10 minutes, ça pouvait prendre 30-60s+ à elle seule, largement
        # plus que le scan optimisé qui suit. Avec -t 35, cette étape
        # devient proportionnelle à ~35s peu importe la durée réelle
        # de la vidéo uploadée.
        repair = subprocess.run(
            [FFMPEG_PATH, '-y', '-i', tmp_path, '-t', '35',
             '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
             '-c:a', 'aac', '-movflags', '+faststart', fixed_path],
            capture_output=True, timeout=300
        )
        video_path = (fixed_path
                      if repair.returncode == 0 and
                      os.path.exists(fixed_path) and
                      os.path.getsize(fixed_path) > 0
                      else tmp_path)

        # ← AJOUT : récupère la vraie durée originale via ffprobe
        # (lecture de métadonnées seule, quasi instantanée — ne décode
        # pas la vidéo) UNIQUEMENT pour un affichage de log correct.
        # Sans ça, comme la vidéo réparée est maintenant tronquée à
        # 35s, le log afficherait à tort "durée: 35s" même pour une
        # vidéo originale de 10 minutes — cette étape corrige juste
        # l'information affichée, sans impact sur le comportement.
        true_duration_label = None
        try:
            probe = subprocess.run(
                [FFMPEG_PATH, '-i', tmp_path],
                capture_output=True, timeout=10, text=True
            )
            import re as _re
            m = _re.search(r"Duration:\s*(\d+):(\d+):(\d+)", probe.stderr or "")
            if m:
                h, mnt, s = map(int, m.groups())
                true_duration_label = f"{h*3600 + mnt*60 + s}s (originale)"
        except Exception:
            pass

        try:
            video_clip = mp.VideoFileClip(video_path)
        except Exception as e:
            for p in [tmp_path, fixed_path]:
                if p and os.path.exists(p):
                    os.remove(p)
            return ({
                'success': False,
                'error':   f'Impossible de lire la vidéo : {e}'
            }, 400)

        total_dur = video_clip.duration
        W_vid     = int(video_clip.size[0])
        H_vid     = int(video_clip.size[1])
        MIN_FACE_PX = 30 * 30

        # ← Limite le scan aux 30 premières secondes de la vidéo, au lieu
        # de parcourir toute sa durée. Le candidat est presque toujours
        # visible dès le début de l'entretien — inutile de décoder et
        # analyser (YOLO + ArcFace) des minutes entières de vidéo juste
        # pour extraire un aperçu du visage. Réduit fortement le temps
        # de traitement et la charge CPU sur ce endpoint.
        SCAN_DURATION_CAP = 30.0
        scan_dur = min(total_dur, SCAN_DURATION_CAP)

        if scan_dur <= 60:
            SAMPLE_STEP = 0.5
        elif scan_dur <= 300:
            SAMPLE_STEP = 1.5
        else:
            SAMPLE_STEP = 2.0

        TOLERANCE        = 0.25
        MERGE_TOLERANCE  = 0.40
        SPATIAL_MERGE_PX = 80
        MAX_SAMPLES      = 8

        sample_times = sorted(set(np.arange(0, scan_dur, SAMPLE_STEP)))

        print(f"[Preview] Durée vidéo: "
              f"{true_duration_label or f'{total_dur:.0f}s (tronquée)'} · "
              f"Scan limité aux {scan_dur:.0f}s premières secondes · "
              f"Frames: {len(sample_times)} (pas={SAMPLE_STEP}s)")

        known_candidates = []

        for t in sample_times:
            try:
                frame_rgb = video_clip.get_frame(t)
            except Exception:
                continue
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            faces = detect_faces(frame_bgr)
            if not faces:
                continue

            for face_img, conf, bbox, _ in faces:
                x1, y1, x2, y2 = bbox
                size = (x2 - x1) * (y2 - y1)
                if size < MIN_FACE_PX:
                    continue

                emb = _arcface_embed_preview(face_img)
                if emb is None:
                    continue

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if cy < H_vid * 0.28:
                    print(f"[Preview] Visage ignoré — miniature bord haut "
                          f"({cx:.0f},{cy:.0f})")
                    continue

                matched_idx = None
                best_sim    = 0.0
                for i, c in enumerate(known_candidates):
                    if not c['embeddings']:
                        continue
                    avg = np.mean(c['embeddings'], axis=0)
                    n   = np.linalg.norm(avg)
                    avg = avg / n if n > 0 else avg
                    sim = float(np.dot(avg, emb))
                    if sim > TOLERANCE and sim > best_sim:
                        best_sim    = sim
                        matched_idx = i

                if matched_idx is not None:
                    c = known_candidates[matched_idx]
                    c['count'] += 1
                    c['cx'] = 0.85 * c['cx'] + 0.15 * cx
                    c['cy'] = 0.85 * c['cy'] + 0.15 * cy
                    c['embeddings'].append(emb)
                    if len(c['embeddings']) > MAX_SAMPLES:
                        c['embeddings'].pop(0)
                    if size > c['best_size']:
                        c['face_img']  = face_img
                        c['bbox']      = [x1, y1, x2, y2]
                        c['best_size'] = size
                else:
                    known_candidates.append({
                        'embeddings': [emb],
                        'face_img':   face_img,
                        'cx':         cx,
                        'cy':         cy,
                        'bbox':       [x1, y1, x2, y2],
                        'count':      1,
                        'best_size':  size,
                        'first_seen': float(t),
                    })
            time.sleep(0)
        video_clip.close()
        for p in [tmp_path, fixed_path]:
            if p and os.path.exists(p):
                os.remove(p)

        merged_any = True
        merge_pass = 0
        while merged_any:
            merged_any = False
            merge_pass += 1
            for i in range(len(known_candidates)):
                if merged_any:
                    break
                for j in range(i + 1, len(known_candidates)):
                    ci = known_candidates[i]
                    cj = known_candidates[j]

                    avg_i = np.mean(ci['embeddings'], axis=0)
                    n_i   = np.linalg.norm(avg_i)
                    avg_i = avg_i / n_i if n_i > 0 else avg_i

                    avg_j = np.mean(cj['embeddings'], axis=0)
                    n_j   = np.linalg.norm(avg_j)
                    avg_j = avg_j / n_j if n_j > 0 else avg_j

                    sim          = float(np.dot(avg_i, avg_j))
                    spatial_dist = np.hypot(ci['cx'] - cj['cx'],
                                            ci['cy'] - cj['cy'])

                    print(f"[Preview] Clusters {i}↔{j}: "
                          f"sim={sim:.3f} dist={spatial_dist:.0f}px")

                    sim_ok      = sim > MERGE_TOLERANCE
                    count_ratio = (min(ci['count'], cj['count']) /
                                   max(ci['count'], cj['count']))
                    spatial_ok  = (spatial_dist < SPATIAL_MERGE_PX and
                                   count_ratio < 0.15)
                    should_merge = sim_ok or spatial_ok

                    if should_merge:
                        reason = []
                        if sim_ok:
                            reason.append(f"sim={sim:.3f}")
                        if spatial_ok:
                            reason.append(f"dist={spatial_dist:.0f}px")
                        print(f"[Preview] 🔗 Fusion clusters {i} et {j} "
                              f"({', '.join(reason)}, pass={merge_pass})")

                        ci['embeddings'].extend(cj['embeddings'])
                        if len(ci['embeddings']) > 10:
                            ci['embeddings'] = ci['embeddings'][-10:]
                        ci['count'] += cj['count']
                        if cj['best_size'] > ci['best_size']:
                            ci['face_img']  = cj['face_img']
                            ci['bbox']      = cj['bbox']
                            ci['best_size'] = cj['best_size']
                        ci['first_seen'] = min(ci['first_seen'],
                                               cj['first_seen'])
                        total_count = ci['count']
                        ci['cx'] = (
                            (ci['cx'] * (ci['count'] - cj['count']) +
                             cj['cx'] * cj['count']) / total_count
                        )
                        ci['cy'] = (
                            (ci['cy'] * (ci['count'] - cj['count']) +
                             cj['cy'] * cj['count']) / total_count
                        )
                        known_candidates.pop(j)
                        merged_any = True
                        break
            time.sleep(0)
        print(f"[Preview] Après fusion: {len(known_candidates)} "
              f"candidat(s) unique(s) (passes={merge_pass})")

        min_count = 1 if scan_dur <= 20 else 2
        valid = [c for c in known_candidates if c['count'] >= min_count]
        valid.sort(key=lambda c: c['first_seen'])

        print(f"[Preview] {len(known_candidates)} cluster(s) brut(s) → "
              f"{len(valid)} candidat(s) retenu(s)")
        print(f"[Preview] DEBUG positions: " +
              ", ".join(f"({c['cx']:.0f},{c['cy']:.0f}) x{c['count']}"
                        for c in valid))

        if not valid:
            return ({
                'success':    False,
                'error':      'Aucun visage identifiable.',
                'candidates': []
            }, 200)

        import uuid
        from app_embedding_cache import store_embedding, clear_old_entries
        session_id = str(uuid.uuid4())[:8]
        clear_old_entries()

        result_candidates = []
        for idx, c in enumerate(valid):
            _, buf = cv2.imencode('.jpg', c['face_img'],
                                  [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

            avg_emb = np.mean(c['embeddings'], axis=0).astype(np.float32)
            norm    = np.linalg.norm(avg_emb)
            avg_emb = avg_emb / norm if norm > 0 else avg_emb
            key     = f"{session_id}_{idx+1}"
            store_embedding(key, avg_emb, c['face_img'])
            print(f"[Preview] Embedding cached: {key} dim={len(avg_emb)}D")

            if W_vid > 0:
                if c['cx'] < W_vid * 0.35:
                    side_label = "Côté Gauche"
                elif c['cx'] > W_vid * 0.65:
                    side_label = "Côté Droit"
                else:
                    side_label = "Centre"
            else:
                side_label = ("Candidat Unique"
                              if len(valid) == 1
                              else f"Position {idx+1}")

            result_candidates.append({
                'id':              idx + 1,
                'face_image':      f"data:image/jpeg;base64,{b64}",
                'center_x':        float(c['cx']),
                'center_y':        float(c['cy']),
                'bbox':            [int(v) for v in c['bbox']],
                'name':            f"Candidat {idx + 1}",
                'side':            side_label,
                'frames_detected': c['count'],
                'first_seen_sec':  c['first_seen'],
                'embedding_key':   key
            })

        print(f"[Preview] {len(result_candidates)} candidat(s) détecté(s)")
        return ({
            'success':    True,
            'candidates': result_candidates,
            'count':      len(result_candidates)
        }, 200)

    except Exception as e:
        for p in [tmp_path, fixed_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        import traceback; traceback.print_exc()
        return ({'success': False, 'error': str(e)}, 500)


@app.post("/extract_candidates_preview")
async def extract_candidates_preview(file: UploadFile = File(...)):
    # ← PATCH ANTI-BLOCAGE : seule la lecture du fichier uploadé (I/O
    # réseau, déjà non-bloquante) reste dans la coroutine. Tout le
    # reste (FFmpeg, décodage, YOLO, ArcFace) part dans un thread via
    # run_in_executor — la boucle asyncio reste libre pendant ce temps.
    file_bytes = await file.read()
    loop = asyncio.get_running_loop()
    payload, status_code = await loop.run_in_executor(
        video_processing_executor, _extract_candidates_preview_sync,
        file_bytes, file.filename
    )
    return JSONResponse(payload, status_code=status_code)


@app.post("/analyze_video_from_url")
async def analyze_video_from_url(request: VideoURLRequest):
    temp_path = None
    try:
        print(f"[analyze_video_from_url] URL: {request.url[:100]}")
        temp_path, filename = await download_video_from_url(
            request.url, request.filename
        )
        print(f"[analyze_video_from_url] Taille: "
              f"{os.path.getsize(temp_path)}")

        class VirtualUploadFile:
            def __init__(self, path, name):
                self.filename = name
                self._path    = path
            async def read(self):
                with open(self._path, 'rb') as f:
                    return f.read()

        result = await analyze_video(
            file=VirtualUploadFile(temp_path, filename),
            target_x=None, target_y=None
        )
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return result

    except Exception as e:
        import traceback
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse({
            'success': False, 'error': str(e),
            'details': traceback.format_exc()
        }, status_code=500)



def _analyze_video_sync(file_bytes: bytes, filename: str,
                        target_x: Optional[float] = None,
                        target_y: Optional[float] = None,
                        embedding_key: Optional[str] = None):
    
    import time
    try:
        start_time = time.time()
        tmp_path   = os.path.join(tempfile.gettempdir(), filename)
        with open(tmp_path, "wb") as buf:
            buf.write(file_bytes)

        print(f"[analyze_video] Démarrage: {filename}")
        if embedding_key:
            print(f"[analyze_video] 🔑 embedding_key reçu: {embedding_key}")

        video_clip = mp.VideoFileClip(tmp_path)
        has_audio  = video_clip.audio is not None

        audio_path        = None
        audio_data        = None
        transcript_text   = ""
        transcript_chunks = []

        if has_audio:
            audio_path = os.path.join(
                tempfile.gettempdir(),
                f"{os.path.splitext(filename)[0]}_audio.wav"
            )
            video_clip.audio.write_audiofile(
                audio_path, verbose=False, logger=None)
            try:
                from moviepy.editor import AudioFileClip
                temp_clip = AudioFileClip(audio_path)
                wav_full  = audio_path + "_full.wav"
                temp_clip.write_audiofile(
                    wav_full, verbose=False, logger=None,
                    fps=16000, nbytes=2, codec='pcm_s16le')
                audio_data, _ = sf.read(wav_full)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                if os.path.exists(wav_full):
                    os.remove(wav_full)
                temp_clip.close()
                print(f"[analyze_video] Audio chargé "
                      f"({len(audio_data)/16000:.1f}s)")
            except Exception as e:
                print(f"[analyze_video] Erreur chargement audio: {e}")
                has_audio = False

        duration = video_clip.duration
        if duration < 5:
            video_clip.close()
            return ({"success": False,
                     "error": f"Vidéo trop courte ({duration:.1f}s). Min: 5s."},
                    400)

        H_vid = int(video_clip.size[1])
        W_vid = int(video_clip.size[0])

        frames_results = []
        visual_history = []
        prev_probs     = None
        sample_rate    = 1.0

        preview_faces    = []
        preview_duration = min(10.0, duration)
        for t_p in np.arange(0, preview_duration, sample_rate):
            try:
                fp     = video_clip.get_frame(t_p)
                fp_bgr = cv2.cvtColor(fp, cv2.COLOR_RGB2BGR)
                fps_p  = detect_faces(fp_bgr)
                if fps_p:
                    fi, _, _, _ = fps_p[0]
                    _, buf = cv2.imencode('.jpg', fi)
                    preview_faces.append(
                        f"data:image/jpeg;base64,"
                        f"{base64.b64encode(buf.tobytes()).decode('utf-8')}"
                    )
            except Exception:
                continue

        # ── TrackingManager NEUF à chaque appel HTTP ──────────────
        tracking_manager = TrackingManager(
            max_age=300, n_init=3,
            shared_arcface=shared_arcface,
            fast_arcface=fast_arcface
        )
        face_analyzer_local = FaceAnalyzer()

        frames_candidate_analyzed = 0
        frames_other_person       = 0
        frames_tracking_lost      = 0
        memo_done = False
        locked    = False

        stored_emb  = None
        stored_face = None

        if embedding_key:
            from app_embedding_cache import get_embedding
            stored_emb, stored_face = get_embedding(embedding_key)
            if stored_emb is not None:
                print(f"[analyze_video] 🎯 Embedding chargé depuis cache "
                      f"— scan vidéo complet démarré")
            else:
                print(f"[analyze_video] ⚠️ Embedding key introuvable — "
                      f"fallback coords")

        def _arcface_embed_quick(face_img) -> Tuple[Optional[np.ndarray], bool]:
          
            if shared_arcface is None or face_img is None or face_img.size == 0:
                return None, False
            h, w = face_img.shape[:2]
            if w < 30 or h < 30:
                return None, False
            try:
                c = np.zeros((640, 640, 3), dtype=np.uint8)
                c[170:470, 170:470] = cv2.resize(face_img, (300, 300),
                    interpolation=cv2.INTER_LINEAR)
                faces = shared_arcface.get(c)
                if faces:
                    best = max(faces, key=lambda f: f.det_score)
                    if best.det_score >= CFG.MIN_DET_SCORE:
                        emb  = best.embedding.astype(np.float32)
                        norm = np.linalg.norm(emb)
                        return (emb/norm if norm > 0 else emb), False
            except Exception:
                pass
            try:
                small = cv2.resize(face_img, (112, 112),
                                   interpolation=cv2.INTER_LINEAR)
                faces = shared_arcface.get(small)
                if faces:
                    best = max(faces, key=lambda f: f.det_score)
                    if best.det_score >= CFG.MIN_DET_SCORE:
                        emb  = best.embedding.astype(np.float32)
                        norm = np.linalg.norm(emb)
                        return (emb/norm if norm > 0 else emb), False
            except Exception:
                pass
            # ← Passe de secours (confiance faible signalée à l'appelant)
            try:
                c = np.zeros((640, 640, 3), dtype=np.uint8)
                c[170:470, 170:470] = cv2.resize(face_img, (300, 300),
                    interpolation=cv2.INTER_LINEAR)
                faces = shared_arcface.get(c)
                if faces:
                    best = max(faces, key=lambda f: f.det_score)
                    if best.det_score >= CFG.MIN_DET_SCORE_FALLBACK:
                        emb  = best.embedding.astype(np.float32)
                        norm = np.linalg.norm(emb)
                        return (emb/norm if norm > 0 else emb), True
            except Exception:
                pass
            try:
                small = cv2.resize(face_img, (112, 112),
                                   interpolation=cv2.INTER_LINEAR)
                faces = shared_arcface.get(small)
                if faces:
                    best = max(faces, key=lambda f: f.det_score)
                    if best.det_score >= CFG.MIN_DET_SCORE_FALLBACK:
                        emb  = best.embedding.astype(np.float32)
                        norm = np.linalg.norm(emb)
                        return (emb/norm if norm > 0 else emb), True
            except Exception:
                pass
            return None, False
        lock_t = None

        if stored_emb is not None:
            print(f"[analyze_video] 🔍 Scan embedding sur toute la vidéo...")

            for t_scan in np.arange(0, min(duration, 60.0), 1.0):
                try:
                    f_scan     = video_clip.get_frame(t_scan)
                    f_scan_bgr = cv2.cvtColor(f_scan, cv2.COLOR_RGB2BGR)
                    faces_scan = detect_faces(f_scan_bgr)
                    if not faces_scan:
                        continue

                    best_sim = 0.0
                    best_fd  = None

                    for fd in faces_scan:
                        tight_s = fd[2]
                        cx_s    = (tight_s[0]+tight_s[2])//2
                        cy_s    = (tight_s[1]+tight_s[3])//2

                        if cy_s < H_vid * 0.28:
                            print(f"[analyze_video] t={t_scan:.0f}s "
                                  f"Miniature ignorée ({cx_s},{cy_s})")
                            continue

                        crop_s = f_scan_bgr[
                            max(0,tight_s[1]):min(H_vid,tight_s[3]),
                            max(0,tight_s[0]):min(W_vid,tight_s[2])]
                        if crop_s.size == 0:
                            continue
                        if float(np.mean(cv2.cvtColor(
                                crop_s, cv2.COLOR_BGR2GRAY))) < 40:
                            continue

                        pad_s = 20
                        fp_s  = f_scan_bgr[
                            max(0,tight_s[1]-pad_s):min(H_vid,tight_s[3]+pad_s),
                            max(0,tight_s[0]-pad_s):min(W_vid,tight_s[2]+pad_s)]
                        emb_s, is_low_conf_s = _arcface_embed_quick(fp_s)
                        if emb_s is None:
                            continue
                        sim_s = float(np.dot(stored_emb, emb_s))
                        effective_threshold_s = (CFG.TOLERANCE_ARCFACE_FALLBACK if is_low_conf_s
                          else CFG.TOLERANCE_ARCFACE)
                        print(f"[analyze_video] t={t_scan:.0f}s "
                           f"Scan ({cx_s},{cy_s}) sim={sim_s:.3f} "
                           f"seuil={effective_threshold_s:.2f}"
                           f"{' [secours]' if is_low_conf_s else ''} "
                           f"{'✅' if sim_s >= effective_threshold_s else '❌'}")

                        if sim_s >= effective_threshold_s and sim_s > best_sim:
                            best_sim = sim_s
                            best_fd  = fd

                    if best_fd is not None:
                        tight_m = best_fd[2]
                        cx_m    = (tight_m[0]+tight_m[2])/2
                        cy_m    = (tight_m[1]+tight_m[3])/2
                        conf_m  = best_fd[1]

                        tracks_m = tracking_manager.tracker.update_tracks(
                            [([tight_m[0], tight_m[1],
                               tight_m[2]-tight_m[0],
                               tight_m[3]-tight_m[1]], conf_m, None)],
                            frame=f_scan_bgr)
                        if tracks_m:
                            bt_m = min(tracks_m, key=lambda t: np.hypot(
                                (t.to_ltrb()[0]+t.to_ltrb()[2])/2 - cx_m,
                                (t.to_ltrb()[1]+t.to_ltrb()[3])/2 - cy_m))
                            tracking_manager.select(
                                bt_m.track_id, list(tight_m))

                            live_emb, _ = _arcface_embed_quick(
                            f_scan_bgr[
                                max(0,int(tight_m[1])-20):min(H_vid,int(tight_m[3])+20),
                                max(0,int(tight_m[0])-20):min(W_vid,int(tight_m[2])+20)])
                            ref_emb = live_emb if live_emb is not None else stored_emb

                            face_live_async = f_scan_bgr[
                                max(0,int(tight_m[1])):min(H_vid,int(tight_m[3])),
                                max(0,int(tight_m[0])):min(W_vid,int(tight_m[2]))]
                            face_for_speed = (face_live_async
                                              if face_live_async.size > 0
                                              else stored_face)

                            tracking_manager.set_reference_embedding(
                                ref_emb, face_img=face_for_speed)
                            tracking_manager.zone.define(
                                cx_m, cy_m, W_vid, H_vid)
                            lock_t    = float(t_scan)
                            locked    = True
                            memo_done = True
                            print(f"[analyze_video] ✅ Candidat trouvé "
                                  f"à t={t_scan:.1f}s sim={best_sim:.3f} "
                                  f"({cx_m:.0f},{cy_m:.0f})")
                            break
                except Exception as e:
                    print(f"[analyze_video] Erreur scan t={t_scan:.1f}: {e}")
                    continue

            if not locked:
                if target_x is not None and target_y is not None:
                    print(f"[analyze_video] ⚠️ Scan embedding échoué → "
                          f"fallback coords ({target_x},{target_y})")
                    stored_emb = None

        if not locked:
            ref_x  = target_x if target_x is not None else W_vid // 2
            ref_y  = target_y if target_y is not None else H_vid // 2
            use_lg = (target_x is None)

            for t_try in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
                          5.0, 10.0, 15.0, 20.0, 30.0]:
                if t_try >= duration:
                    break
                try:
                    f_try     = video_clip.get_frame(t_try)
                    f_try_bgr = cv2.cvtColor(f_try, cv2.COLOR_RGB2BGR)
                    faces_try = detect_faces(f_try_bgr)
                    if not faces_try:
                        continue

                    if use_lg:
                        best = max(faces_try,
                            key=lambda f: (f[2][2]-f[2][0])*(f[2][3]-f[2][1]))
                    else:
                        candidates = []
                        for fd in faces_try:
                            tight   = fd[2]
                            cx_fd   = (tight[0]+tight[2])/2
                            cy_fd   = (tight[1]+tight[3])/2
                            dist    = np.hypot(cx_fd - ref_x, cy_fd - ref_y)
                            cy_diff = abs(cy_fd - ref_y)
                            if dist < 60 and cy_diff < 40:
                                candidates.append((dist, fd))
                        if candidates:
                            best = min(candidates, key=lambda x: x[0])[1]
                        else:
                            print(f"[analyze_video] t={t_try:.1f}s — "
                                  f"aucun visage proche de "
                                  f"({ref_x:.0f},{ref_y:.0f})")
                            continue

                    tight = best[2]
                    conf  = best[1]
                    x1b, y1b, x2b, y2b = tight
                    tracks = tracking_manager.tracker.update_tracks(
                        [([x1b, y1b, x2b-x1b, y2b-y1b], conf, None)],
                        frame=f_try_bgr)
                    if not tracks:
                        continue

                    bt = min(tracks, key=lambda t: np.hypot(
                        (t.to_ltrb()[0]+t.to_ltrb()[2])/2 - (x1b+x2b)/2,
                        (t.to_ltrb()[1]+t.to_ltrb()[3])/2 - (y1b+y2b)/2))
                    tracking_manager.select(bt.track_id, tight)
                    lock_t = float(t_try)
                    locked = True
                    print(f"[analyze_video] Verrouillé à t={t_try}s "
                          f"track_id={bt.track_id} "
                          f"({(x1b+x2b)//2},{(y1b+y2b)//2})")
                    break
                except Exception as e:
                    print(f"[analyze_video] Erreur verrouillage t={t_try}: {e}")
                    continue

        if locked and not memo_done:
            memo_fails  = 0
            lock_t_memo = lock_t if lock_t is not None else 0.0
            for t_mem in np.arange(lock_t_memo,
                                   min(lock_t_memo + 30.0, duration), 0.5):
                if memo_done:
                    break
                if memo_fails >= CFG.MAX_MEMO_FAILS:
                    print("[analyze_video] Mémorisation bloquée")
                    break
                try:
                    f_mem     = video_clip.get_frame(t_mem)
                    f_mem_bgr = cv2.cvtColor(f_mem, cv2.COLOR_RGB2BGR)
                    faces_mem = detect_faces(f_mem_bgr)
                    if not faces_mem:
                        memo_fails += 1
                        continue

                    detections_mem = [
                        (x1, y1, x2, y2, c)
                        for _, c, (x1,y1,x2,y2), _ in faces_mem
                    ]
                    tracking_manager.update(f_mem_bgr, detections_mem)
                    sel_bbox_mem = tracking_manager.get_bbox()
                    if sel_bbox_mem is None:
                        memo_fails += 1
                        continue

                    mx1, my1, mx2, my2 = [max(0,int(v))
                                          for v in sel_bbox_mem]
                    mx2 = min(W_vid, mx2)
                    my2 = min(H_vid, my2)
                    if mx2 <= mx1 or my2 <= my1:
                        memo_fails += 1
                        continue

                    pad      = 20
                    face_mem = f_mem_bgr[
                        max(0,my1-pad):min(H_vid,my2+pad),
                        max(0,mx1-pad):min(W_vid,mx2+pad)]
                    if face_mem.size == 0:
                        memo_fails += 1
                        continue

                    gray_mem = cv2.cvtColor(face_mem, cv2.COLOR_BGR2GRAY)
                    if float(np.mean(gray_mem)) < 40:
                        memo_fails += 1
                        continue

                    done = tracking_manager.memorize_frame(face_mem)
                    if done:
                        cx_mem = (mx1+mx2)/2
                        cy_mem = (my1+my2)/2
                        tracking_manager.zone.define(
                            cx_mem, cy_mem, W_vid, H_vid)
                        tracking_manager.speed.memorize(face_mem)
                        memo_done = True
                        print(f"[analyze_video] ✅ Mémorisation terminée "
                              f"({CFG.MEMORIZE_FRAMES} frames ArcFace)")
                        ok_t, sim_t = tracking_manager.identity.verify(
                            face_img=face_mem)
                        print(f"[analyze_video] Auto-test: sim={sim_t:.3f} "
                              f"{'✅' if ok_t else '❌'}")
                    else:
                        if not tracking_manager.is_memorizing():
                            memo_fails += 1
                except Exception as e:
                    memo_fails += 1
                    print(f"[analyze_video] Erreur mémorisation: {e}")

        if not memo_done:
            print("[analyze_video] ⚠️ Mémorisation incomplète — "
                  "analyse sans vérification stricte")

        start_analysis_t       = lock_t if lock_t is not None else 0.0
        consecutive_lost       = 0
        MAX_LOST_BEFORE_RELOCK = 5
        rej_count              = 0

        print(f"[analyze_video] 📊 Analyse depuis t={start_analysis_t:.1f}s")

        for t in np.arange(start_analysis_t, duration, sample_rate):
            frame     = video_clip.get_frame(t)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            H, W      = frame_bgr.shape[:2]

            faces_all      = detect_faces(frame_bgr)
            zoom_limit     = H * 0.28
            faces_filtered = [
                fd for fd in faces_all
                if (fd[2][1] + fd[2][3]) / 2 >= zoom_limit
            ]
            faces = faces_filtered if faces_filtered else faces_all

            detections = [
                (x1, y1, x2, y2, conf)
                for _, conf, (x1,y1,x2,y2), _ in faces
                if (x2-x1) > 20 and (y2-y1) > 20
            ]

            if (tracking_manager.is_tracking_lost() and
                    consecutive_lost >= MAX_LOST_BEFORE_RELOCK and
                    tracking_manager.last_bbox is not None and faces):
                lb     = tracking_manager.last_bbox
                ref_cx = (lb[0]+lb[2])/2
                ref_cy = (lb[1]+lb[3])/2
                best_r = min(faces, key=lambda f: np.hypot(
                    (f[2][0]+f[2][2])/2 - ref_cx,
                    (f[2][1]+f[2][3])/2 - ref_cy))
                tight_r = best_r[2]
                tr_r    = tracking_manager.tracker.update_tracks(
                    [([tight_r[0], tight_r[1],
                       tight_r[2]-tight_r[0],
                       tight_r[3]-tight_r[1]], best_r[1], None)],
                    frame=frame_bgr)
                if tr_r:
                    bt_r = tr_r[0]
                    tracking_manager.force_track(
                        bt_r.track_id, [float(v) for v in tight_r])
                    consecutive_lost = 0
                    rej_count        = 0
                    print(f"[analyze_video] Recalibrage à t={t:.1f}s")
            else:
                tracking_manager.update(frame_bgr, detections)

            selected_bbox = tracking_manager.get_selected_bbox()

            if selected_bbox is None:
                consecutive_lost     += 1
                frames_tracking_lost += 1
                frames_results.append({
                    'timestamp':     float(t),
                    'tracking_lost': True,
                    'skip_reason':   'tracking_lost',
                    'emotion':       'inconnu',
                    'confidence':    0.0
                })
                continue

            consecutive_lost = 0
            x1, y1, x2, y2  = map(int, selected_bbox)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            cx_cur = (x1+x2)/2
            cy_cur = (y1+y2)/2

            if x2 <= x1 or y2 <= y1:
                frames_tracking_lost += 1
                frames_results.append({
                    'timestamp':     float(t),
                    'tracking_lost': True,
                    'skip_reason':   'bbox_invalide',
                    'emotion':       'inconnu',
                    'confidence':    0.0
                })
                continue

            face_img = frame_bgr[y1:y2, x1:x2]
            if face_img.size == 0:
                frames_tracking_lost += 1
                frames_results.append({
                    'timestamp':     float(t),
                    'tracking_lost': True,
                    'skip_reason':   'face_vide',
                    'emotion':       'inconnu',
                    'confidence':    0.0
                })
                continue

            tracking_manager.last_bbox = [x1, y1, x2, y2]

            gray_c     = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray_c))

            if brightness < CFG.MIN_BRIGHTNESS or brightness > CFG.MAX_BRIGHTNESS:
                frames_tracking_lost += 1
                reason = ('zone_sombre' if brightness < CFG.MIN_BRIGHTNESS
                          else 'zone_surexposee')
                frames_results.append({
                    'timestamp':     float(t),
                    'tracking_lost': True,
                    'skip_reason':   reason,
                    'emotion':       'inconnu',
                    'confidence':    0.0
                })
                continue

            if (memo_done and tracking_manager.zone.defined and
                    not tracking_manager.zone.contains(cx_cur, cy_cur)):
                frames_other_person += 1
                frames_results.append({
                    'timestamp':     float(t),
                    'tracking_lost': True,
                    'skip_reason':   'hors_zone',
                    'emotion':       'inconnu',
                    'confidence':    0.0
                })
                continue

          
            _force_strict_video = False
            if (memo_done and
                    brightness > 160 and
                    lock_t is not None and
                    tracking_manager.last_bbox is not None):
                ref_cx_pos = (tracking_manager.last_bbox[0] +
                              tracking_manager.last_bbox[2]) / 2
                if abs(cx_cur - ref_cx_pos) > 80:
                    print(f"[analyze_video] ⚡ t={t:.1f}s — position "
                          f"suspecte bright={brightness:.0f} "
                          f"cx={cx_cur:.0f} vs ref={ref_cx_pos:.0f} "
                          f"→ vérification stricte forcée")
                    _force_strict_video = True

            if memo_done:
                spd_ok, spd_corr = tracking_manager.speed.is_candidate(face_img)
                if not spd_ok:
                    frames_other_person += 1
                    frames_results.append({
                        'timestamp':     float(t),
                        'tracking_lost': True,
                        'skip_reason':   'autre_personne_hist',
                        'emotion':       'inconnu',
                        'confidence':    0.0
                    })
                    print(f"[analyze_video] t={t:.1f}s — "
                          f"rejet hist corr={spd_corr:.2f}")
                    continue

            if memo_done:
                blur_val = float(cv2.Laplacian(gray_c, cv2.CV_64F).var())

                
                dynamic_threshold = compute_dynamic_threshold(
                    brightness, blur_val)

                print(f"[analyze_video] t={t:.1f}s "
                      f"bright={brightness:.0f} blur={blur_val:.0f} "
                      f"seuil={dynamic_threshold:.2f}")

                pad             = 20
                face_img_padded = frame_bgr[
                    max(0,y1-pad):min(H,y2+pad),
                    max(0,x1-pad):min(W,x2+pad)]

               
                if _force_strict_video:
                    is_candidate, similarity = \
                        tracking_manager.identity.verify(
                            face_img=face_img_padded, strict_global=True)
                    print(f"[analyze_video] 🔒 Vérif. stricte "
                          f"sim={similarity:.3f} "
                          f"{'✅' if is_candidate else '❌'}")
                else:
                    is_candidate, similarity = \
                        tracking_manager.verify_candidate(
                            face_img_padded,
                            threshold=dynamic_threshold
                        )

                if not is_candidate:
                    rej_count += 1
                    total_failure = (similarity == 0.0)
                    tol = CFG.TOTAL_FAILURE_TOLERANCE if total_failure else 2
                    if rej_count < tol:
                        print(f"[analyze_video] t={t:.1f}s — "
                              f"échec isolé ({rej_count}/{tol}) toléré "
                              f"{'[ECHEC TOTAL]' if total_failure else ''}")
                        frames_results.append({
                            'timestamp':     float(t),
                            'tracking_lost': True,
                            'skip_reason':   'uncertain',
                            'emotion':       'inconnu',
                            'confidence':    0.0
                        })
                        continue

                    
                    found_alt_video   = False
                    best_alt_sim_v    = 0.0
                    best_alt_face_v   = None
                    if similarity == 0.0 and faces:
                        print(f"[analyze_video] t={t:.1f}s — Zone noire/"
                              f"échec total → recherche globale parmi "
                              f"{len(faces)} visage(s)")
                        for fd_v in faces:
                            tight_v = fd_v[2]
                            crop_v  = frame_bgr[
                                max(0,tight_v[1]):min(H,tight_v[3]),
                                max(0,tight_v[0]):min(W,tight_v[2])]
                            if crop_v.size == 0:
                                continue
                            if float(np.mean(cv2.cvtColor(
                                    crop_v, cv2.COLOR_BGR2GRAY))) < CFG.MIN_BRIGHTNESS:
                                continue
                            pad_v   = 20
                            fp_v    = frame_bgr[
                                max(0,tight_v[1]-pad_v):min(H,tight_v[3]+pad_v),
                                max(0,tight_v[0]-pad_v):min(W,tight_v[2]+pad_v)]
                            ok_v, sim_v = tracking_manager.identity.verify(
                                face_img=fp_v)
                            if ok_v and sim_v > best_alt_sim_v:
                                best_alt_sim_v  = sim_v
                                best_alt_face_v = fd_v
                                found_alt_video = True

                    if found_alt_video and best_alt_face_v is not None:
                        tight_v = best_alt_face_v[2]
                        cx_v    = (tight_v[0]+tight_v[2])/2
                        cy_v    = (tight_v[1]+tight_v[3])/2
                        conf_v  = best_alt_face_v[1]
                        tracks_v = tracking_manager.tracker.update_tracks(
                            [([tight_v[0], tight_v[1],
                               tight_v[2]-tight_v[0],
                               tight_v[3]-tight_v[1]], conf_v, None)],
                            frame=frame_bgr)
                        if tracks_v:
                            bt_v = min(tracks_v, key=lambda tr: np.hypot(
                                (tr.to_ltrb()[0]+tr.to_ltrb()[2])/2 - cx_v,
                                (tr.to_ltrb()[1]+tr.to_ltrb()[3])/2 - cy_v))
                            tracking_manager.force_track(
                                bt_v.track_id,
                                [float(v) for v in tight_v])
                            # Recalcul des coordonnées pour analyser
                            # CETTE frame avec le bon visage retrouvé
                            x1, y1, x2, y2 = [int(v) for v in tight_v]
                            x1 = max(0, x1); y1 = max(0, y1)
                            x2 = min(W, x2); y2 = min(H, y2)
                            cx_cur, cy_cur = cx_v, cy_v
                            face_img = frame_bgr[y1:y2, x1:x2]
                            face_img_padded = frame_bgr[
                                max(0,y1-pad):min(H,y2+pad),
                                max(0,x1-pad):min(W,x2+pad)]
                            tracking_manager.last_bbox = [x1, y1, x2, y2]
                            similarity = best_alt_sim_v
                            is_candidate = True
                            rej_count = 0
                            print(f"[analyze_video] ✅ Candidat retrouvé "
                                  f"sim={best_alt_sim_v:.3f} "
                                  f"({cx_v:.0f},{cy_v:.0f})")

                    if not is_candidate:
                        frames_other_person += 1
                        frames_results.append({
                            'timestamp':           float(t),
                            'tracking_lost':       True,
                            'skip_reason':         'autre_personne',
                            'identity_similarity': round(similarity, 3),
                            'emotion':             'inconnu',
                            'confidence':          0.0
                        })
                        print(f"[analyze_video] t={t:.1f}s — "
                              f"autre personne (sim={similarity:.2f}) ignorée")
                        continue

                rej_count = 0
                tracking_manager.update_candidate_embedding(face_img_padded)

                
                if (tracking_manager.zone.defined and
                        tracking_manager.zone._zone is not None):
                    old_cx_v = tracking_manager.zone._zone.get('cx', cx_cur)
                    old_cy_v = tracking_manager.zone._zone.get('cy', cy_cur)
                    new_cx_v = old_cx_v * 0.90 + cx_cur * 0.10
                    new_cy_v = old_cy_v * 0.90 + cy_cur * 0.10
                    if (abs(new_cx_v - old_cx_v) > 2 or
                            abs(new_cy_v - old_cy_v) > 2):
                        tracking_manager.zone.define(
                            new_cx_v, new_cy_v, W, H,
                            face_w=float(x2-x1), face_h=float(y2-y1),
                            n_faces=len(faces))
            else:
                similarity = 1.0
                rej_count  = 0

            bbox     = (x1, y1, x2, y2)
            face_rgb = cv2.cvtColor(
                preprocess_face(face_img), cv2.COLOR_BGR2RGB)
            vis_inputs = base_processor(
                images=face_rgb, return_tensors="pt").to(device)

            if USE_ONNX:
                pv_np         = vis_inputs['pixel_values'].cpu().numpy()
                vit_logits, _ = vit_session.run(
                    None, {"pixel_values": pv_np})
                probs = F.softmax(
                    torch.tensor(vit_logits), dim=-1).numpy()[0]
            else:
                with torch.no_grad():
                    probs = F.softmax(
                        model(vis_inputs['pixel_values']), dim=-1
                    ).cpu().numpy()[0]

           
            probs = correct_emotion_probs(face_img, probs)

            final_probs         = probs
            fusion_weights_list = None
            audio_probs_numpy   = None

            if has_audio and audio_data is not None:
                s_s     = int(max(0, (t-1.0)*16000))
                e_s     = int(min(len(audio_data), (t+1.0)*16000))
                segment = audio_data[s_s:e_s]
                if len(segment) > 0:
                    seg_in = audio_processor(
                        segment, sampling_rate=16000,
                        return_tensors="pt").to(device)
                    if USE_ONNX:
                        iv_np = seg_in['input_values'].cpu().numpy()
                        hl, _ = hubert_session.run(
                            None, {"input_values": iv_np})
                        audio_probs_numpy = F.softmax(
                            torch.tensor(hl), dim=-1).numpy()[0]
                    else:
                        with torch.no_grad():
                            audio_probs_numpy = F.softmax(
                                audio_model(seg_in['input_values']),
                                dim=-1).cpu().numpy()[0]

                    a7    = np.zeros(7)
                    a7[0] = audio_probs_numpy[2] * 0.2
                    a7[2] = audio_probs_numpy[3] * 0.2
                    a7[3] = audio_probs_numpy[0] * 1.6
                    a7[6] = audio_probs_numpy[1] * 1.2
                    if np.sum(a7) > 0:
                        a7 /= np.sum(a7)
                    vw = 0.90 if (probs[6]>0.20 or probs[5]>0.20) else 0.70
                    aw = 1.0 - vw
                    final_probs         = vw * probs + aw * a7
                    fusion_weights_list = [vw, aw]

            emotion, conf_score, cal_probs, prev_probs = \
                calibrate_and_smooth_probs(final_probs, prev_probs)

            visual_history.append(cal_probs)
            frames_candidate_analyzed += 1

            emotion_history_so_far = [
                f['emotion'] for f in frames_results
                if not f.get('tracking_lost', False)
            ]

           
            face_analysis_result_v = face_analyzer_local.analyze(frame_bgr)
            face_boost_v = None
            if face_analysis_result_v is not None:
                face_boost_v = face_analyzer_local.get_boost_params(
                    face_analysis_result_v)

            metrics = calculate_candidate_metrics(
                cal_probs,
                audio_probs_numpy,
                history=emotion_history_so_far,
                face_analysis=face_boost_v
            )

            entry = {
                'timestamp':           float(t),
                'emotion':             emotion,
                'emotion_fr':          EMOTION_NAMES_FR.get(emotion, emotion),
                'confidence':          float(conf_score),
                'bbox':                [int(c) for c in bbox],
                'metrics':             metrics,
                'candidate_confirmed': True,
                'identity_similarity': round(similarity, 3)
            }
            if fusion_weights_list:
                entry['fusion_weights'] = fusion_weights_list
            if audio_probs_numpy is not None:
                entry['audio_probs'] = audio_probs_numpy.tolist()
            frames_results.append(entry)
            time.sleep(0)
        video_clip.close()
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        # ← Transcription (Whisper + diarisation) désactivée : plus de
        # déchargement/rechargement GPU↔CPU de audio_model ici — cela
        # supprime aussi le risque de mismatch device avec les
        # inférences ViT concurrentes (ws_analyze_realtime notamment).
        # transcript_text / transcript_chunks restent vides.

        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass

        valid_frames = [f for f in frames_results
                        if not f.get('tracking_lost', False)]

        if not valid_frames:
            return ({
                'success': False,
                'message': (
                    'Aucune frame du candidat analysée. '
                    'Le candidat est-il visible dans la vidéo ? '
                    f'(seuil ArcFace={CFG.TOLERANCE_ARCFACE})'
                )
            }, 200)

        if frames_candidate_analyzed < 5:
            return ({
                'success': False,
                'message': (
                    f'Seulement {frames_candidate_analyzed} frame(s) '
                    f'du candidat — insuffisant.'
                )
            }, 200)

        print(f"[analyze_video] ✅ Résultat:")
        print(f"  Frames candidat   : {frames_candidate_analyzed}")
        print(f"  Autres personnes  : {frames_other_person} (ignorées)")
        print(f"  Tracking perdu    : {frames_tracking_lost} (ignorées)")

        avg_visual = np.mean(visual_history, axis=0)
        audio_probs_all = [
            np.array(f['audio_probs']) for f in frames_results
            if f.get('audio_probs') is not None
        ]
        avg_audio_probs = (np.mean(audio_probs_all, axis=0)
                           if audio_probs_all else None)

        full_emotion_history = [
            f['emotion'] for f in frames_results
            if not f.get('tracking_lost', False)
        ]
        final_metrics = calculate_candidate_metrics(
            avg_visual,
            audio_probs=avg_audio_probs,
            history=full_emotion_history,
            use_full_history=True
        )

        # ← Le ton de la voix (audio_probs) reste analysé normalement.
        # L'ajustement basé sur la cohérence du DISCOURS transcrit est
        # neutralisé automatiquement puisque transcript_text est
        # toujours vide (speech_risk_score restera 0.0 via
        # analyze_speech_deception("") → aucun ajustement appliqué).
        speech_risk_score, speech_flags = analyze_speech_deception(transcript_text)
        coherence_note = None
        if transcript_text and speech_risk_score > 0:
            coherence_penalty = speech_risk_score / 100.0
            comm_before = final_metrics['communication']
            assur_before = final_metrics['assurance_level']

            final_metrics['communication'] = float(max(0, min(100,
                final_metrics['communication'] * (1 - coherence_penalty * 0.35))))
            final_metrics['assurance_level'] = float(max(0, min(100,
                final_metrics['assurance_level'] * (1 - coherence_penalty * 0.25))))

            final_metrics['global_score'] = float(max(0, min(100,
                final_metrics['stress_management']    * 0.25 +
                final_metrics['communication']        * 0.28 +
                final_metrics['assurance_level']      * 0.25 +
                final_metrics['expressivity']         * 0.12 +
                final_metrics['prediction_stability'] * 0.10
            )))

            if speech_risk_score > 35:
                coherence_note = (
                    f"Ton vocal perçu comme posé/confiant, mais le discours "
                    f"transcrit présente des signes d'hésitation ou de "
                    f"sur-justification (score {speech_risk_score:.0f}/100). "
                    f"Communication ajustée de {comm_before:.0f}% à "
                    f"{final_metrics['communication']:.0f}%, assurance de "
                    f"{assur_before:.0f}% à {final_metrics['assurance_level']:.0f}%."
                )

        # ← Soft skills désactivés — plus calculés.
        soft_skills = {}
        inconsistencies = detect_inconsistencies(
            transcript_text, visual_history)
        if coherence_note:
            inconsistencies.append(coherence_note)

        timeline = [
            {"time": f['timestamp'], "emotion": f['emotion'],
             "confidence": f['confidence']}
            for f in frames_results if not f.get('tracking_lost', False)
        ]

        metrics_history = [
            {"time":       f['timestamp'],
             "stress":     f['metrics']['stress_management'],
             "comm":       f['metrics']['communication'],
             "expr":       f['metrics']['expressivity'],
             "speed":      f['metrics'].get('verbal_fluidity_score', 50),
             "conf":       f['metrics']['assurance_level'],
             "model_conf": f['confidence'] * 100}
            for f in frames_results if 'metrics' in f
        ]

        valid_emotions    = [f['emotion']    for f in valid_frames]
        valid_confidences = [f['confidence'] for f in valid_frames]
        valid_timestamps  = [f['timestamp']  for f in valid_frames]

        if len(valid_emotions) < 5:
              risk_level       = "Analyse insuffisante"
              final_score      = None          # ← ne plus mettre 0 : ce n'est pas un score, c'est une absence de mesure
              level_code       = "insufficient"  # ← nouveau, permet au PDF de distinguer "calme" de "non mesuré"
              vis_risk_details = {}
        else:
             vis_risk_score, _, vis_risk_details = calculate_behavioral_tension_signals(
             valid_emotions, valid_confidences, valid_timestamps)
             speech_risk, speech_flags = analyze_speech_patterns(
                transcript_text)
             final_score = vis_risk_score * 0.6 + speech_risk * 0.4
             risk_level  = (
              "Faible — peu de signes de tension observés"
              if final_score < 30 else
                "Modéré — quelques signes de tension, à explorer si pertinent"
              if final_score < 60 else
               "Élevé — signes de tension fréquents (ne préjuge pas de la "
               "sincérité du candidat)"
               )
             level_code = ("low" if final_score < 30 else
                  "medium" if final_score < 60 else "high")  # ← nouveau
             inconsistencies.extend(speech_flags)
        ai_feedback = []

        hs = [t for t in timeline
              if t['emotion'] in ['fear', 'sad']
              and t['confidence'] > 0.6]
        if hs:
            ai_feedback.append({
                "timestamp": hs[0]['time'],
                "reason":    "Appréhension détectée",
                "feedback":  (f"À {hs[0]['time']:.0f}s, signes "
                              f"d'appréhension — posez une question "
                              f"de mise à l'aise.")
            })

        ht = [t for t in timeline
              if t['emotion'] == 'angry'
              and t['confidence'] > 0.70]
        if ht:
            ai_feedback.append({
                "timestamp": ht[0]['time'],
                "reason":    "Concentration élevée",
                "feedback":  (f"À {ht[0]['time']:.0f}s, forte "
                              f"concentration — signe d'implication.")
            })

        jm = [t for t in timeline
              if t['emotion'] == 'happy'
              and t['confidence'] > 0.8]
        if jm:
            ai_feedback.append({
                "timestamp": jm[0]['time'],
                "reason":    "Engagement positif",
                "feedback":  (f"Enthousiasme à {jm[0]['time']:.0f}s — "
                              f"bonne dynamique.")
            })

        sp = [t for t in timeline
              if t['emotion'] == 'surprise'
              and t['confidence'] > 0.6]
        if sp:
            ai_feedback.append({
                "timestamp": sp[0]['time'],
                "reason":    "Réactivité",
                "feedback":  (f"Surprise à {sp[0]['time']:.0f}s — "
                              f"candidat réactif et attentif.")
            })

        if not ai_feedback:
            ai_feedback.append({
                "timestamp": 0,
                "reason":    "Vue d'ensemble",
                "feedback":  "Comportement globalement stable et professionnel."
            })

        pt      = time.time() - start_time
        avg_c   = np.mean([f['confidence'] for f in valid_frames]) * 100
        fps_val = len(frames_results) / pt if pt > 0 else 0
        lat_ms  = (pt / len(frames_results) * 1000
                   if frames_results else 0)

        system_kpis = {
            "real_time_factor":     round(duration / pt, 2) if pt > 0 else 0,
            "avg_confidence":       round(avg_c, 1),
            "fps":                  round(fps_val, 1),
            "latency_ms":           round(lat_ms, 2),
            "processing_time":      round(pt, 2),
            "frames_candidate":     frames_candidate_analyzed,
            "frames_other_person":  frames_other_person,
            "frames_tracking_lost": frames_tracking_lost,
            "identity_protection":  "active" if memo_done else "partial",
            "memorization_done":    memo_done,
            "lock_timestamp":       lock_t,
        }

        deception_timeline = []
        prev_em_dt         = None
        for idx, f in enumerate(frames_results):
            if f.get('tracking_lost', False):
                prev_em_dt = None
                continue
            t_v = f['timestamp']
            em  = f['emotion']
            co  = f['confidence']

            is_signal = (
                (em in ['fear', 'disgust'] and co > 0.55) or
                (em == 'angry' and prev_em_dt == 'fear' and co > 0.55)
            )
            if is_signal:
                deception_timeline.append({
                    "time":        float(t_v),
                    "type":        "Stress Émotionnel",
                    "severity":    "Élevée" if co > 0.75 else "Moyenne",
                    "description": (
                        f"Pic de tension "
                        f"({EMOTION_NAMES_FR.get(em, em)}) "
                        f"{int(co*100)}%.")
                })

            if idx >= 1 and not frames_results[idx-1].get('tracking_lost'):
                pe = frames_results[idx-1]['emotion']
                ce = f['emotion']
                if pe != 'neutral' and ce != 'neutral' and pe != ce:
                    deception_timeline.append({
                        "time":        float(t_v),
                        "type":        "Micro-expression",
                        "severity":    "Moyenne",
                        "description": (
                            f"Changement "
                            f"'{EMOTION_NAMES_FR.get(pe, pe)}'"
                            f" → '{EMOTION_NAMES_FR.get(ce, ce)}'.")
                    })
            prev_em_dt = em

        deception_timeline.sort(key=lambda x: x['time'])

        # ← Transcription désactivée : transcript_formatted reste vide
        # (format_transcript_by_speaker / assign_speakers_to_transcript
        # supprimés, plus jamais appelés — voir fin du fichier).
        transcript_formatted = ""

        response_payload = {
            "success":            True,
            "duration_seconds":   round(duration, 1), 
            "transcript":         transcript_text,
            "transcript_formatted": transcript_formatted,  
            "transcript_chunks":  transcript_chunks,
            "preview_faces":      preview_faces,
            "face_popup":         len(preview_faces) == 0,
            "audio_popup":        not has_audio,
            "frames":             frames_results,
            "timeline":           timeline,
            "metrics_history":    metrics_history,
            "system_kpis":        system_kpis,
            "metrics":            final_metrics,
            "soft_skills":        soft_skills,
            "analysis": {
                "score":              final_score,
                "level":              risk_level,
                "level_code":         level_code,
                "details":            (vis_risk_details
                                       if len(valid_emotions) >= 5 else {}),
                "behavioral_timeline": deception_timeline,
                "deception_timeline": deception_timeline,  # alias rétro-compat
                "disclaimer": (
                    "Ce score reflète la fréquence de signaux de tension "
                    "observés (visage/discours). Il ne mesure PAS la "
                    "sincérité du candidat et ne doit pas être interprété "
                    "comme tel — le stress d'entretien est normal, y "
                    "compris chez des candidats honnêtes."
                ),
            },
            "metric_uncertainty": compute_metric_uncertainty(metrics_history),
            "feedback":           ai_feedback,
            "inconsistencies":    inconsistencies,
            "strengths": {
                k: final_metrics.get(k, 0)
                for k in ['stress_management', 'communication',
                          'assurance_level', 'expressivity',
                          'verbal_fluidity_score']
            },
            "suggestions": [
                *(["Améliorer la gestion du stress "
                   f"({final_metrics.get('stress_management',0):.1f}%)."]
                  if final_metrics.get('stress_management', 0) < 70 else []),
                *(["Travailler la communication "
                   f"({final_metrics.get('communication',0):.1f}%)."]
                  if final_metrics.get('communication', 0) < 70 else []),
                *(["Renforcer l'assurance "
                   f"({final_metrics.get('assurance_level',0):.1f}%)."]
                  if final_metrics.get('assurance_level', 0) < 70 else []),
                *(["Améliorer l'expressivité "
                   f"({final_metrics.get('expressivity',0):.1f}%)."]
                  if final_metrics.get('expressivity', 0) < 70 else []),
            ],
            "tracking": {
                "track_id":            tracking_manager.selected_track_id,
                "tracking_lost":       tracking_manager.is_tracking_lost(),
                "frames_analyzed":     frames_candidate_analyzed,
                "frames_other_person": frames_other_person,
                "identity_active":     memo_done,
                "memorization_done":   memo_done,
                "lock_timestamp":      lock_t,
            }
        }
        return (convert_to_serializable(response_payload), 200)

    except Exception as e:
        print(f"[analyze_video] Erreur: {e}")
        import traceback
        traceback.print_exc()
        for p in [locals().get('tmp_path'), locals().get('audio_path')]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        return ({'success': False, 'error': str(e)}, 500)


@app.post("/analyze_video")
async def analyze_video(
        file: UploadFile = File(...),
        target_x: Optional[float] = Form(None),
        target_y: Optional[float]  = Form(None),
        embedding_key: Optional[str] = Form(None),
):
    file_bytes = await file.read()
    loop = asyncio.get_running_loop()
    payload, status_code = await loop.run_in_executor(
        video_processing_executor, _analyze_video_sync,
        file_bytes, file.filename, target_x, target_y, embedding_key
    )
    return JSONResponse(payload, status_code=status_code)


# ═══════════════════════════════════════════════════════════════════
# WEBSOCKET — helpers
# ═══════════════════════════════════════════════════════════════════

async def _run_sync(loop, pool, fn, *args, **kwargs):
    """
    ← CORRECTIF STRUCTUREL : déporte n'importe quel appel bloquant
    (ArcFace, ViT, MediaPipe...) vers un thread du pool, sans jamais
    changer la logique de l'appelant. Avant ce patch, ws_analyze_
    realtime appelait ces fonctions DIRECTEMENT dans la coroutine —
    chaque frame bloquait la boucle asyncio pendant toute la durée du
    calcul (observé entre 80 et 150ms par frame dans les logs de
    production), empêchant le serveur de traiter toute autre
    connexion WebSocket pendant ce temps.

    Ce helper reproduit exactement le principe déjà utilisé pour
    _decode_frame (`await loop.run_in_executor(executor, _decode_frame,
    frame_b64)`), généralisé aux fonctions avec arguments nommés
    (via functools.partial, puisque run_in_executor ne supporte que
    des arguments positionnels nativement).

    Usage :
        result = await _run_sync(loop, executor, ma_fonction,
                                  arg1, kw=valeur)
    au lieu de :
        result = ma_fonction(arg1, kw=valeur)
    — comportement strictement identique, seule l'exécution est
    déportée vers un thread séparé.
    """
    return await loop.run_in_executor(
        pool, functools.partial(fn, *args, **kwargs))


def _decode_frame(frame_b64: str):
    frame_bytes = base64.b64decode(frame_b64)
    nparr       = np.frombuffer(frame_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _process_audio_sync(audio_bytes: bytes):
    tmp_in = wav_path = None
    try:
        if audio_bytes[:4] == b'RIFF':
            suffix = ".wav"
        elif audio_bytes[:4] == b'\x1aE\xdf\xa3':
            suffix = ".webm"
        else:
            suffix = ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_in = tmp.name
        wav_path = tmp_in + "_conv.wav"
        conv = subprocess.run(
            [FFMPEG_PATH, '-y', '-i', tmp_in,
             '-ar', '16000', '-ac', '1', '-f', 'wav', wav_path],
            capture_output=True, timeout=15
        )
        if conv.returncode != 0 or not os.path.exists(wav_path):
            return {"transcript": "", "audio_probs": None}
        y, _ = sf.read(wav_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if len(y) <= 1600:
            return {"transcript": "", "audio_probs": None}

        # ← Garde-fou RMS existant (rapide) : coupe le cas trivial
        # "silence pur" à coût quasi nul.
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms <= 0.003:
            return {"transcript": "", "audio_probs": None, "status": "Silence"}

        # ← Transcription désactivée — seule l'émotion vocale (HuBERT)
        # est calculée, indépendamment de Whisper.
        transcript = ""

        ai         = audio_processor(y, sampling_rate=16000,
                                     return_tensors="pt").to(device)
        with torch.no_grad():
            ap = F.softmax(
                audio_model(ai['input_values']), dim=-1
            ).cpu().numpy()[0]
        return {"transcript": transcript,
                "audio_probs": ap.tolist(), "status": "Clair"}
    except Exception as e:
        print(f"[Audio] Erreur: {e}")
        return {"transcript": "", "audio_probs": None}
    finally:
        for p in [tmp_in, wav_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

async def _process_audio_async(audio_b64: str, websocket: WebSocket, loop):
    try:
        audio_bytes = base64.b64decode(audio_b64)
        result      = await loop.run_in_executor(
            executor, _process_audio_sync, audio_bytes
        )
        if websocket.client_state.name != "CONNECTED":
            return
        if result.get("transcript"):
            await websocket.send_json({
                "type": "transcript", "text": result["transcript"]
            })
        if result.get("status"):
            await websocket.send_json({
                "type": "audio_status", "status": result["status"]
            })
    except Exception as e:
        print(f"[Audio async] Erreur envoi: {e}")


async def heartbeat(websocket: WebSocket):
    while True:
        try:
            await asyncio.sleep(10)
            if websocket.client_state.name != "CONNECTED":
                break
            await websocket.send_json({"type": "heartbeat"})
        except (WebSocketDisconnect, Exception):
            break


def _extract_embedding_from_face(face_img):
    try:
        face_rgb = cv2.cvtColor(preprocess_face(face_img), cv2.COLOR_BGR2RGB)
        inputs   = base_processor(images=face_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.extract_features(inputs['pixel_values']).cpu().numpy()[0]
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
    except Exception as e:
        print(f"[Embedding] Erreur: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# ws_analyze_realtime — Version Finale v23 + PATCH v6.1 + PATCH RESET
# ═══════════════════════════════════════════════════════════════════
@app.websocket("/ws/analyze_realtime")
async def ws_analyze_realtime(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    tm                 = TrackingManager(shared_arcface=shared_arcface,
                                         fast_arcface=fast_arcface)
    face_analyzer_global.reset()
    face_analyzer      = face_analyzer_global
    emotion_ws_history = []
    active_audio_task  = None
    heartbeat_task     = asyncio.create_task(heartbeat(websocket))

    _frame_counter     = 0
    _arcface_every     = 1
    _last_arcface_ok   = True
    _last_similarity   = 1.0
    _skip_arcface      = False
    _n_candidates      = 1

    # ← AJOUT : cache du dernier résultat MediaPipe (regard, clignement,
    # posture, tension, symétrie) — voir explication au point d'usage
    # plus bas dans la boucle temps réel.
    _last_face_analysis_result = None
    _last_face_boost = {}

    _last_emotion      = 'neutral'
    _last_confidence   = 0.5
    _last_metrics: dict = {}
    _frames_analyzed   = 0
    _metrics_history_ws: list = []

    locked_cx: Optional[float] = None
    locked_cy: Optional[float] = None
    _needs_lock  = False
    _cache_key: Optional[tuple] = None
    _cache_sim: Optional[float] = None
    _rej_count  = 0
    _memo_fails = 0
    audio_path  = None

    _initial_zone_cx: Optional[float] = None
    _initial_zone_cy: Optional[float] = None
    _embedding_key: Optional[str] = None
    _pending_embedding: Optional[tuple] = None

    async def _send_absent(reason: str, bbox, sim: float = 0.0):
        # ← PATCH v7.0 (#19) : expose la cause diagnostiquée plutôt
        # qu'un simple "absent" opaque, pour permettre de distinguer
        # un vrai départ du candidat d'un problème de qualité vidéo.
        fail_reason = getattr(tm.identity, "last_fail_reason", None)
        r = {
            "success":             True,
            "faces_detected":      False,
            "candidate_status":    "absent",
            "warning":             reason,
            "absent_reason":       fail_reason,
            "absent_reason_label": ABSENCE_REASON_LABELS_FR.get(
                fail_reason, "Cause indéterminée"),
            "identity_similarity": round(sim, 3),
            "bbox":                list(bbox) if bbox else [],
            "tracking": {
                "track_id":              tm.track_id,
                "tracking_lost":         True,
                "bbox":                  tm.last_bbox,
                "identity_active":       tm.identity.memorized,
                "memorizing":            tm.is_memorizing(),
                "memorization_progress": tm.identity.progress,
            }
        }
        await websocket.send_json(convert_to_serializable(r))
    def _find_and_lock(faces, frame, frame_w, frame_h,
                       cx_ref, cy_ref,
                       conserve_id=True, first_lock=False,
                       exclure_cx=None, exclure_cy=None):
        nonlocal locked_cx, locked_cy, _needs_lock
        nonlocal _cache_key, _cache_sim, _rej_count, _memo_fails
        nonlocal _initial_zone_cx, _initial_zone_cy

        best_fd   = None
        best_sim  = 0.0
        best_dist = float('inf')

        if first_lock:
            passes   = ["global"]
            dist_max = 150.0
        else:
            passes   = ["zone", "global"] if tm.zone.defined else ["global"]
            dist_max = float('inf')

        for passe in passes:
            for fd in faces:
                tight = fd[2]
                cx_fd = (tight[0]+tight[2])/2
                cy_fd = (tight[1]+tight[3])/2
                dist  = np.hypot(cx_fd - cx_ref, cy_fd - cy_ref)

                if exclure_cx is not None:
                    if abs(cx_fd-exclure_cx) < 15 and abs(cy_fd-exclure_cy) < 15:
                        continue

                if first_lock and dist > dist_max:
                    continue

                if passe == "zone" and not first_lock:
                    if not tm.zone.contains(cx_fd, cy_fd):
                        continue

                crop = frame[max(0,tight[1]):min(frame_h,tight[3]),
                             max(0,tight[0]):min(frame_w,tight[2])]
                if crop.size == 0 or float(np.mean(cv2.cvtColor(
                        crop, cv2.COLOR_BGR2GRAY))) < CFG.MIN_BRIGHTNESS:
                    continue

                if tm.identity.memorized:
                    spd_ok, spd_corr = tm.speed.is_candidate(fd[0])
                    if not spd_ok:
                        continue
                    pad   = 20
                    f_pad = frame[max(0,tight[1]-pad):min(frame_h,tight[3]+pad),
                                  max(0,tight[0]-pad):min(frame_w,tight[2]+pad)]
                    is_global_search = (passe == "global")
                    if tm.identity.use_arcface:
                        ok, sim = tm.identity.verify(
                            face_img=f_pad, strict_global=is_global_search)
                    else:
                        emb     = _extract_embedding_from_face(fd[0])
                        ok, sim = tm.identity.verify(
                            face_img=f_pad, embedding=emb,
                            strict_global=is_global_search)
                    print(f"[WS] Cherche P={passe} ({cx_fd:.0f},{cy_fd:.0f}): "
                          f"sim={sim:.3f} dist={dist:.0f} {'✅' if ok else '❌'}")
                    if ok and sim > best_sim:
                        best_sim = sim; best_fd = fd
                else:
                    if dist < best_dist:
                        best_dist = dist; best_fd = fd; best_sim = 1.0

            if best_fd is not None and tm.identity.memorized:
                break

        if best_fd is None:
            return False, 0.0

        tight = best_fd[2]
        cx    = (tight[0]+tight[2])/2
        cy    = (tight[1]+tight[3])/2
        conf  = best_fd[1]

        tracks = tm.tracker.update_tracks(
            [([tight[0], tight[1], tight[2]-tight[0], tight[3]-tight[1]],
              conf, None)], frame=frame)
        if not tracks:
            return False, 0.0

        bt = min(tracks, key=lambda t: np.hypot(
            (t.to_ltrb()[0]+t.to_ltrb()[2])/2 - cx,
            (t.to_ltrb()[1]+t.to_ltrb()[3])/2 - cy))

        bbox = [float(tight[0]),float(tight[1]),float(tight[2]),float(tight[3])]

        if conserve_id:
            tm.force_track(bt.track_id, bbox)
        else:
            tm.select(bt.track_id, bbox)

        locked_cx = cx; locked_cy = cy
        _needs_lock = False
        _cache_key  = None; _cache_sim = None
        _rej_count  = 0; _memo_fails = 0

        print(f"[WS] ✅ Lock id={bt.track_id} ({cx:.0f},{cy:.0f}) "
              f"sim={best_sim:.3f} first={first_lock} dist={best_dist:.0f}px")
        return True, best_sim

    def _arcface_embed_quick(face_img) -> Tuple[Optional[np.ndarray], bool]:
        """
        ← PATCH v7.1b : aligné sur IdentityManager._arcface_embed — deux
        passes (stricte MIN_DET_SCORE, puis secours MIN_DET_SCORE_FALLBACK)
        au lieu d'un seuil unique 0.35. Retourne (embedding, is_low_conf) :
        is_low_conf=True si le résultat vient de la passe de secours, pour
        que l'appelant exige une similarité plus stricte en compensation
        (CFG.TOLERANCE_ARCFACE_FALLBACK) — sinon le garde-fou ajouté dans
        IdentityManager est contourné par ce chemin de code séparé.
        """
        if shared_arcface is None or face_img is None or face_img.size == 0:
            return None, False
        h, w = face_img.shape[:2]
        if w < 30 or h < 30:
            return None, False
        try:
            c = np.zeros((640, 640, 3), dtype=np.uint8)
            c[170:470, 170:470] = cv2.resize(face_img, (300, 300),
                interpolation=cv2.INTER_LINEAR)
            faces = shared_arcface.get(c)
            if faces:
                best = max(faces, key=lambda f: f.det_score)
                if best.det_score >= CFG.MIN_DET_SCORE:
                    emb  = best.embedding.astype(np.float32)
                    norm = np.linalg.norm(emb)
                    return (emb/norm if norm > 0 else emb), False
        except Exception:
            pass
        try:
            small = cv2.resize(face_img, (112, 112),
                               interpolation=cv2.INTER_LINEAR)
            faces = shared_arcface.get(small)
            if faces:
                best = max(faces, key=lambda f: f.det_score)
                if best.det_score >= CFG.MIN_DET_SCORE:
                    emb  = best.embedding.astype(np.float32)
                    norm = np.linalg.norm(emb)
                    return (emb/norm if norm > 0 else emb), False
        except Exception:
            pass

        # ← Passe de secours (confiance faible signalée à l'appelant)
        try:
            c = np.zeros((640, 640, 3), dtype=np.uint8)
            c[170:470, 170:470] = cv2.resize(face_img, (300, 300),
                interpolation=cv2.INTER_LINEAR)
            faces = shared_arcface.get(c)
            if faces:
                best = max(faces, key=lambda f: f.det_score)
                if best.det_score >= CFG.MIN_DET_SCORE_FALLBACK:
                    emb  = best.embedding.astype(np.float32)
                    norm = np.linalg.norm(emb)
                    return (emb/norm if norm > 0 else emb), True
        except Exception:
            pass
        try:
            small = cv2.resize(face_img, (112, 112),
                               interpolation=cv2.INTER_LINEAR)
            faces = shared_arcface.get(small)
            if faces:
                best = max(faces, key=lambda f: f.det_score)
                if best.det_score >= CFG.MIN_DET_SCORE_FALLBACK:
                    emb  = best.embedding.astype(np.float32)
                    norm = np.linalg.norm(emb)
                    return (emb/norm if norm > 0 else emb), True
        except Exception:
            pass
        return None, False
    try:
        await websocket.send_json({
            "type":    "connected",
            "message": "TalenCube — v23 zoom filter + adaptive threshold + MediaPipe + patch v6.1"
        })

        while True:
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_json(), timeout=15.0)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break

            # ← PATCH RESET : réinitialisation complète du tracking
            # (bouton "Arrêter" ou nouveau fichier/vidéo chargé côté
            # Angular) — remet tout l'état à zéro SANS fermer cette
            # connexion WebSocket.
            if msg.get("reset") is True:
                print("[WS] 🔄 Reset demandé — réinitialisation complète du tracking")
                tm = TrackingManager(shared_arcface=shared_arcface,
                                     fast_arcface=fast_arcface)
                face_analyzer_global.reset()
                emotion_ws_history.clear()
                _frame_counter      = 0
                _last_arcface_ok    = True
                _last_similarity    = 1.0
                _n_candidates       = 1
                # ← AJOUT : évite de réutiliser un résultat MediaPipe
                # de l'ancienne session/candidat après un reset.
                _last_face_analysis_result = None
                _last_face_boost    = {}
                _last_emotion       = 'neutral'
                _last_confidence    = 0.5
                _last_metrics       = {}
                _frames_analyzed    = 0
                locked_cx           = None
                locked_cy           = None
                _needs_lock         = False
                _cache_key          = None
                _cache_sim          = None
                _rej_count          = 0
                _memo_fails         = 0
                _initial_zone_cx    = None
                _initial_zone_cy    = None
                _embedding_key      = None
                _pending_embedding  = None
                _metrics_history_ws.clear()
                await websocket.send_json({"success": True, "type": "reset_ack"})
                continue

            frame_b64      = msg.get("frame")
            audio_b64      = msg.get("audio")
            click_x        = msg.get("click_x") or msg.get("clickX")
            click_y        = msg.get("click_y") or msg.get("clickY")
            is_first_frame = (msg.get("is_first_frame", False) or
                              msg.get("isFirstFrame", False))
            embedding_key  = (msg.get("embedding_key") or
                              msg.get("embeddingKey"))

            if embedding_key:
                _embedding_key = embedding_key
                print(f"[WS] 🔑 embedding_key reçu: {embedding_key}")

            if not frame_b64:
                await websocket.send_json({"success": False,
                                           "error": "Pas de frame"})
                continue

            img = await loop.run_in_executor(executor, _decode_frame, frame_b64)
            if img is None:
                await websocket.send_json({"success": False,
                                           "error": "Image invalide"})
                continue

            H, W = img.shape[:2]

            if click_x is not None and click_y is not None:
                new_cx = float(max(0, min(W-1, click_x)))
                new_cy = float(max(0, min(H-1, click_y)))
                if new_cx != locked_cx or new_cy != locked_cy:
                    _needs_lock = True
                    tm.zone.reset()
                    tm.speed.reset()
                    _initial_zone_cx = None
                    _initial_zone_cy = None
                    _pending_embedding = None
                locked_cx = new_cx; locked_cy = new_cy
                print(f"[WS] Coords ({locked_cx:.0f},{locked_cy:.0f}) {W}x{H}")

            # ← RUN_IN_EXECUTOR (chemin principal) : détection YOLO
            # appelée à CHAQUE frame — déportée vers le pool de threads
            # pour ne plus jamais bloquer la boucle asyncio.
            raw_faces_all = await _run_sync(loop, executor, detect_faces, img)
            zoom_limit    = H * 0.28

            raw_faces_filtered = [
                fd for fd in raw_faces_all
                if (fd[2][1] + fd[2][3]) / 2 >= zoom_limit
            ]

            if raw_faces_filtered:
                raw_faces = raw_faces_filtered
                if len(raw_faces_all) != len(raw_faces_filtered):
                    print(f"[WS] 🎭 Miniatures filtrées: "
                          f"{len(raw_faces_all) - len(raw_faces_filtered)} ignorée(s) "
                          f"→ {len(raw_faces_filtered)} visage(s) retenus")
            else:
                raw_faces = raw_faces_all

            detections = [(x1,y1,x2,y2,c)
                for _,c,(x1,y1,x2,y2),_ in raw_faces
                if (x2-x1) > 20 and (y2-y1) > 20]

            valid_faces = []
            for fd in raw_faces:
                _,_,tight,_ = fd
                crop = img[max(0,tight[1]):min(H,tight[3]),
                           max(0,tight[0]):min(W,tight[2])]
                if crop.size > 0 and float(np.mean(cv2.cvtColor(
                        crop, cv2.COLOR_BGR2GRAY))) >= CFG.MIN_BRIGHTNESS:
                    valid_faces.append(fd)
            faces = valid_faces if valid_faces else raw_faces

            _n_candidates = max(1, len(raw_faces))

            if tm.state == State.ABANDONED:
                tm.reset_tracking()
                _cache_key = None; _cache_sim = None
                _rej_count = 0; _memo_fails = 0

            if (_embedding_key is not None and
                    _pending_embedding is None and
                    not tm.identity.memorized):
                from app_embedding_cache import get_embedding
                stored_emb, stored_face = get_embedding(_embedding_key)
                if stored_emb is not None:
                    _pending_embedding = (stored_emb, stored_face)
                    print(f"[WS] 🎯 Embedding chargé depuis cache "
                          f"— scan actif démarré")
                else:
                    print(f"[WS] ⚠️ Embedding key introuvable — "
                          f"fallback mémorisation normale")
                _embedding_key = None

            if (_pending_embedding is not None and
                    not tm.identity.memorized and raw_faces):

                stored_emb, stored_face = _pending_embedding
                best_fd    = None
                best_sim_s = 0.0

                for fd in raw_faces:
                    tight_s = fd[2]
                    crop_s  = img[max(0,tight_s[1]):min(H,tight_s[3]),
                                  max(0,tight_s[0]):min(W,tight_s[2])]
                    if crop_s.size == 0:
                        continue
                    if float(np.mean(cv2.cvtColor(
                            crop_s, cv2.COLOR_BGR2GRAY))) < CFG.MIN_BRIGHTNESS:
                        continue
                    pad_s  = 20
                    fp_s   = img[max(0,tight_s[1]-pad_s):min(H,tight_s[3]+pad_s),
                                 max(0,tight_s[0]-pad_s):min(W,tight_s[2]+pad_s)]
                    emb_s, is_low_conf_s = await _run_sync(
                        loop, executor, _arcface_embed_quick, fp_s)
                    if emb_s is None:
                        continue
                    sim_s = float(np.dot(stored_emb, emb_s))
                    effective_threshold_s = (
                        CFG.TOLERANCE_ARCFACE_FALLBACK if is_low_conf_s
                        else CFG.TOLERANCE_ARCFACE)
                    cx_s = (tight_s[0]+tight_s[2])//2
                    cy_s = (tight_s[1]+tight_s[3])//2
                    print(f"[WS] 🔍 Scan ({cx_s},{cy_s}) "
                          f"sim={sim_s:.3f} seuil={effective_threshold_s:.2f}"
                          f"{' [secours]' if is_low_conf_s else ''} "
                          f"{'✅' if sim_s >= effective_threshold_s else '❌'}")
                    if sim_s >= effective_threshold_s and sim_s > best_sim_s:
                        best_sim_s = sim_s
                        best_fd    = fd

                if best_fd is None and locked_cx is not None and raw_faces:
                    best_coords = None
                    best_dist_c = float('inf')

                    for fd_c in raw_faces:
                        tight_c = fd_c[2]
                        cx_c = (tight_c[0]+tight_c[2])/2
                        cy_c = (tight_c[1]+tight_c[3])/2
                        dist_c  = np.hypot(cx_c - locked_cx, cy_c - locked_cy)
                        cy_diff = abs(cy_c - locked_cy)

                        if dist_c < 60 and cy_diff < 40:
                            print(f"[WS] Fallback candidat "
                                  f"({cx_c:.0f},{cy_c:.0f}) "
                                  f"dist={dist_c:.0f}px ✅")
                            if dist_c < best_dist_c:
                                best_dist_c = dist_c
                                best_coords = fd_c
                        else:
                            print(f"[WS] Fallback rejeté "
                                  f"({cx_c:.0f},{cy_c:.0f}) "
                                  f"dist={dist_c:.0f}px ❌")

                    if best_coords is not None:
                        print(f"[WS] ⚠️ Embedding faible → fallback coords "
                              f"dist={best_dist_c:.0f}px")
                        best_fd    = best_coords
                        best_sim_s = 0.5

                if best_fd is not None:
                    tight_m = best_fd[2]
                    cx_m    = (tight_m[0]+tight_m[2])/2
                    cy_m    = (tight_m[1]+tight_m[3])/2
                    conf_m  = best_fd[1]

                    _n_candidates = len(raw_faces)
                    _arcface_every = 1 if _n_candidates > 1 else 3
                    print(f"[WS] ArcFace every={_arcface_every} frames "
                          f"(n_candidats={_n_candidates})")

                    tracks_m = tm.tracker.update_tracks(
                        [([tight_m[0], tight_m[1],
                           tight_m[2]-tight_m[0],
                           tight_m[3]-tight_m[1]], conf_m, None)],
                        frame=img)

                    if tracks_m:
                        bt_m = min(tracks_m, key=lambda t: np.hypot(
                            (t.to_ltrb()[0]+t.to_ltrb()[2])/2 - cx_m,
                            (t.to_ltrb()[1]+t.to_ltrb()[3])/2 - cy_m))

                        tm._track_id    = bt_m.track_id
                        tm._initialized = True
                        tm._save_bbox([float(tight_m[0]), float(tight_m[1]),
                                       float(tight_m[2]), float(tight_m[3])])
                        tm.kalman.update(cx_m, cy_m,
                                         tight_m[2]-tight_m[0],
                                         tight_m[3]-tight_m[1])

                        pad_ref = 20
                        fp_ref  = img[
                            max(0,int(tight_m[1])-pad_ref):min(H,int(tight_m[3])+pad_ref),
                            max(0,int(tight_m[0])-pad_ref):min(W,int(tight_m[2])+pad_ref)]
                        live_emb, _ = _arcface_embed_quick(fp_ref)
                        ref_emb  = live_emb if live_emb is not None else stored_emb
                        print(f"[WS] 🔄 Référence depuis frame "
                              f"{'live' if live_emb is not None else 'cache'}")

                        face_live = img[
                            max(0,int(tight_m[1])):min(H,int(tight_m[3])),
                            max(0,int(tight_m[0])):min(W,int(tight_m[2]))]
                        face_for_speed = (face_live
                                          if face_live.size > 0
                                          else stored_face)

                        tm.set_reference_embedding(
                            ref_emb, face_img=face_for_speed)

                        ok_t, sim_t = tm.identity.verify(face_img=fp_ref)
                        print(f"[WS] Auto-test live: sim={sim_t:.3f} "
                              f"{'✅' if ok_t else '❌'}")

                        tm.zone.define(cx_m, cy_m, W, H,
                                       face_w=float(tight_m[2]-tight_m[0]),
                                       face_h=float(tight_m[3]-tight_m[1]),
                                       n_faces=len(raw_faces))
                        _initial_zone_cx = cx_m
                        _initial_zone_cy = cy_m
                        locked_cx        = cx_m
                        locked_cy        = cy_m
                        _pending_embedding = None
                        _needs_lock        = False

                        print(f"[WS] ✅ Candidat trouvé et mémorisé! "
                              f"sim={best_sim_s:.3f} "
                              f"({cx_m:.0f},{cy_m:.0f})")
                    else:
                        result_wait = {
                            "success": True, "faces_detected": False,
                            "candidate_status": "absent",
                            "warning": "Candidat détecté — init tracker...",
                            "tracking": {
                                "track_id": None, "tracking_lost": True,
                                "bbox": None, "identity_active": False,
                                "memorizing": False,
                                "memorization_progress": 0}
                        }
                        await websocket.send_json(
                            convert_to_serializable(result_wait))
                        if (audio_b64 and (not active_audio_task or
                                active_audio_task.done())):
                            active_audio_task = asyncio.create_task(
                                _process_audio_async(
                                    audio_b64, websocket, loop))
                        continue
                else:
                    result_wait = {
                        "success": True, "faces_detected": False,
                        "candidate_status": "absent",
                        "warning": "En attente d'apparition du candidat...",
                        "tracking": {
                            "track_id": None, "tracking_lost": True,
                            "bbox": None, "identity_active": False,
                            "memorizing": False, "memorization_progress": 0}
                    }
                    await websocket.send_json(
                        convert_to_serializable(result_wait))
                    if (audio_b64 and (not active_audio_task or
                            active_audio_task.done())):
                        active_audio_task = asyncio.create_task(
                            _process_audio_async(audio_b64, websocket, loop))
                    continue

            is_tracking_ok = (tm.state == State.TRACKING and
                              tm.identity.memorized)

            should_lock = (
                (_needs_lock or is_first_frame) and
                not tm.identity.memorized and
                not tm.identity.memorizing and
                not is_tracking_ok and
                _pending_embedding is None and
                locked_cx is not None and
                bool(faces) and
                any(np.hypot((f[2][0]+f[2][2])/2 - locked_cx,
                             (f[2][1]+f[2][3])/2 - locked_cy) < 150
                    for f in faces)
            )

            if should_lock:
                print(f"[WS] Tentative verrouillage first={is_first_frame} "
                      f"nb_visages={len(faces)}")
                ok, _ = await _run_sync(
                    loop, executor, _find_and_lock,
                    faces, img, W, H, locked_cx, locked_cy,
                    conserve_id=False, first_lock=is_first_frame)
                if not ok:
                    print("[WS] Verrouillage échoué — attente")

            tm.set_id_rejected(False)
            # ← RUN_IN_EXECUTOR (chemin principal) : tm.update() invoque
            # DeepSort, dont l'embedder MobileNet interne fait aussi du
            # calcul CPU bloquant — appelé à chaque frame.
            await _run_sync(loop, executor, tm.update, img, detections)

            sel_bbox     = tm.get_bbox()
            display_bbox = sel_bbox or tm.last_bbox

            result = {
                "success":          True,
                "faces_detected":   sel_bbox is not None,
                "candidate_status": "unknown",
                "tracking": {
                    "track_id":              tm.track_id,
                    "tracking_lost":         tm.is_lost(),
                    "bbox":                  display_bbox,
                    "identity_active":       tm.identity.memorized,
                    "memorizing":            tm.is_memorizing(),
                    "memorization_progress": tm.identity.progress,
                }
            }

            if tm.is_memorizing() and sel_bbox is not None:
                x1,y1,x2,y2 = [max(0,int(v)) for v in sel_bbox]
                x2 = min(W,x2); y2 = min(H,y2)
                if x2 > x1 and y2 > y1:
                    pad      = 20
                    face_img = img[max(0,y1-pad):min(H,y2+pad),
                                   max(0,x1-pad):min(W,x2+pad)]
                    gray     = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    bright   = float(np.mean(gray))
                    if bright < 40 or face_img.size == 0:
                        _memo_fails += 1
                        result["candidate_status"] = "memorizing"
                        result["warning"] = f"Zone sombre ({bright:.0f})"
                        if _memo_fails >= CFG.MAX_MEMO_FAILS:
                            print("[WS] Mémorisation bloquée → reset")
                            tm.reset_tracking()
                            _needs_lock = True; _memo_fails = 0
                        await websocket.send_json(
                            convert_to_serializable(result))
                        if (audio_b64 and (not active_audio_task or
                                active_audio_task.done())):
                            active_audio_task = asyncio.create_task(
                                _process_audio_async(
                                    audio_b64, websocket, loop))
                        continue
                    done = tm.memorize_frame(face_img)
                    if not done and tm.is_memorizing():
                        _memo_fails += 1
                        print(f"[WS] ArcFace échoué "
                              f"{_memo_fails}/{CFG.MAX_MEMO_FAILS}")
                        if _memo_fails >= CFG.MAX_MEMO_FAILS:
                            print("[WS] ArcFace bloqué → reset")
                            tm.reset_tracking()
                            _needs_lock = True; _memo_fails = 0
                    else:
                        _memo_fails = 0
                    result["candidate_status"] = (
                        "memorized" if done else "memorizing")
                    result["warning"] = (
                        "✅ Candidat mémorisé — analyse démarrée" if done else
                        f"🔍 Mémorisation "
                        f"({tm.identity.progress}/{CFG.MEMORIZE_FRAMES})")
                    result["bbox"] = [x1,y1,x2,y2]
                    if done:
                        _cache_key = None; _cache_sim = None; _memo_fails = 0
                        cx_mem = (x1+x2)/2; cy_mem = (y1+y2)/2
                        _initial_zone_cx = cx_mem; _initial_zone_cy = cy_mem
                        tm.zone.define(cx_mem, cy_mem, W, H,
                                       face_w=float(x2-x1),
                                       face_h=float(y2-y1),
                                       n_faces=len(raw_faces))
                        print(f"[WS] ✅ Zone FIXE: ({cx_mem:.0f},{cy_mem:.0f}) "
                              f"face={x2-x1:.0f}x{y2-y1:.0f}px")
                        ok_t, sim_t = tm.identity.verify(face_img=face_img)
                        print(f"[WS] Auto-test: sim={sim_t:.3f} "
                              f"{'✅' if ok_t else '❌'}")
                await websocket.send_json(convert_to_serializable(result))
                if (audio_b64 and (not active_audio_task or
                        active_audio_task.done())):
                    active_audio_task = asyncio.create_task(
                        _process_audio_async(audio_b64, websocket, loop))
                continue

            if sel_bbox is None:
                if tm.is_lost() and locked_cx is not None:
                    result["candidate_status"] = "absent"
                    result["warning"]          = "Candidat absent — en attente"
                elif locked_cx is None:
                    result["candidate_status"] = "not_selected"
                    result["warning"]          = "Sélectionnez un candidat"
                await websocket.send_json(convert_to_serializable(result))
                if (audio_b64 and (not active_audio_task or
                        active_audio_task.done())):
                    active_audio_task = asyncio.create_task(
                        _process_audio_async(audio_b64, websocket, loop))
                continue

            x1,y1,x2,y2 = [max(0,int(v)) for v in sel_bbox]
            x2 = min(W,x2); y2 = min(H,y2)
            cx_cur = (x1+x2)/2; cy_cur = (y1+y2)/2

            if (x2<=x1 or y2<=y1 or
                    (x2-x1)<CFG.MIN_BBOX_PX or (y2-y1)<CFG.MIN_BBOX_PX):
                tm.set_id_rejected(True); _cache_key=None; _cache_sim=None
                await _send_absent("Bbox invalide", [x1,y1,x2,y2])
                if (audio_b64 and (not active_audio_task or
                        active_audio_task.done())):
                    active_audio_task = asyncio.create_task(
                        _process_audio_async(audio_b64, websocket, loop))
                continue

            pad             = 20
            face_img        = img[y1:y2, x1:x2]
            face_img_padded = img[max(0,y1-pad):min(H,y2+pad),
                                  max(0,x1-pad):min(W,x2+pad)]

            if face_img.size == 0:
                tm.set_id_rejected(True); _cache_key=None; _cache_sim=None
                await _send_absent("Visage vide", [x1,y1,x2,y2])
                if (audio_b64 and (not active_audio_task or
                        active_audio_task.done())):
                    active_audio_task = asyncio.create_task(
                        _process_audio_async(audio_b64, websocket, loop))
                continue

            gray_c = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            bright = float(np.mean(gray_c))

            blur_val_ws = float(cv2.Laplacian(gray_c, cv2.CV_64F).var())

            # ← Appel à la fonction partagée (voir en-tête du fichier)
            # au lieu du calcul dupliqué avec _analyze_video_sync —
            # une seule version de cette règle désormais.
            dynamic_threshold_ws = compute_dynamic_threshold(
                bright, blur_val_ws)

            print(f"[WS] bright={bright:.0f} blur={blur_val_ws:.0f} "
                  f"seuil={dynamic_threshold_ws:.2f}")

            if bright < CFG.MIN_BRIGHTNESS or bright > CFG.MAX_BRIGHTNESS:
                zone_label = ("sombre" if bright < CFG.MIN_BRIGHTNESS
                              else "surexposée")
                tm.set_id_rejected(True)
                _cache_key=None; _cache_sim=None
                found_dark = False
                if raw_faces and tm.identity.memorized:
                    print(f"[WS] 🔦 Zone {zone_label} ({bright:.0f}) → "
                          f"recherche parmi {len(raw_faces)} visage(s)")
                    best_dk_sim = 0.0; best_dk_fd = None
                    for fd_k in raw_faces:
                        t_k = fd_k[2]
                        c_k = img[max(0,t_k[1]):min(H,t_k[3]),
                                  max(0,t_k[0]):min(W,t_k[2])]
                        if c_k.size == 0: continue
                        if float(np.mean(cv2.cvtColor(
                                c_k, cv2.COLOR_BGR2GRAY))) < CFG.MIN_BRIGHTNESS:
                            continue
                        pad_k = 20
                        fp_k  = img[max(0,t_k[1]-pad_k):min(H,t_k[3]+pad_k),
                                    max(0,t_k[0]-pad_k):min(W,t_k[2]+pad_k)]
                        if tm.identity.use_arcface:
                            ok_k, sim_k = await _run_sync(
                                loop, executor, tm.identity.verify,
                                face_img=fp_k, strict_global=True)
                        else:
                            e_k = await _run_sync(
                                loop, executor, _extract_embedding_from_face, fd_k[0])
                            ok_k, sim_k = await _run_sync(
                                loop, executor, tm.identity.verify,
                                face_img=fp_k, embedding=e_k,
                                strict_global=True)
                        cx_k = (t_k[0]+t_k[2])//2; cy_k = (t_k[1]+t_k[3])//2
                        print(f"[WS] 🔦 [STRICT] ({cx_k},{cy_k}): "
                              f"sim={sim_k:.3f} {'✅' if ok_k else '❌'}")
                        if ok_k and sim_k > best_dk_sim:
                            best_dk_sim = sim_k; best_dk_fd = fd_k
                    if best_dk_fd is not None:
                        t_k  = best_dk_fd[2]
                        cx_k = (t_k[0]+t_k[2])/2; cy_k = (t_k[1]+t_k[3])/2
                        tr_k = tm.tracker.update_tracks(
                            [([t_k[0],t_k[1],t_k[2]-t_k[0],
                               t_k[3]-t_k[1]], best_dk_fd[1], None)],
                            frame=img)
                        if tr_k:
                            bt_k = min(tr_k, key=lambda t: np.hypot(
                                (t.to_ltrb()[0]+t.to_ltrb()[2])/2-cx_k,
                                (t.to_ltrb()[1]+t.to_ltrb()[3])/2-cy_k))
                            bx_k = [float(t_k[0]),float(t_k[1]),
                                    float(t_k[2]),float(t_k[3])]
                            tm.force_track(bt_k.track_id, bx_k)
                            locked_cx = cx_k; locked_cy = cy_k
                            _initial_zone_cx = cx_k; _initial_zone_cy = cy_k
                            tm.zone.define(cx_k, cy_k, W, H)
                            _cache_key = None; _cache_sim = None; _rej_count = 0
                            found_dark = True
                            print(f"[WS] ✅ Sorti zone {zone_label} "
                                  f"({cx_k:.0f},{cy_k:.0f}) "
                                  f"sim={best_dk_sim:.3f}")
                            result["tracking"]["bbox"] = bx_k
                            result["tracking"]["tracking_lost"] = False
                            result["faces_detected"] = True
                            f_k = img[max(0,int(t_k[1])):min(H,int(t_k[3])),
                                      max(0,int(t_k[0])):min(W,int(t_k[2]))]
                            if f_k.size > 0:
                                em_k, cf_k, _, _ = await _run_sync(
                                    loop, executor, predict_emotion_enhanced, f_k)
                                gr_k  = cv2.cvtColor(f_k, cv2.COLOR_BGR2GRAY)
                                br_k  = float(np.mean(gr_k))
                                bl_k  = float(cv2.Laplacian(
                                    gr_k, cv2.CV_64F).var())
                                ip_k  = base_processor(
                                    images=cv2.cvtColor(
                                        preprocess_face(f_k),
                                        cv2.COLOR_BGR2RGB),
                                    return_tensors="pt").to(device)
                                with torch.no_grad():
                                    vp_k = F.softmax(
                                        model(ip_k['pixel_values']),
                                        dim=-1).cpu().numpy()[0]
                                mt_k = calculate_candidate_metrics(
                                    vp_k,
                                    history=emotion_ws_history
                                )
                                fs_k = ('Sombre' if br_k<45 else
                                        'Exposé' if br_k>220 else
                                        'Flou' if bl_k<50 else 'Optimal')
                                result.update({
                                    "emotion": str(em_k),
                                    "emotion_fr": str(EMOTION_NAMES_FR.get(
                                        em_k, em_k)),
                                    "emoji": str(EMOTION_EMOJIS.get(em_k,'')),
                                    "confidence": float(cf_k),
                                    "candidate_status": "present",
                                    "identity_similarity": round(
                                        best_dk_sim, 3),
                                    "candidate_metrics": {
                                        k: float(v) for k,v in mt_k.items()},
                                    "bbox": [max(0,int(t_k[0])),
                                             max(0,int(t_k[1])),
                                             min(W,int(t_k[2])),
                                             min(H,int(t_k[3]))],
                                    "reliability": {
                                        "face": {
                                            "brightness": round(br_k,1),
                                            "blur":       round(bl_k,1),
                                            "status":     fs_k},
                                        "audio": {"status": "En traitement..."}}
                                })
                                await websocket.send_json(
                                    convert_to_serializable(result))
                                if (audio_b64 and (not active_audio_task or
                                        active_audio_task.done())):
                                    active_audio_task = asyncio.create_task(
                                        _process_audio_async(
                                            audio_b64, websocket, loop))
                                continue
                if not found_dark:
                    await _send_absent(
                        f"Zone {zone_label} ({bright:.0f})", [x1,y1,x2,y2])
                    if (audio_b64 and (not active_audio_task or
                            active_audio_task.done())):
                        active_audio_task = asyncio.create_task(
                            _process_audio_async(audio_b64, websocket, loop))
                continue

            if tm.zone.defined and not tm.zone.contains(cx_cur, cy_cur):
                print(f"[WS] ⚡ Hors zone ({cx_cur:.0f},{cy_cur:.0f}) → absent")
                tm.set_id_rejected(True); _cache_key=None; _cache_sim=None
                await _send_absent("Candidat hors zone", [x1,y1,x2,y2])
                if (audio_b64 and (not active_audio_task or
                        active_audio_task.done())):
                    active_audio_task = asyncio.create_task(
                        _process_audio_async(audio_b64, websocket, loop))
                continue

            _force_strict_check = False
            if (tm.identity.memorized and
                    bright > 160 and
                    locked_cx is not None and
                    abs(cx_cur - locked_cx) > 80):
                print(f"[WS] ⚡ Position suspecte "
                      f"bright={bright:.0f} "
                      f"cx={cx_cur:.0f} vs ref={locked_cx:.0f} "
                      f"→ vérification stricte forcée")
                _force_strict_check = True

            # ← RUN_IN_EXECUTOR (chemin principal) : comparaison
            # d'histogramme HSV, appelée à chaque frame.
            spd_ok, spd_corr = await _run_sync(
                loop, executor, tm.speed.is_candidate, face_img)
            if not spd_ok:
                print(f"[WS] ⚡ Rejet hist corr={spd_corr:.2f}")
                tm.set_id_rejected(True); _cache_key=None; _cache_sim=None
                await _send_absent(
                    f"Autre personne (hist={spd_corr:.2f})", [x1,y1,x2,y2])
                if (audio_b64 and (not active_audio_task or
                        active_audio_task.done())):
                    active_audio_task = asyncio.create_task(
                        _process_audio_async(audio_b64, websocket, loop))
                continue

            _frame_counter += 1

            # ← Appel à la fonction partagée (voir en-tête du fichier)
            jump_suspect = detect_jump(cx_cur, cy_cur, locked_cx, locked_cy, W)
            if jump_suspect:
                jump_dist = np.hypot(cx_cur - locked_cx, cy_cur - locked_cy)
                print(f"[WS] 🔀 Saut spatial suspect "
                      f"({jump_dist:.0f}px > "
                      f"{CFG.JUMP_DISTANCE_RATIO*W:.0f}px)")

            use_strict = jump_suspect or _force_strict_check
            # ← RUN_IN_EXECUTOR (chemin principal) : vérification
            # ArcFace, appelée à CHAQUE frame — c'est l'appel le plus
            # fréquent de toute la fonction.
            if tm.identity.use_arcface:
                is_candidate, similarity = await _run_sync(
                    loop, executor, tm.identity.verify,
                    face_img=face_img_padded,
                    strict_global=use_strict,
                    threshold=(None if use_strict
                               else dynamic_threshold_ws))
            else:
                emb = await _run_sync(
                    loop, executor, _extract_embedding_from_face, face_img)
                is_candidate, similarity = await _run_sync(
                    loop, executor, tm.identity.verify,
                    face_img=face_img_padded,
                    embedding=emb,
                    strict_global=use_strict,
                    threshold=(None if use_strict
                               else dynamic_threshold_ws))
            if use_strict:
                print(f"[WS] 🔒 Vérif. stricte sim={similarity:.3f} "
                      f"seuil={CFG.TOLERANCE_REID_GLOBAL} "
                      f"{'✅' if is_candidate else '❌'}")
            _last_arcface_ok = is_candidate
            _last_similarity = similarity
            print(f"[WS] ArcFace frame {_frame_counter} "
                  f"sim={similarity:.3f} seuil={dynamic_threshold_ws:.2f} "
                  f"{'✅' if is_candidate else '❌'}")


            if not is_candidate:
                _rej_count += 1

                total_failure = (similarity == 0.0)
                _tol = (CFG.TOTAL_FAILURE_TOLERANCE if total_failure
                        else (5 if _n_candidates == 1 else 3))
                if _rej_count < _tol:
                    print(f"[WS] ⚠️ Échec temporaire "
                          f"({_rej_count}/{_tol}) sim={similarity:.3f} — "
                          f"attente confirmation "
                          f"{'[ECHEC TOTAL]' if total_failure else ''}")
                    result["candidate_status"] = "uncertain"
                    result["warning"] = (
                        f"Vérification... ({_rej_count}/{_tol})")
                    result["bbox"]    = [x1,y1,x2,y2]
                    await websocket.send_json(convert_to_serializable(result))
                    if (audio_b64 and (not active_audio_task or
                            active_audio_task.done())):
                        active_audio_task = asyncio.create_task(
                            _process_audio_async(audio_b64, websocket, loop))
                    continue

                tm.set_id_rejected(True); _cache_key=None; _cache_sim=None

                if similarity == 0.0 and faces:
                    print(f"[WS] Zone noire → recherche globale "
                          f"parmi {len(faces)} visage(s)")
                    found_alt = False; best_alt_sim = 0.0; best_alt_face = None
                    for fd in raw_faces:
                        tight_a = fd[2]
                        crop_a  = img[max(0,tight_a[1]):min(H,tight_a[3]),
                                      max(0,tight_a[0]):min(W,tight_a[2])]
                        if crop_a.size == 0: continue
                        if float(np.mean(cv2.cvtColor(
                                crop_a, cv2.COLOR_BGR2GRAY))) < CFG.MIN_BRIGHTNESS:
                            continue
                        pad_a   = 20
                        f_pad_a = img[
                            max(0,tight_a[1]-pad_a):min(H,tight_a[3]+pad_a),
                            max(0,tight_a[0]-pad_a):min(W,tight_a[2]+pad_a)]
                        if tm.identity.use_arcface:
                            ok_a, sim_a = await _run_sync(
                                loop, executor, tm.identity.verify, face_img=f_pad_a)
                        else:
                            emb_a = await _run_sync(
                                loop, executor, _extract_embedding_from_face, fd[0])
                            ok_a, sim_a = await _run_sync(
                                loop, executor, tm.identity.verify,
                                face_img=f_pad_a, embedding=emb_a)
                        cx_a = (tight_a[0]+tight_a[2])//2
                        cy_a = (tight_a[1]+tight_a[3])//2
                        print(f"[WS] Global ({cx_a},{cy_a}): "
                              f"sim={sim_a:.3f} {'✅' if ok_a else '❌'}")
                        if ok_a and sim_a > best_alt_sim:
                            best_alt_sim  = sim_a
                            best_alt_face = fd
                            found_alt     = True
                    if found_alt and best_alt_face is not None:
                        tight_r  = best_alt_face[2]
                        cx_r     = (tight_r[0]+tight_r[2])/2
                        cy_r     = (tight_r[1]+tight_r[3])/2
                        conf_r   = best_alt_face[1]
                        tracks_r = tm.tracker.update_tracks(
                            [([tight_r[0],tight_r[1],
                               tight_r[2]-tight_r[0],
                               tight_r[3]-tight_r[1]], conf_r, None)],
                            frame=img)
                        if tracks_r:
                            bt_r   = min(tracks_r, key=lambda t: np.hypot(
                                (t.to_ltrb()[0]+t.to_ltrb()[2])/2 - cx_r,
                                (t.to_ltrb()[1]+t.to_ltrb()[3])/2 - cy_r))
                            bbox_r = [float(tight_r[0]),float(tight_r[1]),
                                      float(tight_r[2]),float(tight_r[3])]
                            tm.force_track(bt_r.track_id, bbox_r)
                            locked_cx = cx_r; locked_cy = cy_r
                            _rej_count = 0
                            tm.zone.define(cx_r, cy_r, W, H)
                            _initial_zone_cx = cx_r; _initial_zone_cy = cy_r
                            result["tracking"]["bbox"]          = bbox_r
                            result["tracking"]["tracking_lost"] = False
                            print(f"[WS] ✅ Candidat retrouvé "
                                  f"sim={best_alt_sim:.3f} "
                                  f"({cx_r:.0f},{cy_r:.0f})")
                            face_img_r = img[
                                max(0,int(tight_r[1])):min(H,int(tight_r[3])),
                                max(0,int(tight_r[0])):min(W,int(tight_r[2]))]
                            if face_img_r.size > 0:
                                emotion_r, conf_r2, _, _ = await _run_sync(
                                    loop, executor, predict_emotion_enhanced, face_img_r)
                                gray_r2   = cv2.cvtColor(
                                    face_img_r, cv2.COLOR_BGR2GRAY)
                                bright_r2 = float(np.mean(gray_r2))
                                blur_r2   = float(cv2.Laplacian(
                                    gray_r2, cv2.CV_64F).var())
                                inp_r2    = base_processor(
                                    images=cv2.cvtColor(
                                        preprocess_face(face_img_r),
                                        cv2.COLOR_BGR2RGB),
                                    return_tensors="pt").to(device)
                                with torch.no_grad():
                                    vp_r2 = F.softmax(
                                        model(inp_r2['pixel_values']),
                                        dim=-1).cpu().numpy()[0]
                                metrics_r2 = calculate_candidate_metrics(
                                    vp_r2,
                                    history=emotion_ws_history
                                )
                                fst_r2 = ('Sombre' if bright_r2<45 else
                                          'Exposé' if bright_r2>220 else
                                          'Flou' if blur_r2<50 else 'Optimal')
                                x1r,y1r,x2r,y2r = (
                                    max(0,int(tight_r[0])),
                                    max(0,int(tight_r[1])),
                                    min(W,int(tight_r[2])),
                                    min(H,int(tight_r[3])))
                                result.update({
                                    "emotion": str(emotion_r),
                                    "emotion_fr": str(EMOTION_NAMES_FR.get(
                                        emotion_r, emotion_r)),
                                    "emoji": str(EMOTION_EMOJIS.get(
                                        emotion_r,'')),
                                    "confidence": float(conf_r2),
                                    "candidate_status": "present",
                                    "identity_similarity": round(
                                        best_alt_sim, 3),
                                    "candidate_metrics": {
                                        k: float(v)
                                        for k,v in metrics_r2.items()},
                                    "bbox": [x1r,y1r,x2r,y2r],
                                    "reliability": {
                                        "face": {
                                            "brightness": round(bright_r2,1),
                                            "blur":       round(blur_r2,1),
                                            "status":     fst_r2},
                                        "audio": {
                                            "status": "En traitement..."}}
                                })
                                result["tracking"]["bbox"] = [x1r,y1r,x2r,y2r]
                                result["tracking"]["tracking_lost"] = False
                                result["faces_detected"] = True
                                await websocket.send_json(
                                    convert_to_serializable(result))
                                if (audio_b64 and (not active_audio_task or
                                        active_audio_task.done())):
                                    active_audio_task = asyncio.create_task(
                                        _process_audio_async(
                                            audio_b64, websocket, loop))
                                continue

                reason = ("Zone noire — recherche échouée"
                          if similarity == 0.0
                          else f"Candidat absent (sim={similarity:.2f})")
                print(f"[WS] ❌ {reason}")
                await _send_absent(reason, [x1,y1,x2,y2], similarity)
                if (audio_b64 and (not active_audio_task or
                        active_audio_task.done())):
                    active_audio_task = asyncio.create_task(
                        _process_audio_async(audio_b64, websocket, loop))
                continue

            _rej_count = 0
            # ← RUN_IN_EXECUTOR (chemin principal) : mise à jour de la
            # référence ArcFace, appelée à chaque frame validée.
            await _run_sync(loop, executor, tm.identity.update,
                            face_img=face_img_padded)
            locked_cx = cx_cur; locked_cy = cy_cur

            if tm.zone.defined and tm.zone._zone is not None:
                old_cx = tm.zone._zone.get('cx', cx_cur)
                old_cy = tm.zone._zone.get('cy', cy_cur)
                new_cx = old_cx * 0.90 + cx_cur * 0.10
                new_cy = old_cy * 0.90 + cy_cur * 0.10
                if abs(new_cx - old_cx) > 2 or abs(new_cy - old_cy) > 2:
                    tm.zone.define(new_cx, new_cy, W, H,
                                   face_w=float(x2-x1),
                                   face_h=float(y2-y1),
                                   n_faces=_n_candidates)

            # ← RUN_IN_EXECUTOR (chemin principal) : inférence ViT
            # d'émotion, appelée à chaque frame validée — c'est
            # l'appel le plus coûteux après ArcFace (modèle de deep
            # learning complet).
            emotion, confidence, _, v_probs = await _run_sync(
                loop, executor, predict_emotion_enhanced, face_img)

            if confidence < 0.40:
                result.update({
                    "candidate_status": "uncertain",
                    "warning": f"Confiance faible ({confidence*100:.0f}%)",
                    "bbox":    [x1,y1,x2,y2]
                })
                await websocket.send_json(convert_to_serializable(result))
                if (audio_b64 and (not active_audio_task or
                        active_audio_task.done())):
                    active_audio_task = asyncio.create_task(
                        _process_audio_async(audio_b64, websocket, loop))
                continue

            emotion_ws_history.append(emotion)
            if len(emotion_ws_history) > 30:
                emotion_ws_history.pop(0)

            if _frames_analyzed > 0 and _last_emotion:
                quick_result = {
                    "success":          True,
                    "candidate_status": "present",
                    "emotion":          str(_last_emotion),
                    "emotion_fr":       str(EMOTION_NAMES_FR.get(
                                            _last_emotion, _last_emotion)),
                    "emoji":            str(EMOTION_EMOJIS.get(
                                            _last_emotion, '😐')),
                    "confidence":       float(_last_confidence),
                    "bbox":             [x1, y1, x2, y2],
                    "candidate_metrics": {k: float(v)
                                          for k, v in _last_metrics.items()},
                    "identity_similarity": round(similarity, 3),
                    "partial":          True,
                    "tracking": {
                        "track_id":              tm.track_id,
                        "tracking_lost":         False,
                        "bbox":                  [x1, y1, x2, y2],
                        "identity_active":       tm.identity.memorized,
                        "memorizing":            False,
                        "memorization_progress": tm.identity.progress,
                    }
                }
                await websocket.send_json(
                    convert_to_serializable(quick_result))
            # ← OPTIMISATION PERF (temps réel uniquement) : MediaPipe
            # (FaceLandmarker + PoseLandmarker, 2 inférences) ne tourne
            # plus qu'UNE FRAME SUR DEUX, au lieu de chaque frame. Sur
            # les frames intermédiaires, on réutilise le dernier
            # résultat calculé. Justifié par la nature du signal :
            # regard, clignement, posture, tension et symétrie évoluent
            # nettement plus lentement d'une frame à l'autre que
            # l'expression faciale — contrairement à l'émotion (ViT) et
            # à l'identité (ArcFace), qui elles restent calculées à
            # CHAQUE frame pour ne perdre aucune précision temporelle
            # sur ce qui compte le plus. Réduit d'environ moitié le
            # coût CPU de cette étape spécifique dans le flux temps
            # réel, sans dégrader l'émotion ni l'identité.
            if _frame_counter % 2 == 0:
                face_analysis_result = await _run_sync(
                    loop, executor, face_analyzer.analyze, img)
                face_boost = face_analyzer.get_boost_params(face_analysis_result)
                _last_face_analysis_result = face_analysis_result
                _last_face_boost = face_boost
            else:
                face_analysis_result = _last_face_analysis_result
                face_boost = _last_face_boost

            # ← RUN_IN_EXECUTOR (chemin principal) : deuxième inférence
            # ViT (probabilités brutes pour calculate_candidate_metrics,
            # distincte de predict_emotion_enhanced ci-dessus) — utilise
            # le helper _vit_raw_probs_sync extrait plus haut dans le
            # fichier, désormais déportée comme les autres appels lourds.
            metrics  = calculate_candidate_metrics(
                v_probs,
                history=emotion_ws_history,
                face_analysis=face_boost
            )

            # ← PATCH v7.0 (#22) : accumulation bornée (30 dernières frames
            # ≈ quelques dizaines de secondes) pour calculer une fourchette
            # d'incertitude au lieu d'un chiffre figé. Bornage nécessaire
            # ici : contrairement à _analyze_video_sync (appelé une fois
            # par vidéo), ws_analyze_realtime tourne en continu pendant
            # toute la session — une liste non bornée grandirait sans
            # limite sur un entretien de 45-60 minutes.
            _metrics_history_ws.append(metrics)
            if len(_metrics_history_ws) > 30:
                _metrics_history_ws.pop(0)

            blur_val = float(cv2.Laplacian(gray_c, cv2.CV_64F).var())
            fst      = ('Sombre' if bright < 45 else
                        'Exposé' if bright > 220 else
                        'Flou' if blur_val < 50 else 'Optimal')

            result.update({
                "emotion":             str(emotion),
                "emotion_fr":          str(EMOTION_NAMES_FR.get(
                    emotion, emotion)),
                "emoji":               str(EMOTION_EMOJIS.get(emotion, '')),
                "confidence":          float(confidence),
                "candidate_status":    "present",
                "identity_similarity": round(similarity, 3),
                "candidate_metrics":   {k: float(v) for k,v in metrics.items()},
                "metric_uncertainty":  compute_metric_uncertainty(_metrics_history_ws),
                "bbox":                [x1,y1,x2,y2],
                "reliability": {
                    "face":  {"brightness": round(bright,1),
                              "blur":       round(blur_val,1),
                              "status":     fst},
                    "audio": {"status": "En traitement..."}
                }
            })

            if face_analysis_result is not None:
                result["face_analysis"] = face_analyzer.get_result_dict(
                    face_analysis_result
                )

            _last_emotion    = emotion
            _last_confidence = float(confidence)
            _last_metrics    = metrics
            _frames_analyzed += 1

            await websocket.send_json(convert_to_serializable(result))
            if (audio_b64 and (not active_audio_task or
                    active_audio_task.done())):
                active_audio_task = asyncio.create_task(
                    _process_audio_async(audio_b64, websocket, loop))

    except WebSocketDisconnect:
        print("[WS] Déconnexion client")
    except Exception as e:
        print(f"[WS] Erreur: {e}")
        import traceback; traceback.print_exc()
    finally:
        face_analyzer_global.reset()
        emotion_ws_history.clear()
        _frames_analyzed = 0
        _last_metrics    = {}
        for task in [heartbeat_task, active_audio_task]:
            try:
                if task and not task.done(): task.cancel()
            except Exception: pass
        if audio_path and os.path.exists(audio_path):
            try: os.remove(audio_path)
            except Exception: pass
        try: await websocket.close()
        except Exception: pass
        print("[WS] Session terminée")


@app.post("/analyze_realtime")
async def analyze_realtime(
        frame: UploadFile = File(...),
        audio: Optional[UploadFile] = File(None),
        click_x: Optional[int] = Form(None),
        click_y: Optional[int] = Form(None),
        is_first_frame: Optional[bool] = Form(False)
):
    transcript = ""
    audio_probs = y = v_probs = None
    face_status = 'Aucun visage détecté'
    brightness = blur_val = 0

    try:
        contents = await frame.read()
        nparr    = np.frombuffer(contents, np.uint8)
        img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({'success': False, 'error': 'Image invalide'})

        faces     = detect_faces(img)
        face_img  = bbox = None
        all_faces = []
        emotion   = "None"
        confidence = 0

        if faces:
            all_faces = [list(f[2]) for f in faces]
            # ← FIX : le "else" manquant faisait que la sélection par clic
            # (best = min(...)) était systématiquement écrasée par la
            # sélection du plus grand visage juste après, ET provoquait
            # un appel ViT (predict_emotion_enhanced) en double, son
            # premier résultat n'étant jamais utilisé. Le clic candidat
            # ne fonctionnait donc plus du tout sur cet endpoint.
            if click_x is not None and click_y is not None:
                best = min(faces, key=lambda f: np.hypot(
                    (f[2][0]+f[2][2])/2 - click_x,
                    (f[2][1]+f[2][3])/2 - click_y
                ))
            else:
                best = max(faces,
                           key=lambda x: (x[2][2]-x[2][0])*(x[2][3]-x[2][1]))
            face_img, _, bbox, _ = best
            emotion, confidence, _, _ = predict_emotion_enhanced(face_img)

        if audio:
            audio_contents = await audio.read()
            ct             = audio.content_type or ''
            suffix         = '.ogg' if 'ogg' in ct else '.webm'
            tmp_a = wav_p = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as ta:
                    ta.write(audio_contents)
                    tmp_a = ta.name
                wav_p = tmp_a.replace(suffix, ".wav")
                conv  = subprocess.run(
                    [FFMPEG_PATH, '-y', '-i', tmp_a,
                     '-ar', '16000', '-ac', '1', '-f', 'wav', wav_p],
                    capture_output=True, timeout=15
                )
                if conv.returncode == 0 and os.path.exists(wav_p):
                    y, _ = sf.read(wav_p)
                    if len(y.shape) > 1:
                        y = np.mean(y, axis=1)
                    if len(y) > 1600:
                        # ← Transcription désactivée
                        transcript  = ""
                        ai          = audio_processor(
                            y, sampling_rate=16000, return_tensors="pt"
                        ).to(device)
                        with torch.no_grad():
                            audio_probs = F.softmax(
                                audio_model(ai['input_values']), dim=-1
                            ).cpu().numpy()[0]
            except Exception as e:
                print(f"[Audio] Erreur: {e}")
            finally:
                for p in [tmp_a, wav_p]:
                    if p and os.path.exists(p):
                        try: os.remove(p)
                        except: pass

        if face_img is not None:
            gray       = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            blur_val   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            face_status = (
                'Sombre'  if brightness < 45 else
                'Exposé'  if brightness > 220 else
                'Flou'    if blur_val < 50 else
                'Optimal'
            )
            inputs_v = base_processor(
                images=cv2.cvtColor(
                    preprocess_face(face_img), cv2.COLOR_BGR2RGB
                ),
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                v_probs = F.softmax(
                    model(inputs_v['pixel_values']), dim=-1
                ).cpu().numpy()[0]

        audio_status = 'Micro Inactif'
        if audio and y is not None and len(y) > 0:
            rms = float(np.sqrt(np.mean(y**2)))
            audio_status = 'Silence' if rms < 0.003 else 'Clair ✅'
        elif audio:
            audio_status = 'Pas de voix détectée'

        metrics = calculate_candidate_metrics(v_probs, audio_probs)

        return JSONResponse({
            'success':         True,
            'faces_detected':  face_img is not None,
            'emotion':         emotion,
            'emotion_fr':      EMOTION_NAMES_FR.get(emotion, emotion),
            'emoji':           EMOTION_EMOJIS.get(emotion, ''),
            'color':           EMOTION_COLORS.get(emotion, '#667eea'),
            'confidence':      float(confidence),
            'transcript':      transcript,
            'candidate_metrics': metrics,
            'bbox':            [int(c) for c in bbox] if bbox else [],
            'all_faces':       all_faces,
            'reliability': {
                'face': {'brightness': round(brightness,1),
                         'blur':       round(blur_val,1),
                         'status':     face_status},
                'audio': {'status': audio_status}
            }
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({'success': False, 'error': str(e)},
                            status_code=500)


@app.post("/chatbot")
async def chatbot_interaction(data: dict):
    query   = data.get("query", "").lower()
    context = data.get("context", {})
    metrics = context.get("metrics", {})
    analysis = context.get("analysis", {})
    flags   = context.get("inconsistencies", [])
    stress  = metrics.get("stress_management", 50)
    conf    = metrics.get("assurance_level", 50)
    dscore  = analysis.get("score", 0)

    if any(w in query for w in ["mensonge", "vérité", "tromperie",
                                  "sincérité", "authenticité", "tension"]):
        if dscore < 30:
            r = (f"Peu de signes de tension observés ({dscore:.1f}/100). "
                 f"Cela ne garantit rien sur le fond des réponses — "
                 f"ce n'est qu'un indicateur de confort apparent.")
        elif dscore < 60:
            r = (f"Quelques signes de tension observés ({dscore:.1f}/100). "
                 f"Cela peut refléter du stress d'entretien normal. "
                 f"Si un point vous semble flou, creusez avec une question "
                 f"ouverte plutôt que de vous fier à ce score.")
        else:
            r = (f"Signes de tension fréquents ({dscore:.1f}/100) — "
                 + (" ".join(flags) if flags else "")
                 + " Cela n'indique pas en soi un manque de sincérité : "
                   "posez des questions factuelles de clarification si "
                   "un point du discours vous interpelle.")
    elif any(w in query for w in ["stress", "anxieux", "tension"]):
        r = (f"Gestion du stress : {stress:.1f}%. "
             + ("Très bonne maîtrise." if stress > 70
                else "Proposez une question de mise à l'aise."))
    elif any(w in query for w in ["confiance", "assurance"]):
        r = (f"Assurance : {conf:.1f}%. "
             + ("Profil confiant." if conf > 60
                else "Valorisez un projet personnel."))
    elif any(w in query for w in ["regard", "contact", "yeux"]):
        r = ("Le contact visuel est un bon indicateur d'engagement. "
             "Observez si le candidat maintient le regard durant "
             "les questions difficiles.")
    elif any(w in query for w in ["posture", "corps", "langage"]):
        r = ("La posture droite indique la confiance. "
             "Des épaules affaissées peuvent signaler de la fatigue "
             "ou de l'inconfort.")
    elif any(w in query for w in ["suggestion", "conseil", "question"]):
        r = ("Vérifiez les références." if dscore > 50 else
             "Détendez-le avec des questions ouvertes." if stress < 50 else
             "Profil solide — approfondissez la culture d'entreprise.")
    elif any(w in query for w in ["résumé", "bilan", "profil"]):
        r = (f"Assurance {conf:.0f}%, Stress {stress:.0f}%, "
             f"Points d'attention {dscore:.0f}/100. "
             + ("⚠️ Quelques points à approfondir." if dscore > 60
                else "⚠️ Nervosité à surveiller." if stress < 40
                else "✅ Candidat globalement stable."))
    else:
        r = ("Je suis Nexy. Tapez 'stress', 'assurance', "
             "'regard', 'posture' ou 'suggestions' pour une analyse.")

    return {"response": r}


@app.post("/analyze_imported_video")
async def analyze_imported_video(video_id: int = FastAPIForm(...)):
    try:
        cached = _url_video_cache.get(video_id)
        if not cached:
            return JSONResponse({
                "success": False,
                "error": f"Vidéo {video_id} introuvable."
            }, status_code=404)
        temp_path  = cached["path"]
        filename   = cached["filename"]
        source_url = cached["url"]

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return JSONResponse({'success': False,
                                 'error': 'Impossible d\'ouvrir la vidéo'},
                                status_code=500)
        fps_cv = cap.get(cv2.CAP_PROP_FPS)
        fc     = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        dur    = fc / fps_cv if fps_cv else 0
        cap.release()

        if dur < 5:
            return JSONResponse({
                'success': False,
                'error': f'Vidéo trop courte ({dur:.1f}s).'
            }, status_code=400)

        class VirtualUploadFile:
            def __init__(self, path, name):
                self.filename = "url_" + name
                self._path    = path
            async def read(self):
                with open(self._path, 'rb') as f:
                    return f.read()

        result = await analyze_video(
            file=VirtualUploadFile(temp_path, filename),
            target_x=None, target_y=None
        )

        if hasattr(result, 'body'):
            result_data = (json.loads(result.body)
                           if isinstance(result.body, bytes)
                           else result.body)
        else:
            result_data = result

        if isinstance(result_data, dict):
            result_data["video_id"]       = video_id
            result_data["video_filename"] = filename
            result_data["source_url"]     = source_url

        if os.path.exists(temp_path):
            os.remove(temp_path)
        _url_video_cache.pop(video_id, None)

        return JSONResponse(result_data)

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)},
                            status_code=500)


@app.post("/test-url")
async def test_url(request: dict):
    import urllib.request
    url = request.get("url", "")
    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            http_status    = resp.status
            content_type   = resp.headers.get('Content-Type', 'unknown')
            content_length = resp.headers.get('Content-Length', 'unknown')
            header_bytes   = resp.read(32)
        is_mp4      = b'ftyp' in header_bytes or b'moov' in header_bytes
        is_xml_err  = header_bytes[:5] in [b'<?xml', b'<Erro', b'<Acce']
        return JSONResponse({
            'accessible':     True,
            'http_status':    http_status,
            'content_type':   content_type,
            'content_length': content_length,
            'is_valid_mp4':   is_mp4,
            'is_xml_error':   is_xml_err,
            'diagnostic':     ('MP4 valide' if is_mp4 else
                               'Erreur R2 XML' if is_xml_err else
                               'Format inconnu')
        })
    except urllib.error.HTTPError as e:
        return JSONResponse({
            'accessible': False, 'http_status': e.code,
            'error': str(e)
        })
    except Exception as e:
        return JSONResponse({'accessible': False, 'error': str(e)})


@app.post("/candidate_summary")
async def candidate_summary(
        file: UploadFile = File(None),
        video_url: Optional[str] = Form(None)
) -> JSONResponse:
    if video_url:
        result = await analyze_video_from_url(
            VideoURLRequest(url=video_url)
        )
    elif file:
        result = await analyze_video(file=file, target_x=None, target_y=None)
    else:
        return JSONResponse({"error": "Aucun fichier ou URL."}, status_code=400)

    if isinstance(result, JSONResponse):
        body = result.body
        data = json.loads(body.decode() if isinstance(body, bytes) else body)
        return JSONResponse(content=data)
    return result


@app.get("/dashboard")
async def dashboard_page():
    from fastapi.responses import HTMLResponse
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/report")
async def report_page():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content="<h1>Rapport</h1>")


@app.get("/kpi_dashboard")
async def kpi_dashboard_page():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content="<h1>KPI Dashboard</h1>")


@app.post("/add_training_data")
async def add_training_data(emotion: str, image: UploadFile = File(...)):
    try:
        if emotion not in EMOTION_LABELS:
            return JSONResponse({'error': f'Émotion invalide.'}, status_code=400)
        emotion_dir = Path(Config.TRAINING_DATA_PATH) / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = emotion_dir / f"{ts}.jpg"
        with open(filename, 'wb') as f:
            f.write(await image.read())
        return JSONResponse({'success': True, 'path': str(filename)})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


@app.post("/evaluate")
async def evaluate_model(dataset_path: str):
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import time
        if not os.path.exists(dataset_path):
            return JSONResponse({'error': 'Dossier introuvable'}, status_code=404)
        eval_dataset = EmotionDataset(dataset_path, base_processor)
        eval_loader  = DataLoader(eval_dataset,
                                  batch_size=Config.BATCH_SIZE, shuffle=False)
        model.eval()
        all_preds = []; all_labels = []
        t0 = time.time()
        with torch.no_grad():
            for batch in eval_loader:
                pv  = batch['pixel_values'].to(device)
                lbl = batch['labels'].to(device)
                out = model(pv)
                _, pred = out.max(1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(lbl.cpu().numpy())
        tt  = time.time() - t0
        ns  = len(all_labels)
        report = classification_report(all_labels, all_preds,
                                       target_names=EMOTION_LABELS,
                                       output_dict=True, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds).tolist()
        return JSONResponse({
            "performance": {
                "accuracy":        report["accuracy"],
                "macro_avg_f1":    report["macro avg"]["f1-score"],
                "weighted_avg_f1": report["weighted avg"]["f1-score"]
            },
            "vitesse": {
                "images_evaluees":    ns,
                "temps_total_sec":    round(tt, 2),
                "fps":                round(ns/tt if tt>0 else 0, 1),
                "latence_moyenne_ms": round(tt/ns*1000 if ns>0 else 0, 2)
            },
            "matrice_de_confusion": cm,
            "rapport_detaille":     report
        })
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


@app.get("/health")
async def health():
    # ← Utilise directement MODEL_STATUS, rempli individuellement par
    # chaque bloc de chargement (voir robustesse au démarrage) — reflète
    # précisément quels modèles ont réellement réussi à charger, plutôt
    # que de déduire l'état à partir de vérifications ad-hoc éparpillées.
    all_critical_ok = MODEL_STATUS.get("vit", False) and \
                       MODEL_STATUS.get("yolo", False)
    any_degraded    = not all(MODEL_STATUS.values())

    health_status = {
        "status": ("healthy" if all_critical_ok and not any_degraded
                    else "degraded" if all_critical_ok
                    else "unhealthy"),
        "service": "nexum-ia-python",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "models": dict(MODEL_STATUS),
        "checks": {
            "models_loaded":    model is not None,
            "yolo_loaded":      yolo_model is not None,
            "arcface_loaded":   shared_arcface is not None,
            "mediapipe_loaded": face_analyzer_global.enabled,
            "au_pyfeat_loaded": MODEL_STATUS_AU,
            "device":           str(device),
        }
    }

    if not all_critical_ok:
        # Seuls ViT et YOLO sont considérés critiques (nécessaires à
        # l'analyse d'émotion et à la détection de visage elles-mêmes).
        # Un modèle secondaire indisponible (HuBERT, ArcFace, MediaPipe)
        # dégrade certaines fonctionnalités mais ne doit pas empêcher le
        # serveur de répondre — d'où status "degraded" et non
        # "unhealthy" dans ce cas.
        return JSONResponse(health_status, status_code=503)

    return health_status


@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("NEXUM IA — SERVEUR DÉMARRÉ")
    print("=" * 60)
    print(f"API:       http://localhost:8089")
    print(f"Health:    http://localhost:8089/health")
    print(f"WebSocket: ws://localhost:8089/ws/analyze_realtime")
    print(f"Device:    {device}")
    print(f"MediaPipe: {'✅ actif' if face_analyzer_global.enabled else '❌ inactif'}")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    print("[Server] Arrêt en cours...")
    executor.shutdown(wait=False)
    video_processing_executor.shutdown(wait=False)
    SessionLocal.close_all()
    print("[Server] Arrêté")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("NEXUM IA — SERVEUR DÉMARRÉ")
    print("="*60)
    print("API:       http://localhost:8089")
    print("Dashboard: http://localhost:8089/dashboard")
    print("Health:    http://localhost:8089/health")
    print("WebSocket: ws://localhost:8089/ws/analyze_realtime")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8089,
                log_level="info",
                ws_ping_interval=30, ws_ping_timeout=60)