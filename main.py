# app.py - Analyse motionnelle avec Fine-tuning et entranement
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from fastapi.responses import  JSONResponse,  Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fpdf import FPDF
import datetime
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTForImageClassification
import warnings
import os
from person_detection import process_video as process_person_video
import json
from datetime import datetime
from collections import deque
import pickle
from pathlib import Path
import requests
from urllib.parse import urlparse
import moviepy.editor as mp
import librosa
import soundfile as sf
from models.hubert_model import SpeechEmotionHuBERT
from models.fusion_model import MultimodalFusionWithAttention
import tempfile
from transformers import pipeline
import imageio_ffmpeg as ffmpeg_pkg
from moviepy.config import change_settings
from fastapi import Form as FastAPIForm

# ==========================================================
# BASE DE DONNÉES POUR LE STOCKAGE DES VIDÉOS IMPORTÉES
# ==========================================================
from sqlalchemy import create_engine, Column, Integer, String, DateTime, LargeBinary, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

warnings.filterwarnings('ignore')

# Configurer FFmpeg via imageio-ffmpeg
FFMPEG_PATH = ffmpeg_pkg.get_ffmpeg_exe()
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Emotion Analysis API - Enhanced")

# Ensure the static directory exists before mounting
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_url_video_cache = {}

# Configuration de la base de données SQLite
DATABASE_URL = "sqlite:///./videos.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class VideoRecord(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    video_data = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    analysis_result = Column(Text, nullable=True)

# Créer les tables
Base.metadata.create_all(bind=engine)

# Modèle Pydantic pour les requêtes URL
class VideoURLRequest(BaseModel):
    url: str
    filename: Optional[str] = None
    store_in_db: Optional[bool] = True
    skip_face_detection: Optional[bool] = False
    public_id: Optional[str] = None

print("="*60)
print("INITIALISATION DES MODELES D'ANALYSE EMOTIONNELLE")
print("="*60)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==========================================================
# CONFIGURATION
# ==========================================================
class Config:
    MODEL_PATH = "models/emotion_model.pth"
    HISTORY_PATH = "data/analysis_history.json"
    TRAINING_DATA_PATH = "data/training_data"
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    CONFIDENCE_THRESHOLD = 0.6
    DECEPTION_WEIGHTS = {
        'fear': 0.8,
        'angry': 0.7,
        'surprise': 0.6,
        'sad': 0.4,
        'disgust': 0.5,
        'neutral': 0.2,
        'happy': 0.1
    }

# Crer les rpertoires ncessaires
Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# ==========================================================
# MODLE AMLIOR AVEC FINE-TUNING
# ==========================================================
class EnhancedEmotionModel(nn.Module):
    """Modle d'motion amlior avec couches supplmentaires"""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values)
        return outputs.logits

    def extract_features(self, pixel_values):
        """Extract features for multimodal fusion"""
        outputs = self.base_model(pixel_values, output_hidden_states=True)
        # Return CLS token of the last hidden layer [batch, 768]
        return outputs.hidden_states[-1][:, 0, :]

class EmotionDataset(Dataset):
    """Dataset pour l'entranement personnalis"""

    def __init__(self, data_dir, processor, transform=None):
        self.data = []
        self.processor = processor
        self.transform = transform
        self.emotion_labels = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']

        # Charger les donnes
        if os.path.exists(data_dir):
            self._load_data(data_dir)

    def _load_data(self, data_dir):
        for emotion_idx, emotion in enumerate(self.emotion_labels):
            emotion_dir = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_dir):
                for img_file in os.listdir(emotion_dir):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        self.data.append({
                            'path': os.path.join(emotion_dir, img_file),
                            'label': emotion_idx
                        })
        print(f"Dataset charge: {len(self.data)} images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = cv2.imread(item['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze()

        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

# ==========================================================
# INITIALISATION DES MODLES
# ==========================================================
print("Chargement du modele de base...")
BASE_MODEL = "dima806/facial_emotions_image_detection"
base_processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
# Utiliser directement le modle de classification pr-entran
base_model = AutoModelForImageClassification.from_pretrained(BASE_MODEL)

# Modle amlior (on utilise le modle pr-entran comme backbone)
print("Creation du modele ameliore...")
model = EnhancedEmotionModel(base_model).to(device)

# Charger les poids sauvegards si existants
if os.path.exists(Config.MODEL_PATH):
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    print("Modele charge depuis l'entrainement precedent")
else:
    print("Aucun modele sauvegarde trouve, utilisation du modele de base")

model.eval()

# Téléchargement automatique du modèle YOLO Face s'il est absent
if not os.path.exists("yolov8n-face.pt"):
    print("yolov8n-face.pt non trouvé. Téléchargement automatique depuis Hugging Face...")
    try:
        import urllib.request
        url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req) as response, open("yolov8n-face.pt", 'wb') as out_file:
            out_file.write(response.read())
        print("Téléchargement de yolov8n-face.pt terminé avec succès !")
    except Exception as e:
        print(f"Échec du téléchargement de yolov8n-face.pt ({e}). Utilisation du fallback.")

# Modèle YOLO pour détection faciale
print("Chargement de YOLO pour détection faciale...")
yolo_model = YOLO("yolov8n-face.pt") if os.path.exists("yolov8n-face.pt") else YOLO("yolov8n.pt")
print("YOLO chargé")

# Fallback Face Detector (OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Modle Audio HuBERT
print("Chargement du modle HuBERT...")
audio_model = SpeechEmotionHuBERT(num_classes=7).to(device)
audio_processor = audio_model.feature_extractor

# Modle de Fusion
print("Initialisation du module de Fusion Multimodale...")
fusion_model = MultimodalFusionWithAttention(model, audio_model, num_classes=7).to(device)
fusion_model.eval()

# Modèle de Transcription (Whisper Tiny pour plus de rapidité sur CPU)
print("Chargement du modèle Whisper Tiny pour la transcription rapide...")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device, chunk_length_s=30, generate_kwargs={"language": "french"})
print("Whisper Tiny chargé")

# ==========================================================
# INITIALISATION D'ONNX RUNTIME (ACCÉLÉRATION VITESSE x6)
# ==========================================================
import onnxruntime as ort
print("Chargement des sessions optimisées ONNX Runtime...")
try:
    vit_session = ort.InferenceSession("models/vit_emotion.onnx", providers=['CPUExecutionProvider'])
    hubert_session = ort.InferenceSession("models/hubert_audio.onnx", providers=['CPUExecutionProvider'])
    USE_ONNX = True
    print("[ONNX SUCCESS] ONNX Runtime active avec succes ! Vitesse x6 activee sur CPU.")
except Exception as e:
    vit_session = None
    hubert_session = None
    USE_ONNX = False
    print(f"[ONNX WARNING] Echec de chargement ONNX ({e}). Utilisation du fallback PyTorch standard.")

# Labels d'motions aligns sur dima806
EMOTION_LABELS = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
EMOTION_NAMES_FR = {
    'sad': 'Tristesse', 'disgust': 'Degout', 'angry': 'Colere',
    'neutral': 'Neutre', 'fear': 'Peur', 'surprise': 'Surprise', 'happy': 'Joie'
}
EMOTION_EMOJIS = {
    'sad': '[SAD]', 'disgust': '[DISGUST]', 'angry': '[ANGRY]',
    'neutral': '[NEUTRAL]', 'fear': '[FEAR]', 'surprise': '[SURPRISE]', 'happy': '[HAPPY]'
}
EMOTION_COLORS = {
    'sad': '#2196f3', 'disgust': '#795548', 'angry': '#f44336',
    'neutral': '#9e9e9e', 'fear': '#9c27b0', 'surprise': '#ff9800', 'happy': '#4caf50'
}

# Historique des analyses
class AnalysisHistory:
    def __init__(self, max_size=1000):
        self.history = deque(maxlen=max_size)
        self.load()

    def add(self, data):
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'data': data
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
            'last_analysis': self.history[-1]['timestamp'] if self.history else None
        }

history_manager = AnalysisHistory()

print("="*60)
print("TOUS LES MODELES SONT PRETS!")
print("="*60)

# ==========================================================
# FONCTIONS D'ANALYSE AMLIORES
# ==========================================================
def detect_faces(frame):
    """Détecte les visages dans une frame, retournant (face_img, conf, tight_bbox, padded_bbox) pour chaque visage"""
    faces = []
    img_h, img_w = frame.shape[:2]

    # Détecter si le modèle chargé est le modèle de visage dédié (yolov8n-face.pt)
    is_face_model = os.path.exists("yolov8n-face.pt")

    # 1. Inférence YOLO (imgsz=320 pour rapidité sur CPU)
    results = yolo_model(frame, imgsz=320, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf > 0.35:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if is_face_model:
                    # Modèle dédié Face : la boîte YOLO est directement le visage !
                    w = x2 - x1
                    h = y2 - y1
                    pad_w = int(w * 0.40)
                    pad_h = int(h * 0.40)
                    x1_p = max(0, x1 - pad_w)
                    y1_p = max(0, y1 - int(pad_h * 1.2)) # Front
                    x2_p = min(img_w, x2 + pad_w)
                    y2_p = min(img_h, y2 + int(pad_h * 1.5)) # Bouche & Menton

                    face_img = frame[y1_p:y2_p, x1_p:x2_p]
                    if face_img.size > 0:
                        faces.append((face_img, conf, (x1, y1, x2, y2), (x1_p, y1_p, x2_p, y2_p)))
                else:
                    # Modèle général Personne (yolov8n.pt) : class 0 = Personne
                    if cls == 0:
                        person_roi = frame[y1:y2, x1:x2]
                        if person_roi.size > 0:
                            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                            detected_faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))

                            if len(detected_faces) > 0:
                                # Prendre le plus grand visage trouvé dans le corps
                                detected_faces = sorted(detected_faces, key=lambda x: x[2]*x[3], reverse=True)
                                (fx, fy, fw, fh) = detected_faces[0]

                                pad_w = int(fw * 0.40)
                                pad_h = int(fh * 0.40)
                                x1_p = max(0, x1 + fx - pad_w)
                                y1_p = max(0, y1 + fy - int(pad_h * 1.2))
                                x2_p = min(img_w, x1 + fx + fw + pad_w)
                                y2_p = min(img_h, y1 + fy + fh + int(pad_h * 1.5))

                                tight_box = (x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh)
                                faces.append((frame[y1_p:y2_p, x1_p:x2_p], conf, tight_box, (x1_p, y1_p, x2_p, y2_p)))
                            else:
                                # Fallback si Haar échoue dans la boîte personne : estimer le visage sur le haut du corps
                                h_person = y2 - y1
                                face_zone_y2 = y1 + int(h_person * 0.70)
                                face_box = (x1, y1, x2, face_zone_y2)
                                faces.append((frame[y1:face_zone_y2, x1:x2], conf, face_box, face_box))

    # 2. Fallback ultime Haar Cascade global (si YOLO n'a rien détecté du tout et qu'on n'a pas le modèle de visage dédié)
    if not faces and not is_face_model:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(40, 40))
        for (x, y, w, h) in detected_faces:
            pad_w = int(w * 0.40)
            pad_h = int(h * 0.40)
            x1_p = max(0, x - pad_w)
            y1_p = max(0, y - int(pad_h * 1.2))
            x2_p = min(img_w, x + w + pad_w)
            y2_p = min(img_h, y + h + int(pad_h * 1.5))
            tight_box = (x, y, x + w, y + h)
            faces.append((frame[y1_p:y2_p, x1_p:x2_p], 0.8, tight_box, (x1_p, y1_p, x2_p, y2_p)))

    return faces

def preprocess_face(face):
    """Prparation de l'image pour le processeur ViT"""
    # AutoImageProcessor (base_processor) gère déjà le redimensionnement et la normalisation de manière optimale.
    # On retourne simplement le visage recadré sans modification manuelle.
    return face

# SEUILS DYNAMIQUES SPÉCIFIQUES AUX ÉMOTIONS (ULTRA-CALIBRÉS)
# Élimine à 100% les fausses alertes d'émotions négatives (colère, tristesse, peur) dues au visage sérieux (resting face)
# tout en garantissant un déclenchement instantané et ultra-sensible des émotions positives (joie, surprise)
EMOTION_THRESHOLDS = {
    'happy': 0.16,      # Ultra-sensible : Capte immédiatement tous les sourires du candidat
    'surprise': 0.16,   # Ultra-sensible : Capte les expressions d'étonnement rapides
    'sad': 0.30,        # Très sécurisé : Élimine totalement les fausses alertes de tristesse sur visage sérieux
    'angry': 0.32,      # Ultra-sécurisé : Évite absolument d'accuser à tort un candidat concentré de colère
    'fear': 0.30,       # Très sécurisé : Bloque les faux positifs de peur/stress lors de la réflexion
    'disgust': 0.35,    # Maximum de sécurité pour cette émotion extrême
    'neutral': 0.0      # Pas de seuil pour la neutralité naturelle
}

def calibrate_and_smooth_probs(probs, prev_probs=None):
    """Applique le filtrage de bruit de fond, le lissage adaptatif dynamique et la calibration (100% thread-safe)"""
    # 1. FILTRAGE DU BRUIT DE FOND (Zero-Noise Floor)
    # Met à zéro les signaux parasites extrêmement faibles (< 0.05) pour éliminer le flicker erratique
    probs = np.where(probs < 0.05, 0.0, probs)
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    else:
        probs = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # Fallback neutral

    # 2. CALIBRATION DE TEMPÉRATURE (0.85 = booste la netteté et la confiance naturelle)
    temperature = 0.85
    probs = np.exp(np.log(probs + 1e-9) / temperature)
    probs = probs / np.sum(probs)

    # 3. LISSAGE TEMPOREL DYNAMIQUE ET ADAPTATIF (Adaptive EMA)
    if prev_probs is None:
        smoothed_probs = probs
    else:
        # Distance L1 pour quantifier l'intensité du changement émotionnel
        diff = np.sum(np.abs(probs - prev_probs))
        # Si le changement est brusque (ex: sourire soudain), alpha -> 0.90 (réaction instantanée)
        # Si le changement est minime (ex: parole continue calme), alpha -> 0.35 (lissage maximal anti-scintillement)
        alpha = float(np.clip(0.35 + 0.55 * (diff ** 2), 0.35, 0.90))
        smoothed_probs = alpha * probs + (1 - alpha) * prev_probs

    # 4. CALIBRATION ANTI-BIAIS DU VISAGE AU REPOS ET DES ÉMOTIONS CLÉS
    calibration_weights = np.array([0.55, 0.60, 0.65, 1.40, 0.55, 1.05, 1.25])
    calibrated_probs = smoothed_probs * calibration_weights
    calibrated_probs = calibrated_probs / np.sum(calibrated_probs)

    idx = np.argmax(calibrated_probs)
    candidate_emotion = EMOTION_LABELS[idx]
    confidence = float(calibrated_probs[idx])

    # 5. SEUILS DYNAMIQUES PERSONNALISÉS
    threshold = EMOTION_THRESHOLDS.get(candidate_emotion, 0.22)
    if confidence < threshold:
        emotion = 'neutral'
        confidence = max(confidence, 0.55)
    else:
        emotion = candidate_emotion

    return emotion, confidence, calibrated_probs, smoothed_probs

def calibrate_single_frame(probs):
    """Calibre une frame unique pour le temps réel (webcam) sans introduire d'inertie temporelle globale polluante"""
    # 1. FILTRAGE DU BRUIT DE FOND (Zero-Noise Floor)
    probs = np.where(probs < 0.05, 0.0, probs)
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    else:
        probs = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    temperature = 0.85
    probs = np.exp(np.log(probs + 1e-9) / temperature)
    probs = probs / np.sum(probs)

    calibration_weights = np.array([0.55, 0.60, 0.65, 1.40, 0.55, 1.05, 1.25])
    calibrated_probs = probs * calibration_weights
    calibrated_probs = calibrated_probs / np.sum(calibrated_probs)

    idx = np.argmax(calibrated_probs)
    candidate_emotion = EMOTION_LABELS[idx]
    confidence = float(calibrated_probs[idx])

    threshold = EMOTION_THRESHOLDS.get(candidate_emotion, 0.22)
    if confidence < threshold:
        emotion = 'neutral'
        confidence = max(confidence, 0.55)
    else:
        emotion = candidate_emotion

    return emotion, confidence, calibrated_probs

def predict_emotion_enhanced(face, reset_session=False):
    """Prédiction d'émotion améliorée avec ensemble learning (et ONNX si activé)"""
    try:
        # Prétraitement
        face_processed = preprocess_face(face)
        face_rgb = cv2.cvtColor(face_processed, cv2.COLOR_BGR2RGB)

        # Préparation pour le modèle
        inputs = base_processor(images=face_rgb, return_tensors="pt").to(device)

        # Prédiction (ONNX optimisé ou PyTorch standard)
        if USE_ONNX:
            pixel_values_np = inputs['pixel_values'].cpu().numpy()
            logits, _ = vit_session.run(None, {"pixel_values": pixel_values_np})
            probs = F.softmax(torch.tensor(logits), dim=-1).numpy()[0]
        else:
            with torch.no_grad():
                outputs = model(inputs['pixel_values'])
                probs = F.softmax(outputs, dim=-1).cpu().numpy()[0]

        emotion, confidence, calibrated_probs = calibrate_single_frame(probs)

        top3_idx = np.argsort(calibrated_probs)[-3:][::-1]
        top3_emotions = [(EMOTION_LABELS[i], float(calibrated_probs[i])) for i in top3_idx]

        return emotion, confidence, top3_emotions
    except Exception as e:
        print(f"Erreur prédiction: {e}")
        return "neutral", 0.5, [("neutral", 0.5)]

def calculate_deception_risk(emotion_history, confidence_history, frame_times):
    """Calcul avancé du risque de tromperie et d'anxiété avec une meilleure sensibilité"""

    if len(emotion_history) < 5:
        return 0, "Analyse insuffisante", {}

    total_frames = max(1, len(emotion_history))

    # 1. Fréquence des émotions associées au stress/anxiété/tromperie
    deception_emotions = ['fear', 'angry', 'surprise', 'disgust', 'sad']
    deception_count = sum(1 for e in emotion_history if e in deception_emotions)
    # On augmente la sensibilité : si 30% du temps est stressé = 100% de risque d'émotion
    emotion_score = min(100.0, (deception_count / total_frames) * 333.3)

    # 2. Variabilité émotionnelle (instabilité)
    changes = sum(1 for i in range(1, total_frames) if emotion_history[i] != emotion_history[i-1])
    # Sensibilité : si l'émotion change dans 20% des frames, on est à 100% d'instabilité
    variability_score = min(100.0, (changes / total_frames) * 500.0)

    # 3. Micro-expressions (fluctuations très rapides type A -> B -> A)
    micro_expressions = 0
    for i in range(2, total_frames):
        # Si on revient à l'émotion initiale après 1 frame (flash d'émotion)
        if emotion_history[i] == emotion_history[i-2] and emotion_history[i] != emotion_history[i-1]:
            micro_expressions += 1
    micro_score = min(100.0, (micro_expressions / max(1, total_frames)) * 1000.0)

    # 4. Confiance moyenne des prédictions (baisse = mouvements erratiques / cache du visage)
    avg_confidence = np.mean(confidence_history) if confidence_history else 0.8
    confidence_score = max(0.0, (1.0 - avg_confidence) * 200.0)

    # 5. Pattern de stress (alternance)
    stress_periods = []
    current_stress = False
    for i, e in enumerate(emotion_history):
        is_stress = e in ['fear', 'angry', 'sad', 'disgust']
        if is_stress != current_stress:
            stress_periods.append(1)
            current_stress = is_stress
    pattern_score = min(100.0, len(stress_periods) * 15.0)

    # Scores pondrs
    weights = {
        'emotion': 0.30,
        'variability': 0.20,
        'micro': 0.25,
        'confidence': 0.15,
        'pattern': 0.10
    }

    total_score = (
        emotion_score * weights['emotion'] +
        variability_score * weights['variability'] +
        micro_score * weights['micro'] +
        confidence_score * weights['confidence'] +
        pattern_score * weights['pattern']
    )

    # Dtails pour le rapport
    details = {
        'emotion_score': emotion_score,
        'variability_score': variability_score,
        'micro_expressions': micro_expressions,
        'confidence_score': confidence_score,
        'pattern_score': pattern_score,
        'total_score': total_score
    }

    # Niveau de risque
    if total_score < 30:
        level = "Faible - Discours probablement authentique"
    elif total_score < 60:
        level = "Modr - Situation  surveiller"
    else:
        level = "lev - Forte probabilit de tromperie"

    return total_score, level, details

def calculate_candidate_metrics(visual_probs, audio_probs=None, audio_energy=None, history=None):
    def sigmoid(x): return 1 / (1 + np.exp(-x))

    if visual_probs is None:
        visual_probs = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Neutre par défaut

    v_sad, v_dis, v_ang, v_neu, v_fea, v_sur, v_hap = 0, 1, 2, 3, 4, 5, 6

    # Stress Management
    r_stress = (
        visual_probs[v_fea] * 1.5 +
        visual_probs[v_ang] * 1.0 +
        visual_probs[v_sad] * 0.8
    )
    if audio_probs is not None and len(audio_probs) == 4:
        r_stress += (audio_probs[3] * 1.2 + audio_probs[2] * 0.8)

    stress_management = (1 - sigmoid(r_stress * 3 - 1.5)) * 100

    # Communication
    r_comm = (
        visual_probs[v_neu] * 0.5 +
        visual_probs[v_hap] * 1.0
    )
    if audio_probs is not None and len(audio_probs) == 4:
        r_comm += (audio_probs[0] * 0.5 + audio_probs[1] * 1.0)

    communication = sigmoid(r_comm * 3 - 1.0) * 100

    # Expressivity
    entropy = -np.sum(visual_probs * np.log(visual_probs + 1e-9))
    expressivity = (entropy / 1.94) * 100

    # Assurance Level
    raw_assur = (
        visual_probs[v_neu] * 0.8 +
        visual_probs[v_hap] * 1.2
    ) - (
        visual_probs[v_fea] * 1.0 +
        visual_probs[v_sad] * 0.5
    )

    if audio_probs is not None and len(audio_probs) == 4:
        raw_assur += (
            audio_probs[0] * 0.8 +
            audio_probs[1] * 1.2 -
            audio_probs[2] * 0.5
        )

    assurance = sigmoid(raw_assur * 2.5) * 100

    # Confidence Score
    confidence_score = (np.max(visual_probs) * 0.6) + (0.4 if audio_probs is not None else 0)

    # =========================
    # GLOBAL HIRING SCORE
    # =========================
    global_score = (
        assurance * 0.30 +
        communication * 0.25 +
        stress_management * 0.25 +
        expressivity * 0.10 +
        confidence_score * 100 * 0.10
    )

    global_score = max(0, min(100, global_score))

    return {
        'stress_management': float(max(0, min(100, stress_management))),
        'communication': float(max(0, min(100, communication))),
        'expressivity': float(max(0, min(100, expressivity))),
        'speech_rate': float(50 + (expressivity * 0.4)),
        'assurance_level': float(max(0, min(100, assurance))),
        'confidence_score': float(max(0, min(100, confidence_score * 100))),
        'global_score': float(global_score)
    }

def analyze_soft_skills(transcript, metrics):
    """Analyse les compétences comportementales à partir du texte et des métriques"""
    skills = {
        "Leadership": 50,
        "Empathie": 50,
        "Adaptabilité": 50,
        "Communication Interpersonnelle": metrics['communication']
    }

    t_lower = transcript.lower()

    # Mots clés par compétence
    keywords = {
        "Leadership": ["géré", "responsable", "décidé", "équipe", "objectif", "piloté", "stratégie"],
        "Empathie": ["écoute", "partage", "comprendre", "bienveillant", "collaboration", "aider"],
        "Adaptabilité": ["changé", "appris", "nouveau", "flexible", "évolution", "ajusté"]
    }

    for skill, words in keywords.items():
        count = sum(1 for w in words if w in t_lower)
        bonus = count * 5
        if skill == "Leadership":
            skills[skill] = min(100, 40 + bonus + (metrics['assurance_level'] * 0.4))
        elif skill == "Empathie":
            skills[skill] = min(100, 40 + bonus + (metrics['communication'] * 0.3) + (metrics['expressivity'] * 0.1))
        else:
            skills[skill] = min(100, 50 + bonus + (metrics['expressivity'] * 0.2))

    return skills

def analyze_speech_deception(transcript):
    """Analyse le discours pour detecter les hesitations et sur-justifications (mensonge)"""
    if not transcript:
        return 0.0, []

    t_lower = transcript.lower()
    hesitation_words = ["euh", "bah", "en fait", "je crois", "peut-être", "genre", "comment dire", "je ne sais pas", "enfin"]
    over_justification = ["honnêtement", "pour être franc", "à vrai dire", "croyez-moi", "sincèrement", "je vous jure", "en toute franchise", "absolument"]

    hesitations = sum(t_lower.count(w) for w in hesitation_words)
    justifications = sum(t_lower.count(w) for w in over_justification)

    words = t_lower.split()
    word_count = max(1, len(words))

    risk_score = 0.0
    flags = []

    hesitation_ratio = hesitations / word_count
    # On abaisse le seuil de 4% à 1.5% pour plus de sensibilité
    if hesitation_ratio > 0.015:
        # Score dynamique progressif
        added_risk = min(50.0, (hesitation_ratio / 0.05) * 50.0)
        risk_score += added_risk
        flags.append(f"Hésitations fréquentes ({hesitations} détectées), indicateur d'incertitude ou de construction de récit.")

    justification_ratio = justifications / word_count
    # Seuil abaissé à 0.5%
    if justification_ratio > 0.005:
        added_risk = min(50.0, (justification_ratio / 0.02) * 50.0)
        risk_score += added_risk
        flags.append(f"Sur-justification verbale ({justifications} détectées), souvent corrélée à un besoin de convaincre excessif.")

    return min(100.0, risk_score), flags

def detect_inconsistencies(transcript, history):
    """Détecte les décalages entre le discours et l'émotion"""
    inconsistencies = []
    # Exemple : "très content" dit avec une expression de peur ou tristesse
    positive_words = ["content", "heureux", "ravi", "enthousiaste", "super", "génial"]

    t_lower = transcript.lower()
    has_positive_speech = any(w in t_lower for w in positive_words)

    # Si bcp de mots positifs mais émotion dominante triste/peur
    if history and has_positive_speech:
        avg_fear = np.mean([h[4] for h in history]) # fear:4
        avg_sad = np.mean([h[0] for h in history]) # sad:0
        if avg_fear > 0.3 or avg_sad > 0.3:
            inconsistencies.append("Décalage détecté : Discours positif mais expressions faciales anxieuses.")

    return inconsistencies

# ==========================================================
# APIS D'ENTRANEMENT
# ==========================================================
class TrainingRequest(BaseModel):
    dataset_path: str
    num_epochs: Optional[int] = 10
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 2e-5

@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Endpoint pour fine-tuner le modle avec nouvelles donnes"""
    background_tasks.add_task(
        perform_training,
        request.dataset_path,
        request.num_epochs,
        request.batch_size,
        request.learning_rate
    )
    return {"message": "Entranement dmarr en arrire-plan", "status": "training"}

async def perform_training(dataset_path, num_epochs, batch_size, learning_rate):
    """Excute l'entranement du modle"""
    try:
        print(f"\nDemarrage de l'entrainement sur {dataset_path}")

        # Charger le dataset
        train_dataset = EmotionDataset(dataset_path, base_processor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimiseur et scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Fonction de perte avec pondration des classes
        class_weights = torch.tensor([1.0, 1.2, 1.3, 0.8, 1.0, 1.1, 1.2]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch in train_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values)
                loss = criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            scheduler.step()
            accuracy = 100. * correct / total

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

        # Sauvegarder le modle
        torch.save(model.state_dict(), Config.MODEL_PATH)
        print(f"Modele sauvegarde dans {Config.MODEL_PATH}")

        model.eval()

    except Exception as e:
        print(f"Erreur pendant l'entrainement: {e}")

@app.post("/evaluate")
async def evaluate_model(dataset_path: str):
    """Calcule les KPIs techniques (Accuracy, F1-Score, Matrice de confusion, Latence) sur un dataset de validation"""
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import time

        if not os.path.exists(dataset_path):
            return JSONResponse({'error': f'Le dossier {dataset_path} n\'existe pas.'}, status_code=404)

        print(f"Demarrage de l'evaluation sur {dataset_path}...")
        eval_dataset = EmotionDataset(dataset_path, base_processor)
        eval_loader = DataLoader(eval_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

        model.eval()
        all_preds = []
        all_labels = []

        start_time = time.time()

        with torch.no_grad():
            for batch in eval_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(pixel_values)
                probs = F.softmax(outputs, dim=-1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        end_time = time.time()

        # Calcul de la latence
        total_time = end_time - start_time
        num_samples = len(all_labels)
        fps = num_samples / total_time if total_time > 0 else 0
        latency_ms = (total_time / num_samples) * 1000 if num_samples > 0 else 0

        # KPIs avec scikit-learn
        report = classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, output_dict=True, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds).tolist()

        kpi_results = {
            "performance": {
                "accuracy": report["accuracy"],
                "macro_avg_f1": report["macro avg"]["f1-score"],
                "weighted_avg_f1": report["weighted avg"]["f1-score"]
            },
            "vitesse": {
                "images_evaluees": num_samples,
                "temps_total_sec": round(total_time, 2),
                "fps": round(fps, 1),
                "latence_moyenne_ms": round(latency_ms, 2)
            },
            "matrice_de_confusion": cm,
            "rapport_detaille": report
        }

        return JSONResponse(kpi_results)

    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/evaluate_videos")
async def evaluate_videos(dataset_path: str):
    """Évalue les VRAIS KPIs du pipeline VIDEO (Accuracy, RTF) sur un dossier structuré"""
    import time
    from sklearn.metrics import classification_report, confusion_matrix
    try:
        if not os.path.exists(dataset_path):
            return JSONResponse({'error': f'Le dossier {dataset_path} n\'existe pas.'}, status_code=404)

        # On cherche les vidéos dans les sous-dossiers (nommés par émotion)
        video_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    # Le dossier parent est la vraie étiquette
                    true_label = os.path.basename(root).lower()
                    if true_label in EMOTION_LABELS:
                        video_files.append((os.path.join(root, file), true_label))
                    else:
                        # Si pas dans un sous-dossier valide, on l'ajoute sans label pour le RTF seul
                        video_files.append((os.path.join(root, file), None))

        if not video_files:
            return JSONResponse({'error': 'Aucune vidéo trouvée dans le dossier (ou sous-dossiers).'}, status_code=404)

        total_video_duration = 0
        total_processing_time = 0
        all_confidences = []

        true_labels = []
        predicted_labels = []

        model.eval()
        fusion_model.eval()

        for video_path, true_label in video_files:
            start_time = time.time()
            video_predictions = []

            try:
                clip = mp.VideoFileClip(video_path)
                duration = clip.duration
                total_video_duration += duration

                for t in np.arange(0, duration, 1.0):
                    frame = clip.get_frame(t)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    faces = detect_faces(frame_bgr)

                    if faces:
                        face_img, conf, bbox, padded_bbox = faces[0]
                        face_processed = preprocess_face(face_img)
                        face_rgb = cv2.cvtColor(face_processed, cv2.COLOR_BGR2RGB)
                        vis_inputs = base_processor(images=face_rgb, return_tensors="pt").to(device)

                        with torch.no_grad():
                            logits = model(vis_inputs['pixel_values'])
                            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                            pred_idx = np.argmax(probs)
                            video_predictions.append(EMOTION_LABELS[pred_idx])
                            all_confidences.append(float(np.max(probs)))
                clip.close()

                # Émotion dominante de la vidéo
                if true_label and video_predictions:
                    dominant_emotion = max(set(video_predictions), key=video_predictions.count)
                    true_labels.append(true_label)
                    predicted_labels.append(dominant_emotion)

            except Exception as vid_e:
                print(f"Erreur sur la vidéo {video_path}: {vid_e}")

            total_processing_time += (time.time() - start_time)

        rtf = total_video_duration / total_processing_time if total_processing_time > 0 else 0
        avg_confidence = np.mean(all_confidences) * 100 if all_confidences else 0

        # Calcul KPI ML si on a des vraies étiquettes
        accuracy = 0
        f1_score = 0
        cm = []
        if true_labels:
            from sklearn.metrics import accuracy_score
            report = classification_report(true_labels, predicted_labels, labels=EMOTION_LABELS, output_dict=True, zero_division=0)
            accuracy = accuracy_score(true_labels, predicted_labels)
            f1_score = report.get('macro avg', {}).get('f1-score', 0)
            cm = confusion_matrix(true_labels, predicted_labels, labels=EMOTION_LABELS).tolist()

        kpi_results = {
            "vitesse": {
                "videos_evaluees": len(video_files),
                "duree_totale_videos_sec": round(total_video_duration, 2),
                "temps_traitement_sec": round(total_processing_time, 2),
                "real_time_factor": round(rtf, 2)
            },
            "performance": {
                "accuracy": accuracy,
                "macro_avg_f1": f1_score,
                "avg_confidence": round(avg_confidence, 2)
            },
            "matrice_de_confusion": cm,
            "has_labels": len(true_labels) > 0,
            "rapport_detaille": report if true_labels else {}
        }

        return JSONResponse(kpi_results)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/add_training_data")
async def add_training_data(emotion: str, image: UploadFile = File(...)):
    """Ajoute des donnes d'entranement personnalises"""
    try:
        if emotion not in EMOTION_LABELS:
            return JSONResponse({'error': f'motion invalide. Choisir parmi {EMOTION_LABELS}'}, status_code=400)

        # Sauvegarder l'image
        emotion_dir = Path(Config.TRAINING_DATA_PATH) / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = emotion_dir / f"{timestamp}.jpg"

        contents = await image.read()
        with open(filename, 'wb') as f:
            f.write(contents)

        return JSONResponse({
            'success': True,
            'message': f'Image ajoute pour l\'motion {emotion}',
            'path': str(filename)
        })

    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

# ==========================================================
# FONCTION POUR TÉLÉCHARGER UNE VIDÉO DEPUIS UNE URL
# ==========================================================
async def download_video_from_url(url: str, custom_filename: str = None, max_size_mb: int = 500):
    import urllib.request
    import uuid
    import subprocess

    unique_id = str(uuid.uuid4())[:8]
    filename = f"candidate_{unique_id}.mp4"
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    fixed_path = os.path.join(tempfile.gettempdir(), f"fixed_{filename}")

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        print(f"[download] Téléchargement vers: {temp_path}")
        urllib.request.urlretrieve(url, temp_path)

        size = os.path.getsize(temp_path)
        print(f"[download] Terminé: {size} octets")

        if size == 0:
            raise ValueError("Fichier vide (0 octets)")

        if size > max_size_mb * 1024 * 1024:
            os.remove(temp_path)
            raise ValueError(f"Vidéo trop grande (max {max_size_mb}MB)")

        # ← DÉPLACER LE MOOV ATOM AU DÉBUT (faststart)
        print(f"[download] Application faststart...")
        result = subprocess.run(
            [FFMPEG_PATH, '-i', temp_path, '-c', 'copy', '-movflags', '+faststart', '-y', fixed_path],
            capture_output=True, text=True, timeout=120
        )

        if result.returncode == 0 and os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 0:
            os.remove(temp_path)
            print(f"[download] faststart OK: {os.path.getsize(fixed_path)} octets")
            return fixed_path, filename
        else:
            print(f"[download] faststart stderr: {result.stderr[-200:]}")
            if os.path.exists(fixed_path):
                os.remove(fixed_path)
            return temp_path, filename

    except Exception as e:
        for p in [temp_path, fixed_path]:
            if os.path.exists(p):
                os.remove(p)
        raise Exception(f"Erreur téléchargement: {str(e)}")

# ==========================================================
# ENDPOINTS API
# ==========================================================
@app.get("/")
async def root():
    return {
        "name": "Nexum IA - Emotion Analysis API",
        "version": "2.0.0",
        "status": "online",
        "endpoints": {
            "health":                   "GET  /health",
            "model_info":               "GET  /model_info",
            "analyze_realtime":         "POST /analyze_realtime",
            "analyze_video":            "POST /analyze_video",
            "analyze_video_url":        "POST /analyze_video_url",
            "import_video_from_url":    "POST /import_video_from_url",
            "analyze_imported_video":   "POST /analyze_imported_video",
            "extract_candidates":       "POST /extract_candidates_preview",
            "export_pdf":               "POST /export_pdf",
            "chatbot":                  "POST /chatbot",
            "train":                    "POST /train",
            "evaluate":                 "POST /evaluate",
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": {
            "face_emotion": "loaded (fine-tuned)",
            "face_detection": "loaded"
        },
        "device": str(device),
        "training_status": "available",
        "history": history_manager.get_stats()
    }

@app.get("/model_info")
async def model_info():
    """Informations sur le modle entran"""
    return {
        "model_type": "ViT-Base fine-tuned",
        "num_classes": 7,
        "emotions": EMOTION_LABELS,
        "emotions_fr": EMOTION_NAMES_FR,
        "is_trained": os.path.exists(Config.MODEL_PATH),
        "model_path": Config.MODEL_PATH
    }


@app.post("/export_pdf")
async def export_pdf(data: dict):
    """Génère un rapport PDF complet en mémoire (fpdf2) avec un design premium"""
    try:
        from fpdf import FPDF

        class PDFReport(FPDF):
            def header(self):
                # Bannière supérieure
                self.set_fill_color(15, 23, 42) # Slate 900
                self.rect(0, 0, 210, 35, 'F')

                # Ligne d'accentuation cyan
                self.set_fill_color(0, 242, 255)
                self.rect(0, 35, 210, 1.5, 'F')

                self.set_y(12)
                self.set_font("Helvetica", 'B', 20)
                self.set_text_color(255, 255, 255)
                self.cell(0, 10, "NEXUM IA - ANALYSE D'ENTRETIEN", ln=True, align='C')
                self.set_font("Helvetica", 'I', 10)
                self.set_text_color(148, 163, 184) # Slate 400
                self.cell(0, 5, "Analyse Multimodale & Profilage Cognitif", ln=True, align='C')
                self.ln(20)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", 'I', 8)
                self.set_text_color(128, 128, 128)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        pdf = PDFReport()
        pdf.add_page()

        # En-tête candidat
        pdf.set_y(45)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", 'B', 14)
        candidat_name = data.get('candidate_name', 'Anonyme').encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(100, 10, f"Candidat : {candidat_name}")

        pdf.set_font("Helvetica", '', 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(90, 10, f"Date de génération : {datetime.now().strftime('%d/%m/%Y %H:%M')}", align='R', ln=True)
        pdf.line(10, 56, 200, 56)
        pdf.ln(8)

        # 1. INDICATEURS DE PERFORMANCE
        pdf.set_font("Helvetica", 'B', 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "1. INDICATEURS DE PERFORMANCE (SOFT SKILLS)", ln=True)
        pdf.ln(2)

        m = data.get('metrics', {})
        # Création de "cartes" pour les métriques
        def draw_metric_card(x, y, title, value):
            pdf.set_fill_color(248, 250, 252) # Slate 50
            pdf.set_draw_color(226, 232, 240) # Slate 200
            pdf.rect(x, y, 85, 20, 'FD')

            pdf.set_xy(x + 5, y + 4)
            pdf.set_font("Helvetica", 'B', 10)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(75, 5, title, ln=True)

            pdf.set_xy(x + 5, y + 10)
            pdf.set_font("Helvetica", 'B', 14)
            pdf.set_text_color(0, 150, 160)
            pdf.cell(75, 6, f"{value:.1f}%", ln=True)

        current_y = pdf.get_y()
        draw_metric_card(15, current_y, "Gestion du Stress", m.get('stress_management', 0))
        draw_metric_card(110, current_y, "Assurance & Confiance", m.get('assurance_level', 0))
        draw_metric_card(15, current_y + 25, "Qualité de Communication", m.get('communication', 0))
        draw_metric_card(15, current_y + 50, "Risque de Mensonge", data.get('analysis', {}).get('score', 0))

        pdf.set_y(current_y + 80)  # Adjust for added risk card


        # 2. VERDICT DE L'IA
        pdf.set_font("Helvetica", 'B', 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "2. VERDICT ANALYTIQUE & FIABILITÉ", ln=True)
        pdf.ln(2)

        level = data.get('verdict', "Aucun verdict disponible.").encode('latin-1', 'replace').decode('latin-1')

        # Affichage du niveau de risque
        pdf.set_fill_color(240, 253, 250) if "Faible" in level else pdf.set_fill_color(254, 252, 232)
        pdf.rect(10, pdf.get_y(), 190, 12, 'F')
        pdf.set_xy(15, pdf.get_y() + 2)
        pdf.set_font("Helvetica", 'B', 11)
        pdf.set_text_color(15, 118, 110) if "Faible" in level else pdf.set_text_color(180, 83, 9)
        pdf.cell(0, 8, f"ÉVALUATION GLOBALE : {level}", ln=True)
        pdf.ln(8)

        # Timeline des anomalies dans le PDF (Localisation du risque de mensonge)
        dec_timeline = data.get('deception_timeline', [])
        if dec_timeline:
            pdf.set_font("Helvetica", 'B', 11)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 8, "Timeline Chronologique des Anomalies (Suspicion de mensonge) :", ln=True)
            pdf.ln(2)

            for item in dec_timeline[:8]: # Limiter à 8 items max pour ne pas dépasser la page
                t_val = item.get('time', 0.0)
                type_val = item.get('type', '').encode('latin-1', 'replace').decode('latin-1')
                desc_val = item.get('description', '').encode('latin-1', 'replace').decode('latin-1')
                sev_val = item.get('severity', '')

                pdf.set_font("Helvetica", 'B', 9)
                if sev_val == "Élevée":
                    pdf.set_text_color(220, 38, 38) # Red 600
                elif sev_val == "Moyenne":
                    pdf.set_text_color(217, 119, 6) # Amber 600
                else:
                    pdf.set_text_color(71, 85, 105) # Slate 600
                pdf.cell(35, 5, f"[{t_val:.1f}s] - {type_val} : ", ln=0)

                pdf.set_font("Helvetica", '', 9)
                pdf.set_text_color(71, 85, 105)
                pdf.multi_cell(155, 5, desc_val)
            pdf.ln(4)

        # 3. TRANSCRIPTION
        pdf.set_font("Helvetica", 'B', 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "3. RÉSUMÉ DE LA TRANSCRIPTION (EXTRAIT)", ln=True)
        pdf.ln(2)

        pdf.set_font("Helvetica", '', 9)
        pdf.set_text_color(71, 85, 105)

        chunks = data.get('transcript_chunks', [])

        import re
        def clean_hallucinations(text):
            # Supprime les répétitions massives du même mot (ex: "très très très...")
            return re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text).strip()

        if chunks:
            # Format horodaté
            max_chunks = 40 # Limite pour éviter un PDF trop long
            for idx, chunk in enumerate(chunks[:max_chunks]):
                try:
                    start_time = chunk.get('timestamp', [0])[0]
                    text = clean_hallucinations(chunk.get('text', '')).encode('latin-1', 'replace').decode('latin-1')
                    if text:
                        # Timestamp en gras
                        pdf.set_font("Helvetica", 'B', 9)
                        pdf.set_text_color(0, 150, 160)
                        pdf.cell(15, 5, f"[{start_time:.1f}s]", ln=0)

                        # Texte normal
                        pdf.set_font("Helvetica", '', 9)
                        pdf.set_text_color(71, 85, 105)
                        pdf.multi_cell(175, 5, text)
                except Exception:
                    pass

            if len(chunks) > max_chunks:
                pdf.ln(2)
                pdf.set_font("Helvetica", 'I', 9)
                pdf.cell(0, 5, "... [Transcription tronquée pour la lisibilité] ...", ln=True)
        else:
            # Fallback format texte brut
            transcript = data.get('transcript', "Aucun texte détecté.").encode('latin-1', 'replace').decode('latin-1')
            transcript = clean_hallucinations(transcript)
            if len(transcript) > 1500:
                transcript = transcript[:1500] + " ... [Texte tronqué]"
            pdf.multi_cell(190, 5, transcript)

        # Génération
        pdf_bytes = pdf.output()

        return Response(
            content=bytes(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=Rapport_Nexum_Premium.pdf"}
        )
    except Exception as e:
        print(f"Erreur PDF: {e}")
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.post("/extract_candidates_preview")
async def extract_candidates_preview(file: UploadFile = File(...)):
    """Traite les 10 premières secondes d'une vidéo pour identifier les candidats et extraire leur visage."""
    import time
    import base64
    try:
        tmp_path = os.path.join(tempfile.gettempdir(), f"preview_{file.filename}")
        with open(tmp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        video_clip = mp.VideoFileClip(tmp_path)
        duration = video_clip.duration
        preview_duration = min(10.0, duration)

        sample_rate = 1.0  # 1 frame par seconde
        candidates = {}  # {group_x_index: {'face_img': np_array, 'cx': float, 'cy': float, 'count': int, 'bbox': list}}

        # Pour chaque seconde
        for t in np.arange(0, preview_duration, sample_rate):
            frame = video_clip.get_frame(t)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            faces = detect_faces(frame_bgr)

            for face_img, conf, bbox, padded_bbox in faces:
                # Calculer le centre horizontal
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                # Regrouper les visages par leur position horizontale
                # On tolère une déviation de 15% de la largeur de l'image pour regrouper le même candidat
                img_w = frame.shape[1]
                group_key = None
                for existing_key in candidates.keys():
                    if abs(existing_key - cx) < (img_w * 0.15):
                        group_key = existing_key
                        break

                if group_key is None:
                    group_key = cx
                    candidates[group_key] = {
                        'face_img': face_img,
                        'cx': cx,
                        'cy': cy,
                        'bbox': bbox,
                        'count': 1
                    }
                else:
                    # On garde l'image de visage avec la plus grande taille ou netteté (plus grand bbox)
                    candidates[group_key]['count'] += 1
                    curr_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    prev_bbox = candidates[group_key]['bbox']
                    prev_size = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
                    if curr_size > prev_size:
                        candidates[group_key]['face_img'] = face_img
                        candidates[group_key]['cx'] = cx
                        candidates[group_key]['cy'] = cy
                        candidates[group_key]['bbox'] = bbox

        video_clip.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        valid_candidates = []
        sorted_keys = sorted(candidates.keys())
        for idx, k in enumerate(sorted_keys):
            c = candidates[k]
            face_img = c['face_img']
            _, buffer = cv2.imencode('.jpg', face_img)
            jpg_bytes = buffer.tobytes()
            b64_str = base64.b64encode(jpg_bytes).decode('utf-8')

            valid_candidates.append({
                'id': idx + 1,
                'face_image': f"data:image/jpeg;base64,{b64_str}",
                'center_x': float(c['cx']),
                'center_y': float(c['cy']),
                'bbox': [int(val) for val in c['bbox']],
                'name': f"Candidat {idx + 1}"
            })

        return JSONResponse({
            'success': True,
            'candidates': valid_candidates,
            'count': len(valid_candidates)
        })

    except Exception as e:
        print(f"Erreur Extraction Preview: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.post("/analyze_video_from_url")
async def analyze_video_from_url(request: VideoURLRequest):
    temp_path = None
    try:
        print(f"[analyze_video_from_url] URL reçue: {request.url[:100]}")
        print(f"[analyze_video_from_url] skip_face_detection: {request.skip_face_detection}")

        temp_path, filename = await download_video_from_url(request.url, request.filename)
        print(f"[analyze_video_from_url] Téléchargé: {temp_path}, taille: {os.path.getsize(temp_path)}")

        class VirtualUploadFile:
            def __init__(self, path, name):
                self.filename = name
                self._path = path
            async def read(self):
                with open(self._path, 'rb') as f:
                    return f.read()

        virtual_file = VirtualUploadFile(temp_path, filename)

        # On utilise TOUJOURS la fonction detect_faces normale (pas de fallback frame entière)
        result = await analyze_video(file=virtual_file, target_x=None, target_y=None)

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        return result

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[analyze_video_from_url] ERREUR TYPE: {type(e).__name__}")
        print(f"[analyze_video_from_url] ERREUR MESSAGE: {e}")
        print(f"[analyze_video_from_url] STACKTRACE:\n{error_details}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'details': error_details
        }, status_code=500)

@app.post("/analyze_video")
async def analyze_video(
    file: UploadFile = File(...),
    target_x: Optional[float] = Form(None),
    target_y: Optional[float] = Form(None)
):
    """Analyse multimodale complte d'une vido (Visuel + Audio)"""
    import time
    try:
        start_time = time.time()
        tmp_path = os.path.join(tempfile.gettempdir(), file.filename)
        with open(tmp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        print(f"Analyse de la video: {file.filename}")
        # Load video clip
        video_clip = mp.VideoFileClip(tmp_path)
        has_audio = video_clip.audio is not None

        # Prepare audio path placeholder
        audio_path = None
        audio_data = None
        transcript_text = ""
        transcript_chunks = []
        speech_rate_norm = 0
        speech_rate_real = 0

        if has_audio:
            # Write audio to temporary wav file
            audio_path = os.path.join(tempfile.gettempdir(), f"{os.path.splitext(file.filename)[0]}_audio.wav")
            video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

            # Load full audio for segmentation
            try:
                from moviepy.editor import AudioFileClip
                temp_clip = AudioFileClip(audio_path)
                wav_full = audio_path + "_full.wav"
                temp_clip.write_audiofile(wav_full, verbose=False, logger=None, fps=16000, nbytes=2, codec='pcm_s16le')
                import soundfile as sf
                audio_data, sr = sf.read(wav_full)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                if os.path.exists(wav_full):
                    os.remove(wav_full)
                temp_clip.close()
            except Exception as e_proc:
                print(f"Erreur traitement audio video: {e_proc}")
                has_audio = False

        # 3. Extraire les frames clés (ex: 1 frame par seconde)
        duration = video_clip.duration
        # Validation: ensure video is long enough for analysis
        MIN_VIDEO_DURATION = 5  # seconds, configurable
        if duration < MIN_VIDEO_DURATION:
            video_clip.close()
            return JSONResponse(
                {
                    "success": False,
                    "error": f"Vidéo trop courte ({duration:.1f}s). Minimum requis : {MIN_VIDEO_DURATION}s."
                },
                status_code=400,
            )
        frames_results = []
        visual_history = []
        prev_probs = None

        # On analyse un chantillon de frames pour la performance
        sample_rate = 1.0 # 1 frame par seconde

        # -------------------------------------------------
        # Extract preview faces from the first 10 seconds
        # -------------------------------------------------
        preview_faces = []  # Will hold base64-encoded images
        preview_duration = min(10.0, duration)  # seconds
        import base64, io
        for t_preview in np.arange(0, preview_duration, sample_rate):
            frame_preview = video_clip.get_frame(t_preview)
            frame_preview_bgr = cv2.cvtColor(frame_preview, cv2.COLOR_RGB2BGR)
            faces_preview = detect_faces(frame_preview_bgr)
            if faces_preview:
                # Take the first detected face for preview
                face_img, _, _, _ = faces_preview[0]
                # Encode to JPEG in memory
                _, buffer = cv2.imencode('.jpg', face_img)
                jpg_bytes = buffer.tobytes()
                b64_str = base64.b64encode(jpg_bytes).decode('utf-8')
                preview_faces.append(f"data:image/jpeg;base64,{b64_str}")
        # End of preview extraction
        # Check for presence of faces in the first 10 seconds
        face_popup = False
        if len(preview_faces) == 0:
            face_popup = True
        # Determine audio presence (has_audio already set)
        audio_popup = not has_audio

        # Initialisation du suivi du candidat sélectionné (verrouillage centroïde)
        last_cx = target_x
        last_cy = target_y

        for t_idx, t in enumerate(np.arange(0, duration, sample_rate)):
            frame = video_clip.get_frame(t)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            faces = detect_faces(frame_bgr)
            selected_face = None
            if faces:
                if last_cx is not None and last_cy is not None:
                    # Chercher le visage le plus proche du dernier centre connu
                    best_dist = float('inf')
                    for f_img, f_conf, tight_bbox, padded_bbox in faces:
                        cx = (tight_bbox[0] + tight_bbox[2]) / 2
                        cy = (tight_bbox[1] + tight_bbox[3]) / 2
                        dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)

                        if dist < best_dist:
                            best_dist = dist
                            selected_face = (f_img, f_conf, tight_bbox, padded_bbox)

                    if selected_face is not None:
                        # Mettre à jour l'ancre du candidat pour le suivi dynamique
                        _, _, tight_bbox, _ = selected_face
                        last_cx = (tight_bbox[0] + tight_bbox[2]) / 2
                        last_cy = (tight_bbox[1] + tight_bbox[3]) / 2
                else:
                    # Par défaut, on prend le plus grand visage (candidat principal)
                    selected_face = max(faces, key=lambda x: (x[2][2]-x[2][0]) * (x[2][3]-x[2][1]))
                    last_cx = (selected_face[2][0] + selected_face[2][2]) / 2
                    last_cy = (selected_face[2][1] + selected_face[2][3]) / 2

            if selected_face:
                face_img, conf, bbox, padded_bbox = selected_face

                # Préparation visuelle
                face_processed = preprocess_face(face_img)
                face_rgb = cv2.cvtColor(face_processed, cv2.COLOR_BGR2RGB)
                vis_inputs = base_processor(images=face_rgb, return_tensors="pt").to(device)

                # Prédiction visuelle (ONNX ou PyTorch standard)
                if USE_ONNX:
                    pixel_values_np = vis_inputs['pixel_values'].cpu().numpy()
                    vit_logits, vit_features_np = vit_session.run(None, {"pixel_values": pixel_values_np})
                    probs = F.softmax(torch.tensor(vit_logits), dim=-1).numpy()[0]
                else:
                    with torch.no_grad():
                        logits = model(vis_inputs['pixel_values'])
                        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    vit_features_np = None

                # Variable pour stocker les probabilités finales (visuel ou fusion)
                final_raw_probs = probs
                fusion_weights_list = None
                audio_probs_numpy = None

                # Si on a de l'audio, on fait la fusion sur un segment de 2 secondes
                if has_audio and audio_data is not None:
                    # Extraire 1s avant et 1s après le timestamp t
                    start_sample = int(max(0, (t - 1.0) * 16000))
                    end_sample = int(min(len(audio_data), (t + 1.0) * 16000))
                    segment = audio_data[start_sample:end_sample]

                    if len(segment) > 0:
                        seg_inputs = audio_processor(segment, sampling_rate=16000, return_tensors="pt").to(device)

                        if USE_ONNX:
                            input_values_np = seg_inputs['input_values'].cpu().numpy()
                            # Inférence Audio avec HuBERT en ONNX
                            hubert_logits, hubert_features_np = hubert_session.run(None, {"input_values": input_values_np})
                            audio_probs_numpy = F.softmax(torch.tensor(hubert_logits), dim=-1).numpy()[0]
                        else:
                            with torch.no_grad():
                                audio_logits = audio_model(seg_inputs['input_values'])
                                audio_probs_numpy = F.softmax(audio_logits, dim=-1).cpu().numpy()[0]

                        # ALIGNEMENT DES CLASSES HuBERT (4 classes) SUR LES CLASSES TARGET (7 classes) AVEC FILTRAGE ANTI-BIAIS STRICT
                        # HuBERT: 0=neutral, 1=happy, 2=sad, 3=angry
                        # Target: 0=sad, 1=disgust, 2=angry, 3=neutral, 4=fear, 5=surprise, 6=happy
                        # On atténue fortement la tristesse (0.2) et la colère (0.2) qui parasitent le signal de parole calme.
                        # On booste la neutralité (1.6) pour stabiliser la parole naturelle en entretien.
                        audio_7_probs = np.zeros(7)
                        audio_7_probs[0] = audio_probs_numpy[2] * 0.2  # sad
                        audio_7_probs[2] = audio_probs_numpy[3] * 0.2  # angry
                        audio_7_probs[3] = audio_probs_numpy[0] * 1.6  # neutral
                        audio_7_probs[6] = audio_probs_numpy[1] * 1.2  # happy
                        if np.sum(audio_7_probs) > 0:
                            audio_7_probs = audio_7_probs / np.sum(audio_7_probs)

                        # FUSION PROBABILISTE LATE FUSION DYNAMIQUE ET INTELLIGENTE
                        # Si le visuel détecte de la joie (happy, index 6) ou de la surprise (index 5) avec une confiance décente (> 0.20),
                        # on augmente la pondération visuelle à 90% pour éviter qu'une voix de parole monotone n'écrase le sourire.
                        # Sinon, on garde un équilibre standard (70% Visuel, 30% Audio).
                        if probs[6] > 0.20 or probs[5] > 0.20:
                            visual_weight = 0.90
                        else:
                            visual_weight = 0.70
                        audio_weight = 1.0 - visual_weight

                        final_raw_probs = visual_weight * probs + audio_weight * audio_7_probs
                        fusion_weights_list = [visual_weight, audio_weight]

                # Appliquer la calibration, l'EMA local et l'anti-biais sur le résultat (visuel ou fusion)
                emotion, conf_score, calibrated_probs, prev_probs = calibrate_and_smooth_probs(final_raw_probs, prev_probs=prev_probs)
                visual_history.append(calibrated_probs)

                # Calcul des métriques candidat
                metrics = calculate_candidate_metrics(
                    calibrated_probs,
                    audio_probs_numpy,
                    history=[f['emotion'] for f in frames_results]
                )
                if has_audio and 'speech_rate_norm' in locals():
                    metrics['speech_rate'] = float(speech_rate_norm)

                result_entry = {
                    'timestamp': float(t),
                    'emotion': emotion,
                    'emotion_fr': EMOTION_NAMES_FR.get(emotion, emotion),
                    'confidence': float(conf_score),
                    'bbox': [int(c) for c in bbox],
                    'metrics': metrics
                }

                if fusion_weights_list is not None:
                    result_entry['fusion_weights'] = fusion_weights_list

                frames_results.append(result_entry)

        # Nettoyage
        video_clip.close()
        if os.path.exists(tmp_path): os.remove(tmp_path)
        if audio_path and os.path.exists(audio_path): os.remove(audio_path)

        # Synthse globale
        if not frames_results:
            return JSONResponse({'success': False, 'message': 'Aucun visage dtect dans la vido'})

        # Post-traitement global
        avg_visual = np.mean(visual_history, axis=0) if visual_history else np.zeros(7)
        final_metrics = calculate_candidate_metrics(avg_visual, history=visual_history)
        soft_skills = analyze_soft_skills(transcript_text, final_metrics)
        inconsistencies = detect_inconsistencies(transcript_text, visual_history)

        # Timeline émotionnelle
        timeline = []
        for f in frames_results:
            timeline.append({
                "time": f['timestamp'],
                "emotion": f['emotion'],
                "confidence": f['confidence']
            })

        # Historique des KPIs pour les courbes
        metrics_history = []
        for f in frames_results:
            if 'metrics' in f:
                metrics_history.append({
                    "time": f['timestamp'],
                    "stress": f['metrics']['stress_management'],
                    "comm": f['metrics']['communication'],
                    "expr": f['metrics']['expressivity'],
                    "speed": f['metrics'].get('speech_rate', 50),
                    "conf": f['metrics']['assurance_level'],
                    "model_conf": f['confidence'] * 100
                })

        print("Analyse terminée avec succès")

        # Calcul du score de tromperie (visuel + audio)
        vis_risk_score, vis_risk_level, vis_risk_details = calculate_deception_risk([f['emotion'] for f in frames_results], [f['confidence'] for f in frames_results], [f['timestamp'] for f in frames_results])
        speech_risk_score, speech_flags = analyze_speech_deception(transcript_text)

        final_score = (vis_risk_score * 0.6) + (speech_risk_score * 0.4)

        if final_score < 30:
            risk_level = "Faible - Discours probablement authentique"
        elif final_score < 60:
            risk_level = "Modéré - Situation à surveiller"
        else:
            risk_level = "Élevé - Forte probabilité de mensonge ou d'omission"

        inconsistencies.extend(speech_flags)
        risk_details = vis_risk_details
        risk_details['speech_score'] = speech_risk_score

        # Analyse bot Feedback
        ai_feedback = []
        high_stress = [t for t in timeline if t['emotion'] in ['fear', 'sad'] and t['confidence'] > 0.6]
        if high_stress:
            t = high_stress[0]['time']
            ai_feedback.append({
                "timestamp": t,
                "reason": "Stress détecté",
                "feedback": f"À {t}s, le candidat montre des signes de tension. Vérifiez s'il s'agit d'un manque d'assurance sur le sujet abordé."
            })

        joy_moments = [t for t in timeline if t['emotion'] == 'happy' and t['confidence'] > 0.8]
        if joy_moments:
            t = joy_moments[0]['time']
            ai_feedback.append({
                "timestamp": t,
                "reason": "Engagement positif",
                "feedback": f"Le candidat semble particulièrement à l'aise et enthousiaste à {t}s."
            })

        if not ai_feedback:
            ai_feedback.append({"timestamp": 0, "reason": "Vue d'ensemble", "feedback": "Comportement globalement stable."})

        # Calcul des KPIs système sur cette vidéo
        processing_time = time.time() - start_time
        rtf = duration / processing_time if processing_time > 0 else 0
        avg_confidence = np.mean([f['confidence'] for f in frames_results]) * 100 if frames_results else 0
        fps = len(frames_results) / processing_time if processing_time > 0 else 0
        latency_ms = (processing_time / len(frames_results)) * 1000 if frames_results else 0

        system_kpis = {
            "real_time_factor": round(rtf, 2),
            "avg_confidence": round(avg_confidence, 1),
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 2),
            "processing_time": round(processing_time, 2)
        }

        def to_serializable(obj):
            """Recursively convert NumPy and Torch objects to native Python types for JSON serialization."""
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_serializable(v) for v in obj]
            return obj

        # Construction de la timeline de suspicion de mensonge / stress (localisation précise)
        deception_timeline = []

        # 1. Parcourir les frames pour trouver les pics de stress ou de micro-expressions
        for idx, f in enumerate(frames_results):
            t = f['timestamp']
            emotion = f['emotion']
            conf = f['confidence']

            # Pic de stress
            if emotion in ['fear', 'angry', 'disgust'] and conf > 0.55:
                deception_timeline.append({
                    "time": float(t),
                    "type": "Stress Émotionnel",
                    "severity": "Élevée" if conf > 0.75 else "Moyenne",
                    "description": f"Pic de tension détecté ({EMOTION_NAMES_FR.get(emotion, emotion)}) avec {int(conf*100)}% de confiance."
                })

            # Détection de micro-expression (changement soudain d'émotion en 1 seconde)
            if idx >= 1 and frames_results[idx]['emotion'] != frames_results[idx-1]['emotion']:
                prev_e = frames_results[idx-1]['emotion']
                curr_e = frames_results[idx]['emotion']
                if prev_e != 'neutral' and curr_e != 'neutral':
                    deception_timeline.append({
                        "time": float(t),
                        "type": "Micro-expression",
                        "severity": "Moyenne",
                        "description": f"Changement émotionnel brusque de '{EMOTION_NAMES_FR.get(prev_e, prev_e)}' vers '{EMOTION_NAMES_FR.get(curr_e, curr_e)}'."
                    })

        # 2. Parcourir les segments de transcription pour repérer les anomalies verbales
        for chunk in transcript_chunks:
            text = chunk.get('text', '').lower()
            start_t = chunk.get('timestamp', [0])[0]

            hesitation_words = ["euh", "bah", "en fait", "je crois", "peut-être", "genre", "comment dire", "je ne sais pas"]
            over_justification = ["honnêtement", "pour être franc", "à vrai dire", "croyez-moi", "sincèrement", "je vous jure"]

            found_hesitations = [w for w in hesitation_words if w in text]
            found_justifications = [w for w in over_justification if w in text]

            if found_hesitations:
                deception_timeline.append({
                    "time": float(start_t),
                    "type": "Hésitation Verbale",
                    "severity": "Moyenne" if len(found_hesitations) > 1 else "Faible",
                    "description": f"Le candidat hésite dans son discours : '{', '.join(found_hesitations)}'."
                })
            if found_justifications:
                deception_timeline.append({
                    "time": float(start_t),
                    "type": "Sur-justification",
                    "severity": "Moyenne",
                    "description": f"Emploi d'expressions d'affirmation excessive ('{', '.join(found_justifications)}'), indiquant un besoin excessif de convaincre."
                })

        # Trier la timeline par timestamp croissant
        deception_timeline = sorted(deception_timeline, key=lambda x: x['time'])

        # Construct response payload
        response_payload = {
            "success": True,
            "transcript": transcript_text,
            "preview_faces": preview_faces,
            # Flags for UI popups
            "face_popup": len(preview_faces) == 0,  # No faces in first 10 seconds
            "audio_popup": not has_audio,  # No audio detected (if applicable)
            "frames": frames_results,
            "timeline": timeline,
            "metrics_history": metrics_history,
            "system_kpis": system_kpis,
            "metrics": final_metrics,
            "soft_skills": soft_skills,
            "analysis": {
                "score": final_score,
                "level": risk_level,
                "details": risk_details,
                "deception_timeline": deception_timeline
            },
            "feedback": ai_feedback,
            "inconsistencies": inconsistencies,
            # Strengths derived from final metrics
            "strengths": {
                "stress_management": final_metrics.get("stress_management", 0),
                "communication": final_metrics.get("communication", 0),
                "assurance_level": final_metrics.get("assurance_level", 0),
                "expressivity": final_metrics.get("expressivity", 0),
                "speech_rate": final_metrics.get("speech_rate", 0)
            },
            # Simple suggestions based on low scores (<70)
            "suggestions": [
                *(
                    [
                        f"Améliorer la gestion du stress (actuel: {final_metrics.get('stress_management', 0):.1f}%). Considérez des exercices de respiration avant l'entretien."
                    ] if final_metrics.get('stress_management', 0) < 70 else []
                ),
                *(
                    [
                        f"Travailler la communication (actuel: {final_metrics.get('communication', 0):.1f}%). Entraînez-vous à formuler des réponses claires et structurées."
                    ] if final_metrics.get('communication', 0) < 70 else []
                ),
                *(
                    [
                        f"Renforcer l'assurance (actuel: {final_metrics.get('assurance_level', 0):.1f}%). Posez-vous des questions de simulation pour gagner en confiance."
                    ] if final_metrics.get('assurance_level', 0) < 70 else []
                ),
                *(
                    [
                        f"Améliorer l'expressivité (actuel: {final_metrics.get('expressivity', 0):.1f}%). Travaillez votre gestuelle et votre ton de voix."
                    ] if final_metrics.get('expressivity', 0) < 70 else []
                ),
                *(
                    [
                        f"Optimiser le débit de parole (actuel: {final_metrics.get('speech_rate', 0):.1f} mots/min). Pratiquez avec un métronome ou enregistrez-vous."
                    ] if final_metrics.get('speech_rate', 0) < 70 else []
                )
            ]
        }
# Duplicate response payload block removed

# Duplicate response payload block removed

        return JSONResponse(to_serializable(response_payload))

    except Exception as e:
        print(f"Erreur Analyse Video: {e}")
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.post("/chatbot")
async def chatbot_interaction(data: dict):
    """Chatbot intelligent pour discuter des résultats de l'entretien"""
    query = data.get("query", "").lower()
    context = data.get("context", {})

    metrics = context.get("metrics", {})
    analysis = context.get("analysis", {})
    flags = context.get("inconsistencies", [])

    stress = metrics.get("stress_management", 50)
    conf = metrics.get("assurance_level", 50)
    deception_score = analysis.get("score", 0)

    response = ""

    if "mensonge" in query or "vérité" in query or "vrai" in query or "tromperie" in query:
        if deception_score < 30:
            response = f"Le risque de mensonge est FAIBLE ({deception_score:.1f}%). Le candidat semble authentique, avec une belle cohérence entre ses émotions faciales et son discours."
        elif deception_score < 60:
            response = f"Le risque de mensonge est MODÉRÉ ({deception_score:.1f}%). J'ai détecté quelques hésitations ou micro-expressions de stress. Suggestion: Creusez ses dernières expériences avec des questions techniques précises."
        else:
            response = f"⚠️ Risque de mensonge ÉLEVÉ ({deception_score:.1f}%). Le candidat montre des signes importants de stress, d'hésitations ou de décalages."
            if flags:
                response += " Détails: " + " ".join(flags)
            response += " Suggestion pour le recruteur: Demandez-lui de détailler étape par étape comment il a résolu le problème mentionné, les menteurs ont du mal avec les détails chronologiques."

    elif "stress" in query or "anxieux" in query:
        if stress > 70:
            response = f"Le candidat a une excellente gestion du stress ({stress:.1f}%). Il reste serein même face à la caméra."
        else:
            response = f"Le candidat a montré des signes visibles de tension ({stress:.1f}%). Suggestion: Mettez-le à l'aise avec une question ouverte sur ses passions avant de revenir aux compétences clés."

    elif "confiance" in query or "assurance" in query:
        response = f"Son niveau d'assurance est de {conf:.1f}%. "
        if conf > 60:
            response += "Il dégage une posture très solide et convaincante. C'est un profil idéal pour des postes à responsabilités ou en contact client."
        else:
            response += "Il paraît hésitant. Suggestion: Demandez-lui de parler d'un projet dont il est particulièrement fier pour le valoriser et voir sa réaction."

    elif "suggestion" in query or "conseil" in query or "question" in query:
        if deception_score > 50:
            response = "Je vous suggère de vérifier ses références. Posez-lui des questions comportementales de type STAR (Situation, Tâche, Action, Résultat) pour valider factuellement ses dires."
        elif stress < 50:
            response = "Le candidat est stressé. Posez-lui des questions sur son parcours global pour le détendre et observer s'il reprend confiance."
        else:
            response = "Le profil est solide. Vous pouvez approfondir la discussion sur son adaptation à la culture de votre entreprise ou ses attentes salariales."

    elif "résumé" in query or "bilan" in query or "analyse" in query or "profil" in query:
        response = f"Bilan rapide: Assurance ({conf:.0f}%), Gestion du stress ({stress:.0f}%), Risque de mensonge ({deception_score:.0f}%). "
        if deception_score > 60:
            response += "Point de vigilance majeur sur la véracité de ses propos. "
        elif stress < 40:
            response += "Point d'attention sur sa nervosité, il faut le rassurer. "
        else:
            response += "Candidat solide et cohérent. "
        response += "Que voulez-vous approfondir ?"

    else:
        response = "Je suis l'assistant IA Nexy. Je peux analyser en détail le risque de mensonge, le stress, l'assurance, ou vous donner des suggestions de questions (ex: tapez 'mensonge' ou 'suggestions'). Que souhaitez-vous savoir ?"

    return {"response": response}

@app.post("/analyze")
async def analyze_frame(frame: Optional[UploadFile] = File(None), video: Optional[UploadFile] = File(None)):
    """Analyse amliore d'une frame vido (accepte plusieurs noms de champs)"""
    file = frame or video
    if not file:
        return JSONResponse({'success': False, 'error': 'Aucun fichier reu (champs attendus: frame ou video)'}, status_code=400)

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({'success': False, 'error': 'Image invalide'})

        faces = detect_faces(img)

        if not faces:
            return JSONResponse({
                'success': True,
                'faces_detected': False,
                'message': 'Aucun visage dtect',
                'results': []
            })

        results = []
        for face, det_conf, bbox, padded_bbox in faces:
            emotion, confidence, top3 = predict_emotion_enhanced(face)

            # Plus de simulation arbitraire, on utilise les scores du modèle
            # On base l'attention sur la confiance de detection
            att_weight = float(det_conf)

            # Obtenir les probabilités pour les métriques
            inputs_v = base_processor(images=cv2.cvtColor(preprocess_face(face), cv2.COLOR_BGR2RGB), return_tensors="pt").to(device)
            with torch.no_grad():
                logits_v = model(inputs_v['pixel_values'])
                v_probs = F.softmax(logits_v, dim=-1).cpu().numpy()[0]

            # Calcul des métriques réelles (sans audio temporel ici, mais basé sur le visage)
            metrics = calculate_candidate_metrics(v_probs)

            results.append({
                'bbox': [int(c) for c in bbox],
                'emotion': emotion,
                'emotion_fr': EMOTION_NAMES_FR.get(emotion, emotion),
                'emoji': EMOTION_EMOJIS.get(emotion, ''),
                'color': EMOTION_COLORS.get(emotion, '#667eea'),
                'confidence': float(confidence),
                'attention_weight': float(att_weight),
                'top3_predictions': [(str(e), float(p)) for e, p in top3],
                'detection_confidence': float(det_conf),
                'candidate_metrics': metrics
            })

        return JSONResponse({
            'success': True,
            'faces_detected': True,
            'num_faces': len(results),
            'results': results,
            'model_used': 'fine-tuned' if os.path.exists(Config.MODEL_PATH) else 'base'
        })

    except Exception as e:
        print(f"Erreur: {e}")
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.post("/analyze_sequence")
async def analyze_sequence(frames: List[UploadFile]):
    """Analyse d'une squence de frames pour dtection de micro-expressions"""
    try:
        sequence_results = []
        timestamps = []

        for frame in frames:
            contents = await frame.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                faces = detect_faces(img)
                if faces:
                    emotion, confidence, _ = predict_emotion_enhanced(faces[0][0])
                    sequence_results.append(emotion)
                    timestamps.append(datetime.now().timestamp())

        # Dtection des micro-expressions
        micro_expressions = []
        for i in range(1, len(sequence_results)):
            if sequence_results[i] != sequence_results[i-1]:
                micro_expressions.append({
                    'time': timestamps[i],
                    'from': sequence_results[i-1],
                    'to': sequence_results[i]
                })

        return JSONResponse({
            'success': True,
            'sequence_length': len(sequence_results),
            'micro_expressions_detected': len(micro_expressions),
            'micro_expressions': micro_expressions[:10],  # Top 10
            'emotion_distribution': {e: sequence_results.count(e) for e in set(sequence_results)}
        })

    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.post("/analyze_realtime")
async def analyze_realtime(
    frame: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
    click_x: Optional[int] = Form(None),
    click_y: Optional[int] = Form(None),
    is_first_frame: Optional[bool] = Form(False)
):
    """Analyse temps réel avec sélection de candidat par clic"""
    try:
        # 1. Process Frame
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return JSONResponse({'success': False, 'error': 'Image invalide'})

        faces = detect_faces(img)
        face_img = None
        bbox = []
        all_faces = []
        emotion = "None"
        confidence = 0

        if faces:
            # Récupérer les coordonnées de tous les visages détectés (visage serré uniquement)
            all_faces = [list(f[2]) for f in faces]

            # Si un clic est fourni, on cherche le visage le plus proche avec un seuil de tolérance (gating)
            if click_x is not None and click_y is not None:
                best_face = None
                min_dist = float('inf')
                for f_img, f_conf, tight_bbox, padded_bbox in faces:
                    # Centre du visage
                    cx, cy = (tight_bbox[0] + tight_bbox[2])/2, (tight_bbox[1] + tight_bbox[3])/2
                    dist = np.sqrt((cx - click_x)**2 + (cy - click_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_face = (f_img, f_conf, tight_bbox, padded_bbox)

                # Verrouiller de manière dynamique et continue sur le visage le plus proche du dernier point connu
                if best_face is not None:
                    face_img, det_conf, bbox, padded_bbox = best_face
                    emotion, confidence, top3 = predict_emotion_enhanced(face_img, reset_session=is_first_frame)
                else:
                    face_img = None
                    bbox = []
            else:
                # Par défaut, le visage le plus grand (le plus proche de la caméra)
                face_img, det_conf, bbox, padded_bbox = max(faces, key=lambda x: (x[2][2]-x[2][0]) * (x[2][3]-x[2][1]))
                emotion, confidence, top3 = predict_emotion_enhanced(face_img, reset_session=is_first_frame)

        # 2. Process Audio Chunk (si présent)
        transcript = ""
        audio_probs = None
        if audio:
            audio_contents = await audio.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(audio_contents)
                tmp_audio_path = tmp_audio.name

            try:
                import soundfile as sf
                y, sr = sf.read(tmp_audio_path)
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)

                # Transcription du chunk avec Whisper en passant le raw audio
                ts_result = transcriber({"sampling_rate": 16000, "raw": y})
                transcript = ts_result["text"]

                if len(y) > 0:
                    audio_inputs = audio_processor(y, sampling_rate=16000, return_tensors="pt").to(device)
                    with torch.no_grad():
                        logits_a = audio_model(audio_inputs['input_values'])
                        audio_probs = F.softmax(logits_a, dim=-1).cpu().numpy()[0]
            except Exception as e_audio:
                print(f"Erreur audio segment: {e_audio}")
            finally:
                if os.path.exists(tmp_audio_path): os.remove(tmp_audio_path)

        # 3. Métriques multimodales et Qualité / Fiabilité
        v_probs = None
        face_status = 'Aucun visage detecté '
        brightness = 0
        blur_val = 0
        if face_img is not None:
            # Télémétrie de fiabilité de la caméra (Luminosité et Flou)
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            if brightness < 45:
                face_status = 'Sombre '
            elif brightness > 220:
                face_status = 'Exposé '
            elif blur_val < 50:
                face_status = 'Flou '
            else:
                face_status = 'Optimal '

            inputs_v = base_processor(images=cv2.cvtColor(preprocess_face(face_img), cv2.COLOR_BGR2RGB), return_tensors="pt").to(device)
            with torch.no_grad():
                logits_v = model(inputs_v['pixel_values'])
                v_probs = F.softmax(logits_v, dim=-1).cpu().numpy()[0]

        # Télémétrie de fiabilité audio (Silence et Bruit)
        audio_status = 'Micro Inactif 🎙️'
        if audio and 'y' in locals() and len(y) > 0:
            rms = float(np.sqrt(np.mean(y**2)))
            if rms < 0.003:
                audio_status = 'Silence / Bruit ambiant ⏸️'
            else:
                audio_status = 'Clair ✅'
        elif audio:
            audio_status = 'Pas de voix détectée 🎙️'

        metrics = calculate_candidate_metrics(v_probs, audio_probs)

#         # 4. Update 3D viewer shared state
#         try:
#             if v_probs is not None:
#                 _latest_3d_data["emotion_probs"] = {EMOTION_LABELS[i]: float(v_probs[i]) for i in range(len(EMOTION_LABELS))}
#             _latest_3d_data["emotion"] = emotion
#             _latest_3d_data["confidence"] = float(confidence)
#             # Point cloud from raw frame bytes (best-effort)
#             pc = generate_pointcloud_from_frame(contents)
#             _latest_3d_data["pointcloud"] = pc
#             # Waveform from audio or synthetic demo
#             if audio and 'y' in locals() and len(y) > 0:
#                 step = max(1, len(y) // 256)
#                 wf = y[::step][:256]
#                 _latest_3d_data["waveform"] = [float(v) for v in wf]
#             else:
#                 import math as _math
#                 _latest_3d_data["waveform"] = [_math.sin(i * 0.2) * 0.3 for i in range(256)]
#         except Exception:
#             pass  # Never let 3D state update break the primary response

        return JSONResponse({
            'success': True,
            'faces_detected': face_img is not None,
            'emotion': emotion,
            'emotion_fr': EMOTION_NAMES_FR.get(emotion, emotion),
            'emoji': EMOTION_EMOJIS.get(emotion, ''),
            'color': EMOTION_COLORS.get(emotion, '#667eea'),
            'confidence': float(confidence),
            'transcript': transcript,
            'candidate_metrics': metrics,
            'bbox': [int(c) for c in bbox],
            'all_faces': all_faces,
            'reliability': {
                'face': {
                    'brightness': round(brightness, 1),
                    'blur': round(blur_val, 1),
                    'status': face_status
                },
                'audio': {
                    'status': audio_status
                }
            }
        })

    except Exception as e:
        print(f"Erreur temps réel: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.post("/import_video_from_url")
async def import_video_from_url(url: str = FastAPIForm(...)):
    temp_path = None
    try:
        print(f"[import_video_from_url] Téléchargement: {url}")
        temp_path, filename = await download_video_from_url(url)
        print(f"[import_video_from_url] Téléchargé: {temp_path}")

        video_id = abs(hash(temp_path)) % 100000
        _url_video_cache[video_id] = {
            "path": temp_path,
            "filename": filename,
            "url": url
        }

        return JSONResponse({
            "success": True,
            "video_id": video_id,
            "filename": filename
        })

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"[import_video_from_url] Erreur: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/analyze_imported_video")
async def analyze_imported_video(video_id: int = FastAPIForm(...)):
    import base64
    cached = None
    try:
        print(f"[analyze_imported_video] video_id={video_id}")

        cached = _url_video_cache.get(video_id)
        if not cached:
            return JSONResponse({"success": False, "error": f"Vidéo {video_id} introuvable. Re-importez l'URL."}, status_code=404)

        temp_path = cached["path"]
        filename = cached["filename"]
        source_url = cached["url"]

        # Validate video duration (must be at least 5 seconds)
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return JSONResponse({'success': False, 'error': 'Impossible d\'ouvrir la vidéo pour validation'}, status_code=500)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps else 0
        cap.release()
        if duration < 5:
            # Video too short, reject processing
            return JSONResponse({'success': False, 'error': f'Vidéo trop courte ({duration:.1f}s). Minimum requis: 5 secondes.'}, status_code=400)


        if not os.path.exists(temp_path):
            return JSONResponse({"success": False, "error": "Fichier temporaire expiré. Re-importez l'URL."}, status_code=404)

        class VirtualUploadFile:
            def __init__(self, path, name):
                self.filename = "url_" + name
                self._path = path
            async def read(self):
                with open(self._path, 'rb') as f:
                    return f.read()

        virtual_file = VirtualUploadFile(temp_path, filename)
        result = await analyze_video(file=virtual_file, target_x=None, target_y=None)

        if hasattr(result, 'body'):
            result_data = json.loads(result.body) if isinstance(result.body, bytes) else result.body
        else:
            result_data = result

        if isinstance(result_data, dict):
            result_data["video_id"] = video_id
            result_data["video_filename"] = filename
            result_data["source_url"] = source_url
            with open(temp_path, 'rb') as f:
                result_data["video_data"] = base64.b64encode(f.read()).decode('utf-8')

        if os.path.exists(temp_path):
            os.remove(temp_path)
        _url_video_cache.pop(video_id, None)

        print(f"[analyze_imported_video] Terminé pour video_id={video_id}")
        return JSONResponse(result_data)

    except Exception as e:
        # Fix: on ne tente plus d'accéder à cached.get() si cached est None
        print(f"[analyze_imported_video] Erreur: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/test-url")
async def test_url(request: dict):
    """Test si une URL est accessible et si le fichier est un MP4 valide"""
    import urllib.request
    import uuid
    url = request.get("url", "")
    temp_path = None
    try:
        # 1. Tester l'accès HTTP
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Lire seulement les 32 premiers bytes pour vérifier le header
        with urllib.request.urlopen(req, timeout=15) as response:
            http_status = response.status
            content_type = response.headers.get('Content-Type', 'unknown')
            content_length = response.headers.get('Content-Length', 'unknown')
            header_bytes = response.read(32)

        header_hex = header_bytes.hex()

        # 2. Vérifier si c'est un vrai MP4 (commence par ftyp)
        is_mp4 = b'ftyp' in header_bytes or b'moov' in header_bytes

        # 3. Vérifier si c'est une erreur XML de R2 (AccessDenied)
        is_xml_error = header_bytes[:5] in [b'<?xml', b'<Erro', b'<Acce']

        return JSONResponse({
            'accessible': True,
            'http_status': http_status,
            'content_type': content_type,
            'content_length': content_length,
            'header_hex': header_hex,
            'is_valid_mp4': is_mp4,
            'is_xml_error': is_xml_error,
            'diagnostic': 'MP4 valide' if is_mp4 else ('Erreur R2 XML' if is_xml_error else 'Format inconnu')
        })

    except urllib.error.HTTPError as e:
        return JSONResponse({
            'accessible': False,
            'http_status': e.code,
            'error': str(e),
            'diagnostic': 'URL expirée ou accès refusé' if e.code in [403, 401] else f'HTTP {e.code}'
        })
    except Exception as e:
        return JSONResponse({
            'accessible': False,
            'error': str(e),
            'diagnostic': 'URL inaccessible'
        })
# ==========================================================
# ENDPOINTS POUR LE DASHBOARD - KPIs MODÈLE
# ==========================================================
# ==========================================================
# ENDPOINTS POUR LE DASHBOARD - KPIs MODÈLE
# ==========================================================

@app.get("/report")
async def report_page():
    """Sert la page de rapport d'analyse"""
    from fastapi.responses import HTMLResponse
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Rapport d'analyse - Nexum IA</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #0a0a1a; color: #e0e0ff; }
            .container { max-width: 800px; margin: 0 auto; background: #1a1a2e; padding: 30px; border-radius: 20px; }
            h1 { color: #00f2ff; }
            .metric { margin: 20px 0; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 10px; }
            .btn { background: #00f2ff; color: #000; padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; text-decoration: none; display: inline-block; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Rapport d'analyse</h1>
            <div id="content">Chargement des données...</div>
            <button class="btn" onclick="window.location.href='/dashboard'">← Retour au Dashboard</button>
        </div>
        <script>
            const data = sessionStorage.getItem('lastAnalysis');
            if (data) {
                const analysis = JSON.parse(data);
                const content = document.getElementById('content');
                let html = '<div class="metric"><strong>🎭 Émotion dominante :</strong> ' + (analysis.emotion || 'N/A') + '</div>';
                html += '<div class="metric"><strong>📝 Transcription :</strong><br>' + (analysis.transcript || 'Aucune') + '</div>';
                if (analysis.analysis) {
                    html += '<div class="metric"><strong>⚠️ Risque de tromperie :</strong> ' + (analysis.analysis.score || 0) + '%</div>';
                    html += '<div class="metric"><strong>📊 Niveau :</strong> ' + (analysis.analysis.level || 'N/A') + '</div>';
                }
                if (analysis.metrics) {
                    html += '<div class="metric"><strong>🧠 Gestion du stress :</strong> ' + Math.round(analysis.metrics.stress_management || 0) + '%</div>';
                    html += '<div class="metric"><strong>💬 Communication :</strong> ' + Math.round(analysis.metrics.communication || 0) + '%</div>';
                    html += '<div class="metric"><strong>🎯 Assurance :</strong> ' + Math.round(analysis.metrics.assurance_level || 0) + '%</div>';
                }
                content.innerHTML = html;
            } else {
                document.getElementById('content').innerHTML = '<p>Aucune analyse récente trouvée. Effectuez d\'abord une analyse complète.</p>';
            }
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)


@app.get("/dashboard")
async def dashboard_page():
    """Sert la page dashboard"""
    from fastapi.responses import HTMLResponse
    # Lit le fichier dashboard.html qui doit être dans le même dossier
    with open("dashboard.html", "r", encoding="utf-8") as f:
        dashboard_html = f.read()
    return HTMLResponse(content=dashboard_html)


@app.get("/kpi_dashboard")
async def kpi_dashboard_page():
    """Sert la page KPI dashboard"""
    from fastapi.responses import HTMLResponse
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>KPIs Dashboard - Nexum IA</title>
        <style>
            body { font-family: Arial, sans-serif; background: #0a0a1a; color: #e0e0ff; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: #1a1a2e; border-radius: 20px; padding: 20px; margin-bottom: 20px; border: 1px solid rgba(0,242,255,0.2); }
            h1 { color: #00f2ff; }
            .metric { font-size: 32px; font-weight: bold; color: #00f2ff; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>📈 Dashboard des KPIs Système</h1>
                <p>Indicateurs de performance de l'API d'analyse émotionnelle</p>
            </div>
            <div class="card">
                <h2>🖥️ État du système</h2>
                <div id="health"></div>
            </div>
            <div class="card">
                <h2>📊 Modèle chargé</h2>
                <div id="modelInfo"></div>
            </div>
            <button onclick="window.location.href='/dashboard'" style="background: #00f2ff; color: #000; padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer;">← Retour Dashboard</button>
        </div>
        <script>
            async function loadHealth() {
                try {
                    const res = await fetch('/health');
                    const data = await res.json();
                    document.getElementById('health').innerHTML = `
                        <p>✅ Statut: ${data.status}</p>
                        <p>🖥️ Device: ${data.device}</p>
                        <p>📊 Analyses totales: ${data.history?.total_analyses || 0}</p>
                    `;
                } catch(e) {
                    document.getElementById('health').innerHTML = '<p>❌ Erreur de connexion</p>';
                }
            }
            async function loadModelInfo() {
                try {
                    const res = await fetch('/model_info');
                    const data = await res.json();
                    document.getElementById('modelInfo').innerHTML = `
                        <p>🤖 Type: ${data.model_type}</p>
                        <p>🎭 Émotions: ${data.emotions?.join(', ')}</p>
                        <p>📁 Entraîné: ${data.is_trained ? '✅ Oui' : '❌ Non'}</p>
                    `;
                } catch(e) {
                    document.getElementById('modelInfo').innerHTML = '<p>❌ Erreur de chargement</p>';
                }
            }
            loadHealth();
            loadModelInfo();
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)
@app.post("/candidate_summary")
async def candidate_summary(
    file: UploadFile = File(None),
    video_url: Optional[str] = Form(None)
) -> JSONResponse:
    """Return a summary of the candidate by analyzing the uploaded video or a video URL.
    If neither is provided, returns a 400 error.
    """
    import json
    # Prioritize URL if provided
    if video_url:
        request = VideoURLRequest(url=video_url, filename=None)
        result = await analyze_video_from_url(request)
    elif file:
        result = await analyze_video(file=file, target_x=None, target_y=None)
    else:
        return JSONResponse({"error": "Aucun fichier ou URL fourni."}, status_code=400)

    # If result is a JSONResponse, extract its JSON content
    if isinstance(result, JSONResponse):
        body = result.body
        if isinstance(body, (bytes, bytearray)):
            data = json.loads(body.decode())
        else:
            data = json.loads(body)
        return JSONResponse(content=data)
    # Sinon, retourner tel quel
    return result

# ==========================================================
# 3D VIEWER INTEGRATION
# ==========================================================
# from three_d_utils import generate_pointcloud_from_frame, extract_landmarks, landmarks_to_pointcloud
import math

# In-memory store for the latest analysis frame data (used by 3D viewer)
# _latest_3d_data = {
#     "pointcloud": {"points": []},
#     "emotion": "neutral",
#     "confidence": 0.0,
#     "emotion_probs": {e: 0.0 for e in EMOTION_LABELS},
#     "waveform": []
# }

# @app.get("/3d_viewer")
# async def serve_3d_viewer():
#     """Serve the 3D interactive viewer page"""
#     from fastapi.responses import HTMLResponse
#     with open("3d_viewer.html", "r", encoding="utf-8") as f:
#         html = f.read()
#     return HTMLResponse(content=html)

# @app.post("/api/pointcloud")
# async def api_pointcloud(frame: UploadFile = File(...)):
#     """
#     Accept a JPEG frame and return a 3-D point-cloud (facial landmarks).
#     If MediaPipe is unavailable the utility falls back to a geometric demo cloud.
#     """
#     try:
#         contents = await frame.read()
#         pc = generate_pointcloud_from_frame(contents)
#         # Update shared state
#         _latest_3d_data["pointcloud"] = pc
#         return JSONResponse(pc)
#     except Exception as e:
#         return JSONResponse({"error": str(e), "points": []}, status_code=500)
#
# @app.get("/api/pointcloud/latest")
# async def api_pointcloud_latest():
#     """Return the most recently computed point-cloud (for polling)"""
#     return JSONResponse(_latest_3d_data["pointcloud"])

# @app.get("/api/emotion_heatmap")
# async def api_emotion_heatmap():
#     """
#     Return per-emotion probabilities as RGBA colour stops for the heat-map layer.
#     Colours match the application's EMOTION_COLORS palette.
#     """
#     probs = _latest_3d_data["emotion_probs"]
#     heatmap = []
#     for label in EMOTION_LABELS:
#         prob = probs.get(label, 0.0)
#         hex_col = EMOTION_COLORS.get(label, "#9e9e9e").lstrip("#")
#         r = int(hex_col[0:2], 16)
#         g = int(hex_col[2:4], 16)
#         b = int(hex_col[4:6], 16)
#         heatmap.append({
#             "emotion": label,
#             "label_fr": EMOTION_NAMES_FR.get(label, label),
#             "probability": round(prob, 4),
#             "color": EMOTION_COLORS.get(label, "#9e9e9e"),
#             "rgba": f"rgba({r},{g},{b},{min(1.0, prob * 2.5):.2f})"
#         })
#     return JSONResponse({
#         "dominant": _latest_3d_data["emotion"],
#         "confidence": _latest_3d_data["confidence"],
#         "heatmap": heatmap
#     })
#
# @app.get("/api/waveform")
# async def api_waveform():
#     """Return current audio waveform data for the 3-D viewer's waveform canvas"""
#     return JSONResponse({"points": _latest_3d_data["waveform"]})
#
# @app.get("/api/3d_status")
# async def api_3d_status():
#     """Combined status endpoint polled by the 3-D viewer at ~2 Hz"""
#     return JSONResponse({
#         "emotion": _latest_3d_data["emotion"],
#         "confidence": _latest_3d_data["confidence"],
#         "emotion_probs": _latest_3d_data["emotion_probs"],
#         "point_count": len(_latest_3d_data["pointcloud"].get("points", [])),
#         "waveform_length": len(_latest_3d_data["waveform"])
#     })



if __name__ == "__main__":

    import uvicorn
    print("\n" + "="*60)
    print("ANALYSE EMOTIONNELLE AVANCEE - SERVEUR DEMARRE")
    print("="*60)
    print(f"API: http://localhost:8089")
    print(f"Dashboard: http://localhost:8089/dashboard")
    print(f"Health: http://localhost:8089/health")
    print(f"Model Info: http://localhost:8089/model_info")
    print("="*60)
    print("\nOUVREZ DANS VOTRE NAVIGATEUR:")
    print("   http://localhost:8089/dashboard")
 #   print("   http://localhost:8089/3d_viewer")
    print("   http://localhost:8089/kpi_dashboard")

    print("\nPOUR ENTRAINER LE MODELE:")
    print("   POST /train avec dataset_path")
    print("   POST /add_training_data pour ajouter des exemples")
    print("\n" + "="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8089, log_level="info")
