# train.py - Entraînement multimodal avec datasets Kaggle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from models.vit_model import FacialEmotionViT
from models.hubert_model import SpeechEmotionHuBERT
from models.fusion_model import MultimodalFusionWithAttention

# Configuration
class Config:
    VISUAL_MODEL = "google/vit-base-patch16-224-in21k"
    AUDIO_MODEL = "facebook/hubert-base-ls960"
    NUM_CLASSES = 7
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = "models/multimodal_emotion_model.pth"
    DATASET_PATH = "data/datasets"

class MultimodalEmotionDataset(Dataset):
    """Dataset multimodal pour l'entraînement"""
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    EMOTION_MAP = {e: i for i, e in enumerate(EMOTIONS)}
    
    def __init__(self, data_path, processor_visual, processor_audio, is_train=True):
        self.data = []
        self.processor_visual = processor_visual
        self.processor_audio = processor_audio
        self.data_path = Path(data_path)
        
        self._load_data()
        print(f"📊 Dataset chargé: {len(self.data)} échantillons")
    
    def _load_data(self):
        """Charge les données depuis la structure de dossiers"""
        for emotion in self.EMOTIONS:
            emotion_path = self.data_path / emotion
            if emotion_path.exists():
                for video_file in emotion_path.glob("*.mp4"):
                    self.data.append({
                        'path': str(video_file),
                        'label': self.EMOTION_MAP[emotion],
                        'emotion': emotion
                    })
        
        # Si pas de vidéos, chercher les images
        if len(self.data) == 0:
            for emotion in self.EMOTIONS:
                emotion_path = self.data_path / emotion
                if emotion_path.exists():
                    for img_file in list(emotion_path.glob("*.jpg")) + list(emotion_path.glob("*.png")):
                        self.data.append({
                            'path': str(img_file),
                            'label': self.EMOTION_MAP[emotion],
                            'emotion': emotion
                        })
    
    def _extract_audio(self, video_path):
        """Extrait l'audio d'une vidéo"""
        try:
            audio, sr = librosa.load(video_path, sr=16000, duration=5)
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)))
            else:
                audio = audio[:16000]
            return audio
        except:
            return np.zeros(16000)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Chargement image
        if item['path'].endswith('.mp4'):
            cap = cv2.VideoCapture(item['path'])
            ret, frame = cap.read()
            cap.release()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.imread(item['path'])
            if frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        visual_inputs = self.processor_visual(images=frame, return_tensors="pt")
        
        # Chargement audio
        if item['path'].endswith('.mp4'):
            audio = self._extract_audio(item['path'])
        else:
            audio = np.zeros(16000)
        
        audio_inputs = self.processor_audio(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        return {
            'visual_pixel_values': visual_inputs['pixel_values'].squeeze(),
            'audio_input_values': audio_inputs['input_values'].squeeze(),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Entraîne une époque"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        visual_input = batch['visual_pixel_values'].to(device)
        audio_input = batch['audio_input_values'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(visual_input, audio_input)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(dataloader), accuracy, f1

def validate(model, dataloader, criterion, device):
    """Validation du modèle"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            visual_input = batch['visual_pixel_values'].to(device)
            audio_input = batch['audio_input_values'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(visual_input, audio_input)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(dataloader), accuracy, f1, all_preds, all_labels

def main():
    print("="*60)
    print("🎯 ENTRAÎNEMENT MULTIMODAL - ViT + HuBERT")
    print("="*60)
    
    device = Config.DEVICE
    print(f"📱 Device: {device}")
    
    # Initialisation des modèles
    print("\n📦 Chargement des modèles...")
    visual_model = FacialEmotionViT(num_classes=Config.NUM_CLASSES).to(device)
    audio_model = SpeechEmotionHuBERT(num_classes=Config.NUM_CLASSES).to(device)
    
    # Modèle de fusion
    model = MultimodalFusionWithAttention(
        visual_model=visual_model,
        audio_model=audio_model,
        num_classes=Config.NUM_CLASSES
    ).to(device)
    
    # Optimiseur et scheduler
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    # Fonction de perte pondérée
    class_weights = torch.tensor([1.0, 1.2, 1.3, 0.8, 1.0, 1.1, 1.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Chargement des datasets
    print("\n📊 Chargement des datasets...")
    from transformers import AutoImageProcessor, AutoFeatureExtractor
    
    processor_visual = AutoImageProcessor.from_pretrained(Config.VISUAL_MODEL)
    processor_audio = AutoFeatureExtractor.from_pretrained(Config.AUDIO_MODEL)
    
    # Dataset FER2013
    fer_path = Path(Config.DATASET_PATH) / "fer2013"
    if fer_path.exists():
        train_dataset = MultimodalEmotionDataset(fer_path, processor_visual, processor_audio)
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        print(f"✅ Dataset FER2013 chargé: {len(train_dataset)} samples")
    
    # Dataset EmotionNet
    emotionnet_path = Path(Config.DATASET_PATH) / "emotionnet"
    if emotionnet_path.exists():
        val_dataset = MultimodalEmotionDataset(emotionnet_path, processor_visual, processor_audio)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        print(f"✅ Dataset EmotionNet chargé: {len(val_dataset)} samples")
    
    # Entraînement
    best_accuracy = 0
    history = []
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n📚 Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-"*40)
        
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        if val_loader:
            val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
            print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(), Config.SAVE_PATH)
                print(f"✅ Modèle sauvegardé (Accuracy: {val_acc:.4f})")
        
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss if val_loader else None,
            'val_acc': val_acc if val_loader else None
        })
    
    # Sauvegarde de l'historique
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Entraînement terminé!")
    print(f"📊 Meilleure accuracy: {best_accuracy:.4f}")
    print(f"💾 Modèle sauvegardé: {Config.SAVE_PATH}")

if __name__ == "__main__":
    # D'abord télécharger les datasets
    import subprocess
    subprocess.run(["python", "download_datasets.py"])
    
    # Puis lancer l'entraînement
    main()