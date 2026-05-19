# models/vit_model.py - Vision Transformer pour les émotions faciales
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, ViTForImageClassification

class FacialEmotionViT(nn.Module):
    """Vision Transformer pour la reconnaissance d'émotions faciales"""
    
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        
        self.model_name = "google/vit-base-patch16-224-in21k"
        
        if pretrained:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.backbone = ViTForImageClassification.from_pretrained(self.model_name)
            # Remplacer le classifieur
            hidden_size = self.backbone.config.hidden_size
            self.backbone.classifier = nn.Identity()
        else:
            from transformers import ViTConfig, ViTModel
            config = ViTConfig()
            self.backbone = ViTModel(config)
            hidden_size = config.hidden_size
            self.processor = None
        
        # Couches de fine-tuning
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, pixel_values):
        """Forward pass"""
        outputs = self.backbone(pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, pixel_values):
        """Extrait les embeddings visuels pour la fusion"""
        outputs = self.backbone(pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0, :]
        return features