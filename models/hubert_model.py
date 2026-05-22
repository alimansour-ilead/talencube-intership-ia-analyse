# models/hubert_model.py - HuBERT pour l'analyse audio
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, HubertModel, HubertConfig

class SpeechEmotionHuBERT(nn.Module):
    """HuBERT pour la reconnaissance d'émotions vocales"""
    
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        
        self.model_name = "superb/hubert-base-superb-er"
        
        if pretrained:
            from transformers import AutoModelForAudioClassification
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.backbone = AutoModelForAudioClassification.from_pretrained(self.model_name)
        else:
            config = HubertConfig()
            self.backbone = HubertModel(config)
            self.feature_extractor = None
        
        self.num_classes = num_classes
    
    def forward(self, input_values, attention_mask=None):
        """Forward pass"""
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask
        )
        return outputs.logits
    
    def extract_features(self, input_values, attention_mask=None):
        """Extrait les embeddings audio pour la fusion"""
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Use mean of the last hidden state for features
        features = outputs.hidden_states[-1].mean(dim=1)
        return features