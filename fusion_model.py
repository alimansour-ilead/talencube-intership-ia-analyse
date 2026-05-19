# models/fusion_model.py - Fusion multimodale par attention
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """Attention croisée entre les modalités visuelle et audio"""
    
    def __init__(self, visual_dim=768, audio_dim=768, hidden_dim=512):
        super().__init__()
        
        # Projections pour l'attention
        self.query_proj = nn.Linear(visual_dim, hidden_dim)
        self.key_proj = nn.Linear(audio_dim, hidden_dim)
        self.value_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Projections inverses
        self.visual_out = nn.Linear(hidden_dim, visual_dim)
        self.audio_out = nn.Linear(hidden_dim, audio_dim)
        
        self.scale = hidden_dim ** -0.5
    
    def forward(self, visual_features, audio_features):
        """Attention croisée visuel ↔ audio"""
        
        # Visual attend la représentation Audio
        V_query = self.query_proj(visual_features)
        A_key = self.key_proj(audio_features)
        A_value = self.value_proj(audio_features)
        
        attention_weights = torch.matmul(V_query, A_key.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        attended_audio = torch.matmul(attention_weights, A_value)
        visual_enhanced = self.visual_out(attended_audio)
        
        # Audio attend la représentation Visuelle
        A_query = self.query_proj(audio_features)
        V_key = self.key_proj(visual_features)
        V_value = self.value_proj(visual_features)
        
        attention_weights = torch.matmul(A_query, V_key.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        attended_visual = torch.matmul(attention_weights, V_value)
        audio_enhanced = self.audio_out(attended_visual)
        
        return visual_enhanced + visual_features, audio_enhanced + audio_features

class MultimodalFusionWithAttention(nn.Module):
    """Modèle complet avec fusion par attention"""
    
    def __init__(self, visual_model, audio_model, num_classes=7, fusion_dim=512):
        super().__init__()
        
        self.visual_model = visual_model
        self.audio_model = audio_model
        
        # Attention croisée
        visual_dim = 768  # Dimension ViT
        audio_dim = 768   # Dimension HuBERT
        self.cross_attention = CrossModalAttention(visual_dim, audio_dim)
        
        # Projection fusionnée
        self.fusion_projection = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
        )
        
        # Classifieur final
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)
        
        # Pondération adaptative des modalités
        self.modal_weights = nn.Parameter(torch.ones(2))
        
        self.num_classes = num_classes
    
    def forward(self, visual_input, audio_input, audio_mask=None):
        """Forward pass multimodal"""
        
        # Extraction des features
        visual_features = self.visual_model.extract_features(visual_input)
        audio_features = self.audio_model.extract_features(audio_input, audio_mask)
        
        # Attention croisée
        visual_enhanced, audio_enhanced = self.cross_attention(visual_features, audio_features)
        
        # Pondération adaptative
        weights = F.softmax(self.modal_weights, dim=0)
        visual_weighted = visual_enhanced * weights[0]
        audio_weighted = audio_enhanced * weights[1]
        
        # Fusion concaténée
        fused_features = torch.cat([visual_weighted, audio_weighted], dim=-1)
        
        # Projection et classification
        fused_projected = self.fusion_projection(fused_features)
        logits = self.classifier(fused_projected)
        
        return logits, {
            'visual_features': visual_features,
            'audio_features': audio_features,
            'visual_enhanced': visual_enhanced,
            'audio_enhanced': audio_enhanced,
            'fusion_weights': weights.detach().cpu().numpy()
        }
    
    def predict_with_confidence(self, visual_input, audio_input, audio_mask=None):
        """Prédiction avec score de confiance"""
        logits, features = self.forward(visual_input, audio_input, audio_mask)
        probs = F.softmax(logits, dim=-1)
        confidence, pred = torch.max(probs, dim=-1)
        return pred, confidence, probs, features
        
    def predict_with_confidence_onnx(self, visual_features_np, audio_features_np):
        """Version ONNX optimisée : utilise les features extraites par ONNX pour la fusion"""
        # Convertir en tenseurs PyTorch
        device = self.modal_weights.device
        vis_feat = torch.tensor(visual_features_np).to(device)
        aud_feat = torch.tensor(audio_features_np).to(device)
        
        # Attention croisée
        visual_enhanced, audio_enhanced = self.cross_attention(vis_feat, aud_feat)
        
        # Pondération adaptative
        weights = F.softmax(self.modal_weights, dim=0)
        visual_weighted = visual_enhanced * weights[0]
        audio_weighted = audio_enhanced * weights[1]
        
        # Fusion concaténée
        fused_features = torch.cat([visual_weighted, audio_weighted], dim=-1)
        
        # Projection et classification
        fused_projected = self.fusion_projection(fused_features)
        logits = self.classifier(fused_projected)
        
        probs = F.softmax(logits, dim=-1)
        confidence, pred = torch.max(probs, dim=-1)
        
        return pred, confidence, probs, {
            'visual_features': vis_feat,
            'audio_features': aud_feat,
            'visual_enhanced': visual_enhanced,
            'audio_enhanced': audio_enhanced,
            'fusion_weights': weights
        }