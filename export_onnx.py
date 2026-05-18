import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from main import EnhancedEmotionModel, base_model, SpeechEmotionHuBERT, Config, device

# Définition des wrappers ONNX
class ViTONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, pixel_values):
        logits = self.model(pixel_values)
        features = self.model.extract_features(pixel_values)
        return logits, features

class HuBERTONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_values):
        logits = self.model(input_values)
        features = self.model.extract_features(input_values)
        return logits, features

def main():
    print("=== EXPORT DES MODÈLES VERS ONNX ===")
    
    # 1. Chargement et Export du modèle ViT Visuel
    print("\n[1/2] Préparation du modèle ViT...")
    vit_py = EnhancedEmotionModel(base_model).to(device)
    if os.path.exists(Config.MODEL_PATH):
        vit_py.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print("Poids ViT entraînés chargés !")
    else:
        print("Poids ViT de base utilisés.")
    
    vit_py.eval()
    vit_wrapper = ViTONNXWrapper(vit_py).eval()
    
    dummy_pixel_values = torch.randn(1, 3, 224, 224, device=device)
    vit_onnx_path = "models/vit_emotion.onnx"
    
    print("Exportation de ViT vers ONNX...")
    torch.onnx.export(
        vit_wrapper,
        dummy_pixel_values,
        vit_onnx_path,
        input_names=["pixel_values"],
        output_names=["logits", "features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "features": {0: "batch_size"}
        },
        opset_version=14
    )
    print(f"Modèle ViT exporté avec succès dans: {vit_onnx_path}")
    
    # 2. Chargement et Export du modèle HuBERT Audio
    print("\n[2/2] Préparation du modèle HuBERT...")
    hubert_py = SpeechEmotionHuBERT(num_classes=7).to(device)
    hubert_py.eval()
    hubert_wrapper = HuBERTONNXWrapper(hubert_py).eval()
    
    # HuBERT prend typiquement du son à 16kHz, dummy de 2 secondes (32000 échantillons)
    dummy_input_values = torch.randn(1, 32000, device=device)
    hubert_onnx_path = "models/hubert_audio.onnx"
    
    print("Exportation de HuBERT vers ONNX...")
    torch.onnx.export(
        hubert_wrapper,
        dummy_input_values,
        hubert_onnx_path,
        input_names=["input_values"],
        output_names=["logits", "features"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
            "features": {0: "batch_size"}
        },
        opset_version=14
    )
    print(f"Modèle HuBERT exporté avec succès dans: {hubert_onnx_path}")
    print("\n=== EXPORT ONNX TERMINÉ AVEC SUCCÈS ===")

if __name__ == "__main__":
    main()
