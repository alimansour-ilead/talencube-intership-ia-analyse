# test_large.py
import soundfile as sf
import numpy as np
from transformers import pipeline
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Chargement whisper-large-v3 (peut prendre du temps au premier lancement)...")
t0 = time.time()
large = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                  device=device, chunk_length_s=30,
                  generate_kwargs={"language": "french", "num_beams": 5,
                                    "condition_on_prev_tokens": False})
print(f"Chargé en {time.time()-t0:.1f}s")

audio_data, sr = sf.read("extrait_test.wav")
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)

t0 = time.time()
result = large({"sampling_rate": 16000, "raw": audio_data})
print(f"Transcription ({time.time()-t0:.1f}s pour {len(audio_data)/16000:.0f}s d'audio):")
print(result["text"])