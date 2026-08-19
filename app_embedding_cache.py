"""
Cache partagé des embeddings ArcFace calculés pendant extract_candidates_preview.
Permet de transférer l'identité du candidat choisi vers ws_analyze_realtime
sans recalculer ArcFace au démarrage.
"""
import numpy as np
from typing import Optional, Tuple

_cache: dict = {}  # key → (embedding, face_img)


def store_embedding(key: str, embedding: np.ndarray,
                    face_img=None) -> None:
    """Stocker un embedding avec sa clé."""
    _cache[key] = (embedding, face_img)


def get_embedding(key: str) -> Tuple[Optional[np.ndarray], Optional[object]]:
    """Récupérer et supprimer un embedding (usage unique)."""
    if key in _cache:
        emb, face = _cache.pop(key)
        return emb, face
    return None, None


def has_embedding(key: str) -> bool:
    """Vérifier si un embedding existe dans le cache."""
    return key in _cache


def clear_old_entries(max_size: int = 50) -> None:
    """Supprimer les anciennes entrées si le cache est trop grand."""
    if len(_cache) > max_size:
        keys = list(_cache.keys())
        for k in keys[:len(keys)//2]:
            _cache.pop(k, None)