"""
Cache partagé des embeddings ArcFace calculés pendant extract_candidates_preview.
Permet de transférer l'identité du candidat choisi vers ws_analyze_realtime
sans recalculer ArcFace au démarrage.

← Version Redis : remplace l'ancien dict Python en mémoire locale, qui
ne fonctionnait correctement que sur un seul conteneur. Avec plusieurs
replicas Railway (scaling horizontal), deux requêtes d'un même
utilisateur (extract_candidates_preview puis analyze_video) peuvent
atterrir sur deux conteneurs différents — un dict local ne serait
alors jamais partagé entre eux, causant des "embedding introuvable"
aléatoires. Redis est accessible depuis tous les replicas, réglant
ce problème de fond.
"""
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import redis

# ← REDIS_URL doit être fournie par Railway (variable liée au service
# Redis du même projet). En son absence, on lève une erreur explicite
# au démarrage plutôt qu'un échec silencieux plus tard.
_REDIS_URL = os.environ.get("REDIS_URL")

_redis_client: Optional[redis.Redis] = None
if _REDIS_URL:
    try:
        _redis_client = redis.from_url(_REDIS_URL, socket_connect_timeout=5)
        _redis_client.ping()
        print("[EmbeddingCache] ✅ Connecté à Redis")
    except Exception as e:
        print(f"[EmbeddingCache] ❌ Connexion Redis échouée: {e} "
              f"— le cache d'embeddings ne fonctionnera pas correctement "
              f"avec plusieurs replicas")
        _redis_client = None
else:
    print("[EmbeddingCache] ⚠️ REDIS_URL absente — "
          "cache d'embeddings désactivé (fonctionnera uniquement avec "
          "un seul replica, sans garantie)")

# Durée de vie d'un embedding en cache — le temps que l'utilisateur
# sélectionne un candidat côté frontend. 10 minutes est large.
_TTL_SECONDS = 600
_KEY_PREFIX = "emb_cache:"


def store_embedding(key: str, embedding: np.ndarray, face_img=None) -> None:
    """Stocker un embedding avec sa clé, expirant après _TTL_SECONDS."""
    if _redis_client is None:
        return
    try:
        data = pickle.dumps((embedding, face_img))
        _redis_client.setex(_KEY_PREFIX + key, _TTL_SECONDS, data)
    except Exception as e:
        print(f"[EmbeddingCache] Erreur store_embedding: {e}")


def get_embedding(key: str) -> Tuple[Optional[np.ndarray], Optional[object]]:
    """Récupérer et supprimer un embedding (usage unique)."""
    if _redis_client is None:
        return None, None
    try:
        redis_key = _KEY_PREFIX + key
        data = _redis_client.get(redis_key)
        if data is None:
            return None, None
        _redis_client.delete(redis_key)
        embedding, face_img = pickle.loads(data)
        return embedding, face_img
    except Exception as e:
        print(f"[EmbeddingCache] Erreur get_embedding: {e}")
        return None, None


def has_embedding(key: str) -> bool:
    """Vérifier si un embedding existe dans le cache."""
    if _redis_client is None:
        return False
    try:
        return _redis_client.exists(_KEY_PREFIX + key) > 0
    except Exception as e:
        print(f"[EmbeddingCache] Erreur has_embedding: {e}")
        return False


def clear_old_entries(max_size: int = 50) -> None:
    """
    Conservé pour compatibilité avec le code appelant existant
    (_extract_candidates_preview_sync l'appelle avant de stocker de
    nouveaux embeddings). Ne fait plus rien : Redis gère lui-même
    l'expiration via le TTL fixé dans store_embedding — plus besoin
    de purge manuelle par taille de dictionnaire.
    """
    pass