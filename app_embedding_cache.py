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


# ═══════════════════════════════════════════════════════════════════
# ÉTAT DE SUIVI TEMPS RÉEL — persistance entre reconnexions
# ═══════════════════════════════════════════════════════════════════
# ← AJOUT : distinct du cache d'embedding ci-dessus (qui est à usage
# UNIQUE, pour le transfert extraction→première connexion). Ici, on
# persiste l'état COMPLET du suivi d'un candidat (référence ArcFace +
# zone de position) pendant toute la durée d'une session temps réel,
# mis à jour périodiquement, et LU (jamais supprimé automatiquement)
# à chaque connexion.
#
# Pourquoi c'est nécessaire : avec plusieurs replicas Railway, une
# coupure réseau ou un redémarrage peut faire atterrir la reconnexion
# automatique de Java sur une réplique DIFFERENTE de celle qui avait
# mémorisé le candidat. Sans cet état partagé, la nouvelle réplique
# ne connaît rien du candidat — elle le marque "absent" ou verrouille
# la mauvaise personne, le temps de refaire toute la mémorisation
# initiale (5 frames, plusieurs secondes). Avec cet état dans Redis,
# la nouvelle réplique restaure l'identité déjà connue en quelques
# millisecondes, rendant la reconnexion invisible pour l'utilisateur.
_TRACKING_STATE_TTL_SECONDS = 30 * 60  # 30 min — large marge sur la durée d'un entretien
_TRACKING_KEY_PREFIX = "tracking_state:"


def store_tracking_state(session_key: str, ref_embedding: np.ndarray,
                         zone: Optional[dict] = None) -> None:
    """
    Sauvegarde/rafraîchit l'état de suivi d'un candidat (référence
    ArcFace + zone de position). Appelée après la mémorisation
    initiale, puis périodiquement pendant les mises à jour, pour que
    l'état reste disponible en cas de reconnexion sur une autre
    réplique.
    """
    if _redis_client is None:
        return
    try:
        data = pickle.dumps({"ref": ref_embedding, "zone": zone})
        _redis_client.setex(
            _TRACKING_KEY_PREFIX + session_key,
            _TRACKING_STATE_TTL_SECONDS, data)
    except Exception as e:
        print(f"[TrackingState] Erreur store: {e}")


def get_tracking_state(session_key: str) -> Optional[dict]:
    """
    Récupère l'état de suivi déjà mémorisé pour cette clé, SANS le
    supprimer (contrairement à get_embedding) — plusieurs reconnexions
    successives doivent pouvoir le relire tant que la session dure.
    Retourne None si aucun état n'existe encore (première connexion
    pour ce candidat).
    """
    if _redis_client is None:
        return None
    try:
        data = _redis_client.get(_TRACKING_KEY_PREFIX + session_key)
        if data is None:
            return None
        return pickle.loads(data)
    except Exception as e:
        print(f"[TrackingState] Erreur get: {e}")
        return None


def clear_tracking_state(session_key: str) -> None:
    """
    Supprime explicitement l'état de suivi — à appeler sur un reset
    (changement de candidat) ou une vraie fin de session, pour ne pas
    laisser un état périmé être restauré par erreur sur une future
    connexion sans rapport.
    """
    if _redis_client is None:
        return
    try:
        _redis_client.delete(_TRACKING_KEY_PREFIX + session_key)
    except Exception as e:
        print(f"[TrackingState] Erreur clear: {e}")


# ═══════════════════════════════════════════════════════════════════
# JOBS ASYNCHRONES — analyse vidéo / extraction de candidat
# ═══════════════════════════════════════════════════════════════════
# ← AJOUT : remplace le dictionnaire Python en mémoire locale
# (_async_video_jobs) utilisé par /analyze_video et
# /extract_candidates_preview pour stocker l'état "PROCESSING" puis
# le résultat final d'un job.
#
# Pourquoi c'était nécessaire : avec plusieurs répliques Railway, le
# job est créé sur UNE réplique (celle qui a reçu la requête POST
# initiale), mais la requête de polling suivante (GET .../result/
# {job_id}, envoyée par Java toutes les 2-3s) peut atterrir sur une
# AUTRE réplique — qui n'a jamais entendu parler de ce job_id dans
# son propre dictionnaire local, renvoyant un 404 alors que le job
# est peut-être toujours en cours (ou même déjà terminé) sur la
# réplique d'origine. Java traitait ce 404 comme une réponse
# définitive et arrêtait le polling, stockant {"status":"NOT_FOUND"}
# comme si c'était le résultat réel de l'analyse.
#
# Avec Redis, n'importe quelle réplique peut créer, mettre à jour et
# lire l'état d'un job — le 404 ne se produit plus jamais tant que le
# job existe réellement.
_JOB_TTL_SECONDS = 30 * 60  # 30 min — large marge sur les analyses les plus longues
_JOB_KEY_PREFIX  = "async_job:"


def create_job(job_id: str) -> None:
    """Enregistre un nouveau job comme 'en cours de traitement'."""
    if _redis_client is None:
        return
    try:
        _redis_client.setex(_JOB_KEY_PREFIX + job_id, _JOB_TTL_SECONDS,
                            pickle.dumps({"status": "PROCESSING"}))
    except Exception as e:
        print(f"[AsyncJob] Erreur create: {e}")


def complete_job(job_id: str, payload: dict, status_code: int) -> None:
    """Enregistre le résultat final d'un job (succès ou erreur)."""
    if _redis_client is None:
        return
    try:
        _redis_client.setex(_JOB_KEY_PREFIX + job_id, _JOB_TTL_SECONDS,
                            pickle.dumps({
                                "status": "DONE",
                                "payload": payload,
                                "status_code": status_code
                            }))
    except Exception as e:
        print(f"[AsyncJob] Erreur complete: {e}")


def get_job(job_id: str):
    """
    Retourne l'état d'un job :
    - None si le job n'existe pas (jamais créé, ou expiré après 30 min)
    - "PROCESSING" si le job est encore en cours
    - (payload, status_code) si le job est terminé — l'entrée est
      supprimée de Redis après cette lecture (usage unique, comme le
      comportement d'origine du dictionnaire en mémoire).
    """
    if _redis_client is None:
        return None
    try:
        key  = _JOB_KEY_PREFIX + job_id
        data = _redis_client.get(key)
        if data is None:
            return None
        entry = pickle.loads(data)
        if entry["status"] == "PROCESSING":
            return "PROCESSING"
        _redis_client.delete(key)
        return (entry["payload"], entry["status_code"])
    except Exception as e:
        print(f"[AsyncJob] Erreur get: {e}")
        return None