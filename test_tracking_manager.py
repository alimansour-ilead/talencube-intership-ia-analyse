import io
import logging
import numpy as np
import cv2

import tracking_manager
from tracking_manager import (
    TrackingManager, IdentityManager, State, CFG
)


class _FakeDeepSort:
    def __init__(self, *args, **kwargs):
        pass

    def update_tracks(self, *args, **kwargs):
        return []


tracking_manager.DeepSort = _FakeDeepSort


# ═══════════════════════════════════════════════════════════════════
# 1. Régression ABANDONED
# ═══════════════════════════════════════════════════════════════════

def test_abandoned_log_une_seule_fois_et_garde_last_bbox():
    tm = TrackingManager()
    tm.force_track(1, [10.0, 10.0, 50.0, 50.0])
    assert tm.last_bbox == [10.0, 10.0, 50.0, 50.0]

    # Capture les warnings du logger du module
    log_stream = io.StringIO()
    handler    = logging.StreamHandler(log_stream)
    logger     = logging.getLogger("tracking_manager")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    # Simule beaucoup plus de frames perdues que MAX_REID, pour
    # dépasser largement le seuil ABANDONED plusieurs fois de suite.
    total_frames = CFG.MAX_REID + 50
    for _ in range(total_frames):
        tm._on_lost()

    logger.removeHandler(handler)
    output = log_stream.getvalue()

    assert tm.state == State.ABANDONED, (
        f"État attendu ABANDONED, obtenu {tm.state}")

    # ← Le vrai test de régression : un seul "ABANDONED" loggué,
    # pas un par frame au-delà du seuil.
    occurrences = output.count("ABANDONED")
    assert occurrences == 1, (
        f"Le warning ABANDONED devrait apparaître 1 fois, "
        f"trouvé {occurrences} fois — régression du spam infini.")

    # ← Le vrai test de régression : last_bbox ne doit JAMAIS être
    # effacé lors du passage à ABANDONED, sinon toute réidentification
    # ultérieure devient structurellement impossible.
    assert tm.last_bbox is not None, (
        "last_bbox a été effacé lors du passage à ABANDONED — "
        "régression du bug qui bloquait toute récupération future "
        "du candidat.")


def test_transitions_lost_puis_reid_puis_abandoned():
    """Vérifie l'enchaînement complet des états selon lost_count."""
    tm = TrackingManager()
    tm.force_track(1, [0.0, 0.0, 100.0, 100.0])

    for _ in range(CFG.MAX_LOST):
        tm._on_lost()
    assert tm.state == State.LOST, (
        f"Attendu LOST après {CFG.MAX_LOST} frames perdues, "
        f"obtenu {tm.state}")

    for _ in range(CFG.MAX_REID - CFG.MAX_LOST):
        tm._on_lost()
    assert tm.state == State.REIDENTIFICATION, (
        f"Attendu REIDENTIFICATION après {CFG.MAX_REID} frames, "
        f"obtenu {tm.state}")

    tm._on_lost()
    assert tm.state == State.ABANDONED


# ═══════════════════════════════════════════════════════════════════
# 2. Chemin _combined() (fallback non-ArcFace) — jamais exercé en prod
# ═══════════════════════════════════════════════════════════════════

def _make_face_like_image(hue_shift: int = 0, seed: int = 0) -> np.ndarray:
    """
    Génère une image BGR synthétique 112x112 avec une teinte HSV
    contrôlée, pour pouvoir comparer deux "visages" volontairement
    proches ou éloignés en teinte/histogramme.
    """
    rng   = np.random.RandomState(seed)
    base  = rng.randint(80, 180, size=(112, 112), dtype=np.uint8)
    hsv   = np.zeros((112, 112, 3), dtype=np.uint8)
    hsv[:, :, 0] = (hue_shift) % 180
    hsv[:, :, 1] = 120
    hsv[:, :, 2] = base
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_combined_fallback_favorise_image_similaire():
    """
    Sans ArcFace disponible, IdentityManager doit basculer sur
    _combined() (ratio d'aspect + teinte + histogramme) et donner un
    score plus élevé à une image visuellement proche de la référence
    qu'à une image très différente — comportement jamais vérifié
    depuis longtemps car ce chemin n'est jamais emprunté en pratique
    (ArcFace se charge toujours dans les environnements observés).
    """
    identity = IdentityManager(shared_arcface=None)
    # Force explicitement le mode fallback, indépendamment de la
    # disponibilité réelle d'insightface dans l'environnement de test.
    identity._use_arc = False
    identity.TOLERANCE = CFG.TOLERANCE_VIT

    ref_img = _make_face_like_image(hue_shift=10, seed=1)
    identity._visual_ref(ref_img)

    ref_embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    identity._ref = ref_embedding

    similar_img   = _make_face_like_image(hue_shift=12, seed=2)
    different_img = _make_face_like_image(hue_shift=90, seed=3)

    # Même embedding brut pour isoler l'effet du scoring visuel
    # (ratio/teinte/histogramme) porté par _combined().
    same_embedding = ref_embedding.copy()

    score_similar   = identity._combined(
        similar_img, same_embedding, sim_emb=1.0)
    score_different = identity._combined(
        different_img, same_embedding, sim_emb=1.0)

    assert score_similar >= score_different, (
        f"_combined() devrait favoriser l'image visuellement proche "
        f"(score={score_similar:.3f}) par rapport à l'image très "
        f"différente (score={score_different:.3f}) — le chemin de "
        f"repli non-ArcFace ne se comporte pas comme attendu.")


def test_combined_utilise_bien_le_seuil_pour_verify():
    """
    Vérifie que verify() emprunte bien le chemin _combined() quand
    use_arcface est False, et respecte le seuil donné.
    """
    identity = IdentityManager(shared_arcface=None)
    identity._use_arc  = False
    identity.TOLERANCE = 0.99  # seuil volontairement très strict

    ref_img = _make_face_like_image(hue_shift=10, seed=1)
    identity._visual_ref(ref_img)
    identity._ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Embedding très différent de la référence → sim faible →
    # devrait échouer avec un seuil aussi strict.
    far_embedding = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    ok, score = identity.verify(
        face_img=_make_face_like_image(hue_shift=90, seed=4),
        embedding=far_embedding)

    assert ok is False, (
        f"verify() aurait dû échouer avec un seuil strict de 0.99 "
        f"et un embedding éloigné, mais a retourné ok=True "
        f"(score={score:.3f}).")


if __name__ == "__main__":
    tests = [
        test_abandoned_log_une_seule_fois_et_garde_last_bbox,
        test_transitions_lost_puis_reid_puis_abandoned,
        test_combined_fallback_favorise_image_similaire,
        test_combined_utilise_bien_le_seuil_pour_verify,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {t.__name__} — {e}")
            failed += 1
    print(f"\n{passed} réussi(s), {failed} échoué(s)")