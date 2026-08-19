# au_analyzer.py — VERSION 3 (correction progressive + signal sourire)
# ═══════════════════════════════════════════════════════════════════
# PATCH v7.3 — Remplace le test tout-ou-rien de la v2 par une
# correction PROGRESSIVE, et ajoute un signal de sourire (AU12) pour
# mieux distinguer "parole neutre" de "tension réelle".
#
# PROBLÈME CONCRET DE LA v2 : is_speech_artifact() était un test
# binaire — à mouth_open=0.139 aucune correction, à 0.141 correction
# complète. Sur des frames très proches (même personne, même instant
# presque), ça expliquait le comportement inconsistant observé en
# test réel (parfois "Tension" corrigée, parfois pas, sur des frames
# quasi identiques).
#
# CE QUI CHANGE :
#   1. La correction est maintenant proportionnelle (fonction sigmoïde
#      douce) à la fois à l'ouverture de bouche ET à l'ABSENCE de
#      tension sourcils — plus de bascule brutale à la frontière du
#      seuil.
#   2. Un signal supplémentaire — l'étirement des coins de bouche
#      (AU12, proxy du sourire) — renforce la correction quand présent
#      (un sourire pendant l'ouverture de bouche est un indice fort
#      que ce n'est pas de la colère), sans être requis pour que la
#      correction s'applique (on ne veut pas rater le cas "parle sans
#      sourire, visage neutre par ailleurs").
#   3. Les seuils restent ajustables en haut de fichier, mais leur
#      rôle a changé : ce sont maintenant des points d'inflexion de la
#      sigmoïde, pas des couperets.
#
# INTERFACE PUBLIQUE INCHANGÉE — aucune modification nécessaire dans
# main.py.
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import cv2
from typing import Optional, Dict

IDX_SAD, IDX_DISGUST, IDX_ANGRY, IDX_NEUTRAL, IDX_FEAR, IDX_SURPRISE, IDX_HAPPY = range(7)

# ── Points d'inflexion de la correction progressive ─────────────────
# ⚠️ Toujours des points de départ raisonnables, pas calibrés sur un
# jeu de données annoté — mais la sigmoïde les rend moins sensibles à
# un mauvais réglage exact qu'un seuil dur (v2).
MOUTH_OPEN_MIDPOINT    = 0.14   # point d'inflexion — mouth_open à ce
                                 # niveau ⇒ ~50% de correction appliquée
MOUTH_OPEN_STEEPNESS   = 25.0   # netteté de la transition (plus haut
                                 # = transition plus brusque autour du
                                 # point d'inflexion ; plus bas = plus
                                 # progressif)
BROW_TENSION_MIDPOINT  = 0.45
BROW_TENSION_STEEPNESS = 12.0

SMILE_BONUS_MAX        = 0.25   # bonus de correction max si sourire
                                 # net détecté simultanément (0-1
                                 # ajouté à la force de correction)

TRIGGER_PROB_THRESHOLD = 0.15   # ne vérifier que si angry+disgust dépasse ça
MAX_CORRECTION_STRENGTH = 0.85  # plafond — on ne met jamais à 0 la
                                 # probabilité angry/disgust : même en
                                 # cas de forte suspicion d'articulation,
                                 # on garde une trace, au cas où le
                                 # signal AU se trompe.
CROP_PADDING_RATIO      = 0.35

_face_mesh = None
MODEL_STATUS_AU = False


def load_au_detector() -> bool:
    global _face_mesh, MODEL_STATUS_AU
    try:
        import mediapipe as mp
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.4,
        )
        MODEL_STATUS_AU = True
        print("[AU] ✅ MediaPipe FaceMesh chargé "
              "(correction progressive articulation/émotion active)")
    except Exception as e:
        MODEL_STATUS_AU = False
        _face_mesh = None
        print(f"[AU] ⚠️ Indisponible ({e}) — correction AU désactivée, "
              f"le biais 'parler = colère' ne sera pas corrigé.")
    return MODEL_STATUS_AU


def should_check_speech_artifact(probs: np.ndarray) -> bool:
    if probs is None or len(probs) < 7:
        return False
    return float(probs[IDX_ANGRY] + probs[IDX_DISGUST]) >= TRIGGER_PROB_THRESHOLD


def _pad_crop(face_bgr: np.ndarray, ratio: float = CROP_PADDING_RATIO) -> np.ndarray:
    h, w = face_bgr.shape[:2]
    pad_h, pad_w = int(h * ratio), int(w * ratio)
    return cv2.copyMakeBorder(
        face_bgr, pad_h, pad_h, pad_w, pad_w,
        borderType=cv2.BORDER_REPLICATE
    )


def _extract_landmarks(face_bgr: np.ndarray):
    if _face_mesh is None or face_bgr is None or face_bgr.size == 0:
        return None
    try:
        padded = _pad_crop(face_bgr)
        rgb    = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        result = _face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        return result.multi_face_landmarks[0].landmark
    except Exception as e:
        print(f"[AU] Erreur extraction landmarks: {e}")
        return None


def _mouth_open_score(lm) -> float:
    try:
        mouth_gap   = abs(lm[13].y - lm[14].y)
        interocular = max(1e-4, abs(lm[33].x - lm[263].x))
        return float(mouth_gap / interocular)
    except (IndexError, AttributeError):
        return 0.0


def _brow_tension_score(lm) -> float:
    try:
        brow_dist   = float(abs(lm[65].x - lm[295].x))
        brow_height = float(abs(lm[70].y - lm[159].y))
        return float(
            max(0, 1 - brow_dist * 8) * 0.5 +
            max(0, 1 - brow_height * 12) * 0.5
        )
    except (IndexError, AttributeError):
        return 0.0


def _smile_score(lm) -> float:
    """
    ← NOUVEAU v7.3 : approxime AU12 (releveur de coin de lèvre, le
    muscle du sourire) en comparant la largeur de la bouche (coins,
    landmarks 61/291) à la distance interoculaire — une bouche étirée
    latéralement (relativement à l'écartement des yeux) est un signe
    de sourire, quasi toujours incompatible avec de la colère réelle.
    Retourne un score 0-1 (0 = pas de sourire détecté, 1 = sourire net).
    """
    try:
        mouth_width = abs(lm[61].x - lm[291].x)
        interocular = max(1e-4, abs(lm[33].x - lm[263].x))
        ratio = mouth_width / interocular
        # Ratio neutre typique ~0.9-1.0 ; un vrai sourire dépasse
        # souvent 1.15-1.3. Normalisation douce entre ces bornes.
        return float(np.clip((ratio - 1.00) / 0.30, 0.0, 1.0))
    except (IndexError, AttributeError):
        return 0.0


def _sigmoid(x: float, midpoint: float, steepness: float) -> float:
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def extract_aus(face_bgr: np.ndarray) -> Optional[Dict[str, float]]:
    lm = _extract_landmarks(face_bgr)
    if lm is None:
        return None
    return {
        "mouth_open":   _mouth_open_score(lm),
        "brow_tension": _brow_tension_score(lm),
        "smile":        _smile_score(lm),
    }


def compute_correction_strength(aus: Dict[str, float]) -> float:
    """
    ← NOUVEAU v7.3 : remplace is_speech_artifact() (test binaire) par
    un calcul de force de correction continue entre 0 et
    MAX_CORRECTION_STRENGTH.

    Composition :
      - speech_signal   : sigmoïde sur l'ouverture de bouche — monte
                          doucement autour de MOUTH_OPEN_MIDPOINT.
      - calm_brow_signal: sigmoïde INVERSE sur la tension sourcils —
                          descend doucement autour de
                          BROW_TENSION_MIDPOINT (on veut une correction
                          FORTE quand les sourcils sont calmes, FAIBLE
                          quand ils sont tendus, même si la bouche est
                          ouverte — ça laisse passer une vraie colère
                          exprimée en parlant).
      - smile_bonus     : ajoute jusqu'à SMILE_BONUS_MAX si un sourire
                          net accompagne l'ouverture de bouche.
    """
    mouth_open   = aus.get("mouth_open", 0.0)
    brow_tension = aus.get("brow_tension", 0.0)
    smile        = aus.get("smile", 0.0)

    speech_signal    = _sigmoid(mouth_open, MOUTH_OPEN_MIDPOINT, MOUTH_OPEN_STEEPNESS)
    calm_brow_signal = 1.0 - _sigmoid(brow_tension, BROW_TENSION_MIDPOINT, BROW_TENSION_STEEPNESS)

    base_strength = speech_signal * calm_brow_signal
    smile_bonus   = smile * SMILE_BONUS_MAX * speech_signal

    strength = min(MAX_CORRECTION_STRENGTH, base_strength + smile_bonus)
    return float(strength)


def correct_emotion_probs(face_bgr: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    Point d'entrée principal — signature inchangée.
    """
    if not should_check_speech_artifact(probs):
        return probs

    aus = extract_aus(face_bgr)
    if aus is None:
        return probs

    strength = compute_correction_strength(aus)
    if strength < 0.05:
        return probs

    corrected = probs.copy()
    removed = (corrected[IDX_ANGRY] + corrected[IDX_DISGUST]) * strength
    corrected[IDX_ANGRY]   *= (1 - strength)
    corrected[IDX_DISGUST] *= (1 - strength)

    base = corrected[IDX_NEUTRAL] + corrected[IDX_HAPPY] + 1e-9
    corrected[IDX_NEUTRAL] += removed * (corrected[IDX_NEUTRAL] / base)
    corrected[IDX_HAPPY]   += removed * (corrected[IDX_HAPPY] / base)

    corrected /= corrected.sum()

    print(f"[AU] 🗣️ Correction progressive (force={strength:.2f}) — "
          f"bouche={aus['mouth_open']:.2f} sourcils={aus['brow_tension']:.2f} "
          f"sourire={aus['smile']:.2f} — "
          f"angry {probs[IDX_ANGRY]:.2f}→{corrected[IDX_ANGRY]:.2f}, "
          f"disgust {probs[IDX_DISGUST]:.2f}→{corrected[IDX_DISGUST]:.2f}")

    return corrected


# ── Conservé pour compatibilité (si du code externe l'appelait) ─────
def is_speech_artifact(aus: Dict[str, float]) -> bool:
    """
    ← Conservé pour compatibilité ascendante uniquement — la logique
    réelle utilise maintenant compute_correction_strength() en continu.
    """
    return compute_correction_strength(aus) >= 0.5