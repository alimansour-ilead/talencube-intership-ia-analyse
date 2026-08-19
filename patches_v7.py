import numpy as np
import torch
from collections import Counter
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════
# #17 — VAD (Voice Activity Detection)
# ═══════════════════════════════════════════════════════════════════
_vad_model = None
_vad_get_speech = None
VAD_LOADED = False


def load_vad() -> bool:
    """
    Charge silero-vad une seule fois au démarrage du serveur. À appeler
    depuis app.py juste après le chargement de Whisper.

    silero-vad est un modèle CPU très léger (~2MB, quelques ms par
    segment de 2s) — le coût en latence est négligeable comparé au
    bénéfice (élimination des hallucinations Whisper sur silence/bruit).
    """
    global _vad_model, _vad_get_speech, VAD_LOADED
    try:
        _vad_model, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        _vad_get_speech = vad_utils[0]  # get_speech_timestamps
        VAD_LOADED = True
        print("[VAD] ✅ silero-vad chargé — filtrage anti-hallucination actif")
    except Exception as e:
        VAD_LOADED = False
        print(f"[VAD] ⚠️ Indisponible ({e}) — fallback sur seuil RMS simple")
    return VAD_LOADED


def has_speech(audio_np: np.ndarray, sample_rate: int = 16000,
               min_speech_ms: int = 250) -> bool:
    """
    Détermine si un segment audio contient réellement de la voix,
    AVANT de le transcrire.

    Sans ce filtre, Whisper est appelé même sur du silence pur ou du
    bruit de fond, ce qui produit fréquemment des hallucinations en
    boucle (le modèle "invente" une phrase et la répète indéfiniment).
    C'est la cause directe d'un bug observé en test réel : la
    transcription complète d'un entretien de 5 minutes réduite à une
    seule phrase répétée plusieurs dizaines de fois.

    Si silero-vad n'a pas pu être chargé, on se rabat sur un simple
    seuil RMS plutôt que de bloquer toute transcription.
    """
    if audio_np is None or len(audio_np) == 0:
        return False

    if not VAD_LOADED or _vad_model is None:
        rms = float(np.sqrt(np.mean(audio_np.astype(np.float64) ** 2)))
        return rms > 0.006

    try:
        tensor = torch.from_numpy(audio_np.astype(np.float32))
        timestamps = _vad_get_speech(
            tensor, _vad_model, sampling_rate=sample_rate,
            min_speech_duration_ms=min_speech_ms
        )
        return len(timestamps) > 0
    except Exception as e:
        print(f"[VAD] Erreur analyse segment: {e} — fallback RMS")
        rms = float(np.sqrt(np.mean(audio_np.astype(np.float64) ** 2)))
        return rms > 0.006


# ═══════════════════════════════════════════════════════════════════
# #18 — Anti-hallucination de secours
# ═══════════════════════════════════════════════════════════════════
def dedupe_hallucination(text: str, max_repeats: int = 8,
                          proximity_window: int = 30) -> str:
    """
    ← PATCH : seuil relevé (4→8) ET ajout d'un critère de PROXIMITÉ.
    Avant ce patch, toute répétition d'un même n-gramme (même des mots
    de liaison courants comme "donc", "et donc, c'est") ailleurs dans
    un texte de plusieurs milliers de mots déclenchait une troncature
    de TOUT le reste du texte — un faux positif quasi garanti sur un
    entretien long en français parlé spontané, qui répète naturellement
    ces connecteurs des dizaines de fois sur toute sa durée.

    Une vraie hallucination Whisper (boucle sur bruit/silence) répète
    la MÊME séquence de façon RAPPROCHÉE dans le texte (occurrences
    consécutives ou quasi-consécutives) — pas dispersée sur plusieurs
    milliers de mots. On exige donc désormais que les répétitions
    soient concentrées dans une fenêtre de `proximity_window` mots
    pour déclencher la troncature, en plus d'un nombre de répétitions
    plus élevé.
    """
    if not text or not text.strip():
        return text
    words = text.split()
    if len(words) < max_repeats * 3:
        return text

    for ngram_len in (3, 4, 5, 6):
        if len(words) < ngram_len * max_repeats:
            continue
        grams = [tuple(words[i:i + ngram_len])
                 for i in range(len(words) - ngram_len + 1)]
        counts = Counter(grams)
        most_common_gram, count = counts.most_common(1)[0]
        if count < max_repeats:
            continue

        # ← Positions de toutes les occurrences de ce n-gramme
        positions = [i for i, g in enumerate(grams) if g == most_common_gram]

        # ← Cherche une fenêtre glissante contenant au moins
        # `max_repeats` occurrences RAPPROCHÉES (vraie boucle), pas
        # dispersées sur tout le texte (répétition naturelle du style
        # oral).
        for start in range(len(positions) - max_repeats + 1):
            window_positions = positions[start:start + max_repeats]
            if window_positions[-1] - window_positions[0] <= proximity_window:
                first_idx = window_positions[0]
                cut_at = first_idx + ngram_len * 2
                truncated = ' '.join(words[:cut_at])
                print(f"[Whisper] ⚠️ Hallucination détectée "
                      f"(n-gramme répété {max_repeats}x dans une fenêtre "
                      f"de {proximity_window} mots) — texte tronqué "
                      f"({len(words)} → {cut_at} mots)")
                return truncated + " [...]"
        # Répétitions trouvées mais dispersées (style oral normal) —
        # pas une hallucination, on ne tronque pas.
    return text
def safe_transcribe(transcriber_pipeline, audio_np: np.ndarray,
                     sample_rate: int = 16000) -> dict:
    """
    Point d'entrée unique de transcription, à utiliser PARTOUT à la
    place d'un appel direct à `transcriber(...)`. Applique VAD puis
    anti-hallucination, dans cet ordre.

    `transcriber_pipeline` : l'objet pipeline HuggingFace Whisper déjà
    chargé dans app.py (variable globale `transcriber`).

    Retourne un dict au même format que la pipeline HuggingFace
    ({"text": ..., "chunks": [...]}) pour rester compatible avec le
    code appelant existant (aucun changement requis côté appelant à
    part remplacer le nom de la fonction appelée).
    """
    if transcriber_pipeline is None:
        return {"text": "", "chunks": []}
    if not has_speech(audio_np, sample_rate):
        return {"text": "", "chunks": [], "no_speech": True}
    try:
        result = transcriber_pipeline({"sampling_rate": sample_rate, "raw": audio_np})
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA error" in str(e):
            print(f"[Whisper] ⚠️ VRAM insuffisante — repli sur CPU pour cette transcription")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Force temporairement le pipeline sur CPU pour cet appel
                original_device = transcriber_pipeline.device
                transcriber_pipeline.model.to("cpu")
                transcriber_pipeline.device = torch.device("cpu")
                result = transcriber_pipeline({"sampling_rate": sample_rate, "raw": audio_np})
                # Remet le modèle sur GPU pour les appels suivants
                transcriber_pipeline.model.to(original_device)
                transcriber_pipeline.device = original_device
            except Exception as e2:
                print(f"[Whisper] Erreur repli CPU également: {e2}")
                return {"text": "", "chunks": []}
        else:
            print(f"[Whisper] Erreur transcription: {e}")
            return {"text": "", "chunks": []}
    except Exception as e:
        print(f"[Whisper] Erreur transcription: {e}")
        return {"text": "", "chunks": []}
   
    return result


# ═══════════════════════════════════════════════════════════════════
# #21 — Renommage éthique (mensonge/véracité → signaux comportementaux)
# ═══════════════════════════════════════════════════════════════════
def calculate_behavioral_tension_signals(emotion_history, confidence_history, frame_times):
    """
    Remplace `calculate_deception_risk`. Le calcul mathématique est
    STRICTEMENT IDENTIQUE à l'original (aucune régression) ; seule
    l'interprétation change : ce score reflète la fréquence de
    signaux associés au stress/tension facial (peur, dégoût, tristesse,
    instabilité), PAS une probabilité de mensonge.

    Pourquoi ce changement : la littérature scientifique (Bond &
    DePaulo, Vrij et al.) ne permet pas d'affirmer qu'un signal facial
    ou vocal indique un mensonge — le stress observé en entretien est
    au contraire un phénomène normal et attendu, y compris chez des
    candidats parfaitement honnêtes. Présenter ce score comme une
    "probabilité de tromperie" est trompeur et risqué (biais,
    discrimination indirecte, conformité RGPD/AI Act).
    """
    if len(emotion_history) < 5:
        return 0, "Analyse insuffisante — pas assez de données", {}

    total = max(1, len(emotion_history))
    tension_emotions = ['fear', 'surprise', 'disgust', 'sad']
    tension_count = sum(1 for e in emotion_history if e in tension_emotions)

    for i in range(len(emotion_history) - 1):
        if emotion_history[i] == 'angry' and emotion_history[i + 1] == 'fear':
            tension_count += 1

    emotion_score = min(100.0, (tension_count / total) * 333.3)

    changes = sum(1 for i in range(1, total)
                  if emotion_history[i] != emotion_history[i - 1])
    var_score = min(100.0, (changes / total) * 500.0)

    micro = 0
    for i in range(2, total):
        if (emotion_history[i] == emotion_history[i - 2] and
                emotion_history[i] != emotion_history[i - 1]):
            micro += 1
    micro_score = min(100.0, (micro / max(1, total)) * 1000.0)

    avg_conf = np.mean(confidence_history) if confidence_history else 0.8
    conf_score = max(0.0, (1.0 - avg_conf) * 200.0)

    stress_periods = []
    current_stress = False
    for e in emotion_history:
        is_stress = e in ['fear', 'sad', 'disgust']
        if is_stress != current_stress:
            stress_periods.append(1)
            current_stress = is_stress
    pattern_score = min(100.0, len(stress_periods) * 15.0)

    total_score = (emotion_score * 0.30 + var_score * 0.20 +
                   micro_score * 0.25 + conf_score * 0.15 +
                   pattern_score * 0.10)

    details = {
        'emotion_score': emotion_score,
        'variability_score': var_score,
        'micro_expressions': micro,
        'confidence_score': conf_score,
        'pattern_score': pattern_score,
        'total_score': total_score,
    }

    level = ("Faible — peu de signes de tension observés"
              if total_score < 30 else
              "Modéré — quelques signes de tension, à explorer si pertinent"
              if total_score < 60 else
              "Élevé — signes de tension fréquents (ne préjuge pas de la "
              "sincérité du candidat : le stress d'entretien est normal)")

    return total_score, level, details


def analyze_speech_patterns(transcript):
    """
    Remplace `analyze_speech_deception`. Détecte des marqueurs
    linguistiques (hésitations, sur-justification) — des phénomènes de
    discours normaux liés au stress de l'oral ou au style personnel,
    pas des indices fiables de mensonge.
    """
    if not transcript:
        return 0.0, []
    t_lower = transcript.lower()
    hesitation_w = ["euh", "bah", "en fait", "je crois", "peut-être",
                     "genre", "comment dire", "je ne sais pas", "enfin"]
    justif_w = ["honnêtement", "pour être franc", "à vrai dire",
                 "croyez-moi", "sincèrement", "je vous jure",
                 "en toute franchise", "absolument"]
    hesitations = sum(t_lower.count(w) for w in hesitation_w)
    justifications = sum(t_lower.count(w) for w in justif_w)
    word_count = max(1, len(t_lower.split()))
    marker_score = 0.0
    flags = []

    h_ratio = hesitations / word_count
    if h_ratio > 0.015:
        marker_score += min(50.0, (h_ratio / 0.05) * 50.0)
        flags.append(f"Hésitations fréquentes ({hesitations} détectées) — "
                      f"peut simplement traduire le stress normal de "
                      f"l'exercice, pas un manque de sincérité.")

    j_ratio = justifications / word_count
    if j_ratio > 0.005:
        marker_score += min(50.0, (j_ratio / 0.02) * 50.0)
        flags.append(f"Discours appuyé/insistant ({justifications} "
                      f"marqueurs détectés) — à interpréter avec prudence.")

    return min(100.0, marker_score), flags


# ═══════════════════════════════════════════════════════════════════
# #22 — Affichage d'incertitude (intervalle plutôt qu'un chiffre figé)
# ═══════════════════════════════════════════════════════════════════
def compute_metric_uncertainty(metrics_history: List[dict], window: int = 10) -> dict:
    """
    Calcule, pour chaque métrique clé, un intervalle [min, max] sur les
    N dernières valeurs plutôt qu'un chiffre unique figé.

    Corrige un symptôme observé en test réel : `prediction_stability`
    restait bloqué à 100% en continu pendant de longues périodes — ce
    champ ne reflète pas une vraie confiance du modèle, seulement la
    fréquence de "neutral/happy" dans l'historique récent (c'est
    tautologique : rester neutre fait mécaniquement monter ce score).
    Afficher une fourchette empêche de présenter un signal bruité
    comme une certitude absolue à l'utilisateur final.
    """
    if not metrics_history:
        return {}
    recent = metrics_history[-window:]
    out = {}
    for key in ('stress_management', 'communication', 'assurance_level',
                'expressivity', 'prediction_stability'):
        vals = [m.get(key) for m in recent if m.get(key) is not None]
        if not vals:
            continue
        out[key] = {
            'min': float(min(vals)),
            'max': float(max(vals)),
            'spread': float(max(vals) - min(vals)),
        }
    return out


# ═══════════════════════════════════════════════════════════════════
# #19 — Diagnostic "candidat absent" différencié
# ═══════════════════════════════════════════════════════════════════
def diagnose_absence_reason(faces_detected_count: int,
                             embedding_extracted: bool,
                             similarity: Optional[float],
                             threshold: float) -> str:
    """
    Détermine LAQUELLE des trois causes distinctes explique un statut
    "absent", au lieu de renvoyer systématiquement la même étiquette
    opaque.

    Observé en test réel : un visage parfaitement visible, net et bien
    éclairé était déclaré "CANDIDAT ABSENT" pendant plus d'une minute,
    sans que les logs ne permettent de savoir si c'était :
      (a) YOLO qui ne détectait aucun visage (peu probable ici) ;
      (b) ArcFace qui n'arrivait pas à extraire un embedding
          exploitable (angle, lunettes, reflet, résolution) ;
      (c) un embedding valide mais sous le seuil de similarité (vrai
          changement de personne, ou dérive de la référence
          mémorisée).

    Ce diagnostic rend la cause explicite et actionnable : (b) suggère
    de revoir MIN_DET_SCORE ou la qualité vidéo, (c) suggère de revoir
    TOLERANCE_ARCFACE ou de re-déclencher une mémorisation.
    """
    if faces_detected_count == 0:
        return "no_face_detected"
    if not embedding_extracted:
        return "embedding_extraction_failed"
    if similarity is not None and similarity < threshold:
        return "similarity_below_threshold"
    return "unknown"


ABSENCE_REASON_LABELS_FR = {
    "no_face_detected": "Aucun visage détecté dans l'image",
    "embedding_extraction_failed": (
        "Visage détecté mais qualité insuffisante pour confirmer "
        "l'identité (angle, reflet, résolution)"
    ),
    "similarity_below_threshold": (
        "Visage détecté mais ne correspond pas (ou plus) au candidat "
        "verrouillé"
    ),
    "unknown": "Cause indéterminée",
}


# ═══════════════════════════════════════════════════════════════════
# POINTS D'INTÉGRATION DANS app.py — voir integration_notes.md pour
# les blocs exacts à remplacer (recherche/remplace ligne par ligne).
# ═══════════════════════════════════════════════════════════════════