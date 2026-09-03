# tracking_manager.py — Version 6.0 Professionnelle
# ═══════════════════════════════════════════════════════════════════
# Améliorations v6.0 vs v5.2 :
# - Config : seuils ajustés (HIST_SEUIL 0.25→0.15, ZONE_MARGIN 0.20→0.25)
# - Config : MAX_LOST 30→45, MEMO_MIN_BRIGHT 45→40, MEMO_MIN_SHARPNESS 15→12
# - Config : MEMORIZE_FRAMES 6→5, TOLERANCE_ARCFACE 0.38→0.35
# - SpeedFilter : seuil corrélation 0.25→0.15 + masque HSV élargi
# - IdentityManager.add_frame : seuils adaptatifs selon luminosité
# - IdentityManager.verify : threshold dynamique
# - TrackingManager._on_lost : MAX_LOST 45 + MAX_REID 120
# - update() : alpha adaptatif 0.15/0.08/0.03 selon sim
# ═══════════════════════════════════════════════════════════════════
#
# ═══════════════════════════════════════════════════════════════════
# PATCH v6.1 — corrections tracking multi-visages / changements de plan
# - Config : MAX_BRIGHTNESS ajouté (symétrique de MIN_BRIGHTNESS)
# - Config : TOTAL_FAILURE_TOLERANCE ajouté (échec total ArcFace)
# - Config : JUMP_DISTANCE_RATIO ajouté (saut spatial suspect)
# ═══════════════════════════════════════════════════════════════════
#
# ═══════════════════════════════════════════════════════════════════
# PATCH v6.2 — correction fuite ABANDONED (spam infini + blocage)
# - _on_lost() : le warning "ABANDONED" ne se logue plus qu'UNE SEULE
#   FOIS lors de la transition (avant : répété à chaque frame tant que
#   lost_count continuait d'augmenter, soit des centaines de fois sur
#   une longue vidéo).
# - _on_lost() : last_bbox n'est PLUS effacé lors du passage à
#   ABANDONED (avant : mis à None, ce qui rendait structurellement
#   impossible toute récupération ultérieure du candidat, la logique
#   de recherche globale de secours dans app.py exigeant
#   last_bbox is not None pour se déclencher).
# ═══════════════════════════════════════════════════════════════════

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from filterpy.kalman import KalmanFilter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    MAX_AGE:               int   = 300
    N_INIT:                int   = 3
    NN_BUDGET:             int   = 100
    KALMAN_Q:              float = 0.05
    KALMAN_R:              float = 0.5
    KALMAN_P:              float = 2.0
    MIN_DET_SCORE:         float = 0.35
    MIN_DET_SCORE_FALLBACK: float = 0.20
    TOLERANCE_ARCFACE_FALLBACK: float = 0.32 + 0.10
    TOLERANCE_ARCFACE:     float = 0.32   # ← était 0.35 (plus permissif sur rotation)
    TOLERANCE_VIT:         float = 0.75
    TOLERANCE_REID_GLOBAL: float = 0.55   # ← était 0.60
    UPDATE_ALPHA:          float = 0.05
    UPDATE_STRICT:         float = 0.50   # ← était 0.55
    UPDATE_CONFIRM_FRAMES: int   = 2      # ← était 3 (plus réactif)
    MEMORIZE_FRAMES:       int   = 5
    MEMO_MIN_SHARPNESS:    float = 10.0   # ← était 12.0
    MEMO_MIN_BRIGHT:       float = 35.0   # ← était 40.0
    MAX_MEMO_FAILS:        int   = 40     # ← était 35
    SIM_IMMEDIAT:          float = 0.15
    MAX_REJECTIONS:        int   = 30
    HIST_SEUIL:            float = 0.05   # ← était 0.15 — très bas car rotation visage
    # change beaucoup l'histogramme mais c'est toujours le bon candidat
    ZONE_MARGIN:           float = 0.18   # fallback uniquement
    ZONE_MIN_PX:           int   = 50
    MIN_BBOX_PX:           int   = 30     # ← était 35
    MIN_BRIGHTNESS:        int   = 28     # ← était 30
    MAX_LOST:              int   = 60     # ← était 45 (plus patient)
    MAX_REID:              int   = 150    # ← était 120

    # ── PATCH v6.1 ────────────────────────────────────────────────
    # Plafond de luminosité : au-delà, l'image est cramée (comme
    # MIN_BRIGHTNESS pour le noir). Une frame surexposée doit être
    # traitée comme "zone illisible", pas comme un échec d'identité.
    MAX_BRIGHTNESS:         int   = 195

    # Tolérance spécifique aux échecs TOTAUX d'ArcFace (aucun visage
    # exploitable trouvé, sim=0.0 / bright ou blur extrêmes). Dans ce
    # cas il n'y a rien à "confirmer" en attendant — inutile de
    # gaspiller plusieurs frames avant de déclencher la recherche
    # globale. Distinct de la tolérance normale (visage détecté mais
    # similarité basse), qui elle doit rester patiente (rotation,
    # flou passager sur le bon candidat).
    TOTAL_FAILURE_TOLERANCE: int = 1

    # Distance de saut (en fraction de la largeur de frame) au-delà
    # de laquelle un nouveau bbox est considéré comme un changement
    # de plan / bascule de locuteur potentiel plutôt qu'un mouvement
    # normal du candidat. Déclenche une vérification ArcFace stricte
    # immédiate (seuil global) au lieu du seuil dynamique permissif.
    JUMP_DISTANCE_RATIO:    float = 0.18

    # ── PATCH v6.2 (2/2) — magie numérique regroupée ────────────────
    # Ces seuils étaient auparavant écrits en dur directement dans
    # IdentityManager.add_frame(), sous forme de paliers if/elif sur
    # la luminosité mesurée. Les regrouper ici sous des noms explicites
    # permet de les ajuster à un seul endroit, avec un nom qui décrit
    # leur rôle, plutôt que de devoir deviner lesquels des multiples
    # "60", "100", "8.0" etc. dispersés dans le fichier sont liés entre
    # eux avant de les modifier.
    MEMO_BRIGHTNESS_VERY_DARK:   float = 60.0
    MEMO_BRIGHTNESS_DARK:        float = 100.0
    MEMO_SHARPNESS_VERY_DARK:    float = 8.0
    MEMO_BRIGHT_VERY_DARK:       float = 28.0
    MEMO_SHARPNESS_DARK:         float = 10.0
    MEMO_BRIGHT_DARK:            float = 35.0
    # Seuil de cohérence entre une nouvelle frame candidate et la
    # moyenne des embeddings déjà acceptés pendant la mémorisation.
    # Plus permissif tant que peu de frames sont déjà acceptées (on
    # n'a pas encore une moyenne fiable), plus strict ensuite.
    MEMO_COHERENCE_EARLY:        float = 0.22
    MEMO_COHERENCE_LATER:        float = 0.32
    MEMO_COHERENCE_EARLY_COUNT:  int   = 2


CFG = Config()


class State(Enum):
    SEARCHING        = auto()
    MEMORIZING       = auto()
    TRACKING         = auto()
    LOST             = auto()
    REIDENTIFICATION = auto()
    ABANDONED        = auto()


TrackingState = State


class KalmanPredictor:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([
            [1,0,1,0,0,0,0,0],[0,1,0,1,0,0,0,0],
            [0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
            [0,0,0,0,1,0,1,0],[0,0,0,0,0,1,0,1],
            [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],
        ], dtype=float)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
        ], dtype=float)
        self.kf.R  *= CFG.KALMAN_R
        self.kf.P  *= CFG.KALMAN_P
        self.kf.Q  *= CFG.KALMAN_Q
        self.ready   = False
        self._last_w = 0.0
        self._last_h = 0.0

    def init(self, cx, cy, w, h):
        self.kf.x    = np.array([[cx],[cy],[0.],[0.],[w],[h],[0.],[0.]])
        self._last_w = w
        self._last_h = h
        self.ready   = True

    def update(self, cx, cy, w, h):
        if not self.ready:
            self.init(cx, cy, w, h)
        else:
            self.kf.predict()
            self.kf.update(np.array([[cx],[cy],[w],[h]]))
        self._last_w = w
        self._last_h = h
        return self._extract()

    def predict(self):
        if not self.ready:
            return None
        self.kf.predict()
        return self._extract()

    def _extract(self):
        cx = float(self.kf.x[0,0])
        cy = float(self.kf.x[1,0])
        w  = max(CFG.MIN_BBOX_PX, float(self.kf.x[4,0]))
        h  = max(CFG.MIN_BBOX_PX, float(self.kf.x[5,0]))
        w  = 0.85*self._last_w + 0.15*w if self._last_w > 0 else w
        h  = 0.85*self._last_h + 0.15*h if self._last_h > 0 else h
        return cx, cy, w, h


class SpeedFilter:
    def __init__(self):
        self._ref_hist: Optional[np.ndarray] = None
        self.enabled = False

    def memorize(self, face_img):
        h = self._hist(face_img)
        if h is not None:
            self._ref_hist = h
            self.enabled   = True

    def is_candidate(self, face_img) -> Tuple[bool, float]:
        if not self.enabled or self._ref_hist is None:
            return True, 1.0
        h    = self._hist(face_img)
        corr = self._corr(self._ref_hist, h)
        # ← AMÉLIORATION : seuil abaissé 0.25→0.15
        # Raison : changement éclairage/légère rotation = corrélation basse
        # mais c'est toujours le bon candidat
        return corr >= CFG.HIST_SEUIL, corr

    def reset(self):
        self._ref_hist = None
        self.enabled   = False

    @staticmethod
    def _hist(img):
        try:
            if img is None or img.size == 0:
                return None
            hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # ← AMÉLIORATION : masque HSV plus large pour peau
            mask = cv2.inRange(hsv,
                               np.array([0, 20, 50]),    # était [0,30,60]
                               np.array([25, 255, 255]))
            hist = cv2.calcHist([hsv],[0],mask,[16],[0,180])
            s    = hist.sum()
            return hist/s if s > 0 else hist
        except Exception:
            return None

    @staticmethod
    def _corr(h1, h2) -> float:
        if h2 is None:
            return 1.0
        try:
            return float(cv2.compareHist(
                h1.astype(np.float32),
                h2.astype(np.float32),
                cv2.HISTCMP_CORREL))
        except Exception:
            return 1.0


class ZoneManager:
    def __init__(self):
        self._zone      = None
        self._face_w    = None
        self._face_h    = None
        self._n_faces   = 1      # nombre de candidats détectés (preview)

    def define(self, cx, cy, fw, fh,
               face_w: float = None, face_h: float = None,
               n_faces: int  = 1):
        """
        Zone adaptative basée sur la bbox réelle du visage.

        Si face_w/face_h connus → zone = 2.0x la taille du visage
        Sinon → zone = 15% de la frame (conservatif)

        Si n_faces > 1 (plusieurs candidats détectés) → zone plus petite
        pour éviter de capturer un autre candidat proche.
        """
        self._face_w  = face_w
        self._face_h  = face_h
        self._n_faces = n_faces

        if face_w is not None and face_h is not None:
            # Zone basée sur taille réelle du visage
            # Si plusieurs candidats → plus restrictif
            factor = 1.5 if n_faces > 1 else 2.0
            mx = max(CFG.ZONE_MIN_PX, face_w * factor)
            my = max(CFG.ZONE_MIN_PX, face_h * factor)
        else:
            # Fallback : % de la frame
            pct = 0.12 if n_faces > 1 else 0.18
            mx  = max(CFG.ZONE_MIN_PX, fw * pct)
            my  = max(CFG.ZONE_MIN_PX, fh * pct)

        self._zone = {
            "x_min": max(0,  cx - mx),
            "x_max": min(fw, cx + mx),
            "y_min": max(0,  cy - my),
            "y_max": min(fh, cy + my),
            "cx":    cx,
            "cy":    cy,
        }
        print(f"[ZoneManager] Zone: "
              f"x=[{self._zone['x_min']:.0f},{self._zone['x_max']:.0f}] "
              f"y=[{self._zone['y_min']:.0f},{self._zone['y_max']:.0f}] "
              f"(face={face_w:.0f}x{face_h:.0f}px, "
              f"n_faces={n_faces})"
              if face_w else
              f"[ZoneManager] Zone: "
              f"x=[{self._zone['x_min']:.0f},{self._zone['x_max']:.0f}] "
              f"y=[{self._zone['y_min']:.0f},{self._zone['y_max']:.0f}]")

    def expand(self, fw: float, fh: float):
        """
        Agrandir la zone si le candidat sort temporairement.
        Utilisé après tracking lost pour permettre la réidentification.
        """
        if self._zone is None:
            return
        cx = self._zone['cx']
        cy = self._zone['cy']
        mx = max(CFG.ZONE_MIN_PX, fw * 0.25)
        my = max(CFG.ZONE_MIN_PX, fh * 0.25)
        self._zone['x_min'] = max(0,  cx - mx)
        self._zone['x_max'] = min(fw, cx + mx)
        self._zone['y_min'] = max(0,  cy - my)
        self._zone['y_max'] = min(fh, cy + my)
        print(f"[ZoneManager] Zone élargie: "
              f"x=[{self._zone['x_min']:.0f},{self._zone['x_max']:.0f}] "
              f"y=[{self._zone['y_min']:.0f},{self._zone['y_max']:.0f}]")

    def contains(self, cx, cy) -> bool:
        if self._zone is None:
            return True
        return (self._zone["x_min"] <= cx <= self._zone["x_max"] and
                self._zone["y_min"] <= cy <= self._zone["y_max"])

    def reset(self):
        self._zone    = None
        self._face_w  = None
        self._face_h  = None
        self._n_faces = 1

    def import_zone(self, zone: dict) -> None:
        """
        ← AJOUT : restaure une zone déjà définie (venant d'une autre
        réplique via l'état de tracking partagé), sans repasser par
        define() qui recalcule les marges — la zone était déjà
        correctement calculée par la réplique d'origine.
        """
        if zone:
            self._zone = zone

    @property
    def defined(self) -> bool:
        return self._zone is not None


try:
    from insightface.app import FaceAnalysis as _FA
    _ARCFACE_OK = True
except ImportError:
    _ARCFACE_OK = False


def _frame_quality(face_img) -> Tuple[float, float]:
    try:
        if face_img is None or face_img.size == 0:
            return 0.0, 0.0
        gray       = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return brightness, sharpness
    except Exception:
        return 0.0, 0.0


class IdentityManager:
    def __init__(self, shared_arcface=None, fast_arcface=None):
        self._embeddings      = []
        self._ref             = None
        self._ref_ratio       = None
        self._ref_hue         = None
        self._ref_hist        = None
        self._memorized       = False
        self._memorizing      = False
        self._arcface         = None
        self._use_arc         = False
        self._pending_updates = []
        self.last_fail_reason: Optional[str] = None
        # ← PATCH v6.4 : instance ArcFace "rapide" (det_size réduit),
        # utilisée uniquement pour les vérifications répétées à chaque
        # frame (verify()). L'instance haute précision (self._arcface,
        # det_size=640) reste utilisée pour la mémorisation initiale
        # (add_frame, quelques appels seulement) où la qualité de la
        # référence compte plus que la vitesse. Sans instance dédiée,
        # fallback sur self._arcface (comportement identique à avant).
        self._fast_arcface = fast_arcface

        if shared_arcface is not None:
            self._arcface  = shared_arcface
            self.TOLERANCE = CFG.TOLERANCE_ARCFACE
            self._use_arc  = True
            print(f"[Identity] ✅ ArcFace partagé réutilisé "
                  f"seuil={self.TOLERANCE}"
                  + (" (+ instance rapide verify())"
                     if fast_arcface is not None else ""))
        elif _ARCFACE_OK:
            try:
                self._arcface = _FA(name='buffalo_sc',
                                    providers=['CPUExecutionProvider'])
                self._arcface.prepare(ctx_id=0, det_size=(640,640))
                self.TOLERANCE = CFG.TOLERANCE_ARCFACE
                self._use_arc  = True
                print(f"[Identity] ✅ ArcFace actif seuil={self.TOLERANCE}")
            except Exception as e:
                self.TOLERANCE = CFG.TOLERANCE_VIT
                print(f"[Identity] ⚠️ Fallback ViT ({e})")

        else:
            self.TOLERANCE = CFG.TOLERANCE_VIT

        self.candidate_embedding = None
        self.last_fail_reason: Optional[str] = None
        # ← PATCH v7.1 : mémorise si le DERNIER embedding extrait avec
        # succès (par _arcface_embed / _arcface_embed_fast) provient de
        # la stratégie de secours à seuil abaissé. verify() consulte ce
        # flag pour exiger une similarité plus stricte dans ce cas —
        # sans ça, un embedding de moins bonne qualité pourrait
        # dépasser à tort le seuil normal.
        self._last_embed_low_conf: bool = False

    def set_reference_embedding(self, embedding: np.ndarray,
                                 face_img=None) -> None:
        emb                      = self._norm(embedding)
        self._ref                = emb
        self._memorized          = True
        self._memorizing         = False
        self.candidate_embedding = emb
        self._embeddings         = [emb]
        self._pending_updates    = []
        if face_img is not None:
            self._visual_ref(face_img)
            print(f"[Identity] ✅ Référence chargée depuis cache preview "
                  f"dim={emb.shape[0]}D (avec face_img)")
        else:
            print(f"[Identity] ✅ Référence chargée depuis cache preview "
                  f"dim={emb.shape[0]}D (sans face_img)")

    def _arcface_embed(self, img):
        if self._arcface is None or img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        if w < 30 or h < 30:
            return None

        strategies = []
        try:
            c = np.zeros((640,640,3), dtype=np.uint8)
            c[170:470,170:470] = cv2.resize(img, (300,300),
                                             interpolation=cv2.INTER_LINEAR)
            strategies.append(("S3_c300", c))
        except Exception:
            pass
        try:
            c = np.zeros((640,640,3), dtype=np.uint8)
            c[70:570,70:570] = cv2.resize(img, (500,500),
                                           interpolation=cv2.INTER_LINEAR)
            strategies.append(("S4_c500", c))
        except Exception:
            pass
        try:
            strategies.append(("S2_raw",
                cv2.resize(img, (112,112),
                           interpolation=cv2.INTER_LINEAR)))
        except Exception:
            pass

        # ← PATCH v7.1 : passe 1 — seuil STRICT (comportement d'origine,
        # inchangé). C'est le chemin normal, utilisé dans l'immense
        # majorité des frames.
        self._last_embed_low_conf = False
        for name, frame in strategies:
            try:
                faces = self._arcface.get(frame)
                if not faces:
                    continue
                best = max(faces, key=lambda f: f.det_score)
                if best.det_score < CFG.MIN_DET_SCORE:
                    continue
                print(f"[ArcFace] ✅ {name} score={best.det_score:.3f}")
                emb  = best.embedding.astype(np.float32)
                norm = np.linalg.norm(emb)
                return emb/norm if norm > 0 else emb
            except Exception:
                continue

        # ← PATCH v7.1 : passe 2 — seuil de SECOURS, uniquement si la
        # passe stricte a échoué sur toutes les stratégies. On marque
        # explicitement le résultat comme "confiance faible" pour que
        # verify() applique un seuil de similarité plus exigeant en
        # compensation (voir TOLERANCE_ARCFACE_FALLBACK).
        for name, frame in strategies:
            try:
                faces = self._arcface.get(frame)
                if not faces:
                    continue
                best = max(faces, key=lambda f: f.det_score)
                if best.det_score < CFG.MIN_DET_SCORE_FALLBACK:
                    continue
                print(f"[ArcFace] ⚠️ {name} score={best.det_score:.3f} "
                      f"(détection de secours — confiance faible, "
                      f"similarité exigée renforcée)")
                emb  = best.embedding.astype(np.float32)
                norm = np.linalg.norm(emb)
                self._last_embed_low_conf = True
                return emb/norm if norm > 0 else emb
            except Exception:
                continue

        print("[ArcFace] ❌ Toutes stratégies échouées (strict + secours)")
        return None
    
    def _arcface_embed_fast(self, img):
        """
        ← PATCH v6.4 : variante rapide de _arcface_embed(), utilisée
        pour les vérifications répétées à chaque frame (verify()).

        Utilise une instance ArcFace avec det_size réduit (ex: 320x320
        au lieu de 640x640) si disponible (self._fast_arcface), avec un
        canevas proportionnellement plus petit (mêmes proportions
        visage/canevas que la version haute précision, donc même
        comportement de détection, juste moins de pixels à traiter).

        Le réseau de reconnaissance (embedding 512D) travaille toujours
        sur un crop 112x112 fixe quel que soit det_size — seule l'étape
        de DÉTECTION est accélérée. Le gain vient du fait que la
        détection sur une image 320x320 coûte environ 4x moins cher
        qu'sur 640x640 (proportionnel au nombre de pixels).

        Si aucune instance rapide n'est configurée, retombe sur
        _arcface_embed() (comportement identique à avant v6.4).
        """
        arc = self._fast_arcface if self._fast_arcface is not None \
            else self._arcface
        if arc is None or img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        if w < 30 or h < 30:
            return None

        # Canevas 320x320 (au lieu de 640x640), mêmes proportions
        # visage/canevas que la version haute précision (300/640≈0.469
        # → 150/320≈0.469 ; 500/640≈0.781 → 250/320≈0.781).
        strategies = []
        try:
            c = np.zeros((320,320,3), dtype=np.uint8)
            c[85:235,85:235] = cv2.resize(img, (150,150),
                                          interpolation=cv2.INTER_LINEAR)
            strategies.append(("F_c150", c))
        except Exception:
            pass
        try:
            c = np.zeros((320,320,3), dtype=np.uint8)
            c[35:285,35:285] = cv2.resize(img, (250,250),
                                          interpolation=cv2.INTER_LINEAR)
            strategies.append(("F_c250", c))
        except Exception:
            pass
        try:
            strategies.append(("F_raw",
                cv2.resize(img, (112,112),
                           interpolation=cv2.INTER_LINEAR)))
        except Exception:
            pass

        self._last_embed_low_conf = False
        for name, frame in strategies:
            try:
                faces = arc.get(frame)
                if not faces:
                    continue
                best = max(faces, key=lambda f: f.det_score)
                if best.det_score < CFG.MIN_DET_SCORE:
                    continue
                emb  = best.embedding.astype(np.float32)
                norm = np.linalg.norm(emb)
                return emb/norm if norm > 0 else emb
            except Exception:
                continue

        # ← PATCH v7.1 : même mécanisme de secours que _arcface_embed.
        for name, frame in strategies:
            try:
                faces = arc.get(frame)
                if not faces:
                    continue
                best = max(faces, key=lambda f: f.det_score)
                if best.det_score < CFG.MIN_DET_SCORE_FALLBACK:
                    continue
                emb  = best.embedding.astype(np.float32)
                norm = np.linalg.norm(emb)
                self._last_embed_low_conf = True
                return emb/norm if norm > 0 else emb
            except Exception:
                continue
        return None

    @staticmethod
    def _norm(e):
        n = np.linalg.norm(e)
        return e/n if n > 0 else e

    def _visual_ref(self, img):
        try:
            h, w          = img.shape[:2]
            self._ref_ratio = w/h
            hsv           = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._ref_hue = float(np.mean(hsv[:,:,0]))
            gray          = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist          = cv2.calcHist([gray],[0],None,[8],[0,256]).flatten()
            s             = hist.sum()
            self._ref_hist = hist/s if s > 0 else hist
        except Exception:
            pass

    def start(self):
        self._embeddings      = []
        self._ref             = None
        self._memorized       = False
        self._memorizing      = True
        self.candidate_embedding = None
        self._pending_updates = []
        print(f"[Identity] 🔍 Mémorisation démarrée "
              f"(cible={CFG.MEMORIZE_FRAMES} frames)")

    def add_frame(self, face_img=None, embedding=None) -> bool:
        """
        Version améliorée — seuils adaptatifs selon luminosité.
        Moins stricte si sombre, plus permissive sur la cohérence initiale.
        """
        if not self._memorizing:
            return self._memorized

        if face_img is not None:
            brightness, sharpness = _frame_quality(face_img)

            # ← ADAPTATIF : seuils selon luminosité (constantes nommées
            # dans Config, au lieu de valeurs en dur — voir PATCH v6.2)
            if brightness < CFG.MEMO_BRIGHTNESS_VERY_DARK:
                min_sharpness = CFG.MEMO_SHARPNESS_VERY_DARK
                min_bright    = CFG.MEMO_BRIGHT_VERY_DARK
            elif brightness < CFG.MEMO_BRIGHTNESS_DARK:
                min_sharpness = CFG.MEMO_SHARPNESS_DARK
                min_bright    = CFG.MEMO_BRIGHT_DARK
            else:
                min_sharpness = CFG.MEMO_MIN_SHARPNESS  # 12.0
                min_bright    = CFG.MEMO_MIN_BRIGHT      # 40.0

            if brightness < min_bright:
                print(f"[Identity] Frame rejetée "
                      f"(trop sombre: {brightness:.0f})")
                return False
            if sharpness < min_sharpness:
                print(f"[Identity] Frame rejetée "
                      f"(floue: {sharpness:.0f} < {min_sharpness})")
                return False

        emb = None
        if self._use_arc:
            # ← OPTIMISATION PERF : bascule vers _arcface_embed_fast()
            # (320x320) au lieu de _arcface_embed() (640x640, jusqu'à
            # 6 appels ArcFace en pire cas). Cette fonction est appelée
            # jusqu'à MEMORIZE_FRAMES fois (5) au tout début de chaque
            # session temps réel, AVANT que la moindre analyse ne
            # puisse démarrer — c'était la cause du retard de 10-15s
            # observé par les utilisateurs avant le premier résultat.
            # Le changement reste sûr : la référence finale est une
            # MOYENNE de 5 frames (le bruit d'une extraction moins
            # précise s'atténue déjà par cette moyenne), et update()
            # — déjà optimisé de la même façon — continue d'affiner
            # cette référence tout au long de l'entretien ensuite.
            emb = self._arcface_embed_fast(face_img)
            if emb is None:
                print("[Identity] ArcFace échoué — frame ignorée")
                return False
        elif embedding is not None:
            emb = self._norm(embedding)
        if emb is None:
            return False

        # Cohérence vs frames précédentes
        if self._embeddings:
            mean_so_far = self._norm(
                np.mean(np.stack(self._embeddings), axis=0))
            sim = float(np.dot(mean_so_far, emb))
            # ← ADAPTATIF : seuil cohérence selon nombre de frames
            # déjà acceptées (constantes nommées — voir PATCH v6.2)
            min_coh = (CFG.MEMO_COHERENCE_EARLY
                       if len(self._embeddings) < CFG.MEMO_COHERENCE_EARLY_COUNT
                       else CFG.MEMO_COHERENCE_LATER)
            if sim < min_coh:
                print(f"[Identity] Frame rejetée — incohérente "
                      f"(sim={sim:.2f} < {min_coh})")
                return False

        self._embeddings.append(emb)
        if len(self._embeddings) == 1 and face_img is not None:
            self._visual_ref(face_img)
        print(f"[Identity] Frame "
              f"{len(self._embeddings)}/{CFG.MEMORIZE_FRAMES} acceptée")

        if len(self._embeddings) >= CFG.MEMORIZE_FRAMES:
            self._finalize()
            return True
        return False

    def _finalize(self):
        stacked = np.stack(self._embeddings, axis=0)
        mean    = np.mean(stacked, axis=0)
        norm    = np.linalg.norm(mean)
        self._ref            = mean/norm if norm > 0 else mean
        self._memorized      = True
        self._memorizing     = False
        self.candidate_embedding = self._ref
        sims = [float(np.dot(self._ref, e)) for e in self._embeddings]
        print(f"[Identity] ✅ Ref finalisée dim={self._ref.shape[0]}D "
              f"sur {len(self._embeddings)} frames — "
              f"cohérence moyenne={np.mean(sims):.3f} "
              f"(min={np.min(sims):.3f})")

    def verify(self, face_img=None, embedding=None,
               strict_global: bool = False,
               threshold: float = None) -> Tuple[bool, float]:
        """
        Vérification identité avec threshold dynamique.
        Priorité : threshold > strict_global > TOLERANCE fixe
        """
        if self._ref is None or self._memorizing:
            return True, 1.0

        if threshold is not None:
            seuil = threshold
        elif strict_global:
            seuil = CFG.TOLERANCE_REID_GLOBAL  # 0.60
        else:
            seuil = self.TOLERANCE              # 0.35

        if self._use_arc:
            # ← PATCH v6.4 : embedding rapide (det_size réduit) pour les
            # vérifications courantes — c'est l'appel le plus fréquent
            # (une fois par frame). En mode strict_global (saut spatial
            # suspect, position suspecte), on garde l'embedding haute
            # précision : ces vérifications sont rares et décisives,
            # la vitesse n'y est pas le facteur limitant.
            if face_img is None:
                emb = None
            elif strict_global:
                emb = self._arcface_embed(face_img)
            else:
                emb = self._arcface_embed_fast(face_img)
            if emb is None:
                self.last_fail_reason = "embedding_extraction_failed"
                return False, 0.0

            # ← PATCH v7.1 : si l'embedding vient de la détection de
            # secours (confiance faible), on exige un seuil de
            # similarité renforcé au lieu du seuil normalement
            # applicable — sauf si un seuil explicite plus strict était
            # déjà demandé par l'appelant (ex: strict_global), auquel
            # cas on garde le plus strict des deux.
            effective_seuil = seuil
            if self._last_embed_low_conf:
                effective_seuil = max(seuil, CFG.TOLERANCE_ARCFACE_FALLBACK)

            sim = float(np.dot(self._ref, emb))
            ok  = sim >= effective_seuil
            self.last_fail_reason = None if ok else "similarity_below_threshold"
            print(f"[Identity] verify sim={sim:.3f} "
                  f"seuil={effective_seuil:.2f}"
                  f"{' [secours+renforcé]' if self._last_embed_low_conf else ''} "
                  f"{'(global)' if strict_global else ''} "
                  f"{'✅' if ok else '❌'}")
            return ok, sim
        else:
            if embedding is None:
                return True, 1.0
            emb   = self._norm(embedding)
            sim   = float(np.dot(self._ref, emb))
            score = self._combined(face_img, emb, sim)
            return score >= seuil, score

    def _combined(self, img, emb, sim_emb) -> float:
        W     = (0.55, 0.10, 0.15, 0.20)
        ratio = hue = hist = None
        if img is not None:
            try:
                hh, ww = img.shape[:2]
                ratio  = ww/hh
                hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hue    = float(np.mean(hsv[:,:,0]))
                gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h_raw  = cv2.calcHist(
                    [gray],[0],None,[8],[0,256]).flatten()
                s      = h_raw.sum()
                hist   = h_raw/s if s > 0 else h_raw
            except Exception:
                pass
        sr = (max(0., 1.-abs(ratio-self._ref_ratio)/0.4)
              if ratio and self._ref_ratio else 0.5)
        sh = 0.5
        if hue is not None and self._ref_hue is not None:
            d  = min(abs(hue-self._ref_hue),
                     180-abs(hue-self._ref_hue))
            sh = max(0., 1.-d/30.)
        shi = 0.5
        if hist is not None and self._ref_hist is not None:
            try:
                shi = float(cv2.compareHist(
                    hist.astype(np.float32).reshape(-1,1),
                    self._ref_hist.astype(np.float32).reshape(-1,1),
                    cv2.HISTCMP_CORREL))
            except Exception:
                pass
        return W[0]*sim_emb + W[1]*sr + W[2]*sh + W[3]*shi

    def update(self, face_img=None, embedding=None):
        """
        Mise à jour référence avec alpha adaptatif.
        sim >= 0.80 → alpha 0.15 (mise à jour forte)
        sim >= 0.65 → alpha 0.08 (mise à jour moyenne)
        sim < 0.65  → alpha 0.03 (mise à jour douce)

        ← OPTIMISATION PERF : utilise _arcface_embed_fast (320x320) au
        lieu de _arcface_embed (640x640, jusqu'à 6 appels ArcFace en
        pire cas). Cette méthode est appelée à CHAQUE frame acceptée
        du flux temps réel, juste après que verify() ait DÉJÀ extrait
        un embedding via la version rapide pour la même frame — refaire
        une extraction complète en haute précision ici double le coût
        CPU d'ArcFace pour rien : une mise à jour progressive de
        moyenne glissante n'a pas besoin de la précision maximale,
        contrairement à la mémorisation initiale (add_frame), qui elle
        garde volontairement la version lente/précise.
        """
        if self._ref is None or self._memorizing:
            return
        emb = None
        if self._use_arc and face_img is not None:
            emb = self._arcface_embed_fast(face_img)
        elif not self._use_arc and embedding is not None:
            emb = self._norm(embedding)
        if emb is None:
            self._pending_updates = []
            return

        sim = float(np.dot(self._ref, emb))
        if sim > CFG.UPDATE_STRICT:  # 0.55
            self._pending_updates.append(emb)
            if len(self._pending_updates) < CFG.UPDATE_CONFIRM_FRAMES:
                print(f"[Identity] Update en attente "
                      f"({len(self._pending_updates)}"
                      f"/{CFG.UPDATE_CONFIRM_FRAMES}) "
                      f"sim={sim:.3f}")
                return

            stacked      = np.stack(self._pending_updates, axis=0)
            mean_pending = self._norm(np.mean(stacked, axis=0))
            coherence    = np.mean([float(np.dot(mean_pending, e))
                                    for e in self._pending_updates])
            self._pending_updates = []

            if coherence < 0.52:
                print(f"[Identity] Update REJETÉE — "
                      f"incohérence={coherence:.2f}")
                return

            # Alpha adaptatif selon similarité
            if sim >= 0.80:
                alpha = 0.15
            elif sim >= 0.65:
                alpha = 0.08
            else:
                alpha = 0.03

            new  = (1 - alpha) * self._ref + alpha * mean_pending
            norm = np.linalg.norm(new)
            if norm > 0:
                self._ref                = new / norm
                self.candidate_embedding = self._ref
                print(f"[Identity] ✅ Référence mise à jour "
                      f"alpha={alpha} sim={sim:.3f} "
                      f"cohérence={coherence:.2f}")
        else:
            self._pending_updates = []

    def reset(self):
        self._embeddings      = []
        self._ref             = None
        self._memorized       = False
        self._memorizing      = False
        self.candidate_embedding = None
        self._pending_updates = []
        print("[Identity] Reset complet")

    # ← AJOUT : export/import de la référence mémorisée, pour la
    # persistance inter-répliques (voir app_embedding_cache.
    # store_tracking_state/get_tracking_state). export_ref() est
    # appelée après une mémorisation réussie et périodiquement après
    # chaque update() ; import_ref() est appelée à la connexion, avant
    # de lancer une mémorisation complète, pour vérifier si un état
    # déjà utilisable existe.
    def export_ref(self):
        """Retourne la référence mémorisée (ou None si pas encore prête)."""
        if self._ref is None or self._memorizing:
            return None
        return self._ref.copy()

    def import_ref(self, ref) -> bool:
        """
        Restaure une référence déjà mémorisée (venant d'une autre
        réplique), sans refaire la mémorisation initiale. Retourne
        True si la restauration a réussi.
        """
        if ref is None:
            return False
        try:
            self._ref = np.asarray(ref, dtype=np.float32)
            self._memorized  = True
            self._memorizing = False
            self.candidate_embedding = self._ref
            self._embeddings = [self._ref]
            print(f"[Identity] ✅ Référence restaurée depuis l'état partagé "
                  f"(dim={self._ref.shape[0]}D) — mémorisation ignorée")
            return True
        except Exception as e:
            print(f"[Identity] ⚠️ Échec restauration référence: {e}")
            return False

    @property
    def memorized(self) -> bool:
        return self._memorized

    @property
    def memorizing(self) -> bool:
        return self._memorizing

    @property
    def progress(self) -> int:
        return len(self._embeddings)

    @property
    def use_arcface(self) -> bool:
        return self._use_arc

    @property
    def is_memorized(self) -> bool:
        return self._memorized

    @property
    def is_memorizing(self) -> bool:
        return self._memorizing

    @property
    def memorization_progress(self) -> int:
        return len(self._embeddings)

    @property
    def _n_frames_target(self) -> int:
        return CFG.MEMORIZE_FRAMES


class TrackingManager:
    def __init__(self, max_age=300, n_init=3,
                 nn_budget=100, shared_arcface=None, fast_arcface=None):
        self.tracker  = DeepSort(max_age=max_age, n_init=n_init,
                                  nn_budget=nn_budget,
                                  embedder="mobilenet",
                                  half=False, bgr=True)
        self.identity = IdentityManager(shared_arcface=shared_arcface,
                                        fast_arcface=fast_arcface)
        self.zone     = ZoneManager()
        self.speed    = SpeedFilter()
        self.kalman   = KalmanPredictor()
        self._state   = State.SEARCHING
        self._track_id: Optional[int] = None
        self.last_bbox: Optional[list] = None
        self._last_w      = 0.0
        self._last_h      = 0.0
        self.lost_count   = 0
        self._id_rejected = False
        self.memo_fails   = 0
        self._initialized = False

    @property
    def selected_track_id(self) -> Optional[int]:
        return self._track_id

    @selected_track_id.setter
    def selected_track_id(self, v):
        self._track_id = v

    @property
    def initialized(self) -> bool:
        return self._initialized

    @initialized.setter
    def initialized(self, v):
        self._initialized = v

    @property
    def track_id(self) -> Optional[int]:
        return self._track_id

    @track_id.setter
    def track_id(self, v):
        self._track_id = v

    @property
    def state(self) -> State:
        return self._state

    def set_id_rejected(self, v: bool):
        self._id_rejected = v

    def is_memorizing(self) -> bool:
        return self._state == State.MEMORIZING

    def is_lost(self) -> bool:
        return self._state in (State.LOST,
                               State.REIDENTIFICATION,
                               State.ABANDONED)

    def is_tracking_lost(self) -> bool:
        return self.is_lost()

    def get_bbox(self) -> Optional[list]:
        if self._state in (State.SEARCHING, State.ABANDONED):
            return None
        return self.last_bbox

    def get_selected_bbox(self) -> Optional[list]:
        return self.get_bbox()

    def select(self, track_id: int, bbox=None):
        self._track_id    = track_id
        self.lost_count   = 0
        self._state       = State.MEMORIZING
        self._id_rejected = False
        self.memo_fails   = 0
        self._initialized = True
        self.identity.start()
        if bbox:
            self._save_bbox(bbox)
            x1, y1, x2, y2 = bbox[:4]
            self.kalman.init((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)
        print(f"[Tracker] Verrouillage track_id={track_id}")

    def force_track(self, track_id: int, bbox: list):
        self._track_id    = track_id
        self._state       = State.TRACKING
        self.lost_count   = 0
        self._id_rejected = False
        self.memo_fails   = 0
        self._initialized = True
        self._save_bbox(bbox)
        x1, y1, x2, y2 = bbox[:4]
        cx, cy = (x1+x2)/2, (y1+y2)/2
        self.kalman.update(cx, cy, x2-x1, y2-y1)
        print(f"[Tracker] Force track_id={track_id} (ID conservée)")

    def memorize_frame(self, face_img, embedding=None) -> bool:
        if self._state != State.MEMORIZING:
            return True
        if self.identity.use_arcface:
            done = self.identity.add_frame(face_img=face_img)
        else:
            done = self.identity.add_frame(face_img=face_img,
                                           embedding=embedding)
        if done:
            self._state = State.TRACKING
            self.speed.memorize(face_img)
            print("[Tracker] ✅ Mémorisation → TRACKING")
        return done

    def set_candidate_embedding(self, embedding, face_img=None):
        if self.identity.use_arcface:
            if face_img is None:
                return
            arc_emb = self.identity._arcface_embed(face_img)
            if arc_emb is None:
                return
            self.identity._ref = arc_emb
            self.identity._visual_ref(face_img)
        else:
            emb = self.identity._norm(embedding)
            self.identity._ref = emb
            if face_img is not None:
                self.identity._visual_ref(face_img)
        self.identity._memorized       = True
        self.identity._memorizing      = False
        self.identity.candidate_embedding = self.identity._ref
        if face_img is not None:
            self.speed.memorize(face_img)
        print("[Tracker] ✅ Embedding candidat défini (legacy)")

    def set_reference_embedding(self, embedding: np.ndarray,
                                 face_img=None) -> None:
        self.identity.set_reference_embedding(embedding, face_img=face_img)
        if face_img is not None:
            self.speed.memorize(face_img)
        self._state       = State.TRACKING
        self._initialized = True
        print("[Tracker] ✅ Référence pré-chargée depuis cache "
              "— TRACKING direct")

    def verify_candidate(self, face_img,
                         embedding=None,
                         strict_global: bool = False,
                         threshold: float = None) -> Tuple[bool, float]:
        if self.identity.use_arcface:
            return self.identity.verify(
                face_img=face_img,
                strict_global=strict_global,
                threshold=threshold)
        return self.identity.verify(
            face_img=face_img,
            embedding=embedding,
            strict_global=strict_global,
            threshold=threshold)

    def update_candidate_embedding(self, face_img, embedding=None):
        self.identity.update(face_img=face_img, embedding=embedding)

    @property
    def candidate_embedding(self):
        return self.identity.candidate_embedding

    def update(self, frame, detections) -> list:
        if not detections:
            tracks = self.tracker.update_tracks([], frame=frame)
        else:
            dets   = [([x1, y1, x2-x1, y2-y1], c, None)
                      for x1, y1, x2, y2, c in detections]
            tracks = self.tracker.update_tracks(dets, frame=frame)

        if self._state == State.SEARCHING:
            return tracks

        if self._state == State.MEMORIZING:
            track_found = False
            for t in tracks:
                if t.track_id == self._track_id:
                    bbox = t.to_tlbr()
                    x1, y1, x2, y2 = bbox
                    self._save_bbox([x1, y1, x2, y2])
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    self.kalman.update(cx, cy, x2-x1, y2-y1)
                    self._last_w = x2-x1
                    self._last_h = y2-y1
                    track_found  = True
                    break
            if not track_found and self.last_bbox is not None:
                pred = self.kalman.predict()
                if pred:
                    cx, cy, w, h = pred
                    self._save_bbox([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
            return tracks

        found = next((t for t in tracks
                      if t.is_confirmed() and
                      t.track_id == self._track_id), None)
        if found:
            self._on_found(found)
        else:
            self._on_lost()
        return tracks

    def _on_found(self, track):
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        self.kalman.update(cx, cy, x2-x1, y2-y1)
        self._save_bbox([x1, y1, x2, y2])
        self._last_w      = x2-x1
        self._last_h      = y2-y1
        self.lost_count   = 0
        self._id_rejected = False
        if self._state != State.MEMORIZING:
            self._state = State.TRACKING

    def _on_lost(self):
        """
        Version améliorée — MAX_LOST 45 frames (était 30).
        Tolère mieux les absences courtes (candidat sort puis rentre).

        ← PATCH v6.2 : la transition vers ABANDONED ne loggue plus le
        warning qu'une seule fois (avant : répété à chaque frame tant
        que lost_count continuait d'augmenter, ce qui produisait des
        centaines de lignes identiques sur une longue vidéo). De plus,
        last_bbox n'est plus effacé lors de cette transition — le
        mettre à None rendait structurellement impossible toute
        récupération ultérieure du candidat, la logique de recherche
        globale de secours (dans ws_analyze_realtime / analyze_video)
        exigeant last_bbox is not None pour se déclencher.
        """
        if self._id_rejected:
            return

        self.lost_count += 1
        pred = self.kalman.predict()

        if self.lost_count <= CFG.MAX_LOST:   # 60 frames
            if pred and self.last_bbox:
                cx, cy, w, h = pred
                if self._last_w:
                    w = 0.90 * self._last_w + 0.10 * w
                    h = 0.90 * self._last_h + 0.10 * h
                self._save_bbox([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
            self._state = State.LOST

        elif self.lost_count <= CFG.MAX_REID:  # 150 frames
            if pred and self.last_bbox:
                cx, cy, w, h = pred
                self._save_bbox([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
            self._state = State.REIDENTIFICATION

        else:
            # ← FIX (1/2) : log une seule fois, lors de la transition
            # (pas à chaque frame suivante tant que ABANDONED persiste).
            if self._state != State.ABANDONED:
                logger.warning("[Tracker] ABANDONED après %d frames perdues",
                               self.lost_count)
            self._state = State.ABANDONED
            # ← FIX (2/2) : ne PLUS effacer last_bbox ici. La dernière
            # position connue doit rester disponible pour permettre à
            # une recherche globale de secours de reverrouiller le
            # candidat plus tard dans la vidéo, si les visages
            # redeviennent visibles.

    def reset_tracking(self):
        self.lost_count   = 0
        self._state       = State.SEARCHING
        self.last_bbox    = None
        self._id_rejected = False
        self.memo_fails   = 0
        print("[Tracker] Reset tracking (identité conservée)")

    def reset(self):
        self.identity.reset()
        self.kalman       = KalmanPredictor()
        self._state       = State.SEARCHING
        self._track_id    = None
        self._initialized = False
        self.last_bbox    = None
        self._last_w      = 0.0
        self._last_h      = 0.0
        self.lost_count   = 0
        self._id_rejected = False
        self.memo_fails   = 0
        self.zone.reset()
        self.speed.reset()

    def _save_bbox(self, bbox):
        if bbox is None:
            return
        x1, y1, x2, y2 = map(float, bbox[:4])
        if x2 > x1 and y2 > y1:
            self.last_bbox = [x1, y1, x2, y2]