# face_analyzer.py — Version Professionnelle
# MediaPipe Tasks API (0.10+) + Fallback OpenCV
# Compatible Python 3.14 / Windows / CPU
#
# ═══════════════════════════════════════════════════════════════
# INSTALLATION — Télécharger les modèles (une seule fois) :
#
# 1. Créer le dossier : models/mediapipe/
#
# 2. Télécharger face_landmarker.task (3MB) :
#    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
#
# 3. Télécharger pose_landmarker_lite.task (4MB) :
#    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
#
# Si les modèles sont absents → fallback OpenCV automatique ✅
# ═══════════════════════════════════════════════════════════════
#
# ═══════════════════════════════════════════════════════════════
# PATCH v1.1 — logging des exceptions réelles avant fallback
# Avant : chaque méthode (_gaze, _blink, _tension, _symmetry, _posture)
# avalait silencieusement TOUTE exception et retournait une valeur
# neutre par défaut (souvent 0.5). Un vrai bug (ex: MediaPipe change
# la numérotation de ses landmarks, division par zéro imprévue) était
# alors indiscernable d'un simple manque de données — les métriques
# se figeaient à une valeur neutre sans jamais signaler pourquoi.
# Le comportement de résilience (toujours retourner une valeur par
# défaut, ne jamais faire planter l'analyse) reste inchangé — seule
# la visibilité du problème réel change.
# ═══════════════════════════════════════════════════════════════

import cv2
import numpy as np
import os
from typing import Optional, Dict

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core import base_options as mp_base
    _MP_TASKS_OK = True
except ImportError:
    _MP_TASKS_OK = False

FACE_MODEL_PATH = os.path.join("models", "mediapipe", "face_landmarker.task")
POSE_MODEL_PATH = os.path.join("models", "mediapipe", "pose_landmarker_lite.task")


class FaceAnalyzer:
    """
    Analyse comportementale complète du visage.

    MODE PROFESSIONNEL (modèles .task présents) :
    ✅ 478 points faciaux
    ✅ Gaze réel (direction regard, contact visuel)
    ✅ Clignement précis (EAR + blendshapes)
    ✅ Tension sourcils
    ✅ Posture épaules
    ✅ Symétrie visage

    MODE FALLBACK (sans modèles) :
    ✅ Qualité frame (brightness/blur)
    ✅ Mouvement tête (optical flow)
    ✅ Stabilité (Haar cascade)
    ✅ Clignement approximatif (Haar yeux)
    """

    def __init__(self):
        self._mode      = "fallback"
        self._face_lm   = None
        self._pose_lm   = None
        self._prev_gray = None
        self.enabled    = True

        self._gaze_history       = []
        self._blink_history      = []
        self._tension_history    = []
        self._posture_history    = []
        self._movement_history   = []
        self._stability_history  = []
        self._brightness_history = []

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')

        self._try_load_tasks()

    def _try_load_tasks(self):
        if not _MP_TASKS_OK:
            print("[FaceAnalyzer] 📦 Fallback OpenCV actif")
            return

        face_ok = os.path.exists(FACE_MODEL_PATH)
        pose_ok = os.path.exists(POSE_MODEL_PATH)

        if not face_ok:
            print(f"[FaceAnalyzer] ⚠️ face_landmarker.task absent")
            print(f"  → Placer dans : {FACE_MODEL_PATH}")
            print("  → URL : https://storage.googleapis.com/mediapipe-models/"
                  "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

        if not pose_ok:
            print(f"[FaceAnalyzer] ⚠️ pose_landmarker_lite.task absent")
            print(f"  → Placer dans : {POSE_MODEL_PATH}")
            print("  → URL : https://storage.googleapis.com/mediapipe-models/"
                  "pose_landmarker/pose_landmarker_lite/float16/1/"
                  "pose_landmarker_lite.task")

        if face_ok:
            try:
                opts = mp_vision.FaceLandmarkerOptions(
                    base_options=mp_base.BaseOptions(
                        model_asset_path=FACE_MODEL_PATH),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False,
                )
                self._face_lm = mp_vision.FaceLandmarker.create_from_options(opts)
                print("[FaceAnalyzer] ✅ FaceLandmarker chargé (478 points)")
            except Exception as e:
                print(f"[FaceAnalyzer] ❌ FaceLandmarker: {e}")

        if pose_ok:
            try:
                opts = mp_vision.PoseLandmarkerOptions(
                    base_options=mp_base.BaseOptions(
                        model_asset_path=POSE_MODEL_PATH),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._pose_lm = mp_vision.PoseLandmarker.create_from_options(opts)
                print("[FaceAnalyzer] ✅ PoseLandmarker chargé (posture)")
            except Exception as e:
                print(f"[FaceAnalyzer] ❌ PoseLandmarker: {e}")

        if self._face_lm is not None:
            self._mode = "professional"
            print("[FaceAnalyzer] 🎯 Mode PROFESSIONNEL actif")
        else:
            print("[FaceAnalyzer] 📦 Mode FALLBACK OpenCV actif")

    # ════════════════════════════════════════════════════════════
    # MÉTHODE PRINCIPALE
    # ════════════════════════════════════════════════════════════
    def analyze(self, frame_bgr) -> Optional[Dict]:
        if not self.enabled or frame_bgr is None or frame_bgr.size == 0:
            return None
        try:
            if self._mode == "professional":
                return self._analyze_pro(frame_bgr)
            return self._analyze_fallback(frame_bgr)
        except Exception as e:
            print(f"[FaceAnalyzer] Erreur: {e}")
            return self._analyze_fallback(frame_bgr)

    # ════════════════════════════════════════════════════════════
    # MODE PROFESSIONNEL
    # ════════════════════════════════════════════════════════════
    def _analyze_pro(self, frame_bgr) -> Dict:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        face_result = self._face_lm.detect(mp_image)

        if not face_result.face_landmarks:
            return self._analyze_fallback(frame_bgr)

        lm = face_result.face_landmarks[0]
        H, W = frame_bgr.shape[:2]

        result = {
            'gaze':     self._gaze(lm),
            'blink':    self._blink(lm, face_result.face_blendshapes),
            'tension':  self._tension(lm),
            'symmetry': self._symmetry(lm),
            'quality':  self._quality(frame_bgr),
            'posture':  self._posture(mp_image)
                        if self._pose_lm else
                        {'droite': None, 'posture_ratio': 0.5, 'score': 0.5},
        }
        result['comfort_score'] = self._comfort_pro(result)
        result['mode'] = 'professional'
        return result

    def _gaze(self, lm) -> Dict:
        try:
            left_ratio = (
                (lm[468].x - lm[33].x) /
                max(0.001, abs(lm[133].x - lm[33].x))
            )
            right_ratio = (
                (lm[473].x - lm[263].x) /
                max(0.001, abs(lm[362].x - lm[263].x))
            )
            gr = float((left_ratio + right_ratio) / 2)

            if gr < 0.35:   direction = 'gauche'
            elif gr > 0.65: direction = 'droite'
            else:           direction = 'caméra'

            vr = (lm[468].y - lm[159].y) / max(0.001, abs(lm[145].y - lm[159].y))
            if float(vr) > 0.7:
                direction = 'bas'

            contact = bool(direction == 'caméra')
            self._gaze_history.append(1 if contact else 0)
            if len(self._gaze_history) > 30:
                self._gaze_history.pop(0)
            cr = float(np.mean(self._gaze_history))

            return {
                'contact_visuel': contact,
                'direction':      direction,
                'ratio':          round(gr, 3),
                'contact_ratio':  round(cr, 2),
                'score':          cr
            }
        except Exception as e:
            # ← PATCH v1.1 : logguer la vraie cause avant de retomber
            # sur le défaut neutre, pour ne plus confondre un bug
            # silencieux avec un simple manque de données.
            print(f"[FaceAnalyzer] ⚠️ erreur _gaze: {e}")
            return {'contact_visuel': None, 'direction': 'inconnu',
                    'contact_ratio': 0.5, 'score': 0.5}

    def _blink(self, lm, blendshapes) -> Dict:
        try:
            blink = False
            rate  = 15.0

            if blendshapes and len(blendshapes) > 0:
                bs  = {b.category_name: b.score for b in blendshapes[0]}
                val = float((bs.get('eyeBlinkLeft', 0) +
                             bs.get('eyeBlinkRight', 0)) / 2)
                blink = bool(val > 0.4)
            else:
                ear = abs(lm[159].y - lm[145].y) / \
                      max(0.001, abs(lm[33].x - lm[133].x))
                blink = bool(float(ear) < 0.20)

            self._blink_history.append(1 if blink else 0)
            if len(self._blink_history) > 300:
                self._blink_history.pop(0)
            rate = float(sum(self._blink_history) *
                         60 / max(1, len(self._blink_history)))

            if 12 <= rate <= 22:  score = 1.0
            elif rate > 30:       score = 0.4
            elif rate < 8:        score = 0.6
            else:                 score = 0.75

            return {
                'blink':     bool(blink),
                'eyes_open': bool(not blink),
                'rate':      round(rate, 1),
                'score':     float(score)
            }
        except Exception as e:
            print(f"[FaceAnalyzer] ⚠️ erreur _blink: {e}")
            return {'blink': False, 'eyes_open': True,
                    'rate': 15.0, 'score': 0.5}

    def _tension(self, lm) -> Dict:
        try:
            brow_dist   = float(abs(lm[65].x - lm[295].x))
            brow_height = float(abs(lm[70].y - lm[159].y))
            tension     = float(
                max(0, 1 - brow_dist * 8) * 0.5 +
                max(0, 1 - brow_height * 12) * 0.5
            )
            self._tension_history.append(tension)
            if len(self._tension_history) > 30:
                self._tension_history.pop(0)
            avg = float(np.mean(self._tension_history))
            return {'tension': round(tension, 3),
                    'avg_tension': round(avg, 3),
                    'score': float(1 - avg)}
        except Exception as e:
            print(f"[FaceAnalyzer] ⚠️ erreur _tension: {e}")
            return {'tension': 0.0, 'avg_tension': 0.0, 'score': 0.5}

    def _symmetry(self, lm) -> Dict:
        try:
            eye_cx   = float((lm[33].x + lm[263].x) / 2)
            nose_off = float(abs(lm[4].x - eye_cx))
            mdiff    = float(abs(lm[61].y - lm[291].y))
            sym      = float(1 - min(1.0, nose_off * 5 + mdiff * 10))
            return {'symmetry': round(sym, 3),
                    'mouth_diff': round(mdiff, 3),
                    'score': sym}
        except Exception as e:
            print(f"[FaceAnalyzer] ⚠️ erreur _symmetry: {e}")
            return {'symmetry': 1.0, 'mouth_diff': 0.0, 'score': 0.5}

    def _posture(self, mp_image) -> Dict:
        try:
            pr = self._pose_lm.detect(mp_image)
            if not pr.pose_landmarks:
                return {'droite': None, 'posture_ratio': 0.5, 'score': 0.5}

            lm = pr.pose_landmarks[0]
            # ← FIX : mp.tasks.vision.PoseLandmark n'existe pas dans la
            # nouvelle API MediaPipe Tasks (contrairement à l'ancienne
            # mp.solutions.pose.PoseLandmark) — provoquait une erreur à
            # CHAQUE frame ("has no attribute 'PoseLandmark'"), rendant
            # l'analyse de posture silencieusement indisponible en
            # continu. Les landmarks de pose sont une simple liste
            # indexée par position ; 11=épaule gauche, 12=épaule droite
            # sont les indices standards BlazePose, stables et
            # documentés indépendamment de la version de l'API.
            sh_l = lm[11]
            sh_r = lm[12]

            diff   = float(abs(sh_l.y - sh_r.y))
            droite = bool(diff < 0.05)
            conf   = float(max(0, 1 - (sh_l.y + sh_r.y) / 2))

            self._posture_history.append(1 if droite else 0)
            if len(self._posture_history) > 30:
                self._posture_history.pop(0)
            pr_ratio = float(np.mean(self._posture_history))

            return {
                'droite':        droite,
                'shoulder_diff': round(diff, 3),
                'posture_ratio': round(pr_ratio, 2),
                'score':         float(pr_ratio * 0.6 + conf * 0.4)
            }
        except Exception as e:
            print(f"[FaceAnalyzer] ⚠️ erreur _posture: {e}")
            return {'droite': None, 'posture_ratio': 0.5, 'score': 0.5}

    def _quality(self, frame_bgr) -> Dict:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        b    = float(np.mean(gray))
        bv   = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        self._brightness_history.append(b)
        if len(self._brightness_history) > 30:
            self._brightness_history.pop(0)

        bs = 1.0 if 60 <= b <= 200 else (b/60 if b < 60 else
             max(0.3, 1-(b-200)/100))
        ls = min(1.0, bv/100) if bv < 200 else 1.0

        st = ('Sombre' if b < 45 else 'Exposé' if b > 220
              else 'Flou' if bv < 50 else 'Optimal')

        return {
            'brightness': round(b, 1),
            'blur_val':   round(bv, 1),
            'status':     str(st),
            'score':      float(bs*0.5 + ls*0.5)
        }

    def _comfort_pro(self, a) -> float:
        w = {'gaze': 0.30, 'blink': 0.12, 'tension': 0.18,
             'posture': 0.25, 'symmetry': 0.08, 'quality': 0.07}
        s = tw = 0.0
        for k, ww in w.items():
            if k in a and a[k].get('score') is not None:
                s  += float(a[k]['score']) * ww
                tw += ww
        return float(np.clip(s/tw*100, 0, 100) if tw else 50)

    # ════════════════════════════════════════════════════════════
    # MODE FALLBACK
    # ════════════════════════════════════════════════════════════
    def _analyze_fallback(self, frame_bgr) -> Dict:
        gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        result = {
            'quality':   self._quality(frame_bgr),
            'movement':  self._movement(gray),
            'stability': self._stability(gray),
            'blink':     self._blink_cv(gray),
        }
        result['comfort_score'] = self._comfort_fallback(result)
        result['mode'] = 'fallback'
        return result

    def _movement(self, gray) -> Dict:
        mv = 0.0
        if self._prev_gray is not None:
            try:
                pts = cv2.goodFeaturesToTrack(
                    self._prev_gray, maxCorners=50,
                    qualityLevel=0.3, minDistance=7, blockSize=7)
                if pts is not None and len(pts) > 5:
                    cp, st, _ = cv2.calcOpticalFlowPyrLK(
                        self._prev_gray, gray, pts, None)
                    if cp is not None:
                        gp = pts[st == 1]
                        gc = cp[st == 1]
                        if len(gp) > 0:
                            mv = float(np.mean(
                                np.linalg.norm(gc - gp, axis=1)))
            except Exception as e:
                print(f"[FaceAnalyzer] ⚠️ erreur _movement (optical flow): {e}")

        self._prev_gray = gray.copy()
        self._movement_history.append(mv)
        if len(self._movement_history) > 20:
            self._movement_history.pop(0)

        avg = float(np.mean(self._movement_history))
        sc  = (1.0 if avg < 2 else 0.8 if avg < 5
               else 0.6 if avg < 10 else 0.3)

        return {
            'movement':     round(mv, 2),
            'avg_movement': round(avg, 2),
            'agitation':    bool(avg > 8),
            'score':        float(sc)
        }

    def _stability(self, gray) -> Dict:
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return {'face_detected': False, 'centered': None,
                    'face_size': 0.0, 'stability_ratio': 0.5, 'score': 0.5}

        H, W = gray.shape[:2]
        fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        cx = float(fx + fw/2)
        cy = float(fy + fh/2)
        cdx = abs(cx - W/2) / (W/2)
        cdy = abs(cy - H/2) / (H/2)
        cen = bool(cdx < 0.3 and cdy < 0.3)
        fr  = float((fw*fh)/(W*H))

        self._stability_history.append(1 if cen else 0)
        if len(self._stability_history) > 20:
            self._stability_history.pop(0)
        sr  = float(np.mean(self._stability_history))
        sc  = float(max(0, 1-cdx-cdy)*0.6 + min(1.0,fr*10)*0.2 + sr*0.2)

        return {
            'face_detected':   True,
            'centered':        cen,
            'face_size':       round(fr, 3),
            'stability_ratio': round(sr, 2),
            'score':           sc
        }

    def _blink_cv(self, gray) -> Dict:
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        ok = False
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
            eyes = self.eye_cascade.detectMultiScale(gray[fy:fy+fh, fx:fx+fw])
            ok   = bool(len(eyes) >= 2)
        return {'eyes_open': bool(ok), 'blink': bool(not ok),
                'score': 1.0 if ok else 0.5}

    def _comfort_fallback(self, a) -> float:
        w = {'quality': 0.30, 'movement': 0.30,
             'stability': 0.25, 'blink': 0.15}
        s = tw = 0.0
        for k, ww in w.items():
            if k in a and a[k].get('score') is not None:
                s  += float(a[k]['score']) * ww
                tw += ww
        return float(np.clip(s/tw*100, 0, 100) if tw else 50)

    # ════════════════════════════════════════════════════════════
    # INTERFACES UNIFIÉES pour app.py
    # ════════════════════════════════════════════════════════════
    def get_boost_params(self, analysis: Optional[Dict]) -> Dict:
        """
        Extrait les paramètres de boost pour calculate_candidate_metrics.
        Fonctionne en mode pro ET fallback.
        """
        default = {
            'stability_score': 0.5, 'quality_score': 0.5,
            'movement_score':  0.5, 'blink_score':   0.5,
            'gaze_score':      0.5, 'contact_ratio': 0.5,
            'posture_score':   0.5, 'tension_score': 0.5,
        }
        if analysis is None:
            return default

        mode = analysis.get('mode', 'fallback')

        if mode == 'professional':
            return {
                'stability_score': float(analysis.get(
                    'quality', {}).get('score', 0.5)),
                'quality_score':   float(analysis.get(
                    'quality', {}).get('score', 0.5)),
                'movement_score':  0.7,
                'blink_score':     float(analysis.get(
                    'blink', {}).get('score', 0.5)),
                'gaze_score':      float(analysis.get(
                    'gaze', {}).get('score', 0.5)),
                'contact_ratio':   float(analysis.get(
                    'gaze', {}).get('contact_ratio', 0.5)),
                'posture_score':   float(analysis.get(
                    'posture', {}).get('score', 0.5)),
                'tension_score':   float(analysis.get(
                    'tension', {}).get('score', 0.5)),
            }
        else:
            return {
                'stability_score': float(analysis.get(
                    'stability', {}).get('score', 0.5)),
                'quality_score':   float(analysis.get(
                    'quality', {}).get('score', 0.5)),
                'movement_score':  float(analysis.get(
                    'movement', {}).get('score', 0.5)),
                'blink_score':     float(analysis.get(
                    'blink', {}).get('score', 0.5)),
                'gaze_score': 0.5, 'contact_ratio': 0.5,
                'posture_score': 0.5, 'tension_score': 0.5,
            }

    def get_result_dict(self, analysis: Optional[Dict]) -> Dict:
        """
        Retourne dict JSON-serializable pour ws_analyze_realtime.
        Tous les types sont Python natifs — aucune erreur JSON.
        """
        if analysis is None:
            return {}

        mode = analysis.get('mode', 'fallback')
        q    = analysis.get('quality', {})
        d    = {
            'mode':          str(mode),
            'brightness':    float(q.get('brightness', 0)),
            'blur':          float(q.get('blur_val', 0)),
            'status':        str(q.get('status', 'Inconnu')),
            'comfort_score': float(analysis.get('comfort_score', 50)),
        }

        if mode == 'professional':
            g = analysis.get('gaze', {})
            b = analysis.get('blink', {})
            p = analysis.get('posture', {})
            t = analysis.get('tension', {})
            s = analysis.get('symmetry', {})
            d.update({
                'gaze_direction': str(g.get('direction', 'inconnu')),
                'eye_contact':    float(g.get('contact_ratio', 0.5)),
                'blink_rate':     float(b.get('rate', 15)),
                'eyes_open':      bool(b.get('eyes_open', True)),
                'posture_droite': bool(p['droite'])
                                  if p.get('droite') is not None else None,
                'brow_tension':   float(t.get('avg_tension', 0)),
                'symmetry':       float(s.get('symmetry', 1.0)),
            })
        else:
            m  = analysis.get('movement', {})
            st = analysis.get('stability', {})
            b  = analysis.get('blink', {})
            d.update({
                'movement':      float(m.get('avg_movement', 0)),
                'agitation':     bool(m.get('agitation', False)),
                'face_centered': bool(st['centered'])
                                 if st.get('centered') is not None else None,
                'eyes_open':     bool(b.get('eyes_open', True)),
            })

        return d

    def reset(self):
        self._gaze_history       = []
        self._blink_history      = []
        self._tension_history    = []
        self._posture_history    = []
        self._movement_history   = []
        self._stability_history  = []
        self._brightness_history = []
        self._prev_gray          = None
        print("[FaceAnalyzer] Reset historique")