import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize DeepSORT tracker with cosine metric
tracker = DeepSort(max_cosine_distance=0.3, max_age=30, n_init=3)


# Initialize YOLOv8 model for person detection
# Assuming the model weights are available via ultralytics hub (default 'yolov8n.pt')
model = YOLO('yolov8n.pt')  # You can change to a larger model if needed

def process_video(video_path: str, conf_thresh: float = 0.3):
    """Detect persons in a video and assign persistent IDs using DeepSORT.

    Args:
        video_path: Path to the video file.
        conf_thresh: Confidence threshold for YOLO detections.

    Returns:
        A dict mapping person_id -> {'frames': int, 'bbox': list of last bbox}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    person_stats = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        # YOLO inference
        results = model(frame, verbose=False)[0]
        # Filter for person class (class id 0 in COCO)
        detections = []
        for r in results.boxes:
            cls = int(r.cls)
            if cls != 0:  # skip non‑person
                continue
            conf = float(r.conf)
            if conf < conf_thresh:
                continue
            xyxy = r.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, 0))
        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            # Update stats
            if track_id not in person_stats:
                person_stats[track_id] = {"frames": 0, "bbox": ltrb}
            person_stats[track_id]["frames"] += 1
            person_stats[track_id]["bbox"] = ltrb
    cap.release()
    return person_stats
