"""
live_detector.py
----------------
Real-time frame processing for live webcam detection.
Reuses existing severity logic and bounding box utilities.
Never loads YOLO — receives the shared model from main.py.
"""

import time
import base64
import cv2
import numpy as np

from services.video_utils import classify_severity, draw_box
from services.tracker import CentroidTracker

CONFIDENCE_THRESHOLD = 0.5
PROCESS_WIDTH        = 640          # resize before inference for performance
tracker              = CentroidTracker(max_disappeared=8, iou_threshold=0.3)
_frame_counter       = 0


def reset_tracker():
    """Call when a new live session starts."""
    global _frame_counter
    tracker.reset()
    _frame_counter = 0


def process_live_frame(frame_bytes: bytes, model) -> dict:
    """
    Decode a raw JPEG frame, run YOLOv8, track detections,
    and return metadata + a base64-encoded annotated JPEG.

    Args:
        frame_bytes: Raw JPEG bytes from the browser.
        model:       Shared YOLOv8 model loaded at startup.

    Returns:
        dict with keys: frame, timestamp, defects, annotated_frame
    """
    global _frame_counter
    _frame_counter += 1
    timestamp = round(time.time(), 3)

    # ── Decode ────────────────────────────────────────────────────────────────
    np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode frame")

    orig_h, orig_w = frame.shape[:2]

    # ── Resize for inference (keep aspect ratio) ──────────────────────────────
    scale  = PROCESS_WIDTH / orig_w
    inf_w  = PROCESS_WIDTH
    inf_h  = int(orig_h * scale)
    small  = cv2.resize(frame, (inf_w, inf_h))
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # ── YOLOv8 inference ──────────────────────────────────────────────────────
    results = model.predict(source=rgb, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # ── Extract detections in ORIGINAL frame coords ───────────────────────────
    raw_bboxes = []
    raw_meta   = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Scale back to original resolution
            ox1 = int(x1 / scale)
            oy1 = int(y1 / scale)
            ox2 = int(x2 / scale)
            oy2 = int(y2 / scale)
            w   = ox2 - ox1
            h   = oy2 - oy1

            conf      = float(round(box.conf[0].item(), 4))
            class_id  = int(box.cls[0].item())
            cls_name  = model.names[class_id]
            severity  = classify_severity(w, h)

            raw_bboxes.append([ox1, oy1, w, h])
            raw_meta.append({
                "class_name": cls_name,
                "confidence": conf,
                "severity":   severity,
            })

    # ── Track ─────────────────────────────────────────────────────────────────
    tracked = tracker.update(raw_bboxes)   # {track_id: [x,y,w,h]}

    # ── Annotate original frame + build metadata ──────────────────────────────
    defects = []
    for i, (bbox, meta) in enumerate(zip(raw_bboxes, raw_meta)):
        track_id = _find_track_id(bbox, tracked) or (i + 1)

        draw_box(
            frame,
            track_id=track_id,
            class_name=meta["class_name"],
            confidence=meta["confidence"],
            bbox=bbox,
            severity=meta["severity"],
        )

        defects.append({
            "id":         track_id,
            "type":       meta["class_name"],
            "confidence": meta["confidence"],
            "severity":   meta["severity"],
            "bbox":       bbox,
        })

    # ── Encode annotated frame as JPEG → base64 ───────────────────────────────
    _, buf      = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64_frame   = base64.b64encode(buf).decode("utf-8")

    return {
        "frame":           _frame_counter,
        "timestamp":       timestamp,
        "defects":         defects,
        "annotated_frame": b64_frame,   # data URI src for <img>
    }


# ── Internal helper ───────────────────────────────────────────────────────────

def _find_track_id(bbox, tracked: dict):
    from services.tracker import _iou
    best_id, best_score = None, 0.0
    for tid, tbbox in tracked.items():
        score = _iou(bbox, tbbox)
        if score > best_score:
            best_score = score
            best_id    = tid
    return best_id if best_score > 0.1 else None
