"""
video_detector.py
-----------------
Core video processing pipeline.
Reads a video file frame by frame, runs YOLOv8 inference,
tracks detections across frames, draws annotations,
and writes the processed video to disk.
"""

import time
import cv2
import numpy as np

from services.tracker import CentroidTracker
from services.video_utils import classify_severity, draw_box

CONFIDENCE_THRESHOLD = 0.5
OUTPUT_VIDEO_PATH    = "processed_video.mp4"


def process_video(video_path: str, model) -> dict:
    """
    Process a video file end-to-end.

    Args:
        video_path: Path to the input video file.
        model:      Loaded YOLOv8 model (shared from app startup).

    Returns:
        summary dict with metadata, per-frame results, and output path.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # ── Video properties ───────────────────────────────────────────────────
    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Output writer ──────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    # ── Tracker ────────────────────────────────────────────────────────────
    tracker = CentroidTracker(max_disappeared=15, iou_threshold=0.3)

    # ── Accumulators ───────────────────────────────────────────────────────
    all_frame_data   = []
    all_confidences  = []
    unique_track_ids = set()
    frame_idx        = 0
    start_time       = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # ── Inference ─────────────────────────────────────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = model.predict(source=frame_rgb,
                                  conf=CONFIDENCE_THRESHOLD,
                                  verbose=False)

        # ── Extract raw detections ────────────────────────────────────────
        raw_detections = []   # [x, y, w, h]
        raw_meta       = []   # {class_name, confidence, severity}

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                w = x2 - x1
                h = y2 - y1
                conf      = float(round(box.conf[0].item(), 4))
                class_id  = int(box.cls[0].item())
                cls_name  = model.names[class_id]
                severity  = classify_severity(w, h)

                raw_detections.append([x1, y1, w, h])
                raw_meta.append({
                    "class_name": cls_name,
                    "confidence": conf,
                    "severity":   severity,
                })

        # ── Update tracker ────────────────────────────────────────────────
        tracked = tracker.update(raw_detections)
        # tracked: {track_id: [x, y, w, h]}

        # Build a mapping: bbox tuple → track_id for annotation
        bbox_to_id = {tuple(bbox): tid for tid, bbox in tracked.items()}

        # ── Annotate frame + collect metadata ──────────────────────────────
        frame_defects = []

        for i, (bbox, meta) in enumerate(zip(raw_detections, raw_meta)):
            # Find matching track ID (best IoU match among tracked objects)
            track_id = _find_track_id(bbox, tracked)

            if track_id is not None:
                unique_track_ids.add(track_id)

            tid = track_id or (i + 1)

            draw_box(
                frame,
                track_id=tid,
                class_name=meta["class_name"],
                confidence=meta["confidence"],
                bbox=bbox,
                severity=meta["severity"],
            )

            all_confidences.append(meta["confidence"])

            frame_defects.append({
                "id":         tid,
                "type":       meta["class_name"],
                "confidence": meta["confidence"],
                "severity":   meta["severity"],
                "bbox":       bbox,
            })

        # ── Write annotated frame ─────────────────────────────────────────
        writer.write(frame)

        # ── Store frame metadata ──────────────────────────────────────────
        all_frame_data.append({
            "frame":         frame_idx,
            "timestamp":     round(timestamp, 3),
            "total_defects": len(frame_defects),
            "defects":       frame_defects,
        })

        frame_idx += 1

    # ── Cleanup ────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    processing_time = round(time.time() - start_time, 2)

    avg_conf = (round(sum(all_confidences) / len(all_confidences), 4)
                if all_confidences else 0.0)

    return {
        "summary": {
            "total_frames":        total_frames,
            "frames_processed":    frame_idx,
            "total_unique_defects": len(unique_track_ids),
            "average_confidence":  avg_conf,
            "processing_time_sec": processing_time,
            "output_video":        OUTPUT_VIDEO_PATH,
            "fps":                 round(fps, 2),
            "resolution":          f"{width}x{height}",
        },
        "frames": all_frame_data,
    }


# ── Internal helper ────────────────────────────────────────────────────────────

def _find_track_id(bbox, tracked: dict):
    """Return the track ID whose bbox has the highest IoU with `bbox`."""
    from services.tracker import _iou
    best_id  = None
    best_iou = 0.0
    for tid, tbbox in tracked.items():
        score = _iou(bbox, tbbox)
        if score > best_iou:
            best_iou = score
            best_id  = tid
    return best_id if best_iou > 0.1 else None
