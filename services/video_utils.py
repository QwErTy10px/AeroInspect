"""
video_utils.py
--------------
Shared utility functions for video processing.
Reuses the severity classification logic from main.py.
"""

import cv2


# ── Severity (same logic as image detection) ─────────────────────────────────

SEVERITY_AREA_THRESHOLD = 10_000   # px² — must match main.py


def classify_severity(width: int, height: int) -> str:
    """Minor if bbox area < threshold, else Severe."""
    return "Minor" if (width * height) < SEVERITY_AREA_THRESHOLD else "Severe"


# ── Severity → red-based colour (BGR for OpenCV) ──────────────────────────────

def severity_color_bgr(severity: str):
    if severity == "Severe":
        return (0, 0, 239)    # #EF0000 — bright red
    if severity == "Moderate":
        return (0, 113, 248)  # #F87100 — medium red (orange-red)
    return (0, 165, 252)      # #FCA500 — light red (salmon)


# ── Draw a single annotated box on a frame ────────────────────────────────────

def draw_box(frame, track_id: int, class_name: str,
             confidence: float, bbox: list, severity: str):
    """
    Draw a bounding box and label pill on `frame` (in-place).
    bbox = [x, y, w, h] in pixel coords.
    """
    x, y, w, h = [int(v) for v in bbox]
    color = severity_color_bgr(severity)

    # Box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Corner ticks
    tick = min(w, h, 16)
    corners = [
        ((x, y + tick), (x, y), (x + tick, y)),
        ((x + w - tick, y), (x + w, y), (x + w, y + tick)),
        ((x, y + h - tick), (x, y + h), (x + tick, y + h)),
        ((x + w - tick, y + h), (x + w, y + h), (x + w, y + h - tick)),
    ]
    for (p1, vertex, p2) in corners:
        cv2.line(frame, p1, vertex, color, 2, cv2.LINE_AA)
        cv2.line(frame, vertex, p2, color, 2, cv2.LINE_AA)

    # Label: "#{id} {class} {conf}% {severity}"
    label = f"#{track_id} {class_name} {confidence*100:.0f}% {severity}"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Pill background — clamp so it doesn't go above frame top
    pill_y = max(y - th - baseline - 6, 0)
    cv2.rectangle(frame, (x, pill_y), (x + tw + 8, pill_y + th + baseline + 4),
                  color, -1)
    cv2.putText(frame, label, (x + 4, pill_y + th + 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame
