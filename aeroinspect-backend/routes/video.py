"""
routes/video.py
---------------
FastAPI router for the video detection endpoint.
Mounts at /detect-video.
Does NOT touch the existing /detect image endpoint.
"""

import os
import json
import tempfile
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from services.video_detector import process_video

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
OUTPUT_VIDEO_PATH         = "processed_video.mp4"
OUTPUT_META_PATH          = "detection_report.json"


@router.post("/detect-video", summary="Detect defects in an uploaded video")
async def detect_video(file: UploadFile = File(...)):
    """
    POST /detect-video

    Accepts a multipart video upload (mp4 / avi / mov / mkv).
    Processes every frame through the YOLOv8 model with object tracking.
    Returns a JSON summary + metadata for all frames.

    The processed video can be downloaded via GET /video/processed.
    The detection report can be downloaded via GET /video/report.
    """
    from main import model   # model loaded once at startup in main.py

    # ── Validate extension ────────────────────────────────────────────────
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. "
                   f"Allowed: {ALLOWED_VIDEO_EXTENSIONS}"
        )

    # ── Save upload to a temp file ────────────────────────────────────────
    try:
        suffix = f".{ext}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as exc:
        logger.error("Failed to save uploaded video: %s", exc)
        raise HTTPException(status_code=400,
                            detail="Failed to read uploaded video file.")

    # ── Process video ─────────────────────────────────────────────────────
    try:
        result = process_video(tmp_path, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Video processing error: %s", exc)
        raise HTTPException(status_code=500,
                            detail=f"Video processing failed: {exc}")
    finally:
        # Always clean up temp upload
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # ── Save detection report to disk ─────────────────────────────────────
    try:
        with open(OUTPUT_META_PATH, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save detection report: %s", exc)

    return JSONResponse(content=result)


@router.get("/video/processed", summary="Download the processed video")
async def download_processed_video():
    """GET /video/processed — streams the last processed video file."""
    if not os.path.exists(OUTPUT_VIDEO_PATH):
        raise HTTPException(status_code=404,
                            detail="No processed video found. "
                                   "Run /detect-video first.")
    return FileResponse(
        OUTPUT_VIDEO_PATH,
        media_type="video/mp4",
        filename="processed_video.mp4",
    )


@router.get("/video/report", summary="Download the detection report (JSON)")
async def download_report():
    """GET /video/report — returns the full per-frame detection metadata."""
    if not os.path.exists(OUTPUT_META_PATH):
        raise HTTPException(status_code=404,
                            detail="No detection report found. "
                                   "Run /detect-video first.")
    return FileResponse(
        OUTPUT_META_PATH,
        media_type="application/json",
        filename="detection_report.json",
    )
