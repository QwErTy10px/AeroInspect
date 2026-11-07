"""
routes/live_detection.py
------------------------
WebSocket endpoint for real-time webcam glass defect detection.
Does NOT touch /detect or /detect-video.
"""

import json
import logging
import base64

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.live_detector import process_live_frame, reset_tracker

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/live-detection")
async def live_detection(websocket: WebSocket):
    """
    WebSocket: /ws/live-detection

    Protocol (client → server):
        Binary message: raw JPEG frame bytes

    Protocol (server → client):
        JSON message: {
            frame, timestamp, defects, annotated_frame (base64 JPEG)
        }
        OR error JSON: { error: "..." }

    The client sends frames; the server responds with annotated results.
    """
    from main import model   # shared model, loaded once at startup

    await websocket.accept()
    reset_tracker()
    logger.info("Live detection WebSocket connected")

    try:
        while True:
            # Receive raw JPEG bytes from the browser
            data = await websocket.receive()

            # Support both bytes and base64-encoded text frames
            if "bytes" in data and data["bytes"]:
                frame_bytes = data["bytes"]
            elif "text" in data and data["text"]:
                # Browser sent base64 text — decode it
                try:
                    frame_bytes = base64.b64decode(data["text"])
                except Exception:
                    await websocket.send_text(
                        json.dumps({"error": "Invalid base64 frame"})
                    )
                    continue
            else:
                continue

            # Process frame
            try:
                result = process_live_frame(frame_bytes, model)
                await websocket.send_text(json.dumps(result))
            except ValueError as exc:
                await websocket.send_text(json.dumps({"error": str(exc)}))
            except Exception as exc:
                logger.error("Frame processing error: %s", exc)
                await websocket.send_text(
                    json.dumps({"error": "Detection failed for this frame"})
                )

    except WebSocketDisconnect:
        logger.info("Live detection WebSocket disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
    finally:
        reset_tracker()
