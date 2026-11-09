"""
Glass Defect Detection API
--------------------------
FastAPI backend using YOLOv8 for detecting defects in glass images.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from routes.video import router as video_router
from routes.live_detection import router as live_router


# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_PATH = "runs/detect/glass_defect-2/weights/best.pt"        # Path to the YOLOv8 weights file
CONFIDENCE_THRESHOLD = 0.5         # Minimum confidence to keep a detection
SEVERITY_AREA_THRESHOLD = 10_000   # px²: below → Minor, at/above → Severe
OUTPUT_PATH = "output.jpg"         # Where the annotated image is saved
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ─── Global model holder ──────────────────────────────────────────────────────

model: YOLO | None = None


# ─── 1. Model Loading ─────────────────────────────────────────────────────────

def load_model(model_path: str) -> YOLO:
    """
    Load the YOLOv8 model from disk.
    Called ONCE at startup – never per request.
    """
    logger.info("Loading YOLOv8 model from: %s", model_path)
    loaded = YOLO(model_path)
    logger.info("Model loaded successfully.")
    return loaded


# ─── Lifespan: startup / shutdown ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    global model
    model = load_model(MODEL_PATH)
    yield
    # Nothing special needed on shutdown


app = FastAPI(
    title="Glass Defect Detection API",
    description="Detect defects in glass images using YOLOv8.",
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(video_router)
app.include_router(live_router)
# ─── 2. Image Preprocessing ───────────────────────────────────────────────────

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode raw bytes into an OpenCV BGR image.

    Args:
        file_bytes: Raw bytes from the uploaded file.

    Returns:
        BGR numpy array ready for YOLO inference.

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image. The file may be corrupt or unsupported.")

    # YOLO expects RGB; OpenCV reads BGR – convert before inference
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


# ─── 3. Severity Classification ───────────────────────────────────────────────

def classify_severity(width: int, height: int) -> str:
    """
    Classify defect severity based on bounding-box area.

    Args:
        width:  Bounding box width  (pixels)
        height: Bounding box height (pixels)

    Returns:
        "Minor" if area < SEVERITY_AREA_THRESHOLD, else "Severe".
    """
    area = width * height
    return "Minor" if area < SEVERITY_AREA_THRESHOLD else "Severe"


# ─── 4. Detection Logic ───────────────────────────────────────────────────────

def run_detection(image_rgb: np.ndarray) -> list[dict]:
    """
    Run YOLOv8 inference and extract structured defect records.

    Args:
        image_rgb: RGB image array.

    Returns:
        List of dicts with keys: type, confidence, bbox, severity.
    """
    results = model.predict(
        source=image_rgb,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    defects = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            # xyxy → x1, y1, x2, y2
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w = x2 - x1
            h = y2 - y1

            confidence = float(round(box.conf[0].item(), 4))
            class_id   = int(box.cls[0].item())
            class_name = model.names[class_id]
            severity   = classify_severity(w, h)

            defects.append({
                "type":       class_name,
                "confidence": confidence,
                "bbox":       [x1, y1, w, h],   # [x, y, width, height]
                "severity":   severity,
            })

    return defects


# ─── 5. Output Processing ─────────────────────────────────────────────────────

def draw_and_save(image_rgb: np.ndarray, defects: list[dict], output_path: str) -> None:
    """
    Annotate the image with bounding boxes and labels, then save to disk.

    Args:
        image_rgb:   Original RGB image.
        defects:     List of detection dicts from run_detection().
        output_path: File path to write the annotated image.
    """
    # Work in BGR for OpenCV drawing functions
    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for defect in defects:
        x, y, w, h = defect["bbox"]
        severity    = defect["severity"]
        label       = f"{defect['type']} {defect['confidence']:.2f} [{severity}]"

        # Choose colour by severity
        color = (0, 0, 255) if severity == "Severe" else (0, 200, 100)  # Red / Green

        # Draw rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Draw filled label background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x, y - text_h - baseline - 4), (x + text_w, y), color, -1)

        # Draw label text
        cv2.putText(
            annotated, label,
            (x, y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    cv2.imwrite(output_path, annotated)
    logger.info("Annotated image saved to: %s", output_path)


# ─── 6. Helper: file-type validation ──────────────────────────────────────────

def validate_file_extension(filename: str) -> None:
    """Raise HTTP 400 if the file extension is not in ALLOWED_EXTENSIONS."""
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )


# ─── 7. API Endpoint ──────────────────────────────────────────────────────────

@app.post("/detect", summary="Detect defects in an uploaded glass image")
async def detect_defects(file: UploadFile = File(...)) -> JSONResponse:
    """
    POST /detect

    Accepts a multipart image upload (jpg / jpeg / png).
    Returns JSON with a list of detected defects.

    Response schema:
    {
        "defects": [
            {
                "type":       "crack",
                "confidence": 0.91,
                "bbox":       [x, y, width, height],
                "severity":   "Minor" | "Severe"
            },
            ...
        ]
    }
    """
    # --- Validate file type ---------------------------------------------------
    validate_file_extension(file.filename)

    # --- Read raw bytes -------------------------------------------------------
    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    # --- Preprocess -----------------------------------------------------------
    try:
        image_rgb = preprocess_image(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # --- Run inference --------------------------------------------------------
    try:
        defects = run_detection(image_rgb)
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # --- Annotate & save ------------------------------------------------------
    try:
        draw_and_save(image_rgb, defects, OUTPUT_PATH)
    except Exception as exc:
        # Non-fatal: log but still return detection results
        logger.warning("Could not save annotated image: %s", exc)

    return JSONResponse(content={"defects": defects})


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    """Returns 200 OK if the service is running and model is loaded."""
    return {"status": "ok", "model_loaded": model is not None}
