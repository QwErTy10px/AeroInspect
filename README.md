# Glass Defect Detection API

A minimal, production-ready FastAPI backend that uses **YOLOv8** and **OpenCV**
to detect defects in glass images.

---

## Project Structure

```
glass_defect_detection/
├── main.py           ← FastAPI application (all logic lives here)
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts, downloads `yolov8n.pt` automatically on first run (if not
already present), and loads the model **once** before accepting requests.

### 3. Test the endpoint

```bash
curl -X POST "http://localhost:8000/detect" \
     -F "file=@/path/to/your/glass_image.jpg"
```

**Example response:**
```json
{
  "defects": [
    {
      "type": "crack",
      "confidence": 0.91,
      "bbox": [120, 45, 200, 80],
      "severity": "Minor"
    },
    {
      "type": "scratch",
      "confidence": 0.76,
      "bbox": [300, 200, 250, 180],
      "severity": "Severe"
    }
  ]
}
```

The annotated image is also saved locally as **`output.jpg`**.

---

## API Reference

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| POST   | `/detect` | Upload an image and get defect data  |
| GET    | `/health` | Verify the service and model status  |

### POST `/detect`

| Parameter | Type            | Description                          |
|-----------|-----------------|--------------------------------------|
| `file`    | `multipart/form-data` | Image file (jpg, jpeg, png)   |

### Severity thresholds

| Bounding-box area (px²) | Severity |
|-------------------------|----------|
| < 10 000                | Minor    |
| ≥ 10 000                | Severe   |

Adjust `SEVERITY_AREA_THRESHOLD` in `main.py` to match your dataset.

---

## Using a Custom Model

Replace `MODEL_PATH = "yolov8n.pt"` in `main.py` with the path to your
fine-tuned weights, e.g.:

```python
MODEL_PATH = "models/glass_defect_yolov8.pt"
```

---

## Interactive API Docs

Once the server is running, visit:

- **Swagger UI** – http://localhost:8000/docs  
- **ReDoc**      – http://localhost:8000/redoc
