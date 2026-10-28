from pathlib import Path
from datetime import datetime, timezone
import io
import json
import sqlite3
import uuid

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "structural_damage_dnn.keras"
METRICS_PATH = ROOT / "outputs" / "evaluation_metrics.json"
DB_PATH = ROOT / "outputs" / "inspections.db"
MAX_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(
    title="Structural Damage Screening API",
    description="Educational concrete crack screening demo; not an engineering safety assessment.",
    version="1.0.0",
)
model = None
class_names = []
metrics = {}

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities_json TEXT NOT NULL,
                disclaimer TEXT NOT NULL
            )
        """)
        conn.commit()

@app.on_event("startup")
def startup():
    global model, class_names, metrics
    init_db()
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        class_names = metrics.get("class_names", [])
    else:
        print("Model not found yet. Train with: python src/train.py")

@app.get("/api/health")
def health():
    return {"status": "ready" if model is not None else "model_not_trained",
            "model_loaded": model is not None}

@app.get("/api/model-info")
def model_info():
    return {
        "model_loaded": model is not None,
        "classes": class_names,
        "test_accuracy": metrics.get("test_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
    }

@app.get("/api/inspections")
def inspections(limit: int = 50):
    limit = max(1, min(limit, 200))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, filename, predicted_class, confidence, "
            "probabilities_json, disclaimer FROM inspections "
            "ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) | {"probabilities": json.loads(row["probabilities_json"])}
            for row in rows]

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model unavailable. Train it first.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Upload a JPEG, PNG, or WebP image.")
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "Image exceeds 10 MB.")
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image = image.resize((224, 224))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Invalid image file.")

    array = np.asarray(image, dtype=np.float32)[None, ...]
    probs = model.predict(array, verbose=0)[0]
    idx = int(np.argmax(probs))
    predicted = class_names[idx]
    confidence = float(probs[idx])
    prob_map = {name: float(p) for name, p in zip(class_names, probs)}
    disclaimer = (
        "Image classification only. This result is not an engineering diagnosis "
        "or a determination that a building is safe."
    )
    recommendation = (
        "A crack-like pattern was predicted. Arrange review by a qualified "
        "professional; urgent concerns should be assessed promptly."
        if predicted == "crack"
        else "No crack was predicted in this image. This does not rule out "
             "damage or establish that the structure is safe."
    )
    record_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO inspections VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record_id, created_at, Path(file.filename or "upload").name,
             predicted, confidence, json.dumps(prob_map), disclaimer)
        )
        conn.commit()

    return {
        "id": record_id,
        "created_at": created_at,
        "predicted_class": predicted,
        "confidence": confidence,
        "probabilities": prob_map,
        "recommendation": recommendation,
        "disclaimer": disclaimer,
    }
