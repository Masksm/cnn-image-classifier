import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Reuse the tested prediction pipeline from the existing research project.
from src.predict import predict_image

app = FastAPI(title="Terrain AI Prediction API", version="1.0.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@app.get("/")
def root():
    return {"service": "Terrain AI Prediction API", "status": "online"}


@app.get("/health")
def health():
    model_path = Path(__file__).resolve().parent.parent / "model" / "best_inception_full_finetuned.keras"
    return {"status": "ok", "model_file_present": model_path.is_file()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image format. Use JPG, PNG, WEBP or BMP.")
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    try:
        predictions = predict_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="The trained model file is not available on the prediction server.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The model could not process this image.") from exc
    return {"predictions": predictions}
