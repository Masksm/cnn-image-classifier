# Terrain AI Prediction API

FastAPI service for the existing CNN scene-classification model. It reuses `src/predict.py` and the trained model at `model/best_inception_full_finetuned.keras`.

## Required files

The deployment needs these repository paths available together:

- `backend/main.py`
- `backend/requirements.txt`
- `src/predict.py`
- `model/best_inception_full_finetuned.keras` (Git LFS object, approximately 269 MB)

The service starts from the repository root with:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

## API

- `GET /` service status
- `GET /health` model-file availability
- `POST /predict` multipart form field named `file`; returns `{"predictions":[{"class":"forest","confidence":0.98}]}`

Set `ALLOWED_ORIGINS` to the exact deployed frontend origin before production. The default wildcard is intended only for initial setup/testing.

## Deployment note

This API requires a Python host that supports TensorFlow and can download Git LFS model files. It is deliberately separate from the Next.js frontend because the full TensorFlow model is too large and resource-intensive to bundle into a Vercel serverless function.
