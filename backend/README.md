# Terrain AI Prediction API

FastAPI service for the existing CNN scene-classification model. It reuses `src/predict.py` and the trained model at `model/best_inception_full_finetuned.keras`.

## Deploy as a Hugging Face Docker Space

1. Create a new Space on Hugging Face.
2. Choose **Docker** as the Space SDK.
3. Upload this folder's `Dockerfile` to the root of that Space. The Dockerfile clones the public GitHub project and fetches only the selected model from Git LFS, not the image dataset or the other model checkpoints.
4. Wait for the image to build. TensorFlow and the 269 MB model make the first build relatively large and slow.
5. Once the Space is running, its root URL is the API URL. Check `/health` and then `/docs`.
6. Set `ALLOWED_ORIGINS` to the deployed Vercel website origin.

## API

- `GET /` service status
- `GET /health` model-file availability
- `POST /predict` multipart form field named `file`; returns `{"predictions":[{"class":"forest","confidence":0.98}]}`

The frontend expects the API base URL in Vercel's `NEXT_PUBLIC_API_URL` environment variable. Set it to the Space URL, without a trailing slash, then redeploy the frontend.

## Local run

From the repository root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

The API is intentionally separate from Vercel because the TensorFlow runtime and full model are too large for a typical Vercel serverless function bundle.
