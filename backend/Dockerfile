FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_LFS_SKIP_SMUDGE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Sparse checkout avoids downloading the image dataset and unrelated project files.
# LFS smudging is disabled so only the selected 269 MB model is fetched below.
RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/Masksm/cnn-image-classifier.git /app/project \
    && cd /app/project \
    && git sparse-checkout set backend src model \
    && git lfs install \
    && git lfs pull --include="model/best_inception_full_finetuned.keras" --exclude=""

WORKDIR /app/project
RUN pip install -r backend/requirements.txt

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
