# Single-container image for Brahma AI. Bundles ffmpeg so the compositor and
# outro detector work out of the box.
FROM python:3.11-slim

# ffmpeg (+ ffprobe) is a hard runtime dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY BRAHMA_AI_Logo.jpg ./
RUN pip install --no-cache-dir -e .

# Assets and outputs are mounted at runtime (see `make docker-run`); create the
# mount points so the app can write even if nothing is mounted.
RUN mkdir -p adv_video sample_video outputs/cache

EXPOSE 8501

# Credentials are provided at runtime via a mounted configs/ dir + env.
ENV GOOGLE_APPLICATION_CREDENTIALS=configs/bq_creds.json \
    VERTEX_LOCATION=us-central1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
