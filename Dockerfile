FROM python:3.11-slim

WORKDIR /app

# Limit thread spawning to avoid CPU/RAM spikes on startup and during inference.
# On a 2-core VPS each model loader otherwise creates dozens of threads.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

# Install torch CPU before everything else (separate layer for caching)
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image layer
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B'); CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')"

# Copy application source
COPY main.py .
COPY structural_patterns.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
