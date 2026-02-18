FROM python:3.11-slim

WORKDIR /app

# Install torch CPU before everything else (separate layer for caching)
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image layer
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"

# Copy application source
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
