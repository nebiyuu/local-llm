FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by PyMuPDF
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirments.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirments.txt

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', device='cpu')"

# Copy app files
COPY extract.py .
COPY ui.py .

# Gradio port
EXPOSE 7860


CMD ["python", "ui.py"]