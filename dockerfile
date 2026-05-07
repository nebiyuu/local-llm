FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by PyMuPDF
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY pyproject.toml .

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('rasyosef/RoBERTa-Amharic-Embed-Medium', device='cpu')"

# Gradio port
EXPOSE 7860

CMD ["python", "-m", "docchat"]
