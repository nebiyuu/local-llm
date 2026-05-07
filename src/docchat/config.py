import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    embedding_model: str = "rasyosef/RoBERTa-Amharic-Embed-Medium"
    models_dir: str = "./data/models"
    cache_dir: str = "./model_cache"

    chunk_size: int = 800
    chunk_overlap: int = 150

    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "documents"

    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "qwen3.5:0.8b"

    n_retrieval_chunks: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "DOCCHAT_"


settings = Settings()