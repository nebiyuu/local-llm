import os
from sentence_transformers import SentenceTransformer
from ..config import settings

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = settings.embedding_model
        models_dir = settings.models_dir

        models_path = None
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    models_path = item_path
                    break

        if models_path and os.path.exists(models_path):
            print(f"Model found in {models_path}")
            _model = SentenceTransformer(models_path, device="cpu")
        else:
            print(f"Model not found in {models_dir}, downloading to cache folder {settings.cache_dir}")
            _model = SentenceTransformer(model_name, device="cpu", cache_folder=settings.cache_dir)
    return _model


def embed_chunks(texts: list[str]):
    model = get_model()
    embeddings = model.encode(texts)
    return embeddings