# AGENTS.md

## Setup

```bash
pip install -e ".[dev]"    # editable install + pytest
```

Docker uses `requirements.txt` — keep it in sync with `pyproject.toml` dependencies.

## Run

```bash
python -m docchat          # entrypoint: src/docchat/__main__.py → ui/interface.py:main()
```

Requires Ollama on `localhost:11434` with model `qwen3.5:0.8b` (configurable via `DOCCHAT_OLLAMA_URL` / `DOCCHAT_OLLAMA_MODEL`).

- **Self-contained**: `docker compose up` (includes Ollama service)
- **Standalone Docker**: `docker run -p 7860:7860 --add-host=host.docker.internal:host-gateway docchat`

## Tests

```bash
pytest                     # from project root, no extra flags
```

- Must install package first: `pip install -e ".[dev]"`
- First run downloads `rasyosef/RoBERTa-Amharic-Embed-Medium` (~hundreds of MB) — slow
- `tests/test_core/test_chunker.py`, `tests/test_llm/test_client.py`, `tests/test_ui/test_interface.py` define **inline copies** of the functions they test — intentional isolation, do not "refactor" to imports
- Integration tests in `tests/test_integration/test_rag_pipeline.py` import from `docchat.*` and exercise the real embedding model

## Configuration

- `pydantic-settings` with env prefix `DOCCHAT_`; `.env` file auto-loaded
- Module-level singleton: `from docchat.config import settings`

## Architecture

```
src/docchat/
  core/     pdf.py, chunker.py, embedding.py   # PDF→text, word-based chunking, sentence-transformers
  store/    collection.py                      # ChromaDB DocumentCollection class (primary interface)
  llm/      client.py                          # Ollama streaming client (generator, yields (chunk, thinking))
  ui/       interface.py, styles.py, styles.css  # Gradio UI
  config.py                                    # Settings (env-aware)
```

- **CSS**: `interface.py` imports `from .styles import CSS` — the string in `styles.py` is the source of truth. `styles.css` is a standalone duplicate not used by code.
- **Chunking**: word-based (`.split()`), not token-based.
- **ChromaDB**: `DocumentCollection` class is the primary interface. The module-level `get_collection()` helper exists but is secondary.
- **LLM client**: `ask_ollama` is a generator yielding `(chunk, thinking)` tuples; `None` as chunk signals end-of-stream.

## Gotchas

- No CI, no lint config, no formatter, no typecheck config exist in this repo.
- Embedding model is Amharic-specific (`rasyosef/RoBERTa-Amharic-Embed-Medium`). Store tests use `np.random.rand(N, 384)` vectors to avoid needing the real model.
- Dockerfile pre-downloads the embedding model at build time, copies `src/` and `pyproject.toml`.
- `data/` and `.env` are gitignored; ChromaDB persists to `data/chroma_db/` by default.
- Running the app loads the embedding model into memory on startup (`interface.py:main()` calls `get_model()`).

## Refactoring history

See `agents/plan.md` — documents the migration from the original flat prototype (old `ui.py` + `extract.py`) to the current package structure. All items listed have been implemented.
