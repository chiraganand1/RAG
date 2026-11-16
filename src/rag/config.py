import os
from pathlib import Path

INDEX_DIR = Path("data/index")
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")