import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from rag.config import INDEX_DIR, EMBEDDINGS_FILE, CHUNKS_FILE, EMBED_MODEL_NAME
from rag.ingest import extract_text, chunk_text
INDEX_DIR.mkdir(parents=True, exist_ok=True)
def build_index(pdf_path: str):
    """Stage 1-3: extract, chunk, embed, save to disk."""
    print(f"Extracting text from {pdf_path}...")
    text = extract_text(pdf_path)
    print(f"Extracted {len(text):,} characters")

    print("Chunking...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    print(f"Loading embedding model ({EMBED_MODEL_NAME})...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Embedding chunks (this is the slow step, runs once)...")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)

    np.save(EMBEDDINGS_FILE, embeddings)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)

    print(f"Saved {len(chunks)} chunk embeddings to {EMBEDDINGS_FILE}")
    print("Index build complete. You can now run --ask")
def load_index():
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError("No index found. Run with --build first.")
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)
    return embeddings, chunks