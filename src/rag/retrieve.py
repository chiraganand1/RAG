from rag.index import load_index
import numpy as np
from sentence_transformers import SentenceTransformer
from rag.config import EMBED_MODEL_NAME

def retrieve(query: str, top_k: int = 5) -> list[str]:
    """Stage 4: embed the query, find the most similar chunks via cosine similarity."""
    embeddings, chunks = load_index()

    model = SentenceTransformer(EMBED_MODEL_NAME)
    query_vec = model.encode([query])[0]

    # cosine similarity: normalize both sides, then dot product
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = chunk_norms @ query_norm

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]