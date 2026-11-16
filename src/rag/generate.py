from rag.retrieve import retrieve
from anthropic import Anthropic
from rag.config import LLM_MODEL

def answer(query: str, top_k: int = 5) -> str:
    """Stage 4-5: retrieve context, then ask Claude to answer grounded in it only."""
    context_chunks = retrieve(query, top_k=top_k)
    context = "\n\n---\n\n".join(context_chunks)

    client = Anthropic()  # reads ANTHROPIC_API_KEY from environment
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so directly rather than guessing.

Context:
{context}

Question: {query}"""

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text