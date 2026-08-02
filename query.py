"""
query.py
--------
Takes a user question, retrieves the most relevant chunks from the FAISS
index, builds a grounded prompt, and calls a local free LLM (via Ollama)
to generate an answer.

"""

import sys
import pickle
import argparse

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

INDEX_DIR = "index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"   # swap for "mistral", "phi3", etc. -- whatever you've pulled
TOP_K = 4                 # number of chunks to retrieve


def load_index():
    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["metadata"]


def retrieve(question: str, model, index, chunks, metadata, top_k: int = TOP_K):
    """Embed the question and return the top_k most similar chunks."""
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": chunks[idx],
            "source": metadata[idx]["source"],
            "score": float(score),
        })
    return results


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a grounded prompt that instructs the model to answer ONLY from
    the retrieved context. This is the single most important guardrail
    against hallucination in the whole pipeline -- without an explicit
    "don't know? say so" instruction, the model will confidently
    fabricate answers when the retrieved chunks don't actually contain
    the answer.
    """
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )

    prompt = f"""You are an internal knowledge base assistant. Answer the question
using ONLY the context provided below. If the context does not contain
enough information to answer, say "I don't have enough information to
answer that" instead of guessing.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def generate_answer(prompt: str) -> str:
    if not OLLAMA_AVAILABLE:
        return ("[Ollama not installed] Install with 'pip install ollama' and run "
                "'ollama pull llama3' locally, then re-run this script.")

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def answer_question(question: str, verbose: bool = True):
    model = SentenceTransformer(EMBEDDING_MODEL)
    index, chunks, metadata = load_index()

    retrieved = retrieve(question, model, index, chunks, metadata)

    if verbose:
        print("\n--- Retrieved chunks ---")
        for i, r in enumerate(retrieved, 1):
            print(f"{i}. (score={r['score']:.3f}) {r['source']}")
            print(f"   {r['text'][:150]}...\n")

    prompt = build_prompt(question, retrieved)
    answer = generate_answer(prompt)

    print("--- Answer ---")
    print(answer)
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the RAG knowledge base.")
    parser.add_argument("question", type=str, help="Question to ask the knowledge base")
    parser.add_argument("--quiet", action="store_true", help="Hide retrieved chunk debug output")
    args = parser.parse_args()

    answer_question(args.question, verbose=not args.quiet)
