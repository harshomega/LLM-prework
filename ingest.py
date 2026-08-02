"""
ingest.py
---------
Loads documents from ./data, chunks them, generates embeddings using a
free local sentence-transformers model, and builds a FAISS index for
fast similarity search.

"""

import os
import glob
import json
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

DATA_DIR = "data"
INDEX_DIR = "index"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks to preserve context across boundaries
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # free, small, fast (~80MB, runs on CPU)


def load_documents(data_dir: str) -> list[dict]:
    """Load .txt, .md, and .pdf files from data_dir into raw text documents."""
    documents = []

    for filepath in glob.glob(os.path.join(data_dir, "**/*"), recursive=True):
        if os.path.isdir(filepath):
            continue

        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext in (".txt", ".md"):
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            elif ext == ".pdf":
                reader = PdfReader(filepath)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)

            else:
                continue  # skip unsupported file types

            if text.strip():
                documents.append({"source": filepath, "text": text})

        except Exception as e:
            print(f"[WARN] Failed to load {filepath}: {e}")

    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.

    Overlap matters here: without it, a sentence that spans a chunk
    boundary gets cut in half and neither chunk retrieves it correctly.
    This was one of the first bugs I hit -- retrieval kept missing
    obviously-relevant answers because the key sentence was split
    across two chunks with no shared context.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # step forward, keeping overlap

    return [c.strip() for c in chunks if c.strip()]


def build_index():
    print(f"Loading documents from ./{DATA_DIR} ...")
    documents = load_documents(DATA_DIR)

    if not documents:
        print(f"No documents found in ./{DATA_DIR}. Add .txt/.md/.pdf files and re-run.")
        return

    print(f"Loaded {len(documents)} documents. Chunking...")

    all_chunks = []       # flat list of chunk text
    chunk_metadata = []   # parallel list: which source file each chunk came from

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_metadata.append({"source": doc["source"]})

    print(f"Created {len(all_chunks)} chunks. Loading embedding model "
          f"'{EMBEDDING_MODEL}' (first run downloads it, ~80MB, then it's cached)...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings...")
    embeddings = model.encode(
        all_chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalize so we can use inner product = cosine similarity
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product on normalized vectors == cosine similarity
    index.add(embeddings.astype(np.float32))

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))

    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": chunk_metadata}, f)

    print(f"Done. Indexed {len(all_chunks)} chunks from {len(documents)} documents.")
    print(f"Index saved to ./{INDEX_DIR}/")


if __name__ == "__main__":
    build_index()
