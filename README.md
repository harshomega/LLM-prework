# Internal Knowledge Base Q&A Bot (RAG)

A retrieval-augmented generation system that answers questions from a folder
of internal documents, built entirely with free/open-source tools.

## Stack (100% free)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store:** FAISS (`IndexFlatIP`)
- **LLM:** Ollama running a local model 
- **Orchestration:** plain Python — no LangChain/LlamaIndex, so every step is explicit

## Setup

```bash
pip install -r requirements.txt

# Install Ollama (https://ollama.com) and pull a model:
ollama pull llama3
```

**Option A — if already have docs:** drop your `.txt`, `.md`, or `.pdf`
files into `data/`, then:

```bash
python ingest.py                       
python query.py "your question here"   
```

**Option B — scrape a website into docs first:**

```bash
python scrape.py urls.txt             

python ingest.py                      
python query.py "your question here"
```

`scrape.py` uses Selenium headless Chrome rather than plain
`requests` + BeautifulSoup because many internal wikis render content with JavaScript — a plain HTTP GET
only returns the empty shell HTML before JS runs. Selenium waits for
the page to actually render before extracting text.

## Architecture

```
 website URLs                data/*.txt,.pdf
      │                             ▲
      ▼                             │
 [scrape.py: Selenium]  ──── saves scraped pages as .txt ──┘
      │
      ▼
 [chunk_text]  ── overlapping chunks (500 chars, 50 overlap)
      │
      ▼
 [SentenceTransformer] ── embeds each chunk (384-dim vectors)
      │
      ▼
 [FAISS IndexFlatIP]  ── stores normalized vectors for cosine similarity
      │
      ▼ (at query time)
 question ──▶ [embed question] ──▶ [FAISS search, top-k=4]
                                          │
                                          ▼
                              [build grounded prompt]
                                          │
                                          ▼
                              [Ollama local LLM] ──▶ answer
```

## Key design decisions 

1. **Why sentence-transformers over OpenAI embeddings:** free, runs locally,
   no rate limits, no data leaving your machine. Trade-off: lower embedding
   quality than `text-embedding-3-large`, so retrieval is slightly less
   precise on nuanced queries.

2. **Why FAISS over Pinecone/Weaviate:** zero infra cost, no network latency,
   good enough for a knowledge base of a few thousand chunks. Trade-off: no
   built-in persistence/scaling story past a single machine — would need to
   migrate to a hosted vector DB if the corpus grew past ~1M vectors or
   needed multi-user concurrent writes.

3. **Chunking with overlap:** without overlap, sentences spanning a chunk
   boundary get split and neither chunk retrieves correctly. This is a real
   bug worth mentioning as your "mistake and lesson" in Section 4.

4. **Explicit "don't know" instruction in the prompt:** the single biggest
   lever against hallucination. Worth mentioning as a technical decision
   with a clear trade-off more "I don't know" responses vs. fewer
   confidently wrong answers.

5. **Why Selenium over requests+BeautifulSoup for scraping:** most modern
   internal knowledge tools render content
   client-side with JavaScript. `requests.get()` only sees the pre-render
   HTML. Trade-off: Selenium is much heavier and
   slower per page — fine for a few hundred pages, but would need a
   headless-browser pool or async scraping if scraping thousands of pages.

6. **Politeness delay between requests:** a 2-second delay between page
   loads in `scrape.py` avoids hammering the target server and getting
   rate-limited or IP-banned mid-crawl a real issue I'd hit if scraping
   too fast.

## Known bottleneck (good scaling story for Section 3)

`IndexFlatIP` does an exact, brute-force search — O(n) per query. Fine for
small corpora, but retrieval latency grows linearly with chunk count. At
scale, the fix is switching to an approximate index like `IndexIVFFlat` or
`IndexHNSWFlat`, which trade a small amount of recall for much faster
lookups on large corpora.
