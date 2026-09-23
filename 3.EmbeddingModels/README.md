# 3. Embedding Models

## What is an Embedding?

An **embedding** is a way to convert text into a list of numbers (called a **vector**). The magic is that similar pieces of text end up with similar numbers, so you can mathematically measure how "close" two pieces of text are in meaning.

**Example:**
- "The cat sat on the mat" → `[0.12, -0.43, 0.87, ...]` (384 numbers)
- "A feline rested on a rug" → `[0.11, -0.41, 0.85, ...]` (very similar!)
- "The stock market crashed" → `[0.93, 0.21, -0.64, ...]` (very different)

This is the foundation of:
- **Semantic search** — find documents by meaning, not just keywords
- **RAG (Retrieval-Augmented Generation)** — feed relevant documents to an LLM as context
- **Recommendation systems** — find similar items

## How similarity is measured

The standard measure is **cosine similarity** — a number between -1 and 1:
- `1.0` = identical meaning
- `0.0` = unrelated
- `-1.0` = opposite meaning

The `sklearn.metrics.pairwise.cosine_similarity` function handles this calculation.

## Backends covered

| Provider | Model | Notes |
|----------|-------|-------|
| **OpenAI** | `text-embedding-3-large` | Cloud API, high quality |
| **HuggingFace** | `all-MiniLM-L6-v2` | Runs locally, free, fast |
| **Ollama** | `nomic-embed-text` | Runs locally, no API key needed |

---

## Files in this folder

| File | What it covers | Requires |
|------|---------------|---------|
| `1_EmbeddingModels_OpenAI.ipynb` | Embed a single string with OpenAI; inspect the vector | `OPENAI_API_KEY` |
| `2_Embedding_OpenAI_Docs.ipynb` | Embed multiple documents at once with `embed_documents()` | `OPENAI_API_KEY` |
| `3_EmbeddingModels_OpenSource.ipynb` | Embed text using a local HuggingFace model (no API key needed) | None |
| `4_Document_Similarity.ipynb` | Real semantic search — embed 100 story pages, find the most relevant one for a query | `OPENAI_API_KEY` |
| `5_Embedding_Models_Ollama.ipynb` | Embed text with Ollama; split long documents into chunks first | Ollama + `nomic-embed-text` |
| `embedding.py` | Standalone Python script version of the embedding demo | — |

---

## Two key methods

```python
# embed_query — for a single search query
vector = embedding_model.embed_query("What is the capital of France?")
# returns: [0.12, -0.43, 0.87, ...]  (one list of floats)

# embed_documents — for a batch of documents
vectors = embedding_model.embed_documents(["Paris is in France", "Rome is in Italy"])
# returns: [[...], [...]]  (one list per document)
```

Always use `embed_query` for the search question and `embed_documents` for the knowledge base.

---

## Prerequisites

- Virtual environment activated
- For OpenAI notebooks: `OPENAI_API_KEY` in `.env`
- For Ollama notebook: Ollama running + `ollama pull nomic-embed-text`
- For open-source notebook: no extra setup (model downloads automatically)

---

## How to run

```bash
source .venv/bin/activate
jupyter notebook
```

Open notebooks in order (1 → 5) for the best learning progression.

---

## What you'll learn

1. What an embedding vector looks like
2. How to embed a single string vs. a list of strings
3. How cosine similarity finds the most semantically related document
4. How to chunk long documents before embedding (for RAG pipelines)
5. How to use free local models instead of paid cloud APIs

---

## Key code pattern (semantic search)

```python
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = ["Paris is the capital of France", "Berlin is the capital of Germany"]
query = "Where is the Eiffel Tower?"

doc_vectors = model.embed_documents(documents)
query_vector = model.embed_query(query)

scores = cosine_similarity([query_vector], doc_vectors)[0]
best_match_index = scores.argmax()
print(documents[best_match_index])  # → "Paris is the capital of France"
```
