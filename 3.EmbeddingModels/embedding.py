#!/usr/bin/env python3
"""
RAG over a PDF — Interactive QnA CLI
- PyPDFLoader + RecursiveCharacterTextSplitter
- OllamaEmbeddings (default: mxbai-embed-large)
- Cosine similarity (sklearn) retrieval
- ChatOllama (default: llama3.1:latest) answering

Install:
  pip install langchain langchain-community langchain-ollama pypdf scikit-learn numpy

Ollama models:
  ollama pull mxbai-embed-large
  ollama pull llama3.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_EMBED_MODEL = "mxbai-embed-large"
DEFAULT_LLM_MODEL = "llama3.1:latest"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5


@dataclass
class Corpus:
    texts: List[str]
    chunks: List[Document]
    embeddings: np.ndarray
    embed_model_name: str
    chunk_size: int
    chunk_overlap: int


# -----------------------------
# Cache helpers
# -----------------------------
def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _file_fingerprint(pdf_path: Path) -> str:
    st = pdf_path.stat()
    raw = f"{pdf_path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _cache_key(
    pdf_path: Path, embed_model: str, chunk_size: int, chunk_overlap: int
) -> str:
    fp = _file_fingerprint(pdf_path)
    raw = f"{fp}|{embed_model}|{chunk_size}|{chunk_overlap}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _cache_paths(cache_dir: Path, key: str) -> Dict[str, Path]:
    base = cache_dir / f"ragpdf_{key}"
    return {
        "meta": base.with_suffix(".meta.json"),
        "chunks": base.with_suffix(".chunks.jsonl"),
        "emb": base.with_suffix(".emb.npy"),
    }


def _save_cache(paths: Dict[str, Path], corpus: Corpus) -> None:
    meta = {
        "embed_model_name": corpus.embed_model_name,
        "chunk_size": corpus.chunk_size,
        "chunk_overlap": corpus.chunk_overlap,
        "num_chunks": len(corpus.texts),
        "emb_shape": list(corpus.embeddings.shape),
        "emb_dtype": str(corpus.embeddings.dtype),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with paths["chunks"].open("w", encoding="utf-8") as f:
        for doc in corpus.chunks:
            rec = {"page_content": doc.page_content, "metadata": doc.metadata}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    np.save(paths["emb"], corpus.embeddings)


def _load_cache(paths: Dict[str, Path]) -> Tuple[List[Document], np.ndarray]:
    chunks: List[Document] = []
    with paths["chunks"].open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunks.append(
                Document(page_content=rec["page_content"], metadata=rec.get("metadata") or {})
            )
    embeddings = np.load(paths["emb"])
    return chunks, embeddings


# -----------------------------
# Core RAG
# -----------------------------
def build_corpus(
    pdf_path: str,
    embed_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> Corpus:
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    cache_paths = None
    if cache_dir is not None:
        _safe_mkdir(cache_dir)
        key = _cache_key(pdf, embed_model_name, chunk_size, chunk_overlap)
        cache_paths = _cache_paths(cache_dir, key)

    if use_cache and cache_paths and cache_paths["meta"].exists() and cache_paths["chunks"].exists() and cache_paths["emb"].exists():
        chunks, embeddings = _load_cache(cache_paths)
        texts = [d.page_content for d in chunks]
        return Corpus(
            texts=texts,
            chunks=chunks,
            embeddings=np.asarray(embeddings),
            embed_model_name=embed_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # 1) Load PDF
    loader = PyPDFLoader(str(pdf))
    docs = loader.load()

    # 2) Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    texts = [d.page_content for d in chunks]

    # 3) Embed
    embedding_model = OllamaEmbeddings(model=embed_model_name)
    document_embeddings = embedding_model.embed_documents(texts)
    embeddings = np.asarray(document_embeddings, dtype=np.float32)

    corpus = Corpus(
        texts=texts,
        chunks=chunks,
        embeddings=embeddings,
        embed_model_name=embed_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if cache_paths and use_cache:
        _save_cache(cache_paths, corpus)

    return corpus


def retrieve_top_k(
    query: str,
    corpus: Corpus,
    embedding_model: OllamaEmbeddings,
    k: int,
) -> Tuple[List[str], List[float], np.ndarray]:
    query_emb = np.asarray(embedding_model.embed_query(query), dtype=np.float32).reshape(1, -1)
    scores = cosine_similarity(query_emb, corpus.embeddings)[0]
    top_k_idx = np.argsort(scores)[::-1][:k]
    top_k_texts = [corpus.texts[i] for i in top_k_idx]
    top_k_scores = [float(scores[i]) for i in top_k_idx]
    return top_k_texts, top_k_scores, top_k_idx


def answer_query(
    query: str,
    corpus: Corpus,
    embedding_model: OllamaEmbeddings,
    llm_model_name: str,
    top_k: int,
    show_sources: bool = False,
) -> str:
    top_k_texts, top_k_scores, top_k_idx = retrieve_top_k(
        query=query,
        corpus=corpus,
        embedding_model=embedding_model,
        k=top_k,
    )

    context_blocks: List[str] = []
    for rank, (idx, text, score) in enumerate(zip(top_k_idx, top_k_texts, top_k_scores), start=1):
        meta = corpus.chunks[int(idx)].metadata or {}
        page = meta.get("page", "unknown")
        context_blocks.append(f"[Chunk {rank} | page={page} | score={score:.4f}]\n{text}")

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful financial analyst assistant. "
        "Answer the question based ONLY on the provided context from a PDF. "
        "If the answer is not explicitly in the context, say you don't know. "
        "Do not hallucinate numbers or details."
    )

    user_prompt = (
        f"Context from the PDF (top {top_k} relevant chunks):\n"
        f"\"\"\"\n{context}\n\"\"\"\n\n"
        f"Question: {query}\n\n"
        "Instructions:\n"
        "- Use only the context above.\n"
        "- If the answer is partially present, clearly specify what is known and what is not.\n"
        "- Be concise but clear."
    )

    llm = ChatOllama(model=llm_model_name)
    resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    answer = resp.content.strip()

    if show_sources:
        sources_lines = []
        for rank, (idx, score) in enumerate(zip(top_k_idx, top_k_scores), start=1):
            meta = corpus.chunks[int(idx)].metadata or {}
            page = meta.get("page", "unknown")
            sources_lines.append(f"- Chunk {rank}: page={page}, score={score:.4f}")
        answer += "\n\nSources:\n" + "\n".join(sources_lines)

    return answer


# -----------------------------
# CLI
# -----------------------------
HELP_TEXT = """
Commands:
  /help                 Show this help
  /quit or /exit        Exit
  /topk N               Set top-k retrieval (e.g., /topk 8)
  /sources on|off       Toggle printing sources
  /model NAME           Set LLM model name (e.g., /model llama3.1:latest)
  /stats                Show corpus stats

Ask questions by typing normally.
""".strip()


def run_cli(args: argparse.Namespace) -> int:
    pdf_path = args.pdf
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None

    print("Loading PDF and building embeddings...")
    try:
        corpus = build_corpus(
            pdf_path=pdf_path,
            embed_model_name=args.embed_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            cache_dir=cache_dir,
            use_cache=not args.no_cache,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Ready. Chunks: {len(corpus.texts)}")
    print(f"Embed model: {args.embed_model}")
    print(f"LLM model:   {args.llm_model}")
    print("Type /help for commands.\n")

    embedding_model = OllamaEmbeddings(model=args.embed_model)

    top_k = args.top_k
    llm_model = args.llm_model
    show_sources = args.show_sources

    while True:
        try:
            q = input("q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not q:
            continue

        if q.startswith("/"):
            parts = q.split()
            cmd = parts[0].lower()

            if cmd in ("/quit", "/exit"):
                print("Bye.")
                return 0

            if cmd == "/help":
                print(HELP_TEXT + "\n")
                continue

            if cmd == "/topk":
                if len(parts) != 2 or not parts[1].isdigit():
                    print("Usage: /topk N\n")
                    continue
                top_k = max(1, int(parts[1]))
                print(f"top_k set to {top_k}\n")
                continue

            if cmd == "/sources":
                if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
                    print("Usage: /sources on|off\n")
                    continue
                show_sources = parts[1].lower() == "on"
                print(f"sources {'ON' if show_sources else 'OFF'}\n")
                continue

            if cmd == "/model":
                if len(parts) < 2:
                    print("Usage: /model NAME\n")
                    continue
                llm_model = " ".join(parts[1:])
                print(f"LLM model set to: {llm_model}\n")
                continue

            if cmd == "/stats":
                print(
                    f"PDF:         {pdf_path}\n"
                    f"Chunks:      {len(corpus.texts)}\n"
                    f"Chunk size:  {corpus.chunk_size}\n"
                    f"Overlap:     {corpus.chunk_overlap}\n"
                    f"Embed model: {corpus.embed_model_name}\n"
                    f"Top-k:       {top_k}\n"
                    f"LLM model:   {llm_model}\n"
                    f"Sources:     {'ON' if show_sources else 'OFF'}\n"
                )
                continue

            print("Unknown command. Type /help\n")
            continue

        # Normal question
        print("\n--- ANSWER ---")
        try:
            ans = answer_query(
                query=q,
                corpus=corpus,
                embedding_model=embedding_model,
                llm_model_name=llm_model,
                top_k=top_k,
                show_sources=show_sources,
            )
            print(ans)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
        print()

    # unreachable
    # return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive RAG QnA over a PDF using Ollama embeddings + ChatOllama."
    )
    p.add_argument("--pdf", required=True, help="Path to the PDF file")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Ollama chat model")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--show-sources", action="store_true", help="Append retrieved chunk refs")
    p.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "rag_pdf_cli"),
        help="Cache directory for chunks/embeddings",
    )
    p.add_argument("--no-cache", action="store_true", help="Disable cache usage")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run_cli(args))


if __name__ == "__main__":
    main()