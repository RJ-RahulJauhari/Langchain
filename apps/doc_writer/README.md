# Doc Writer App — AI-powered Codebase Documentation Generator

## What is this?

This is a **Streamlit web app** that takes a zip file of any codebase, reads through the code files, and uses an LLM to automatically generate documentation for it.

Think of it as an AI assistant that can read an entire project and write explanations for what the code does, how it's structured, and how to use it.

---

## How it works

```
1. You upload a .zip file of your project
2. The app extracts the files and reads the code
3. The code is split into chunks and embedded (stored in Chroma)
4. The LLM studies the codebase by querying the vector store
5. The LLM generates structured documentation
6. You download the output as a file
```

It uses the same **RAG** technique as the chatbot, but applied to code files instead of documents.

---

## Files in this folder

| File | Description |
|------|-------------|
| `doc_writer.py` | The main Streamlit app |
| `sample_output/index.html` | An example of the kind of output this tool can generate — a professional HTML portfolio page |

---

## Tech stack

| Component | What it is |
|-----------|-----------|
| **Streamlit** | Web UI |
| **ChatOllama** | Local LLM that reads code and writes documentation |
| **OllamaEmbeddings** | Embeds code chunks for semantic search |
| **Chroma** | Vector database for storing code embeddings |
| **RecursiveCharacterTextSplitter** | Breaks code files into chunks |
| **zipfile** (Python stdlib) | Extracts the uploaded zip |

---

## Prerequisites

- Virtual environment activated
- Ollama running (`ollama serve`)
- A model pulled: `ollama pull llama3.1` (or whichever you prefer)
- The embedding model: `ollama pull nomic-embed-text`

---

## How to run

From the **project root**:

```bash
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\activate             # Windows

streamlit run apps/doc_writer/doc_writer.py
```

Then open `http://localhost:8501` in your browser.

---

## How to use it

1. Open the app in your browser
2. Upload a `.zip` file of the codebase you want documented
3. Select what kind of documentation you want (README, API docs, architecture overview, etc.)
4. Click **"Generate Documentation"**
5. Read or download the output

---

## What you'll learn from reading this code

- How to handle file uploads in Streamlit
- How to process zip archives in Python
- How to build a RAG pipeline over code files instead of text documents
- How to use Chroma as a persistent vector store
- How to structure a multi-step LLM workflow with progress feedback
