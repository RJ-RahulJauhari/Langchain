# Chatbot App — RAG-powered Conversational Assistant

## What is this?

This is a fully working **Streamlit web app** that runs a chatbot in your browser. What makes it special is that it uses **RAG (Retrieval-Augmented Generation)** — you can upload your own documents (PDFs, DOCX files, or paste text), and the chatbot will answer questions specifically about those documents.

## What is RAG?

**RAG = Retrieval-Augmented Generation**

The problem with plain LLMs is that they only know what they were trained on. If you want to ask questions about *your* documents — a PDF manual, a contract, meeting notes — the model has no idea what's in them.

RAG fixes this:

```
1. You upload a document
2. The app splits it into small chunks
3. Each chunk is embedded (converted to a vector)
4. Chunks are stored in a vector database (Chroma)

When you ask a question:
5. Your question is embedded
6. The most relevant chunks are retrieved from Chroma
7. Those chunks are sent to the LLM as context
8. The LLM answers based on your document, not just its training data
```

It's like giving the AI a set of notes to read before answering, every single time you ask something.

---

## Tech stack

| Component | What it is |
|-----------|-----------|
| **Streamlit** | Creates the web UI in Python — no frontend knowledge needed |
| **ChatOllama** | The local LLM that generates responses (runs via Ollama) |
| **OllamaEmbeddings** | Converts text chunks and queries into vectors |
| **Chroma** | Vector database that stores and searches the embeddings |
| **RecursiveCharacterTextSplitter** | Splits long documents into smaller chunks |
| **PyPDFLoader** | Loads and reads PDF files |

---

## Prerequisites

- Virtual environment activated
- Ollama running (`ollama serve`)
- The model you want to chat with pulled (e.g. `ollama pull llama3.1`)
- The embedding model pulled: `ollama pull nomic-embed-text`

---

## How to run

From the **project root**:

```bash
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\activate             # Windows

streamlit run apps/chatbot/chatbot.py
```

Then open your browser at `http://localhost:8501`.

---

## How to use it

1. Open the app in your browser
2. In the sidebar, upload a PDF or DOCX file (or paste text directly)
3. Click **"Process Document"**
4. Once processed, type your question in the chat box
5. The chatbot will answer based on the content of your document

Without uploading a document, it works as a plain conversational chatbot using the Ollama model.

---

## What you'll learn from reading this code

- How to build a Streamlit app with a chat interface
- How to load and split documents with LangChain loaders and text splitters
- How to create and query a Chroma vector store
- How RAG retrieval works in practice (embed → store → retrieve → generate)
- How to maintain conversation history in a Streamlit session
