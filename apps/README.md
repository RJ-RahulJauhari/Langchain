# Apps — Full Streamlit Applications

This folder contains complete, runnable web applications built with LangChain and Streamlit. These are not just notebooks — they are real apps you can open in a browser and use.

## What is Streamlit?

**Streamlit** is a Python library that lets you build interactive web apps entirely in Python — no HTML, CSS, or JavaScript required. You write a `.py` file, run `streamlit run yourfile.py`, and a web page appears in your browser.

It's the fastest way to turn a Python AI script into a usable web app.

---

## Apps in this folder

| Folder | App | What it does |
|--------|-----|-------------|
| `chatbot/` | RAG Chatbot | Upload documents (PDF/DOCX), then ask questions about them |
| `doc_writer/` | Codebase Doc Writer | Upload a zip of a codebase, get AI-generated documentation |

---

## How to run any app

From the **project root**:

```bash
# Activate the virtual environment first
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# Run the chatbot
streamlit run apps/chatbot/chatbot.py

# Run the doc writer
streamlit run apps/doc_writer/doc_writer.py
```

Both apps open at `http://localhost:8501` by default.

---

## Common requirements

Both apps need:
1. **Ollama running** in the background (`ollama serve`)
2. **Models pulled** — at minimum: `ollama pull nomic-embed-text`
3. **Virtual environment activated** with all packages installed

See each app's own `README.md` for app-specific details.
