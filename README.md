# LangChain Learning — GenAI Playground

A structured, hands-on learning repository for building GenAI applications with LangChain. Covers everything from basic LLM calls to agents, tool calling, chains, embeddings, and vector stores — using OpenAI, Ollama (local), and HuggingFace backends.

---

## What's Inside

| Folder / File | What It Covers |
|---|---|
| `1.LLMs/` | Basic LLM calls with OpenAI (`gpt-3.5-turbo-instruct`) |
| `2.ChatModels/` | Chat models via OpenAI, Ollama (local), and HuggingFace |
| `3.EmbeddingModels/` | Text embeddings with OpenAI, open-source models, Ollama, and document similarity |
| `4.Prompt/` | Prompt templates, message placeholders, temperature tuning, and a chatbot UI |
| `5.StructuredOutput/` | Extracting structured data using Pydantic, TypedDict, and `with_structured_output` |
| `6.Chains/` | Sequential, parallel, and conditional chains with LangChain LCEL |
| `8. Runnables/` | LangChain Runnables — the building blocks of LCEL pipelines |
| `17. Tool Calling/` | Defining and binding custom tools to models |
| `18. Agents/` | Building agents with tool use (DuckDuckGo search + Ollama) |
| `chatbot.py` | Streamlit chatbot app with RAG (Chroma vector store + Ollama) |
| `doc_writer.py` | Streamlit app to study a codebase and generate documentation via LLM |

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed locally (for local model notebooks)
- OpenAI API key (for OpenAI notebooks)
- HuggingFace token (for HuggingFace notebooks)

---

## Environment Setup

### 1. Clone / open the project

```bash
cd "Rahul Jauhari/Personal Projects/GenAI - Learning/Langchain"
```

### 2. Create and activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root (one already exists — fill in your keys):

```
OPENAI_API_KEY="sk-..."
HUGGINGFACEHUB_ACCESS_TOKEN="hf_..."
```

### 5. Pull Ollama models (for local model notebooks)

```bash
# Install Ollama first: https://ollama.com/download
ollama pull llama3.1        # used in Tool Calling notebook
ollama pull gemma4:26b      # used in Agents notebook
ollama pull nomic-embed-text  # used in embedding notebooks
```

---

## Running Notebooks

Start Jupyter and open any notebook:

```bash
jupyter notebook
# or
jupyter lab
```

Navigate to the numbered folder and open the `.ipynb` file.

**Recommended learning order:**

1. `1.LLMs/1_LLM_Demo.ipynb` — Basic LLM invocation
2. `2.ChatModels/` — Chat models (OpenAI → Ollama → HuggingFace)
3. `3.EmbeddingModels/` — Embeddings and document similarity
4. `4.Prompt/` — Prompt templates, chatbot, temperature
5. `5.StructuredOutput/` — Getting structured JSON back from models
6. `6.Chains/` — Chaining steps together with LCEL
7. `8. Runnables/` — Understanding the Runnable abstraction
8. `17. Tool Calling/` — Binding tools to models
9. `18. Agents/` — Full agent loop with live tool use

---

## Running the Streamlit Apps

Both apps require Ollama running locally.

### Chatbot (RAG-enabled)

A conversational chatbot that can ingest PDFs/DOCX files and answer questions about them using Chroma vector search.

```bash
streamlit run chatbot.py
```

### Codebase Doc Writer

Upload a zip of a codebase and let the LLM study and document it for you.

```bash
streamlit run doc_writer.py
```

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core LangChain framework |
| `langchain-openai` | OpenAI LLM / embeddings integration |
| `langchain-ollama` | Local Ollama model integration |
| `langchain-huggingface` | HuggingFace model integration |
| `langchain-community` | Community loaders, vector stores |
| `chromadb` | Chroma vector store (used in chatbot & doc writer) |
| `faiss-cpu` | FAISS vector store |
| `transformers` | HuggingFace transformers |
| `streamlit` | Web UI for the chatbot and doc writer apps |
| `python-dotenv` | Loads `.env` API keys |

---

## Notes

- Notebooks that use **OpenAI** require a valid `OPENAI_API_KEY` in `.env`.
- Notebooks that use **Ollama** require the Ollama daemon running (`ollama serve`) and the relevant model pulled.
- Notebooks that use **HuggingFace** require a valid `HUGGINGFACEHUB_ACCESS_TOKEN` in `.env`.
- The `.venv` directory is already set up — activate it before running anything.
