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

- Python 3.11+ — [python.org/downloads](https://www.python.org/downloads/)
- [Ollama](https://ollama.com) installed locally (for local model notebooks)
- OpenAI API key (for OpenAI notebooks)
- HuggingFace token (for HuggingFace notebooks)

---

## Environment Setup

### 1. Clone / open the project

**macOS / Linux**
```bash
cd "Rahul Jauhari/Personal Projects/GenAI - Learning/Langchain"
```

**Windows (Command Prompt or PowerShell)**
```powershell
cd "Rahul Jauhari\Personal Projects\GenAI - Learning\Langchain"
```

### 2. Create and activate the virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script with an execution policy error, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Once activated, your terminal prompt will show `(.venv)`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This works the same on all platforms once the venv is activated.

### 4. Set up environment variables

**macOS / Linux**
```bash
cp .env.example .env
```

**Windows (Command Prompt)**
```cmd
copy .env.example .env
```

**Windows (PowerShell)**
```powershell
Copy-Item .env.example .env
```

Then open `.env` in any text editor and fill in your keys:

```
OPENAI_API_KEY="sk-..."
HUGGINGFACEHUB_ACCESS_TOKEN="hf_..."
```

### 5. Set up Ollama (local LLM runtime)

Several notebooks and both Streamlit apps run models **locally** through Ollama — no API key needed, everything stays on your machine.

#### Install Ollama

**macOS**
```bash
brew install ollama
# or download the desktop app from https://ollama.com/download
```

**Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**

1. Go to [https://ollama.com/download](https://ollama.com/download)
2. Download the Windows installer (`OllamaSetup.exe`)
3. Run the installer — Ollama installs as a background service and adds `ollama` to your PATH
4. Open a new Command Prompt or PowerShell window and verify:
   ```powershell
   ollama --version
   ```

#### Start the Ollama daemon

Ollama must be running in the background before any notebook or app can talk to it.

**macOS / Linux**
```bash
ollama serve
```

**Windows** — Ollama runs automatically as a system tray app after installation. If it's not running, search for "Ollama" in the Start menu and launch it. You can also start it from the terminal:
```powershell
ollama serve
```

Verify it's up (all platforms):
```bash
curl http://localhost:11434
# should return: Ollama is running
```

On **Windows** without `curl`, use PowerShell:
```powershell
Invoke-WebRequest -Uri http://localhost:11434 -UseBasicParsing
```

> Ollama listens on `http://localhost:11434` by default.

#### Pull the models used in this repo

Run these in any terminal (all platforms):

```bash
ollama pull llama3.1          # Tool Calling notebook (17)
ollama pull gemma4:26b        # Agents notebook (18)
ollama pull nomic-embed-text  # Embedding notebooks (3), chatbot.py, doc_writer.py
```

Model weights are stored at:
- macOS / Linux: `~/.ollama/models`
- Windows: `C:\Users\<YourName>\.ollama\models`

Large models (gemma4:26b is ~17 GB) take a while on first pull — make sure you have disk space.

#### List and manage models

```bash
ollama list              # see downloaded models
ollama rm <model-name>   # delete a model to free disk space
ollama show llama3.1     # inspect a model's info
```

#### Test a model from the terminal

```bash
ollama run llama3.1
# type a prompt and press Enter; Ctrl+D or /bye to exit
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
