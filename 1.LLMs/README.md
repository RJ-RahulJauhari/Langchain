# 1. LLMs — Large Language Models

## What is an LLM?

A **Large Language Model (LLM)** is an AI model that has been trained on huge amounts of text. It can read a piece of text you give it (called a **prompt**) and generate a meaningful response.

Think of it like a very smart autocomplete — you give it the beginning of a sentence or a question, and it continues or answers it based on everything it has learned.

Examples of LLMs:
- **GPT-3.5**, **GPT-4** by OpenAI
- **Llama**, **Gemma**, **Qwen** (open-source, can run locally via Ollama)

## LLM vs. Chat Model — what's the difference?

| LLM | Chat Model |
|-----|-----------|
| Takes a single text string as input | Takes a list of messages (like a conversation) |
| Good for completion tasks ("continue this text…") | Good for back-and-forth conversations |
| e.g. `OpenAI(model="gpt-3.5-turbo-instruct")` | e.g. `ChatOpenAI(model="gpt-4o")` |

This folder covers the **plain LLM** style. Chat models are in `2.ChatModels/`.

## How LangChain fits in

LangChain gives you a consistent Python interface to talk to any LLM. Instead of learning a different SDK for every provider, you just call `.invoke("your prompt")` on any model and get a response back.

---

## Files in this folder

| File | What it does |
|------|-------------|
| `1_LLM_Demo.ipynb` | Connects to OpenAI's `gpt-3.5-turbo-instruct` model, sends a question, and prints the answer |

---

## Prerequisites

- OpenAI API key in `.env` as `OPENAI_API_KEY`
- Virtual environment activated (`source .venv/bin/activate`)

---

## How to run

1. Open a terminal in the project root and activate the venv:
   ```bash
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `1.LLMs/1_LLM_Demo.ipynb` and run the cells top-to-bottom.

---

## What you'll learn

- How to import LangChain's OpenAI wrapper
- How to create an LLM object and configure it
- How to call `.invoke()` to send a prompt and get a response
- What the raw response object looks like

---

## Key code pattern

```python
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
response = llm.invoke("What is the capital of India?")
print(response)
```

That's it — 3 lines to talk to an LLM.
