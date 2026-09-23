# 2. Chat Models

## What is a Chat Model?

A **Chat Model** is an LLM that is designed to have a conversation. Instead of taking a single string prompt, it takes a **list of messages** — just like a WhatsApp or iMessage chat thread.

Each message has a **role**:
- `SystemMessage` — instructions for the AI (e.g. "You are a helpful assistant")
- `HumanMessage` — what the user says
- `AIMessage` — what the AI replied previously (used to give the model memory of past turns)

This structure lets the model understand context across multiple turns of conversation.

## Why use Chat Models over plain LLMs?

- They support **multi-turn conversations** (the model remembers what was said before)
- They can be given **a system personality** via the system message
- Most modern models (GPT-4o, Llama 3, Gemma) are released as chat models

## Backends covered

| Provider | What it is | Cost |
|----------|-----------|------|
| **OpenAI** (`ChatOpenAI`) | Cloud-hosted models like GPT-4o | Paid API |
| **Ollama** (`ChatOllama`) | Runs open-source models locally on your machine | Free |
| **HuggingFace** (`ChatHuggingFace` / `HuggingFaceEndpoint`) | Cloud or local open-source models | Free tier available |

---

## Files in this folder

| File | What it does | Requires |
|------|-------------|---------|
| `1_ChatModels_OpenAI.ipynb` | Sends a message to GPT-4o, reads the response | `OPENAI_API_KEY` |
| `2_OllamaChatModel.ipynb` | Talks to a local Ollama model with a system message | Ollama running locally |
| `3_HuggingFace_ChatModel.ipynb` | Uses HuggingFace models (via their API router) | `HUGGINGFACEHUB_ACCESS_TOKEN` |

---

## Prerequisites

- Virtual environment activated
- For notebook 1: `OPENAI_API_KEY` in `.env`
- For notebook 2: Ollama installed and running (`ollama serve`), model pulled (e.g. `ollama pull qwen3-coder:30b` or any model you prefer)
- For notebook 3: `HUGGINGFACEHUB_ACCESS_TOKEN` in `.env`

---

## How to run

```bash
source .venv/bin/activate
jupyter notebook
```

Open any notebook in this folder and run cells top-to-bottom.

---

## What you'll learn

- The difference between `SystemMessage`, `HumanMessage`, and `AIMessage`
- How to send a multi-message conversation to a chat model
- How to read `.content` from the response
- How to swap between cloud (OpenAI) and local (Ollama) models with almost no code change

---

## Key code pattern

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("What is the best way to learn Python?"),
]

response = model.invoke(messages)
print(response.content)   # the text the AI generated
```

With Ollama (local), the only change is:
```python
from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3.1")
```

Everything else stays the same — that's the power of LangChain's unified interface.
