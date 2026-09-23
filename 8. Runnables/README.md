# 8. Runnables — Understanding LangChain's Core Abstraction

## What is a Runnable?

Everything in LangChain — prompts, models, output parsers, custom functions — is a **Runnable**. A Runnable is simply something that:

1. Takes an **input**
2. Does something
3. Returns an **output**

The genius of this design is that every Runnable has the same interface (`.invoke()`, `.stream()`, `.batch()`), so you can chain them together with the `|` pipe operator regardless of what's inside.

```
PromptTemplate → ChatOllama → StrOutputParser
  (Runnable)      (Runnable)     (Runnable)
      └─────────────┴──────────────┘
              one big chain (also a Runnable!)
```

## Why does this matter?

Before LangChain introduced the Runnable concept, chaining steps together required a lot of boilerplate. Understanding how Runnables work under the hood helps you:

- Debug chains when something goes wrong
- Build your own custom steps
- Understand why `|` works between different LangChain objects

---

## Files in this folder

| File | What it covers |
|------|---------------|
| `Langchain-Normal.ipynb` | **Build a mini LangChain from scratch** — implements a custom `MyLLM` class and `MyPromptTemplate` without using LangChain at all. This shows you exactly what LangChain is doing under the hood. |

---

## What the "from scratch" notebook demonstrates

The notebook builds:

```python
class MyLLM:
    def predict(self, prompt):
        # ... returns a response dict

class MyPromptTemplate:
    def format(self, input_dict):
        # ... fills in the template and returns a string
```

By building these yourself, you see that LangChain is just a well-designed wrapper around the same core ideas:
- A template fills in variables → produces a prompt string
- An LLM takes that string → produces a response
- A parser takes that response → extracts what you need

---

## Prerequisites

- Virtual environment activated
- No API keys or Ollama needed for this notebook (it uses a mock LLM)

---

## How to run

```bash
source .venv/bin/activate
jupyter notebook
```

Open `8. Runnables/Langchain-Normal.ipynb` and run cells top-to-bottom.

---

## What you'll learn

1. What a "Runnable" really is at its core
2. How LangChain wraps LLMs and prompt templates in a consistent interface
3. Why the `|` pipe operator works — it's just composing Runnables
4. The mental model for understanding any LangChain chain

---

## Key insight

When you write:
```python
chain = prompt | model | parser
chain.invoke({"topic": "AI"})
```

LangChain is doing exactly this under the hood:
```python
filled_prompt = prompt.format({"topic": "AI"})
raw_response  = model.predict(filled_prompt)
clean_output  = parser.parse(raw_response)
```

The `|` operator just makes this pipeline readable and composable.
