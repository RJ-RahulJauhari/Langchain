# 4. Prompt Engineering

## What is a Prompt?

A **prompt** is the text (or set of messages) you send to an LLM to tell it what you want. The quality of your prompt has a huge impact on the quality of the response.

**Bad prompt:** "Tell me about dogs"
**Better prompt:** "You are a veterinarian. Explain in simple terms, for a new dog owner, the 3 most important vaccinations a puppy needs and why."

Prompt engineering is the skill of crafting prompts that consistently get useful, accurate, and well-structured responses.

## What is a Prompt Template?

Rather than hardcoding your prompt as a fixed string, a **PromptTemplate** lets you define a reusable template with **placeholders** (variables) that you fill in at runtime.

```python
# Instead of this (hardcoded):
"Explain Virat Kohli as if I'm 5 years old"

# You use a template:
template = "Explain {topic} as if I'm 5 years old"
# Then fill it: topic = "Virat Kohli"
```

This is powerful because you can reuse the same template for hundreds of different inputs.

## PromptTemplate vs. ChatPromptTemplate

| PromptTemplate | ChatPromptTemplate |
|---------------|-------------------|
| Produces a single string | Produces a list of messages |
| Used with plain LLMs | Used with chat models |
| `template = "Tell me about {topic}"` | `[("system", "You are a {domain} expert"), ("human", "Explain {topic}")]` |

## What is Temperature?

**Temperature** controls how creative/random the model's responses are:
- `0.0` = very deterministic, always picks the most likely next word (good for facts)
- `1.0` = more creative and varied (good for brainstorming, writing)
- `2.0` = very random, sometimes incoherent (usually too high)

Think of it like asking someone to name a colour:
- Temperature 0: "Red" (always the same)
- Temperature 1: "Turquoise", "Crimson", "Mauve" (varied and interesting)
- Temperature 2: "Banana" (makes no sense!)

---

## Files in this folder

| File | What it covers |
|------|---------------|
| `prompt_templates.ipynb` | Creating reusable `PromptTemplate` and `ChatPromptTemplate`; filling in variables |
| `chatbot.ipynb` | A simple memory chatbot — maintains conversation history in a list |
| `message_placeholder.ipynb` | Using `MessagesPlaceholder` to inject prior conversation history into a prompt |
| `temperature.ipynb` | How `temperature` affects response creativity; comparing low vs. high values |
| `prompt_ui.py` | Streamlit web app version of the chatbot |
| `prompt_ui.ipynb` | Notebook version of the Streamlit chatbot UI |
| `chat_history.txt` | Sample chat history used by `message_placeholder.ipynb` |

---

## Prerequisites

- Virtual environment activated
- `OPENAI_API_KEY` in `.env` (for temperature.ipynb)
- Ollama running (for chatbot.ipynb and prompt_ui.py)

---

## How to run

```bash
source .venv/bin/activate
jupyter notebook
```

To run the Streamlit chatbot UI:
```bash
streamlit run 4.Prompt/prompt_ui.py
```

---

## What you'll learn

1. How to create a `PromptTemplate` with variables
2. How to create a `ChatPromptTemplate` for multi-turn chat
3. How to keep a conversation history so the model remembers past messages
4. How `MessagesPlaceholder` lets you inject dynamic history into a fixed template
5. How temperature changes the style and randomness of responses

---

## Key code patterns

### Basic PromptTemplate
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain {topic} in simple terms.")
])

prompt = template.invoke({"domain": "cricket", "topic": "LBW rule"})
# prompt.messages → [SystemMessage(...), HumanMessage(...)]
```

### Simple chatbot with memory
```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

chat_history = [SystemMessage("You are a helpful assistant.")]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(user_input))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(response.content))
    print("Bot:", response.content)
```

The key insight: the model has no built-in memory — you give it the full history every time you call `.invoke()`. The `chat_history` list IS the memory.
