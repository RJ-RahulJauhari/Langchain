# 17. Tool Calling

## What is Tool Calling?

Out of the box, an LLM can only generate text. It cannot:
- Browse the internet
- Run calculations
- Check the weather
- Read a file
- Call an API

**Tool calling** (also called **function calling**) fixes this. You define Python functions and "give" them to the model. When the model decides it needs to use one, it tells you — you run the function — and you give the result back to the model.

The model doesn't actually run the code. It just says: *"Call `multiply` with `a=5` and `b=10`"*. Your Python code does the actual work and feeds the result back.

## How it works step by step

```
1. You define a tool (a Python function)
2. You attach it to the model with .bind_tools([tool])
3. You send a message to the model
4. If the model needs to use a tool:
   a. It returns a special response with tool_calls (NOT regular text)
   b. You extract the tool call, run your function with those arguments
   c. You send the result back to the model
   d. The model uses the result to write its final answer
```

## The `@tool` decorator

LangChain makes defining tools easy with the `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers, multiplies them together."""
    return a * b
```

The docstring is important — it's what the model reads to understand what the tool does and when to use it.

---

## Files in this folder

| File | What it covers |
|------|---------------|
| `ToolCallingLangchain.ipynb` | Define a custom `multiply` tool with `@tool`; bind it to a model with `.bind_tools()`; inspect `tool_calls` in the response |

---

## Prerequisites

- Virtual environment activated
- Ollama running + `ollama pull llama3.1` (or any tool-calling-capable model)

---

## How to run

```bash
source .venv/bin/activate
jupyter notebook
```

Open `17. Tool Calling/ToolCallingLangchain.ipynb` and run cells top-to-bottom.

---

## What you'll learn

1. How to define a tool with the `@tool` decorator
2. What `.name`, `.description`, and `.args` look like on a tool
3. How to bind tools to a model with `.bind_tools([tool_list])`
4. What the model's response looks like when it wants to call a tool (`response.tool_calls`)
5. How the model can decide to call multiple tools in one response

---

## Key code pattern

```python
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers, multiplies them together."""
    return a * b

model = ChatOllama(model="llama3.1")

# Give the tool to the model — it can now choose to call it
model_with_tools = model.bind_tools([multiply])

response = model_with_tools.invoke("What is 5 multiplied by 10?")

# If the model chose to use the tool:
print(response.tool_calls)
# → [{'name': 'multiply', 'args': {'a': 5, 'b': 10}, 'id': '...', 'type': 'tool_call'}]

# You then run the tool yourself and send the result back
result = multiply.invoke(response.tool_calls[0]["args"])
print(result)  # → 50
```

> **Note:** Tool calling requires a model that supports it. Not all models do. `llama3.1`, `qwen2.5`, and `gemma` are good choices with Ollama.
