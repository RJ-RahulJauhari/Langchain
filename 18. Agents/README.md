# 18. Agents

## What is an Agent?

An **Agent** is an LLM that can:
1. Receive a goal in natural language
2. **Decide on its own** which tools to use and in what order
3. Keep acting until it has enough information to answer

The key difference from tool calling (folder 17) is **autonomy**:

| Tool Calling | Agent |
|-------------|-------|
| You decide when to call tools | The model decides when to call tools |
| Fixed, pre-written sequence | Dynamic — adapts based on what it finds |
| One round of tool calls | Multiple rounds until the goal is met |

Think of an agent like a research assistant: you give it a question, and it goes off, searches the web, reads results, follows up on interesting leads, and comes back with a complete answer — all on its own.

## How an agent loop works

```
User goal → Agent
              ↓
         Think: "Do I need a tool?"
              ↓ Yes
         Choose tool + arguments
              ↓
         Tool runs, returns result
              ↓
         Think: "Do I have enough info?"
              ↓ No → loop back
              ↓ Yes
         Generate final answer → User
```

This loop is called the **ReAct pattern** (Reason + Act).

## Tools used in this folder

| Tool | What it does |
|------|-------------|
| `DuckDuckGoSearchRun` | Searches the web using DuckDuckGo — no API key required |

---

## Files in this folder

| File | What it covers |
|------|---------------|
| `Agent.ipynb` | Create an agent with `create_agent()`, give it the DuckDuckGo search tool, and watch it autonomously answer questions that require live web lookups |

---

## Prerequisites

- Virtual environment activated
- Ollama running + `ollama pull gemma4:26b` (or a smaller model like `llama3.1`)
- The `ddgs` package (installed automatically in the notebook with `!pip install -U ddgs`)

---

## How to run

```bash
source .venv/bin/activate
jupyter notebook
```

Open `18. Agents/Agent.ipynb` and run cells top-to-bottom.

> **Tip:** `gemma4:26b` is a large model (~17 GB). If you don't have it, change the model name to `llama3.1` or `qwen2.5:latest` — they also work well for agents.

---

## What you'll learn

1. How to install and use `DuckDuckGoSearchRun` as an agent tool
2. How `create_agent()` wires a model and a list of tools into an agent loop
3. How to invoke the agent with `{"messages": [{"role": "user", "content": "..."}]}`
4. How the agent decides whether to search or answer directly
5. How to display the agent's markdown response with `IPython.display.Markdown`

---

## Key code pattern

```python
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

model = ChatOllama(model="llama3.1")
tools = [DuckDuckGoSearchRun()]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a helpful assistant. Use tools when you need up-to-date information.",
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What is the latest news in AI?"}]
})

# The final message in the list is the agent's answer
print(response["messages"][-1].content)
```

---

## Agents vs. Chains — when to use which

| Use a Chain when... | Use an Agent when... |
|--------------------|---------------------|
| The steps are fixed and known in advance | The steps depend on what the model finds |
| You need predictable, reproducible behavior | You need flexibility and adaptation |
| Speed and cost matter (no extra LLM calls) | Correctness matters more than speed |
| Example: summarise → translate | Example: research a topic and write a report |
