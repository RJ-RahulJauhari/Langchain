# 6. Chains — Connecting Steps Together

## What is a Chain?

A **chain** is a sequence of steps connected together, where the output of one step becomes the input of the next.

Think of it like an assembly line:

```
Raw material → Machine A → Machine B → Finished product
     ↕              ↕           ↕
 Your text    → Prompt    → LLM   → Parsed answer
```

In LangChain, you build chains using the `|` operator (called **pipe**):

```python
chain = prompt | model | output_parser
result = chain.invoke({"topic": "AI"})
```

This is called **LCEL** — LangChain Expression Language. It's the modern way to build LangChain applications.

## The 3 core building blocks

| Block | What it does | Example |
|-------|-------------|---------|
| `PromptTemplate` | Fills in your template with variables | `"Tell me 5 facts about {technology}"` |
| `ChatOllama` / `ChatOpenAI` | Sends the filled prompt to the LLM | Gets the AI's response |
| `StrOutputParser` | Extracts just the text from the response | Turns `AIMessage` → plain string |

## Types of chains

### Sequential Chain
Steps run **one after another**, each using the previous step's output.

```
Prompt → LLM → Parser → result
```

### Parallel Chain
Multiple chains run **at the same time** on the same input, and their results are combined.

```
             ┌→ notes_chain →┐
input ───────┤               ├→ merge_chain → final_result
             └→ quiz_chain  →┘
```

### Conditional (Branch) Chain
The chain **routes to different paths** depending on some condition — like an if/else in code.

```
input → classify_sentiment → if positive → thank_you_chain
                           → if negative → improvement_chain
```

---

## Files in this folder

| File | What it covers |
|------|---------------|
| `SequentialChain.ipynb` | Basic `prompt | model | parser` pipeline; how to visualise the chain graph |
| `ParallelChain.ipynb` | `RunnableParallel` — generate notes AND questions from the same text simultaneously |
| `ParallelChain-1.ipynb` | Advanced parallel chain with web search (Tavily) + notes + quiz generator |
| `ConditionalChain.ipynb` | `RunnableBranch` — classify customer feedback sentiment, then route to different response generators |

---

## Prerequisites

- Virtual environment activated
- Ollama running + models pulled
- `ParallelChain-1.ipynb` additionally needs `TAVILY_API_KEY` in `.env`

---

## How to run

```bash
source .venv/bin/activate
jupyter notebook
```

Recommended order: `SequentialChain.ipynb` → `ParallelChain.ipynb` → `ConditionalChain.ipynb` → `ParallelChain-1.ipynb`

---

## What you'll learn

1. How to use the `|` pipe operator to chain steps
2. How `StrOutputParser` turns a raw `AIMessage` into a plain string
3. How to run multiple chains in parallel with `RunnableParallel`
4. How to branch logic with `RunnableBranch`
5. How to visualise a chain's structure with `.get_graph().print_ascii()`

---

## Key code patterns

### Sequential
```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

prompt = PromptTemplate(template="Tell me 5 facts about {technology}", input_variables=["technology"])
model = ChatOllama(model="llama3.1")
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"technology": "AI"})
print(result)
```

### Parallel
```python
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel({
    "notes":     notes_prompt     | model | StrOutputParser(),
    "questions": questions_prompt | model | StrOutputParser(),
})
result = parallel_chain.invoke({"data": "some text..."})
print(result["notes"])
print(result["questions"])
```

### Conditional (Branch)
```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", thank_you_chain),
    (lambda x: x["sentiment"] == "negative", improvement_chain),
    RunnableLambda(lambda x: "Neutral feedback received"),  # required default
)
```

> Note: `RunnableBranch` **always requires a default** as its last argument. Omitting it raises a `TypeError`.
