# 5. Structured Output

## The problem: LLMs return plain text

By default, an LLM returns a blob of text. For example:

```
"The sentiment of this review is positive. The main pros are the battery life 
and the comfort. The cons include the bulky design."
```

This is fine for humans to read, but very hard for code to work with. You can't do `response["sentiment"]` on a string.

## The solution: Structured Output

**Structured output** forces the LLM to return data in a specific shape — like a Python dictionary or a typed object — that your code can directly use.

```python
# Instead of this messy string...
"The sentiment is positive, pros are battery and comfort, cons are bulky design"

# You get this clean, usable object...
{
    "sentiment": "positive",
    "pros": ["battery life", "comfort"],
    "cons": ["bulky design"]
}
```

You define the shape you want using either **Pydantic** or **TypedDict**, and LangChain handles getting the model to fill it in correctly.

---

## Two ways to define the shape

### 1. Pydantic BaseModel (recommended)

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    summary: str = Field(description="A brief summary")
    pros: Optional[list[str]] = Field(description="List of positives")
    cons: Optional[list[str]] = Field(description="List of negatives")
```

- Gives you **validation** (if the model returns the wrong type, it raises an error)
- Supports **default values**, **descriptions**, and **constraints**
- Returns a proper Python object: `review.sentiment`, `review.pros`

### 2. TypedDict (simpler, no validation)

```python
from typing import TypedDict, Literal, Annotated

class Review(TypedDict):
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "The sentiment"]
    summary: Annotated[str, "A brief summary"]
```

- Simpler to write
- Returns a plain Python `dict`: `review["sentiment"]`
- No automatic validation

---

## Files in this folder

| File | What it covers |
|------|---------------|
| `Pydantic.ipynb` | What Pydantic is; how to define a schema with `BaseModel`; how LangChain uses it to extract structured data |
| `TypedDict.ipynb` | What `TypedDict` is; simpler alternative to Pydantic |
| `with_structured_output.ipynb` | How to use `model.with_structured_output(Schema)` to make any LLM return structured data |

---

## Prerequisites

- Virtual environment activated
- Ollama running + a model pulled (notebooks use `qwen3-coder:30b` and `qwen2.5:latest`)
- Or swap the model name to any model you have available

---

## How to run

```bash
source .venv/bin/activate
jupyter notebook
```

Recommended order: `Pydantic.ipynb` → `TypedDict.ipynb` → `with_structured_output.ipynb`

---

## What you'll learn

1. What Pydantic is and why it's useful for data validation
2. How to define a schema (the "shape" of data you want)
3. How to use `.with_structured_output()` to attach the schema to a model
4. The difference between Pydantic (object with `.sentiment`) and TypedDict (dict with `["sentiment"]`)

---

## Key code pattern

```python
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Literal, Optional

class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    summary: str = Field(description="A brief summary of the review")
    pros: Optional[list[str]] = Field(description="The positives mentioned")
    cons: Optional[list[str]] = Field(description="The negatives mentioned")

model = ChatOllama(model="qwen2.5:latest")

# Attach the schema — the model will now always return data in this shape
review_model = model.with_structured_output(Review)

result = review_model.invoke("Great headphones! Amazing battery but a bit heavy.")
print(result.sentiment)   # "positive"
print(result.pros)        # ["Amazing battery"]
print(result.cons)        # ["a bit heavy"]
```
