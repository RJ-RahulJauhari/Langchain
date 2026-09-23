"""Streamlit app for studying codebases and generating documentation."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import pathlib
import re
import shutil
import time
import uuid
import zipfile
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain.schema import Document  # noqa: F401
from langchain_community.embeddings import OllamaEmbeddings  # noqa: F401
from langchain_community.vectorstores import Chroma  # noqa: F401
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================
# App config & paths
# ============================
st.set_page_config(page_title="🧭 Codebase Study & Doc Writer", layout="wide")

APP_ROOT = pathlib.Path(".code_study").resolve()
APP_ROOT.mkdir(parents=True, exist_ok=True)

JOBS_FILE = APP_ROOT / "jobs.json"
CODEBASE_DIRS = APP_ROOT / "codebases"
OUT_DIR_ROOT = APP_ROOT / "outputs"
VSTORE_DIR = APP_ROOT / "vstore"

for path in (CODEBASE_DIRS, OUT_DIR_ROOT, VSTORE_DIR):
    path.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = os.getenv("CODE_STUDY_MODEL", "deepseek-v3.1:671b-cloud")
DEFAULT_TEMP = float(os.getenv("CODE_STUDY_TEMP", "0.3"))

# File context extraction defaults
DEFAULT_FILE_CHUNK = 4000
DEFAULT_FILE_OVERLAP = 500

# Module doc generation
DEFAULT_MODULE_BATCH = 1
DEFAULT_MAX_FILES_IN_PROMPT = 60  # number of file summaries for module prompt
DEFAULT_FILE_SUMMARY_PASSES = 1   # passes per chunk (single pass over all chunks by default)


# ============================
# Helpers
# ============================
def _hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _load_jobs() -> dict:
    if JOBS_FILE.exists():
        try:
            return json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_jobs(payload: dict) -> None:
    JOBS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _stream_save(uploaded, dest: pathlib.Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    uploaded.seek(0)
    with open(dest, "wb") as handle:
        while True:
            data = uploaded.read(chunk_size)
            if not data:
                break
            handle.write(data)
    uploaded.seek(0)


def _read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""


# ============================
# Model (cached)
# ============================
@st.cache_resource
def get_model(name: str, temperature: float) -> ChatOllama:
    """Return a cached Ollama chat model instance."""
    return ChatOllama(model=name, temperature=temperature)


# ============================
# Discovery & structure
# ============================
def discover_files(root: pathlib.Path, exts: List[str], exclude_dirs: List[str]) -> List[pathlib.Path]:
    include_exts = {ext.lower() for ext in exts}
    excluded = {directory.strip().lower() for directory in exclude_dirs}
    matches: List[pathlib.Path] = []

    for path in root.rglob("*"):
        if path.is_dir():
            parts = [part.lower() for part in path.parts]
            if any(target in parts for target in excluded):
                continue
            continue

        if path.suffix.lower() in include_exts:
            parts = [part.lower() for part in path.parts]
            if any(target in parts for target in excluded):
                continue
            matches.append(path)

    return matches


def build_tree_summary(root: pathlib.Path, files: List[pathlib.Path]) -> Tuple[dict, str]:
    tree: dict = {}
    root_str = str(root.resolve())

    def add_path(path: pathlib.Path) -> None:
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        node = tree
        for index, part in enumerate(parts):
            if index == len(parts) - 1:
                node.setdefault("__files__", []).append(part)
            else:
                node = node.setdefault(part, {})

    for file_path in files:
        add_path(file_path)

    def walk(node: dict, prefix: str = "") -> List[str]:
        lines: List[str] = []
        directories = sorted(k for k in node.keys() if k != "__files__")
        file_entries = sorted(node.get("__files__", []))

        for directory in directories:
            lines.append(f"{prefix}{directory}/")
            lines.extend(walk(node[directory], prefix + "  "))

        for filename in file_entries:
            lines.append(f"{prefix}└─ {filename}")

        return lines

    sitemap_lines = [f"root: {root_str}", ""]
    sitemap_lines.extend(walk(tree))
    return tree, "\n".join(sitemap_lines)


# ============================
# LLM prompts
# ============================
def prompt_plan_modules(sitemap: str) -> str:
    template = """
You are a senior software architect. Given this repository sitemap, design a module plan
for high-quality documentation. Output ONLY JSON with this shape:

{{
  "modules": [
    {{
      "name": "string, kebab-or-snake case preferred",
      "description": "what this module is about and how it relates to others",
      "includes": ["glob or path patterns to collect files for this module"],
      "depends_on": ["other module names, if any"],
      "priority": 1
    }}
  ],
  "notes": "any global remarks"
}}

Repository sitemap:
{sitemap}

Rules:
- Prefer 5–20 modules for mid-sized codebases (adjust as needed).
- Use folder structure to propose module boundaries.
- Use simple patterns like "src/api/**", "app/**", "**/*.py" where appropriate.
- Set "priority" roughly in build/runtime dependency order.
""".strip()
    return template.format(sitemap=sitemap)


def prompt_refine_file_json(relpath: str, prior_json: Optional[dict], chunk: str) -> str:
    """Return a prompt that asks the LLM to refine a file summary for one chunk."""
    prior = json.dumps(prior_json, ensure_ascii=False) if prior_json else "null"
    template = """
Act as a precise code summarizer. You are summarizing ONE source file incrementally.
Given the prior JSON summary (may be null) and THIS code chunk, return ONLY JSON with this shape:

{{
  "file": "{relpath}",
  "purpose": "one-paragraph responsibility",
  "key_components": [
    {{"name": "...", "kind": "class|function|const|type|interface", "signature": "...", "role": "short"}}
  ],
  "apis": [{{"name": "...", "method": "GET|POST|...", "path": "...", "input": "...", "output": "..."}}],
  "dependencies": ["module or package names"],
  "config": ["important env vars or settings"],
  "interactions": ["other files/modules this touches (from imports/calls/notes)"],
  "notes": ["pitfalls, TODOs, errors, performance hints"]
}}

Rules:
- Merge with prior JSON; do not duplicate entries.
- Only include facts visible in the chunk; if unknown, omit.
- Keep it compact but accurate.

PRIOR_JSON:
{prior}

CHUNK (truncated if huge):
```code
{chunk}
```
""".strip()
    return template.format(relpath=relpath, prior=prior, chunk=chunk[:7000])


def prompt_module_doc(
    module_name: str,
    module_desc: str,
    file_cards: List[dict],
    deps: List[str],
    er_hint_md: str,
    mvc_hint: dict,
    api_snippets: List[dict],
    prior_doc: str,
) -> str:
    outline = """
If an MVC pattern is detected, structure the document as:

Overview

Controllers (entry points, routing, responsibilities)

Services / Use Cases (business logic, shared utilities)

Models / Entities / Persistence (schema, validation, storage)

Views / Templates (rendering, UI composition)

Request Flow (Controller → Service → Data) with step-by-step narration

Data flow & module interactions

Configuration, logging, error handling

Performance / scaling considerations

Examples (if obvious)

Otherwise, use a sensible structure grouping by responsibilities.
""".strip()
    mvc_flow_guidance = ""
    if mvc_hint.get("is_mvc"):
        mvc_flow_guidance = (
            "\n\nEmphasize the end-to-end flow for each surfaced API: describe how controllers validate input, "
            "hand work to services, and how services leverage models/storage, including responses, "
            "side effects, and cross-cutting concerns (auth, errors)."
        )
    prior_note = "(none)" if not prior_doc else prior_doc[-12000:]
    return (
        """You are writing DETAILED engineering documentation for the module {module_name}.

Module description:
{module_desc}

Upstream/related modules: {deps}
Database/ER hints (if present):
{er_hint}

MVC hint (heuristic): {mvc_hint}

Below are file cards distilled from the actual file contents in this module:
{file_cards}

API-related code excerpts to ground your explanations:
{api_snippets}

Prior documentation (append new sections, do NOT repeat existing material):
{prior_note}
{outline}
Write precise Markdown covering:

Overview & purpose

How this module connects to others (data/control flow)

Public APIs / CLI / entry points (from cards)

For each API endpoint, explain:
- Why the endpoint exists / business need
- Detailed Controller → Service → Data flow (include auth, validation, helpers)
- Request structure and expected response payload(s)
- Status codes returned and when
- Key helpers/utilities invoked (middleware, validators, shared services)
- Provide a short code block sourced from the provided snippets to anchor the explanation
- Which frontend component(s) or flows consume this endpoint, if evident from imports or file relationships

Important classes/functions with signatures & interactions (from cards)

Configuration & environment variables

Error handling, logging, observability

If DB used: tables/relations relevant to this module

Minimal examples (only if obvious)

Note uncertainties explicitly as "Not clear from available context".

Do not invent names or APIs not present in the file cards.
{mvc_flow_guidance}
""".strip()
    ).format(
        module_name=module_name,
        module_desc=module_desc,
        deps=", ".join(deps) if deps else "(none)",
        er_hint=er_hint_md or "(none)",
        mvc_hint=json.dumps(mvc_hint, ensure_ascii=False),
        file_cards=json.dumps(file_cards, ensure_ascii=False)[:16000],
        outline=outline,
        mvc_flow_guidance=mvc_flow_guidance,
        api_snippets=json.dumps(api_snippets, ensure_ascii=False)[:16000],
        prior_note=prior_note,
    )


def prompt_module_overview(
    module_name: str,
    module_desc: str,
    deps: List[str],
    er_hint_md: str,
    file_cards: List[dict],
    endpoints_summary: List[dict],
    mvc_hint: dict,
    prior_doc_tail: str,
) -> str:
    file_cards_json = json.dumps(file_cards, ensure_ascii=False)[:16000]
    file_cards_json = file_cards_json.replace("{", "{{").replace("}", "}}").replace("\n", " ")
    endpoints_json = json.dumps(endpoints_summary, ensure_ascii=False)[:8000]
    endpoints_json = endpoints_json.replace("{", "{{").replace("}", "}}").replace("\n", " ")
    prior_tail = (prior_doc_tail or "(none)").replace("{", "{{").replace("}", "}}")
    return (
        """You are updating documentation for module {module_name}.

Existing trailing documentation (do not repeat):
{prior_doc_tail}

Module description:
{module_desc}

Related modules / dependencies: {deps}
Database/ER hints: {er_hint}
MVC hint: {mvc_hint}

File card summaries:
{file_cards}

Endpoint summary (for reference only, do not deep dive yet):
{endpoints}

Write a concise Markdown section that introduces this module. Cover:
- High-level purpose and responsibilities
- Architecture or patterns (reference MVC hint if relevant)
- Key components/files (grouped by role)
- How this module connects to others and data sources
- A table listing the discovered endpoints with method, path, handler

Keep the tone engineering-focused. Do not document individual endpoints in depth; subsequent steps will append endpoint sections.
""".strip()
    ).format(
        module_name=module_name,
        module_desc=module_desc,
        deps=", ".join(deps) if deps else "(none)",
        er_hint=er_hint_md or "(none)",
        mvc_hint=json.dumps(mvc_hint, ensure_ascii=False),
        file_cards=file_cards_json,
        endpoints=endpoints_json,
        prior_doc_tail=prior_tail,
    )


def prompt_endpoint_doc(
    module_name: str,
    module_desc: str,
    deps: List[str],
    mvc_hint: dict,
    er_hint_md: str,
    endpoint: dict,
    file_card: dict,
    prior_doc_tail: str,
) -> str:
    methods = ", ".join(endpoint.get("methods", []))
    handler = endpoint.get("handler") or "(anonymous handler)"
    functions_info = json.dumps(endpoint.get("functions", []), ensure_ascii=False)[:4000]
    functions_info = functions_info.replace("{", "{{").replace("}", "}}").replace("\n", " ")
    file_card_json = json.dumps(file_card or {}, ensure_ascii=False)[:6000]
    file_card_json = file_card_json.replace("{", "{{").replace("}", "}}").replace("\n", " ")
    code = (endpoint.get("code", "") or "")[:4000]
    code_safe = code.replace("{", "{{").replace("}", "}}")
    prior_tail = (prior_doc_tail or "(none)").replace("{", "{{").replace("}", "}}")
    language = endpoint.get("language") or ""
    return (
        """You are appending documentation for module {module_name}. Continue where the existing text left off:
{prior_tail}

Create a NEW Markdown section for the endpoint below. Do not repeat previously documented endpoints.

Endpoint:
- Methods: {methods}
- Path: {path}
- Source file: {file}
- Handler: {handler}
- Function details (line numbers): {functions}
- Related module description: {module_desc}
- Module dependencies: {deps}
- MVC hint: {mvc_hint}
- ER hint: {er_hint}
- File summary: {file_card}

Code excerpt (use this as the authoritative context):
```{language}
{code}
```

Write Markdown covering:
- Heading `### {methods} {path}`
- Business need / why this endpoint exists
- Step-by-step flow: controller → services/helpers → data access (cite helpers, middleware, validations)
- Request schema (body, params, query) and response payload(s)
- Status codes and when they occur (success and failure)
- Mention authentication/authorization and error handling paths
- Highlight helpers/utilities/middleware invoked (with purpose)
- Note how the frontend or other modules interact with this endpoint if observable
- Include a concise code block (≤ 30 lines) from the snippet to anchor discussion
- Call out uncertainties as "Not clear from available context" if needed

Return ONLY the new Markdown section.
""".strip()
    ).format(
        module_name=module_name,
        module_desc=module_desc,
        deps=", ".join(deps) if deps else "(none)",
        mvc_hint=json.dumps(mvc_hint, ensure_ascii=False),
        er_hint=er_hint_md or "(none)",
        methods=methods,
        path=endpoint.get("path", "(unknown)"),
        file=endpoint.get("file", "(unknown)"),
        handler=handler,
        functions=functions_info,
        file_card=file_card_json,
        code=code_safe,
        language=language or "text",
        prior_tail=prior_tail,
    )


# ============================
# JSON extractor
# ============================
def extract_first_json(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


# ============================
# Context extraction
# ============================
def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=[
            "\nclass ",
            "\ndef ",
            "\nasync def ",
            "\n# ",
            "\n// ",
            "\n/**",
            "\n/*",
            "\nfunction ",
            "\nexport ",
            "\nif ",
            "\nfor ",
            "\nwhile ",
            "\n\n",
        ],
    )
    return splitter.split_text(text or "")


SIG_RE = re.compile(
    r"""
    (?:(?:def|class)\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:\(.*?\))?\s*:)
    |
    (?:(?:export\s+)?(?:const|let|var|function|class)\s+[A-Za-z_][A-Za-z0-9_]*)
    |
    (?:interface\s+[A-Za-z_][A-Za-z0-9_]*\s*\{)
    |
    (?:type\s+[A-Za-z_][A-Za-z0-9_]*\s*=)
    |
    (?:struct\s+[A-Za-z_][A-Za-z0-9_]*\s*\{)
    """,
    re.VERBOSE,
)

HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]


def _indent_level(line: str) -> int:
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip())


def _find_python_block_end(lines: List[str], func_index: int) -> int:
    base_indent = _indent_level(lines[func_index])
    last = func_index
    for idx in range(func_index + 1, len(lines)):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        indent = _indent_level(lines[idx])
        if indent <= base_indent and not stripped.startswith("@"):
            break
        last = idx
    return last + 1


def _find_js_block_end(lines: List[str], start_index: int) -> int:
    brace_balance = 0
    paren_balance = 0
    last = start_index
    seen_brace = False
    for idx in range(start_index, len(lines)):
        line = lines[idx]
        brace_balance += line.count("{") - line.count("}")
        if "{" in line:
            seen_brace = True
        paren_balance += line.count("(") - line.count(")")
        if line.strip():
            last = idx
        if seen_brace:
            if brace_balance <= 0 and idx > start_index:
                break
        else:
            if idx > start_index and paren_balance <= 0 and ";" in line:
                break
    return last + 1


def _language_from_suffix(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    if ext == ".py":
        return "python"
    if ext in (".ts", ".tsx"):
        return "typescript"
    if ext in (".js", ".jsx"):
        return "javascript"
    return ""


def _extract_functions_with_bounds(text: str, suffix: str) -> List[dict]:
    lines = text.splitlines()
    functions: List[dict] = []
    suffix = suffix.lower()
    py_pattern = re.compile(r'^\s*(async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')
    js_named_pattern = re.compile(
        r'^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('
    )
    js_const_pattern = re.compile(
        r'^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?(?:function\s*\(|\([^)]*\)\s*=>)'
    )
    js_default_pattern = re.compile(
        r'^\s*export\s+default\s+(?:async\s+)?function(?:\s+([A-Za-z_][A-Za-z0-9_]*))?\s*\('
    )

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if suffix == ".py":
            match = py_pattern.match(line)
            if match:
                name = match.group(2)
                decorator_start = idx
                while decorator_start > 0 and lines[decorator_start - 1].strip().startswith("@"):
                    decorator_start -= 1
                start_line = decorator_start + 1
                end_line = _find_python_block_end(lines, idx)
                snippet = "\n".join(lines[decorator_start:end_line])[:4000]
                functions.append(
                    {
                        "name": name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "code": snippet,
                        "language": "python",
                    }
                )
                idx = end_line
                continue
        if suffix in (".js", ".jsx", ".ts", ".tsx"):
            match = js_named_pattern.match(line)
            source = "named"
            if not match:
                match = js_const_pattern.match(line)
                source = "const"
            if not match:
                match = js_default_pattern.match(line)
                source = "default"
            if match:
                if source == "default":
                    name = match.group(1) or "default"
                else:
                    name = match.group(1)
                start_idx = idx
                while start_idx > 0 and lines[start_idx - 1].strip().startswith("export"):
                    start_idx -= 1
                start_line = start_idx + 1
                end_line = _find_js_block_end(lines, idx)
                snippet = "\n".join(lines[start_idx:end_line])[:4000]
                functions.append(
                    {
                        "name": name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "code": snippet,
                        "language": _language_from_suffix(f"dummy{suffix}"),
                    }
                )
                idx = end_line
                continue
        idx += 1
    return functions


def _parse_methods_from_args(arg_text: str) -> List[str]:
    matches = re.findall(
        r"['\"](get|post|put|delete|patch|options|head)['\"]",
        arg_text,
        flags=re.IGNORECASE,
    )
    methods: List[str] = []
    for item in matches:
        candidate = item.upper()
        if candidate not in methods:
            methods.append(candidate)
    return methods or ["GET"]


def _extract_endpoints(text: str, path: pathlib.Path, functions: List[dict]) -> List[dict]:
    lines = text.splitlines()
    endpoints: List[dict] = []
    suffix = path.suffix.lower()
    language = _language_from_suffix(path.as_posix())
    func_lookup = {f.get("name"): f for f in functions if f.get("name")}
    method_decorator_re = re.compile(
        r'^@(?:(?:[A-Za-z_][A-Za-z0-9_]*\.)*)(?:router|app|api|bp)[\w.]*\.(get|post|put|delete|patch|options|head)\s*\((?P<args>.*)$',
        re.IGNORECASE,
    )
    route_decorator_re = re.compile(
        r'^@(?:(?:[A-Za-z_][A-Za-z0-9_]*\.)*)(?:router|app|api|bp)[\w.]*\.route\s*\((?P<args>.*)$',
        re.IGNORECASE,
    )
    path_re = re.compile(r"['\"]([^'\"]+)['\"]")
    decorator_buffer: List[dict] = []

    if suffix == ".py":
        for idx, line in enumerate(lines):
            stripped = line.strip()
            method_match = method_decorator_re.match(stripped)
            if method_match:
                args = method_match.group("args") or ""
                path_match = path_re.search(args)
                route_path = path_match.group(1) if path_match else "(unknown)"
                methods = [method_match.group(1).upper()]
                decorator_buffer.append(
                    {"methods": methods, "path": route_path, "line": idx + 1}
                )
                continue
            route_match = route_decorator_re.match(stripped)
            if route_match:
                args = route_match.group("args") or ""
                path_match = path_re.search(args)
                route_path = path_match.group(1) if path_match else "(unknown)"
                methods = _parse_methods_from_args(args)
                decorator_buffer.append(
                    {"methods": [m.upper() for m in methods], "path": route_path, "line": idx + 1}
                )
                continue
            def_match = re.match(r'^\s*(async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', line)
            if def_match:
                func_name = def_match.group(2)
                if decorator_buffer:
                    func_info = func_lookup.get(func_name)
                    if func_info:
                        start_line = func_info.get("start_line", decorator_buffer[0]["line"])
                        end_line = func_info.get("end_line", start_line)
                        snippet = func_info.get("code", "") or "\n".join(
                            lines[start_line - 1 : end_line]
                        )
                    else:
                        start_line = decorator_buffer[0]["line"]
                        end_line = _find_python_block_end(lines, idx)
                        snippet = "\n".join(lines[start_line - 1 : end_line])
                    snippet = snippet[:4000]
                    functions_involved = []
                    if func_info:
                        functions_involved.append(
                            {
                                "name": func_info["name"],
                                "start_line": func_info["start_line"],
                                "end_line": func_info["end_line"],
                            }
                        )
                    for deco in decorator_buffer:
                        endpoints.append(
                            {
                                "methods": deco["methods"],
                                "path": deco["path"],
                                "handler": func_name,
                                "start_line": start_line,
                                "end_line": end_line,
                                "decorator_line": deco["line"],
                                "functions": functions_involved,
                                "code": snippet,
                                "language": "python",
                                "source": "python_decorator",
                            }
                        )
                    decorator_buffer = []
                else:
                    decorator_buffer = []
        decorator_buffer = []

    express_pattern = re.compile(
        r'\b(router|app)\.(get|post|put|delete|patch|options|head)\s*\(\s*([\'\"])([^\'\"]+)\3\s*,',
        re.IGNORECASE,
    )
    for idx, line in enumerate(lines):
        match = express_pattern.search(line)
        if not match:
            continue
        method = match.group(2).upper()
        route_path = match.group(4)
        snippet_lines: List[str] = []
        paren_balance = 0
        brace_balance = 0
        for offset in range(idx, len(lines)):
            snippet_lines.append(lines[offset])
            paren_balance += lines[offset].count("(") - lines[offset].count(")")
            brace_balance += lines[offset].count("{") - lines[offset].count("}")
            if paren_balance <= 0 and brace_balance <= 0 and offset > idx:
                break
        end_line = idx + len(snippet_lines)
        snippet = "\n".join(snippet_lines)[:4000]
        flat_call = " ".join(snippet_lines)
        handler_name = None
        if "=>" in flat_call:
            handler_name = "inline"
        else:
            handler_match = re.search(r'([A-Za-z_][A-Za-z0-9_\.]+)\s*\)\s*;?\s*$', flat_call)
            if handler_match:
                handler_name = handler_match.group(1)
        functions_involved = []
        if handler_name and handler_name != "inline":
            lookup = handler_name.split(".")[-1]
            info = func_lookup.get(lookup)
            if info:
                functions_involved.append(
                    {
                        "name": info["name"],
                        "start_line": info["start_line"],
                        "end_line": info["end_line"],
                    }
                )
        endpoints.append(
            {
                "methods": [method],
                "path": route_path,
                "handler": handler_name,
                "start_line": idx + 1,
                "end_line": end_line,
                "functions": functions_involved,
                "code": snippet,
                "language": language or _language_from_suffix(path.as_posix()),
                "source": "js_route",
            }
        )

    return endpoints


def summarize_file_shallow(path: pathlib.Path, max_preview_chars: int = 1200) -> dict:
    text = _read_text(path)
    preview = text[:max_preview_chars]
    signatures = SIG_RE.findall(text)[:40]
    functions = _extract_functions_with_bounds(text, path.suffix)
    endpoints = _extract_endpoints(text, path, functions)
    return {
        "file": path.as_posix(),
        "size": path.stat().st_size if path.exists() else 0,
        "preview": preview,
        "signatures": signatures,
        "functions": functions,
        "endpoints": endpoints,
    }


# ============================
# DB / ER detection (heuristic)
# ============================
SQL_CREATE_RE = re.compile(r"CREATE\s+TABLE\s+`?([A-Za-z0-9_]+)`?\s*\(", re.IGNORECASE)
SQL_REF_RE = re.compile(r"REFERENCES\s+`?([A-Za-z0-9_]+)`?", re.IGNORECASE)
ALCHEMY_TABLE_RE = re.compile(r"__tablename__\s*=\s*['\"]([A-Za-z0-9_]+)['\"]")
ALCHEMY_FK_RE = re.compile(r"ForeignKey\(\s*['\"]([A-Za-z0-9_.]+)['\"]")
DJANGO_MODEL_RE = re.compile(r"class\s+([A-Za-z0-9_]+)\s*\(\s*models\.Model\s*\)")
DJANGO_FK_RE = re.compile(r"models\.ForeignKey\(\s*['\"]([A-Za-z0-9_]+)['\"]")
PRISMA_MODEL_RE = re.compile(r"\bmodel\s+([A-Za-z_][A-Za-z0-9_]*)\s+\{")
MONGOOSE_REF_RE = re.compile(r"ref\s*:\s*['\"]([A-Za-z0-9_]+)['\"]")


def detect_er(files: List[pathlib.Path]) -> Tuple[dict, str]:
    graph = {"tables": {}}  # table -> {"refs": set()}

    def ensure_table(name: Optional[str]) -> None:
        if not name:
            return
        graph["tables"].setdefault(name, {"refs": set()})

    for file_path in files:
        text = _read_text(file_path)
        if not text:
            continue

        for match in SQL_CREATE_RE.finditer(text):
            ensure_table(match.group(1))

        for match in SQL_REF_RE.finditer(text):
            ensure_table(match.group(1))

        for match in ALCHEMY_TABLE_RE.finditer(text):
            ensure_table(match.group(1))

        for match in DJANGO_MODEL_RE.finditer(text):
            ensure_table(match.group(1))

        for match in PRISMA_MODEL_RE.finditer(text):
            ensure_table(match.group(1))

        created_tables = re.findall(r"CREATE\s+TABLE\s+`?([A-Za-z0-9_]+)`?\s*\(", text, flags=re.IGNORECASE)
        for creator in created_tables:
            refs = re.findall(r"REFERENCES\s+`?([A-Za-z0-9_]+)`?", text, flags=re.IGNORECASE)
            if not refs:
                continue
            ensure_table(creator)
            for referenced in refs:
                ensure_table(referenced)
                graph["tables"].setdefault(creator, {"refs": set()})
                graph["tables"][creator]["refs"].add(referenced)

        for match in ALCHEMY_FK_RE.finditer(text):
            ref_table = match.group(1)
            if ref_table and "." in ref_table:
                ref_table = ref_table.split(".")[0]
            ensure_table(ref_table)

        for match in DJANGO_FK_RE.finditer(text):
            ensure_table(match.group(1))

        for match in MONGOOSE_REF_RE.finditer(text):
            ensure_table(match.group(1))

    for table in list(graph["tables"].keys()):
        refs = graph["tables"][table].get("refs", set())
        graph["tables"][table]["refs"] = sorted(refs)

    lines = ["digraph ER {", "  rankdir=LR;", "  node [shape=box];"]
    for table in sorted(graph["tables"].keys()):
        lines.append(f'  "{table}";')
    for table, info in graph["tables"].items():
        for referenced in info.get("refs", []):
            lines.append(f'  "{table}" -> "{referenced}";')
    lines.append("}")

    return graph, "\n".join(lines)


# ============================
# MVC detection (heuristic)
# ============================
def detect_mvc_for_files(relpaths: List[str]) -> dict:
    """Group files into coarse MVC buckets."""
    buckets = {
        "controllers": [],
        "models": [],
        "views": [],
        "routes": [],
        "templates": [],
        "services": [],
    }

    for relpath in relpaths:
        lower = relpath.lower()
        parts = lower.split("/")
        name = parts[-1] if parts else lower

        if "controller" in name or "controllers" in parts or name.endswith("controller.java") or "http/controllers" in lower:
            buckets["controllers"].append(relpath)

        if "model" in name or "models" in parts or "entity" in name or "entities" in parts or name.endswith(".prisma"):
            buckets["models"].append(relpath)

        if "views" in parts or "templates" in parts or name.endswith((".html", ".twig", ".ejs", ".tsx")):
            buckets["views"].append(relpath)
            if "templates" in parts:
                buckets["templates"].append(relpath)

        if "routes" in parts or name in {"urls.py", "router.ts", "routes.ts", "routes.js"}:
            buckets["routes"].append(relpath)

        if "service" in name or "services" in parts or "repository" in name:
            buckets["services"].append(relpath)

        if name == "views.py":
            buckets["views"].append(relpath)
        if name == "models.py":
            buckets["models"].append(relpath)

    hits = sum(1 for key in ("controllers", "models", "views") if buckets[key])
    buckets["is_mvc"] = bool(hits >= 2 or (buckets["routes"] and hits >= 1))
    return buckets


# ============================
# Job & artifacts
# ============================
def ensure_job(root: pathlib.Path, title: str, include_exts: List[str], exclude_dirs: List[str]) -> dict:
    jobs = _load_jobs()
    for job in jobs.values():
        if job.get("root") == str(root.resolve()) and not job.get("finished", False):
            return job

    job_id = f"{_hash(str(root.resolve()))}-{str(uuid.uuid4())[:6]}"
    out_dir = OUT_DIR_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "id": job_id,
        "root": str(root.resolve()),
        "title": title or root.name,
        "created_at": time.time(),
        "include_exts": include_exts,
        "exclude_dirs": exclude_dirs,
        "finished": False,
        "structure_done": False,
        "plan_done": False,
        "map_done": False,
        "contexts_done": False,
        "er_done": False,
        "modules_done": [],
        "out_dir": str(out_dir),
        "sitemap_md": "sitemap.md",
        "tree_json": "tree.json",
        "plan_json": "modules_plan.json",
        "map_json": "modules_map.json",
        "contexts_json": "file_contexts.jsonl",
        "er_json": "er_graph.json",
        "er_dot": "er_graph.dot",
        "modules_meta_json": "modules_meta.json",
    }
    jobs[job_id] = job
    _save_jobs(jobs)
    return job


def save_artifact(job: dict, name: str, data, binary: bool = False) -> pathlib.Path:
    out_dir = pathlib.Path(job["out_dir"])
    path = out_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        with open(path, "wb") as handle:
            handle.write(data)
    else:
        if isinstance(data, (dict, list)):
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            path.write_text(str(data), encoding="utf-8")
    return path


# ============================
# Steps 1–5 (analyze → ER)
# ============================
def step_analyze_structure(job: dict, files: List[pathlib.Path]) -> Tuple[dict, str]:
    tree, sitemap = build_tree_summary(pathlib.Path(job["root"]), files)
    save_artifact(job, job["tree_json"], tree)
    save_artifact(job, job["sitemap_md"], sitemap)
    job["structure_done"] = True
    jobs = _load_jobs()
    jobs[job["id"]] = job
    _save_jobs(jobs)
    return tree, sitemap


def step_plan_modules(job: dict, model: ChatOllama, sitemap_md: str) -> dict:
    response = model.invoke([HumanMessage(content=prompt_plan_modules(sitemap_md))])
    plan = extract_first_json(getattr(response, "content", "") or "")
    if not plan or "modules" not in plan:
        plan = {
            "modules": [
                {
                    "name": "main",
                    "description": "All files",
                    "includes": ["**/*"],
                    "depends_on": [],
                    "priority": 1,
                }
            ],
            "notes": "fallback",
        }
    save_artifact(job, job["plan_json"], plan)
    job["plan_done"] = True
    jobs = _load_jobs()
    jobs[job["id"]] = job
    _save_jobs(jobs)
    return plan


def step_map_files_to_modules(job: dict, plan: dict, files: List[pathlib.Path]) -> dict:
    root = pathlib.Path(job["root"])
    module_map: Dict[str, List[str]] = {}
    include_exts = [ext.lower() for ext in job["include_exts"]]

    for module in plan.get("modules", []):
        name = module["name"]
        patterns = module.get("includes", []) or []
        collected: set[pathlib.Path] = set()

        for pattern in patterns:
            try:
                collected.update(pathlib.Path(match) for match in glob.glob(str(root / pattern), recursive=True))
            except Exception:
                continue

        rel_paths: List[str] = []
        for absolute in collected:
            path = pathlib.Path(absolute)
            if path.is_file() and path.suffix.lower() in include_exts:
                try:
                    rel_paths.append(path.relative_to(root).as_posix())
                except Exception:
                    continue

        if not rel_paths:
            all_rel = [file.relative_to(root).as_posix() for file in files]
            module_key = name.replace(" ", "-").replace("_", "-").lower()
            for rel in all_rel:
                if module_key in rel.replace("\\", "/").lower():
                    rel_paths.append(rel)

        module_map[name] = sorted(set(rel_paths))

    save_artifact(job, job["map_json"], module_map)
    job["map_done"] = True
    jobs = _load_jobs()
    jobs[job["id"]] = job
    _save_jobs(jobs)
    return module_map


def step_extract_contexts(
    job: dict,
    files: List[pathlib.Path],
    chunk_size: int,
    overlap: int,
    max_chunk_chars: int = 4000,
) -> str:
    root = pathlib.Path(job["root"])
    out_path = pathlib.Path(job["out_dir"]) / job["contexts_json"]
    with open(out_path, "w", encoding="utf-8") as handle:
        for file_path in files:
            rel = file_path.relative_to(root).as_posix()
            text = _read_text(file_path)
            chunks = chunk_text(text, chunk_size, overlap)
            shallow = summarize_file_shallow(file_path)
            shallow["chunks"] = [chunk[:max_chunk_chars] for chunk in chunks]
            shallow["relpath"] = rel
            handle.write(json.dumps(shallow, ensure_ascii=False) + "\n")

    job["contexts_done"] = True
    jobs = _load_jobs()
    jobs[job["id"]] = job
    _save_jobs(jobs)
    return str(out_path)


def load_file_contexts(job: dict) -> Dict[str, dict]:
    ctx_path = pathlib.Path(job["out_dir"]) / job["contexts_json"]
    contexts: Dict[str, dict] = {}
    if not ctx_path.exists():
        return contexts
    with open(ctx_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            contexts[obj["relpath"]] = obj
    return contexts


def step_detect_er(job: dict, files: List[pathlib.Path]) -> Tuple[dict, str]:
    graph, dot = detect_er(files)
    save_artifact(job, job["er_json"], graph)
    save_artifact(job, job["er_dot"], dot)
    job["er_done"] = True
    jobs = _load_jobs()
    jobs[job["id"]] = job
    _save_jobs(jobs)
    return graph, dot


# ============================
# File summary refinement (uses full file chunks)
# ============================
def _dedupe_by_key(dicts: List[dict], key: str) -> List[dict]:
    seen: Dict[str, dict] = {}
    for entry in dicts:
        item_key = str(entry.get(key))
        if item_key not in seen:
            seen[item_key] = entry
    return list(seen.values())


def refine_file_summary(model: ChatOllama, relpath: str, chunks: List[str], passes: int = 1) -> dict:
    state: Optional[dict] = None
    for _ in range(max(1, passes)):
        for chunk in chunks:
            prompt = prompt_refine_file_json(relpath, state, chunk)
            response = model.invoke([HumanMessage(content=prompt)])
            candidate = extract_first_json(getattr(response, "content", "") or "")
            if not candidate:
                continue

            if state is None:
                state = candidate
                continue

            state["file"] = relpath
            if candidate.get("purpose"):
                state["purpose"] = candidate["purpose"]

            def merge_list(key: str, name_key: Optional[str] = None) -> None:
                existing = state.get(key, []) or []
                incoming = candidate.get(key, []) or []
                if name_key:
                    state[key] = _dedupe_by_key(existing + incoming, name_key)
                else:
                    combined = {json.dumps(item, ensure_ascii=False) for item in (existing + incoming)}
                    state[key] = [json.loads(item) for item in combined]

            merge_list("key_components", "name")
            merge_list("apis", "name")
            merge_list("dependencies")
            merge_list("config")
            merge_list("interactions")
            merge_list("notes")

    return state or {
        "file": relpath,
        "purpose": "",
        "key_components": [],
        "apis": [],
        "dependencies": [],
        "config": [],
        "interactions": [],
        "notes": [],
    }


# ============================
# Module doc generation (uses refined file summaries)
# ============================
def load_modules_meta(job: dict) -> dict:
    path = pathlib.Path(job["out_dir"]) / job["modules_meta_json"]
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_modules_meta(job: dict, meta: dict) -> None:
    save_artifact(job, job["modules_meta_json"], meta)


def step_generate_module_docs(
    job: dict,
    model: ChatOllama,
    plan: dict,
    module_map: dict,
    file_contexts: Dict[str, dict],
    er_graph: dict,
    modules_to_do: List[str],
    max_files_in_prompt: int,
    file_summary_passes: int,
) -> List[str]:
    out_dir = pathlib.Path(job["out_dir"]) / "modules"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_modules_meta(job)
    er_hint_md = ""
    if er_graph and er_graph.get("tables"):
        tables = sorted(er_graph["tables"].keys())
        er_hint_md = "Tables detected: " + ", ".join(tables)

    completed: List[str] = []
    plan_modules = {module["name"]: module for module in plan.get("modules", [])}

    for module_name in modules_to_do:
        if module_name not in plan_modules:
            continue

        st.write(f"▶ Preparing **{module_name}** …")
        module = plan_modules[module_name]
        rels: List[str] = module_map.get(module_name, []) or []
        mvc_hint = detect_mvc_for_files(rels)

        file_cards: List[dict] = []
        file_card_map: Dict[str, dict] = {}
        functions_map: Dict[str, List[dict]] = {}
        module_endpoints: List[dict] = []
        endpoint_keys_seen: set[str] = set()

        file_cache_dir = pathlib.Path(job["out_dir"]) / "file_cards" / module_name
        file_cache_dir.mkdir(parents=True, exist_ok=True)

        for index, rel in enumerate(rels, start=1):
            st.caption(f"  - Summarizing {index}/{len(rels)}: `{rel}`")
            safe_rel = re.sub(r"[^a-zA-Z0-9_\-.]+", "_", rel)
            cache_path = file_cache_dir / f"{safe_rel}.json"

            ctx = file_contexts.get(rel)
            chunks = ctx.get("chunks", []) if ctx else []

            card = None
            if cache_path.exists():
                try:
                    card = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    card = None
            if card is None and ctx:
                card = refine_file_summary(model, rel, chunks or [], passes=file_summary_passes)
                cache_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
            if card is None:
                continue

            file_cards.append(card)
            file_card_map[rel] = card

            if ctx:
                functions_map[rel] = ctx.get("functions", []) or []
                for endpoint in ctx.get("endpoints", []) or []:
                    ep = dict(endpoint)
                    ep["file"] = rel
                    methods = [m.upper() for m in ep.get("methods", [])]
                    ep["methods"] = methods or ["GET"]
                    doc_key = "{}:{}:{}:{}".format(
                        rel,
                        "|".join(ep["methods"]),
                        ep.get("path", ""),
                        ep.get("handler") or ep.get("start_line", 0),
                    )
                    if doc_key in endpoint_keys_seen:
                        continue
                    endpoint_keys_seen.add(doc_key)
                    ep["doc_key"] = doc_key
                    ep["language"] = ep.get("language") or _language_from_suffix(rel)
                    ep["code"] = (ep.get("code") or "")[:4000]
                    module_endpoints.append(ep)

        if len(file_cards) > max_files_in_prompt:
            priority_set = set(
                mvc_hint.get("controllers", [])
                + mvc_hint.get("models", [])
                + mvc_hint.get("views", [])
                + mvc_hint.get("routes", [])
                + mvc_hint.get("services", [])
            )
            prioritized = [card for card in file_cards if card.get("file") in priority_set]
            prioritized.extend(card for card in file_cards if card.get("file") not in priority_set)
            file_cards_for_prompt = prioritized[:max_files_in_prompt]
        else:
            file_cards_for_prompt = file_cards

        module_endpoints.sort(
            key=lambda ep: (ep.get("path", ""), ep.get("methods", []), ep.get("handler") or "")
        )

        safe_name = re.sub(r"[^a-zA-Z0-9_\-.]+", "_", module_name)
        md_path = out_dir / f"{safe_name}.md"
        current_doc = ""
        if md_path.exists():
            try:
                current_doc = md_path.read_text(encoding="utf-8")
            except Exception:
                current_doc = ""

        module_meta = meta.get(module_name, {})
        processed_endpoints = set(module_meta.get("processed_endpoints", []))
        history = module_meta.get("history", [])
        endpoint_details = module_meta.get("endpoint_details", {})
        overview_generated = module_meta.get("overview_generated", False)

        def append_chunk(chunk: str) -> bool:
            nonlocal current_doc
            chunk = (chunk or "").strip()
            if not chunk:
                return False
            existing_text = current_doc.rstrip()
            with open(md_path, "a", encoding="utf-8") as handle:
                if existing_text and not existing_text.endswith("\n\n"):
                    handle.write("\n\n")
                handle.write(chunk)
                handle.write("\n")
            if current_doc:
                if not current_doc.endswith("\n\n"):
                    current_doc = current_doc.rstrip("\n") + "\n\n" + chunk
                else:
                    current_doc = current_doc.rstrip("\n") + chunk
            else:
                current_doc = chunk
            return True

        def update_meta(last_chunk: str | None = None) -> None:
            entry = {
                "files": rels,
                "mvc": mvc_hint,
                "doc_path": md_path.as_posix(),
                "generated_at": time.time(),
                "overview_generated": overview_generated,
                "processed_endpoints": sorted(processed_endpoints),
                "endpoints_total": len(module_endpoints),
                "available_endpoints": module_endpoints,
                "functions_by_file": functions_map,
                "history": history,
                "endpoint_details": endpoint_details,
                "last_chunk_preview": (last_chunk or current_doc[-500:]).strip() if (last_chunk or current_doc) else "",
            }
            meta[module_name] = entry
            save_modules_meta(job, meta)

        endpoints_summary = [
            {
                "methods": ep.get("methods", []),
                "path": ep.get("path"),
                "file": ep.get("file"),
                "handler": ep.get("handler"),
            }
            for ep in module_endpoints[:50]
        ]

        if not overview_generated:
            prior_tail = current_doc[-4000:] if current_doc else ""
            overview_msg = model.invoke(
                [
                    HumanMessage(
                        content=prompt_module_overview(
                            module_name=module_name,
                            module_desc=module.get("description", ""),
                            deps=module.get("depends_on", []),
                            er_hint_md=er_hint_md,
                            file_cards=file_cards_for_prompt,
                            endpoints_summary=endpoints_summary,
                            mvc_hint=mvc_hint,
                            prior_doc_tail=prior_tail,
                        )
                    )
                ]
            )
            overview_chunk = getattr(overview_msg, "content", "") or ""
            if append_chunk(overview_chunk):
                history.append(
                    {
                        "generated_at": time.time(),
                        "type": "overview",
                        "chars": len(overview_chunk.strip()),
                    }
                )
                overview_generated = True
                update_meta(overview_chunk[:500])

        endpoints_pending = [
            ep for ep in module_endpoints if ep.get("doc_key") not in processed_endpoints
        ]

        if endpoints_pending:
            for endpoint in endpoints_pending:
                methods_display = ", ".join(endpoint.get("methods", []))
                st.caption(
                    f"    • Documenting {methods_display} {endpoint.get('path', '(unknown)')} ({endpoint.get('file')})"
                )
                file_card = file_card_map.get(endpoint.get("file", ""), {})
                prior_tail = current_doc[-6000:] if current_doc else ""
                endpoint_msg = model.invoke(
                    [
                        HumanMessage(
                            content=prompt_endpoint_doc(
                                module_name=module_name,
                                module_desc=module.get("description", ""),
                                deps=module.get("depends_on", []),
                                mvc_hint=mvc_hint,
                                er_hint_md=er_hint_md,
                                endpoint=endpoint,
                                file_card=file_card,
                                prior_doc_tail=prior_tail,
                            )
                        )
                    ]
                )
                endpoint_chunk = getattr(endpoint_msg, "content", "") or ""
                if append_chunk(endpoint_chunk):
                    processed_endpoints.add(endpoint["doc_key"])
                    endpoint_details[endpoint["doc_key"]] = {
                        "methods": endpoint.get("methods", []),
                        "path": endpoint.get("path"),
                        "file": endpoint.get("file"),
                        "handler": endpoint.get("handler"),
                        "start_line": endpoint.get("start_line"),
                        "end_line": endpoint.get("end_line"),
                        "functions": endpoint.get("functions", []),
                    }
                    history.append(
                        {
                            "generated_at": time.time(),
                            "type": "endpoint",
                            "endpoint": endpoint.get("doc_key"),
                            "chars": len(endpoint_chunk.strip()),
                        }
                    )
                    update_meta(endpoint_chunk[:500])

        module_complete = False
        if module_endpoints:
            module_complete = len(processed_endpoints) >= len(module_endpoints)
        else:
            module_complete = overview_generated

        if module_complete:
            completed.append(module_name)

        update_meta()

    job["modules_done"] = sorted(set(job.get("modules_done", [])) | set(completed))
    if set(job["modules_done"]) >= set(module_map.keys()):
        job["finished"] = True
    jobs = _load_jobs()
    jobs[job["id"]] = job
    _save_jobs(jobs)

    return completed


# ============================
# UI
# ============================
def main() -> None:
    st.title("🧭 Codebase Study & Documentation Writer")
    st.caption("Tip: For huge uploads (>200 MB), set [server] maxUploadSize in .streamlit/config.toml.")

    col_left, col_right = st.columns(2)
    with col_left:
        root_str = st.text_input("Local codebase path (absolute or relative)", value="")
    with col_right:
        uploaded_zip = st.file_uploader("…or upload a .zip (supports very large files with config)", type=["zip"])

    root_path: Optional[pathlib.Path] = None
    if uploaded_zip is not None:
        with st.spinner("Saving & extracting .zip…"):
            try:
                job_id_hint = _hash(uploaded_zip.name + str(time.time())) + f"-{str(uuid.uuid4())[:6]}"
                base = CODEBASE_DIRS / job_id_hint
                base.mkdir(parents=True, exist_ok=True)
                zip_path = base / uploaded_zip.name
                _stream_save(uploaded_zip, zip_path)
                with zipfile.ZipFile(zip_path, "r", allowZip64=True) as zip_ref:
                    zip_ref.extractall(base / "src")
                root_path = base / "src"
                st.success(f"Extracted to: {root_path}")
            except Exception as exc:
                st.error(f"Zip extraction failed: {exc}")
                return
    elif root_str.strip():
        root_path = pathlib.Path(root_str).expanduser().resolve()
        if not root_path.exists() or not root_path.is_dir():
            st.error("Provided path does not exist or is not a directory.")
            return

    include_exts_default = [
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".java",
        ".go",
        ".rb",
        ".rs",
        ".cs",
        ".cpp",
        ".h",
        ".hpp",
        ".m",
        ".swift",
        ".kt",
        ".php",
        ".scala",
        ".lua",
        ".sql",
        ".prisma",
        ".json",
        ".yaml",
        ".yml",
    ]
    include_exts = st.text_input("File extensions (comma-separated)", value=",".join(include_exts_default))
    extensions = [
        ext.strip() if ext.strip().startswith(".") else "." + ext.strip()
        for ext in include_exts.split(",")
        if ext.strip()
    ]

    exclude_dirs_default = ".git,node_modules,dist,build,__pycache__,.venv,venv,.idea,.vscode"
    exclude_dirs = st.text_input("Exclude directories (comma-separated)", value=exclude_dirs_default)
    excluded_dirs = [directory.strip() for directory in exclude_dirs.split(",") if directory.strip()]

    st.divider()

    model_name = st.text_input(
        "Model name",
        value=DEFAULT_MODEL,
        help="An Ollama model ID (must be available locally).",
    )
    temperature = st.slider("Temperature", 0.0, 1.5, DEFAULT_TEMP, 0.05)

    per_file_chunk = st.number_input(
        "Per-file chunk (context extract)",
        min_value=400,
        max_value=4000,
        value=DEFAULT_FILE_CHUNK,
        step=100,
    )
    per_file_overlap = st.number_input(
        "Per-file overlap",
        min_value=0,
        max_value=800,
        value=DEFAULT_FILE_OVERLAP,
        step=50,
    )
    max_chunk_chars = st.number_input(
        "Max chars per chunk stored",
        min_value=1000,
        max_value=12000,
        value=4000,
        step=500,
    )

    modules_per_batch = st.number_input(
        "Modules per batch (generation)",
        min_value=1,
        max_value=20,
        value=DEFAULT_MODULE_BATCH,
        step=1,
    )
    max_files_in_prompt = st.number_input(
        "Max file cards per module prompt",
        min_value=10,
        max_value=200,
        value=DEFAULT_MAX_FILES_IN_PROMPT,
        step=5,
    )
    file_summary_passes = st.number_input(
        "File summary passes (per chunk)",
        min_value=1,
        max_value=3,
        value=DEFAULT_FILE_SUMMARY_PASSES,
        step=1,
    )

    if not root_path:
        st.info("Choose a local path or upload a zip to begin.")
        return

    job = ensure_job(root_path, title=root_path.name, include_exts=extensions, exclude_dirs=excluded_dirs)
    out_dir = pathlib.Path(job["out_dir"])
    st.markdown(f"**Job ID:** `{job['id']}`  •  **Root:** `{job['root']}`  •  **Output:** `{out_dir}`")

    files = discover_files(root_path, extensions, excluded_dirs)
    st.write(f"Matched **{len(files)}** files for study.")

    generate_docs = st.button("🚀 Generate Full Documentation", type="primary")

    with st.sidebar:
        st.header("📊 Pipeline Status")
        st.write(f"- Structure: {'✅' if job['structure_done'] else '⏳'}")
        st.write(f"- Module plan: {'✅' if job['plan_done'] else '⏳'}")
        st.write(f"- File mapping: {'✅' if job['map_done'] else '⏳'}")
        st.write(f"- Contexts: {'✅' if job['contexts_done'] else '⏳'}")
        st.write(f"- ER diagram: {'✅' if job['er_done'] else '⏳'}")
        st.write(f"- Modules completed: {len(job.get('modules_done', []))}")
        if job.get("finished"):
            st.success("🎉 Documentation complete")
        st.divider()
        if st.button("🧹 Clear cached file cards for next modules"):
            cache_root = out_dir / "file_cards"
            try:
                shutil.rmtree(cache_root)
                st.success("Cleared.")
            except FileNotFoundError:
                st.info("Nothing to clear.")
            except Exception as exc:
                st.error(f"Failed to clear cache: {exc}")

    def load_ctx_if_needed() -> Dict[str, dict]:
        return load_file_contexts(job)

    def do_structure():
        with st.status("Step 1: Analyzing file structure…", expanded=True) as status:
            tree, sitemap_md = step_analyze_structure(job, files)
            st.write("Built tree and sitemap.")
            st.code(sitemap_md[:4000])
            status.update(label="Structure analysis done ✅", state="complete")
        return tree, sitemap_md

    def do_plan(model_obj: ChatOllama, sitemap_text: str):
        with st.status("Step 2: Planning modules with LLM…", expanded=True) as status:
            plan = step_plan_modules(job, model_obj, sitemap_text)
            st.json(plan)
            status.update(label="Module plan created ✅", state="complete")
        return plan

    def do_map(plan_data: dict):
        with st.status("Step 3: Mapping files to modules…", expanded=True) as status:
            mapping = step_map_files_to_modules(job, plan_data, files)
            st.json(mapping)
            status.update(label="File → module map saved ✅", state="complete")
        return mapping

    def do_contexts():
        with st.status("Step 4: Extracting file contexts from FULL content…", expanded=True) as status:
            path_written = step_extract_contexts(
                job,
                files,
                int(per_file_chunk),
                int(per_file_overlap),
                int(max_chunk_chars),
            )
            st.write(f"Saved contexts: `{path_written}`")
            status.update(label="File contexts ready ✅", state="complete")
        return load_ctx_if_needed()

    def do_er():
        with st.status("Step 5: Detecting database & ER graph…", expanded=True) as status:
            graph, dot = step_detect_er(job, files)
            st.json(graph)
            st.write("Graphviz DOT:")
            st.code(dot[:4000], language="dot")
            try:
                st.graphviz_chart(dot)
            except Exception:
                st.info("Graphviz preview unavailable, DOT saved.")
            status.update(label="ER detection done ✅", state="complete")
        return graph, dot

    def do_modules(model_obj: ChatOllama, plan_data: dict, mapping: dict, contexts: Dict[str, dict], er_graph_data: dict):
        module_names = [module["name"] for module in plan_data.get("modules", [])]
        done = set(job.get("modules_done", []))
        todo = [name for name in module_names if name not in done][: int(modules_per_batch)]
        if not todo:
            st.success("All modules already generated.")
            return []
        with st.status(
            f"Step 6: Generating docs for modules: {', '.join(todo)} …",
            expanded=True,
        ) as status:
            generated = step_generate_module_docs(
                job,
                model_obj,
                plan_data,
                mapping,
                contexts,
                er_graph_data,
                todo,
                max_files_in_prompt=int(max_files_in_prompt),
                file_summary_passes=int(file_summary_passes),
            )
            st.write("Generated:", generated)
            status.update(label="Module batch generated ✅", state="complete")
        modules_dir = pathlib.Path(job["out_dir"]) / "modules"
        for name in generated:
            safe_name = re.sub(r"[^a-zA-Z0-9_\-.]+", "_", name)
            path = modules_dir / f"{safe_name}.md"
            if path.exists():
                st.markdown(f"- **{name}** → `{path}`")
        return generated

    model = get_model(model_name, float(temperature))

    if generate_docs:
        if job["structure_done"]:
            sitemap_path = out_dir / job["sitemap_md"]
            sitemap_md = sitemap_path.read_text(encoding="utf-8") if sitemap_path.exists() else ""
            if not sitemap_md:
                _, sitemap_md = do_structure()
        else:
            _, sitemap_md = do_structure()

        plan_path = out_dir / job["plan_json"]
        if job["plan_done"] and plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        else:
            plan = do_plan(model, sitemap_md)

        map_path = out_dir / job["map_json"]
        if job["map_done"] and map_path.exists():
            module_map = json.loads(map_path.read_text(encoding="utf-8"))
        else:
            module_map = do_map(plan)

        if job["contexts_done"]:
            contexts = load_ctx_if_needed()
            if not contexts:
                contexts = do_contexts()
        else:
            contexts = do_contexts()

        er_path = out_dir / job["er_json"]
        if job["er_done"] and er_path.exists():
            er_graph = json.loads(er_path.read_text(encoding="utf-8"))
        else:
            er_graph, _ = do_er()

        while True:
            generated = do_modules(model, plan, module_map, contexts, er_graph)
            if not generated:
                break

        st.toast("Documentation pipeline finished.", icon="✅")


if __name__ == "__main__":
    main()
