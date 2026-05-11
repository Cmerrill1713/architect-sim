"""Extract LLM call sites from Go, Python, and Rust source files.

Scans source code for LLM invocation patterns, classifies task type from
surrounding context, and extracts configuration (model, temperature, max_tokens).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..models import LLMCall


# ---------------------------------------------------------------------------
# Task-type classification patterns
# ---------------------------------------------------------------------------

TASK_CLASSIFIERS: dict[str, list[re.Pattern]] = {
    "classification": [
        re.compile(r"classify|categorize|label|detect|isA|type_of|intent", re.I),
        re.compile(r"\btrue\b|\bfalse\b|\byes\b|\bno\b|\bbool\b", re.I),
        re.compile(r"pick one|choose|select|which", re.I),
    ],
    "extraction": [
        re.compile(r"extract|parse|find|locate|identify|NER|entity", re.I),
        re.compile(r"JSON|json\.Unmarshal|json_schema|structured", re.I),
        re.compile(r"regex|pattern|field", re.I),
    ],
    "generation": [
        re.compile(r"generate|create|write|compose|draft|synthesize", re.I),
        re.compile(r"response|reply|answer|explain|describe", re.I),
        re.compile(r"creative|story|poem|essay", re.I),
    ],
    "planning": [
        re.compile(r"plan|strategy|steps|breakdown|decompose", re.I),
        re.compile(r"goal|objective|task|workflow|pipeline", re.I),
        re.compile(r"reason|think|analyze|evaluate", re.I),
    ],
    "routing": [
        re.compile(r"route|dispatch|delegate|forward|select.*model", re.I),
        re.compile(r"classify.*route|intent.*route", re.I),
    ],
    "code": [
        re.compile(r"code|function|implement|refactor|debug|fix", re.I),
        re.compile(r"golang|python|rust|javascript", re.I),
        re.compile(r"diff|patch|edit|modify", re.I),
    ],
}


# ---------------------------------------------------------------------------
# Structured output detection
# ---------------------------------------------------------------------------

STRUCTURED_OUTPUT_RE = re.compile(
    r"json_schema|json_mode|function_call|functions|tool_choice"
    r"|json\.Unmarshal|json\.NewDecoder|json_object"
    r"|response_format.*json|ResponseFormat|structured_output"
    r"|json\.loads|\.json\(\)|\.parse\(",
    re.I,
)


# ---------------------------------------------------------------------------
# Go LLM call patterns
# ---------------------------------------------------------------------------

# client.Complete(ctx, prompt)  /  client.CompleteWithOptions(ctx, prompt, opts)
GO_COMPLETE_RE = re.compile(
    r"(\w+)\.(Complete|CompleteWithOptions)\(",
    re.MULTILINE,
)

# HTTP posts to /v1/chat/completions
GO_CHAT_COMPLETIONS_HTTP_RE = re.compile(
    r"""(?:http\.Post|http\.NewRequest)\s*\([^)]*(?:/v1/chat/completions)""",
    re.MULTILINE,
)

# Any URL string containing /v1/chat/completions (covers fmt.Sprintf, string concat)
GO_CHAT_COMPLETIONS_URL_RE = re.compile(
    r"""/v1/chat/completions""",
    re.MULTILINE,
)

# Model field in Go struct literal or JSON
GO_MODEL_FIELD_RE = re.compile(
    r"""(?:Model:\s*"([^"]+)"|"model"\s*:\s*"([^"]+)")""",
    re.MULTILINE,
)

# Temperature in Go
GO_TEMPERATURE_RE = re.compile(
    r"""(?:Temperature:\s*([\d.]+)|"temperature"\s*:\s*([\d.]+))""",
    re.MULTILINE,
)

# MaxTokens in Go
GO_MAX_TOKENS_RE = re.compile(
    r"""(?:MaxTokens:\s*(\d+)|"max_tokens"\s*:\s*(\d+))""",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Python LLM call patterns
# ---------------------------------------------------------------------------

# OpenAI client: client.chat.completions.create(...)
PY_OPENAI_RE = re.compile(
    r"""(\w+)\.chat\.completions\.create\(""",
    re.MULTILINE,
)

# Legacy OpenAI: openai.ChatCompletion.create(...)
PY_OPENAI_LEGACY_RE = re.compile(
    r"""openai\.ChatCompletion\.create\(""",
    re.MULTILINE,
)

# requests.post to /v1/chat/completions
PY_REQUESTS_CHAT_RE = re.compile(
    r"""requests\.(post|put)\s*\([^)]*(?:/v1/chat/completions)""",
    re.MULTILINE | re.DOTALL,
)

# httpx to /v1/chat/completions
PY_HTTPX_CHAT_RE = re.compile(
    r"""(?:httpx|aiohttp)\.\w+\.(post|put)\s*\([^)]*(?:/v1/chat/completions)""",
    re.MULTILINE | re.DOTALL,
)

# LangChain ChatOpenAI
PY_LANGCHAIN_RE = re.compile(
    r"""ChatOpenAI\(""",
    re.MULTILINE,
)

# Ollama via LangChain or direct
PY_OLLAMA_RE = re.compile(
    r"""Ollama\(""",
    re.MULTILINE,
)

# Generic /v1/chat/completions in Python (catch-all)
PY_CHAT_URL_RE = re.compile(
    r"""/v1/chat/completions""",
    re.MULTILINE,
)

# model="..." in Python
PY_MODEL_RE = re.compile(
    r"""model\s*=\s*["']([^"']+)["']""",
    re.MULTILINE,
)

# temperature=... in Python
PY_TEMPERATURE_RE = re.compile(
    r"""temperature\s*=\s*([\d.]+)""",
    re.MULTILINE,
)

# max_tokens=... in Python
PY_MAX_TOKENS_RE = re.compile(
    r"""max_tokens\s*=\s*(\d+)""",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Rust LLM call patterns
# ---------------------------------------------------------------------------

RUST_CHAT_RE = re.compile(
    r"""\.post\s*\(\s*(?:format!\s*\()?[^)]*(?:/v1/chat/completions)""",
    re.MULTILINE,
)

RUST_MODEL_RE = re.compile(
    r'"model"\s*:\s*"([^"]+)"',
    re.MULTILINE,
)

RUST_TEMPERATURE_RE = re.compile(
    r'"temperature"\s*:\s*([\d.]+)',
    re.MULTILINE,
)

RUST_MAX_TOKENS_RE = re.compile(
    r'"max_tokens"\s*:\s*(\d+)',
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# System prompt extraction
# ---------------------------------------------------------------------------

# Go: system prompt in string literal or variable assignment
SYSTEM_PROMPT_RE = re.compile(
    r"""(?:system[Pp]rompt|systemMsg|sysPrompt|SYSTEM_PROMPT)\s*[:=]\s*(?:`([^`]{1,500})`|"([^"]{1,500})")""",
    re.MULTILINE,
)

# Python: role=system content
PY_SYSTEM_MSG_RE = re.compile(
    r'"role"\s*:\s*"system"[^}]*"content"\s*:\s*"([^"]{1,500})"',
    re.MULTILINE | re.DOTALL,
)

# Python keyword style: role="system"
PY_SYSTEM_MSG_KW_RE = re.compile(
    r"""role\s*=\s*["']system["'][^)]*content\s*=\s*["']([^"']{1,500})["']""",
    re.MULTILINE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# File extension mapping
# ---------------------------------------------------------------------------

LANG_EXTENSIONS: dict[str, list[str]] = {
    "go":     [".go"],
    "python": [".py"],
    "rust":   [".rs"],
}

SKIP_DIRS = {"vendor", "node_modules", "__pycache__", ".venv", "venv", "target", ".git"}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _line_number(content: str, pos: int) -> int:
    """Return 1-based line number for a byte offset."""
    return content[:pos].count("\n") + 1


def _context_window(content: str, pos: int, window: int = 50) -> str:
    """Return lines surrounding *pos*, +/- *window* lines."""
    lines = content.splitlines(keepends=True)
    target = _line_number(content, pos) - 1  # 0-based index
    start = max(0, target - window)
    end = min(len(lines), target + window + 1)
    return "".join(lines[start:end])


def _narrow_context(content: str, pos: int, window: int = 10) -> str:
    """Return a narrow context (+/- *window* lines) for config extraction.

    Useful for extracting model/temperature/max_tokens that belong to *this*
    specific call rather than a neighboring one.
    """
    return _context_window(content, pos, window)


def _classify_task(context: str) -> str:
    """Classify the LLM call's task type from surrounding source context.

    Scores each category by how many classifier patterns match.  Returns the
    highest-scoring category, or "unknown" if nothing matches.
    """
    scores: dict[str, int] = {}
    for category, patterns in TASK_CLASSIFIERS.items():
        score = sum(1 for p in patterns if p.search(context))
        if score > 0:
            scores[category] = score

    if not scores:
        return "unknown"
    return max(scores, key=scores.get)


def _estimate_complexity(task_type: str, has_structured: bool, prompt_len: int) -> str:
    """Estimate LLM call complexity from heuristics."""
    if task_type in ("classification", "routing"):
        return "simple"
    if task_type in ("planning", "code"):
        return "complex"
    if task_type == "extraction" or has_structured:
        return "moderate"
    if prompt_len > 300:
        return "complex"
    return "moderate"


def _extract_system_prompt(context: str) -> str:
    """Try to extract the first ~100 chars of a system prompt from context."""
    for pattern in (SYSTEM_PROMPT_RE, PY_SYSTEM_MSG_RE, PY_SYSTEM_MSG_KW_RE):
        m = pattern.search(context)
        if m:
            text = next((g for g in m.groups() if g), "")
            return text[:100].strip()
    return ""


def _has_structured_output(context: str) -> bool:
    return bool(STRUCTURED_OUTPUT_RE.search(context))


def _first_group(*groups: str | None) -> str:
    """Return the first non-None group value, or empty string."""
    for g in groups:
        if g is not None:
            return g
    return ""


# ---------------------------------------------------------------------------
# Per-language scanners
# ---------------------------------------------------------------------------

def _extract_config(
    content: str,
    pos: int,
    model_re: re.Pattern,
    temp_re: re.Pattern,
    maxtok_re: re.Pattern,
    multi_group: bool = True,
) -> tuple:
    """Extract model, temperature, max_tokens from context around a call site.

    Tries a narrow window (+/- 10 lines) first to get config that belongs to
    *this* call.  Falls back to the broader window (+/- 50 lines).

    Returns (model, temperature, max_tokens, narrow_ctx, broad_ctx).
    """
    narrow = _narrow_context(content, pos)
    broad = _context_window(content, pos)

    def _search_config(ctx: str):
        model = ""
        m = model_re.search(ctx)
        if m:
            model = _first_group(*m.groups()) if multi_group else m.group(1)
        temp = -1.0
        m = temp_re.search(ctx)
        if m:
            try:
                val = _first_group(*m.groups()) if multi_group else m.group(1)
                temp = float(val)
            except ValueError:
                pass
        max_tok = -1
        m = maxtok_re.search(ctx)
        if m:
            try:
                val = _first_group(*m.groups()) if multi_group else m.group(1)
                max_tok = int(val)
            except ValueError:
                pass
        return model, temp, max_tok

    # Try narrow first
    model, temp, max_tok = _search_config(narrow)
    # Fill gaps from broad
    if not model or temp < 0 or max_tok < 0:
        bm, bt, bmt = _search_config(broad)
        if not model:
            model = bm
        if temp < 0:
            temp = bt
        if max_tok < 0:
            max_tok = bmt

    return model, temp, max_tok, narrow, broad


def _scan_go(service_name: str, content: str, file_path: str) -> list[LLMCall]:
    """Find LLM call sites in a Go source file."""
    calls: list[LLMCall] = []
    seen_lines: set[int] = set()

    def _add(pos: int, call_type: str):
        line = _line_number(content, pos)
        if line in seen_lines:
            return
        seen_lines.add(line)

        model, temp, max_tok, narrow, broad = _extract_config(
            content, pos, GO_MODEL_FIELD_RE, GO_TEMPERATURE_RE, GO_MAX_TOKENS_RE,
        )

        sys_prompt = _extract_system_prompt(broad)
        structured = _has_structured_output(broad)
        task = _classify_task(broad)
        complexity = _estimate_complexity(task, structured, len(sys_prompt))

        calls.append(LLMCall(
            service_name=service_name,
            file_path=file_path,
            line_number=line,
            call_type=call_type,
            model=model,
            temperature=temp,
            max_tokens=max_tok,
            task_type=task,
            system_prompt_preview=sys_prompt,
            has_structured_output=structured,
            estimated_complexity=complexity,
        ))

    # Pattern 1: client.Complete / CompleteWithOptions
    for m in GO_COMPLETE_RE.finditer(content):
        method = m.group(2)
        ct = "complete" if method == "Complete" else "complete_with_options"
        _add(m.start(), ct)

    # Pattern 2: HTTP calls to /v1/chat/completions (direct http.Post / NewRequest)
    for m in GO_CHAT_COMPLETIONS_HTTP_RE.finditer(content):
        _add(m.start(), "chat_completions")

    # Pattern 3: Catch remaining /v1/chat/completions references in Go
    # (e.g., fmt.Sprintf, string variables) -- but only in lines that look
    # like they are *making* a request (not just defining a constant).
    for m in GO_CHAT_COMPLETIONS_URL_RE.finditer(content):
        line = _line_number(content, m.start())
        if line not in seen_lines:
            # Check if the surrounding line has request-related keywords
            line_start = content.rfind("\n", 0, m.start()) + 1
            line_end = content.find("\n", m.end())
            line_text = content[line_start:line_end if line_end != -1 else len(content)]
            if re.search(r"Post|Request|Do\(|Send|fetch|call|invoke|Sprintf", line_text, re.I):
                _add(m.start(), "chat_completions")

    # Pattern 4: Standalone model field references that indicate an LLM call
    # Only if we haven't already found a call on that line.
    # We look for model field assignments near HTTP or Complete calls to avoid
    # false positives on unrelated structs.

    return calls


def _scan_python(service_name: str, content: str, file_path: str) -> list[LLMCall]:
    """Find LLM call sites in a Python source file."""
    calls: list[LLMCall] = []
    seen_lines: set[int] = set()

    def _add(pos: int, call_type: str):
        line = _line_number(content, pos)
        if line in seen_lines:
            return
        seen_lines.add(line)

        model, temp, max_tok, narrow, broad = _extract_config(
            content, pos, PY_MODEL_RE, PY_TEMPERATURE_RE, PY_MAX_TOKENS_RE,
            multi_group=False,
        )

        sys_prompt = _extract_system_prompt(broad)
        structured = _has_structured_output(broad)
        task = _classify_task(broad)
        complexity = _estimate_complexity(task, structured, len(sys_prompt))

        calls.append(LLMCall(
            service_name=service_name,
            file_path=file_path,
            line_number=line,
            call_type=call_type,
            model=model,
            temperature=temp,
            max_tokens=max_tok,
            task_type=task,
            system_prompt_preview=sys_prompt,
            has_structured_output=structured,
            estimated_complexity=complexity,
        ))

    # OpenAI client
    for m in PY_OPENAI_RE.finditer(content):
        _add(m.start(), "chat_completions")

    # Legacy OpenAI
    for m in PY_OPENAI_LEGACY_RE.finditer(content):
        _add(m.start(), "chat_completions")

    # requests.post to chat/completions
    for m in PY_REQUESTS_CHAT_RE.finditer(content):
        _add(m.start(), "chat_completions")

    # httpx / aiohttp
    for m in PY_HTTPX_CHAT_RE.finditer(content):
        _add(m.start(), "chat_completions")

    # LangChain ChatOpenAI
    for m in PY_LANGCHAIN_RE.finditer(content):
        _add(m.start(), "langchain")

    # Ollama
    for m in PY_OLLAMA_RE.finditer(content):
        _add(m.start(), "ollama")

    # Catch-all: remaining /v1/chat/completions URLs in request-like lines
    for m in PY_CHAT_URL_RE.finditer(content):
        line = _line_number(content, m.start())
        if line not in seen_lines:
            line_start = content.rfind("\n", 0, m.start()) + 1
            line_end = content.find("\n", m.end())
            line_text = content[line_start:line_end if line_end != -1 else len(content)]
            if re.search(r"post|put|request|fetch|send|aiohttp|httpx", line_text, re.I):
                _add(m.start(), "chat_completions")

    return calls


def _scan_rust(service_name: str, content: str, file_path: str) -> list[LLMCall]:
    """Find LLM call sites in a Rust source file."""
    calls: list[LLMCall] = []
    seen_lines: set[int] = set()

    def _add(pos: int, call_type: str):
        line = _line_number(content, pos)
        if line in seen_lines:
            return
        seen_lines.add(line)

        model, temp, max_tok, narrow, broad = _extract_config(
            content, pos, RUST_MODEL_RE, RUST_TEMPERATURE_RE, RUST_MAX_TOKENS_RE,
            multi_group=False,
        )

        sys_prompt = _extract_system_prompt(broad)
        structured = _has_structured_output(broad)
        task = _classify_task(broad)
        complexity = _estimate_complexity(task, structured, len(sys_prompt))

        calls.append(LLMCall(
            service_name=service_name,
            file_path=file_path,
            line_number=line,
            call_type=call_type,
            model=model,
            temperature=temp,
            max_tokens=max_tok,
            task_type=task,
            system_prompt_preview=sys_prompt,
            has_structured_output=structured,
            estimated_complexity=complexity,
        ))

    for m in RUST_CHAT_RE.finditer(content):
        _add(m.start(), "chat_completions")

    return calls


# ---------------------------------------------------------------------------
# Language dispatch
# ---------------------------------------------------------------------------

_SCANNERS = {
    "go":     (_scan_go,     [".go"]),
    "python": (_scan_python, [".py"]),
    "rust":   (_scan_rust,   [".rs"]),
}


def _should_skip(path: Path) -> bool:
    """Return True if the path is inside a directory we should skip."""
    parts = set(path.parts)
    return bool(parts & SKIP_DIRS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_llm_calls(service_name: str, source_dir: str, language: str) -> list[LLMCall]:
    """Extract all LLM call sites from a service's source code.

    Args:
        service_name: Logical service name (e.g. "agi-core").
        source_dir:   Root directory of the service source.
        language:     One of "go", "python", "rust".

    Returns:
        List of LLMCall objects, one per detected call site.
    """
    if language not in _SCANNERS:
        return []

    scanner_fn, extensions = _SCANNERS[language]
    calls: list[LLMCall] = []
    source_path = Path(source_dir)

    for ext in extensions:
        for src_file in sorted(source_path.rglob(f"*{ext}")):
            if _should_skip(src_file):
                continue
            # Skip test files
            if src_file.name.endswith("_test.go") or src_file.name.startswith("test_"):
                continue
            if src_file.stem.endswith("_test"):
                continue

            try:
                content = src_file.read_text(errors="replace")
            except (OSError, IOError):
                continue

            calls.extend(scanner_fn(service_name, content, str(src_file)))

    return calls


def extract_llm_calls_all_languages(service_name: str, source_dir: str) -> list[LLMCall]:
    """Extract LLM calls across all supported languages in a source directory.

    Useful when the service language is unknown or mixed.
    """
    all_calls: list[LLMCall] = []
    for lang in _SCANNERS:
        all_calls.extend(extract_llm_calls(service_name, source_dir, lang))
    return all_calls


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_llm_report(calls: list[LLMCall]) -> str:
    """Format LLM call analysis as a markdown report.

    Includes:
    - Total call sites by service
    - Task type distribution
    - Complexity distribution
    - Model usage
    - Actionable recommendations
    """
    if not calls:
        return "# LLM Call Analysis\n\nNo LLM call sites detected.\n"

    lines: list[str] = []
    lines.append("# LLM Call Analysis\n")
    lines.append(f"**Total call sites**: {len(calls)}\n")

    # -- By service ----------------------------------------------------------
    by_service: Counter = Counter(c.service_name for c in calls)
    lines.append("## Call Sites by Service\n")
    lines.append("| Service | Count |")
    lines.append("|---------|------:|")
    for svc, count in by_service.most_common():
        lines.append(f"| {svc} | {count} |")
    lines.append("")

    # -- Task type distribution -----------------------------------------------
    by_task: Counter = Counter(c.task_type for c in calls)
    total = len(calls)
    lines.append("## Task Type Distribution\n")
    lines.append("| Task Type | Count | % |")
    lines.append("|-----------|------:|----:|")
    for task, count in by_task.most_common():
        pct = count / total * 100
        bar = "#" * int(pct / 5)
        lines.append(f"| {task} | {count} | {pct:.1f}% {bar} |")
    lines.append("")

    # -- Complexity distribution ----------------------------------------------
    by_complexity: Counter = Counter(c.estimated_complexity for c in calls)
    lines.append("## Complexity Distribution\n")
    lines.append("| Complexity | Count | % |")
    lines.append("|------------|------:|----:|")
    for comp in ("simple", "moderate", "complex"):
        count = by_complexity.get(comp, 0)
        pct = count / total * 100
        lines.append(f"| {comp} | {count} | {pct:.1f}% |")
    lines.append("")

    # -- Model usage ----------------------------------------------------------
    models_used = [c for c in calls if c.model]
    if models_used:
        by_model: Counter = Counter(c.model for c in models_used)
        lines.append("## Model Usage\n")
        lines.append("| Model | Count | Services |")
        lines.append("|-------|------:|----------|")
        for model, count in by_model.most_common():
            svcs = sorted({c.service_name for c in calls if c.model == model})
            lines.append(f"| `{model}` | {count} | {', '.join(svcs)} |")
        lines.append("")
    else:
        lines.append("## Model Usage\n")
        lines.append("No explicit model names detected (may use defaults or env vars).\n")

    # -- Call type breakdown ---------------------------------------------------
    by_type: Counter = Counter(c.call_type for c in calls)
    lines.append("## Call Type Breakdown\n")
    lines.append("| Call Type | Count |")
    lines.append("|-----------|------:|")
    for ct, count in by_type.most_common():
        lines.append(f"| {ct} | {count} |")
    lines.append("")

    # -- Structured output usage -----------------------------------------------
    structured_count = sum(1 for c in calls if c.has_structured_output)
    lines.append("## Structured Output\n")
    lines.append(f"- Calls with structured output (JSON mode / function calling): "
                 f"**{structured_count}** ({structured_count / total * 100:.1f}%)")
    lines.append(f"- Calls without: **{total - structured_count}**\n")

    # -- Temperature summary ---------------------------------------------------
    temp_calls = [c for c in calls if c.temperature >= 0]
    if temp_calls:
        lines.append("## Temperature Settings\n")
        temps = [c.temperature for c in temp_calls]
        lines.append(f"- Specified: {len(temp_calls)} / {total}")
        lines.append(f"- Range: {min(temps):.2f} - {max(temps):.2f}")
        lines.append(f"- Mean: {sum(temps) / len(temps):.2f}\n")

    # -- Detailed call list (top 30) -------------------------------------------
    lines.append("## Detailed Call Sites (top 30)\n")
    lines.append("| # | Service | File | Line | Type | Model | Task | Complexity |")
    lines.append("|---|---------|------|-----:|------|-------|------|------------|")
    for i, c in enumerate(calls[:30], 1):
        fname = Path(c.file_path).name
        model_str = f"`{c.model}`" if c.model else "-"
        lines.append(
            f"| {i} | {c.service_name} | {fname} | {c.line_number} "
            f"| {c.call_type} | {model_str} | {c.task_type} | {c.estimated_complexity} |"
        )
    if len(calls) > 30:
        lines.append(f"\n*...and {len(calls) - 30} more call sites.*\n")
    lines.append("")

    # -- Recommendations -------------------------------------------------------
    lines.append("## Recommendations\n")
    recs: list[str] = []

    simple_count = by_complexity.get("simple", 0)
    if simple_count > 0:
        recs.append(
            f"- **{simple_count} simple call(s)** (classification/routing) could "
            f"use a smaller model (4B-8B) for lower latency and cost."
        )

    complex_count = by_complexity.get("complex", 0)
    if complex_count > 0:
        recs.append(
            f"- **{complex_count} complex call(s)** (planning/code) should use the "
            f"densest available model for accuracy."
        )

    no_model = sum(1 for c in calls if not c.model)
    if no_model > 0:
        recs.append(
            f"- **{no_model} call(s)** have no explicit model specified. "
            f"Consider making model selection explicit for routing control."
        )

    no_temp = sum(1 for c in calls if c.temperature < 0)
    if no_temp > 0:
        recs.append(
            f"- **{no_temp} call(s)** have no explicit temperature. "
            f"Classification/extraction calls should use temperature=0 for "
            f"determinism."
        )

    if not structured_count and by_task.get("extraction", 0) > 0:
        recs.append(
            "- Extraction calls detected without structured output mode. "
            "Consider using JSON mode or function calling for reliability."
        )

    if not recs:
        recs.append("- No immediate optimization opportunities detected.")

    lines.extend(recs)
    lines.append("")

    return "\n".join(lines)
