# Phase 2: drift_runtime — Implementation Spec

## Overview

`drift_runtime` is the Python package that makes transpiled Drift programs actually execute. When `drift build` converts a `.drift` file to `.py`, the output starts with `import drift_runtime`. This module provides every function and class that the transpiled Python references.

**Goal:** After Phase 2, `drift run examples/hello.drift` produces real output. `ai.ask` calls a real LLM. `fetch` makes real HTTP requests. Pipelines process real data.

**Tech Stack:** Python 3.11+, `httpx` for HTTP, `anthropic` SDK for AI calls (with OpenAI as optional fallback), `json`/`csv` from stdlib for file I/O. Minimal dependencies.

**Architecture:** Single Python package `drift_runtime/` installed alongside the `drift/` compiler package. Organized into submodules by domain.

---

## Project Structure

```
drift_runtime/
  __init__.py          # Exports: ai, fetch, read, save, query, env, config, log
  ai.py                # AI primitives: ask, classify, embed, see, predict, enrich, score
  data.py              # fetch, read, save, query, merge
  config.py            # Loads drift.config YAML
  env.py               # Environment variable access (already handled by os.environ in transpiler, but helper utilities here)
  types.py             # Confident type, schema utilities
  pipeline.py          # Pipeline helper functions (deduplicate, group, etc.)
  exceptions.py        # Runtime error types
```

---

## drift.config

Every Drift project can have a `drift.config` file in the project root (YAML format). The runtime reads this at startup.

```yaml
name: my-project
version: 0.1.0

ai:
  provider: anthropic          # "anthropic" or "openai"
  default_model: claude-sonnet-4-5-20250929
  fallback_model: claude-haiku-4-5-20251001
  cache: true
  max_retries: 3
  timeout: 30                  # seconds

data:
  output_dir: ./output

secrets:
  source: env                  # "env" = read from environment variables
```

**If no drift.config exists**, use sensible defaults:
- provider: anthropic
- model: claude-sonnet-4-5-20250929
- cache: false
- max_retries: 2
- timeout: 30
- output_dir: ./output
- secrets source: env

**Config loading:** On first access to any `drift_runtime` function, load config from `drift.config` in the current working directory. Cache it for the duration of the program. Use PyYAML (`pyyaml`) for parsing.

---

## Module: drift_runtime.ai

This is the core module. Every AI primitive in Drift maps to a function here.

### Setup

The module initializes an AI client on first use based on `drift.config`:
- If provider is `anthropic`: use the `anthropic` Python SDK
- If provider is `openai`: use the `openai` Python SDK
- API key comes from environment: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

### Functions

#### `ai.ask(prompt: str, schema=None, context: dict = None) -> str | dict`

The primary AI inference function.

**Without schema:**
```python
result = drift_runtime.ai.ask("What is 2 + 2?")
# result = "4" (string response)
```

**With schema:**
```python
result = drift_runtime.ai.ask(
    "Analyze this property",
    schema=DealScore,
    context={"address": "123 Main St", "price": 285000}
)
# result = DealScore(address="123 Main St", arv=350000, ...) — populated dataclass instance
```

**Implementation:**
1. Build the messages array: `[{"role": "user", "content": prompt}]`
2. If `context` is provided, append it to the prompt: `f"{prompt}\n\nContext:\n{json.dumps(context, indent=2)}"`
3. If `schema` is provided, append to the system message: `f"Respond with valid JSON matching this schema: {schema_to_json_description(schema)}"`
4. Call the AI provider's API with the configured model
5. If `schema` is provided, parse the JSON response and construct the dataclass instance
6. If JSON parsing fails, retry once with a prompt asking for valid JSON
7. Handle retries according to `max_retries` in config
8. Return the string response or populated dataclass

#### `ai.classify(input: str, categories: list[str]) -> str`

Classifies input into one of the provided categories.

**Implementation:**
1. Build prompt: `f"Classify the following text into exactly one of these categories: {categories}\n\nText: {input}\n\nRespond with only the category name, nothing else."`
2. Call the AI provider
3. Strip whitespace from response
4. Validate response is one of the categories (if not, retry once)
5. Return the category string

#### `ai.embed(input: str) -> list[float]`

Generates an embedding vector.

**Implementation:**
- Anthropic: Use `anthropic` SDK's embedding endpoint if available, otherwise fall back to a prompt-based approximation or raise `NotImplementedError` with a helpful message
- OpenAI: Use `openai.embeddings.create(model="text-embedding-3-small", input=input)`
- Return list of floats

#### `ai.see(input: str | bytes, prompt: str) -> str`

Analyzes an image with AI vision.

**Implementation:**
1. If `input` is a file path string, read the file and base64 encode it
2. If `input` is bytes, base64 encode directly
3. Build a multimodal message with the image and prompt
4. Call the AI provider with vision-capable model
5. Return string response

#### `ai.predict(prompt: str, schema=None) -> dict | ConfidentValue`

Similar to `ai.ask` but semantically for predictions. If the transpiled code references `confident`, return a `ConfidentValue`.

**Implementation:**
1. Same as `ai.ask` but append to prompt: `"Also provide a confidence score between 0.0 and 1.0 for your prediction."`
2. Parse the response
3. If schema is provided, return populated dataclass
4. If `confident` type was used, return `ConfidentValue(value=..., confidence=...)`

#### `ai.enrich(items: list, prompt: str) -> list`

Enriches a list of items using AI. Used in pipelines.

**Implementation:**
1. For each item in the list (or batch them for efficiency):
   - Build prompt: `f"{prompt}\n\nItem: {json.dumps(item)}"`
   - Call AI provider
   - Parse response and merge enrichment data into the item
2. Return the enriched list
3. For large lists (>20 items), batch by sending multiple items per call

#### `ai.score(items: list, prompt: str) -> list`

Scores a list of items using AI. Used in pipelines.

**Implementation:**
1. For each item (or in batches):
   - Build prompt: `f"{prompt}\n\nItem: {json.dumps(item)}\n\nRespond with only a number."`
   - Call AI provider
   - Parse the numeric score
   - Add `score` field to the item
2. Return the scored list

---

## Module: drift_runtime.data

### `fetch(url: str, headers: dict = None, params: dict = None) -> list | dict`

Makes HTTP GET requests.

**Implementation:**
1. Use `httpx.get(url, headers=headers, params=params, timeout=config.timeout)`
2. If response is JSON (check Content-Type), parse and return
3. If response is CSV/text, return as string
4. Raise `DriftRuntimeError` on HTTP errors with helpful message
5. Support retries on 429/500/502/503/504

### `read(path: str) -> list | dict | str`

Reads files from disk.

**Implementation:**
- `.csv` → Parse with `csv.DictReader`, return `list[dict]`
- `.json` → Parse with `json.load`, return parsed object
- `.txt` / `.md` → Return string content
- `.pdf` → Attempt text extraction with basic approach, or return bytes with a note that PDF support requires additional packages
- Other → Return string content
- Raise `FileNotFoundError` with helpful message if file doesn't exist

### `save(data, path: str) -> None`

Saves data to disk.

**Implementation:**
1. Create parent directories if they don't exist (`os.makedirs`)
2. Determine format from file extension:
   - `.json` → `json.dump(data, indent=2)`
   - `.csv` → `csv.DictWriter` if data is list of dicts
   - `.txt` / `.md` → Write string directly
   - Dataclass → Convert to dict first, then save based on extension
3. Print `f"Saved: {path}"` to confirm

### `query(sql: str, source) -> list[dict]`

Executes SQL queries.

**Implementation:**
- For v1, support SQLite only: `import sqlite3`
- `source` should be a database path string or connection object
- Execute query, return results as list of dicts
- If source is a more complex database URL, raise `NotImplementedError` with message about future database support

### `merge(sources: list[list]) -> list`

Combines multiple lists into one.

**Implementation:**
- Simply concatenate: `result = []; for s in sources: result.extend(s); return result`

---

## Module: drift_runtime.types

### `ConfidentValue`

```python
@dataclass
class ConfidentValue:
    value: any
    confidence: float  # 0.0 to 1.0
    
    def __repr__(self):
        return f"ConfidentValue({self.value}, confidence={self.confidence:.0%})"
    
    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.value > other
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.value < other
        return NotImplemented
```

Supports comparison operators so `if estimate > 300000:` works naturally.

### `schema_to_json_description(schema_class) -> str`

Takes a dataclass and produces a JSON schema description string for AI prompting.

```python
def schema_to_json_description(cls) -> str:
    fields = {}
    for f in dataclasses.fields(cls):
        fields[f.name] = f.type.__name__ if hasattr(f.type, '__name__') else str(f.type)
    return json.dumps(fields, indent=2)
```

### `parse_ai_response_to_schema(response: str, schema_class) -> object`

Parses an AI JSON response into a dataclass instance.

```python
def parse_ai_response_to_schema(response: str, schema_class):
    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove first line
        text = text.rsplit("```", 1)[0]  # remove last fence
    
    data = json.loads(text)
    return schema_class(**data)
```

---

## Module: drift_runtime.pipeline

Helper functions used by transpiled pipeline code.

### `deduplicate(items: list, key: str) -> list`

```python
def deduplicate(items: list, key: str) -> list:
    seen = {}
    for item in items:
        k = item[key] if isinstance(item, dict) else getattr(item, key)
        if k not in seen:
            seen[k] = item
    return list(seen.values())
```

### `group_by(items: list, key: str) -> list[dict]`

```python
def group_by(items: list, key: str) -> list[dict]:
    groups = {}
    for item in items:
        k = item[key] if isinstance(item, dict) else getattr(item, key)
        groups.setdefault(k, []).append(item)
    return [{"key": k, "items": v} for k, v in groups.items()]
```

---

## Module: drift_runtime.config

```python
import os
import yaml  # pyyaml

_config = None

def get_config() -> dict:
    global _config
    if _config is not None:
        return _config
    
    config_path = os.path.join(os.getcwd(), "drift.config")
    
    defaults = {
        "ai": {
            "provider": "anthropic",
            "default_model": "claude-sonnet-4-5-20250929",
            "fallback_model": "claude-haiku-4-5-20251001",
            "cache": False,
            "max_retries": 2,
            "timeout": 30,
        },
        "data": {
            "output_dir": "./output",
        },
        "secrets": {
            "source": "env",
        },
    }
    
    if os.path.exists(config_path):
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        # Deep merge user_config into defaults
        _config = deep_merge(defaults, user_config)
    else:
        _config = defaults
    
    return _config
```

---

## Module: drift_runtime.exceptions

```python
class DriftRuntimeError(Exception):
    """Base runtime error."""
    pass

class DriftAIError(DriftRuntimeError):
    """AI inference failed."""
    pass

class DriftNetworkError(DriftRuntimeError):
    """HTTP request failed."""
    pass

class DriftFileError(DriftRuntimeError):
    """File operation failed."""
    pass

class DriftConfigError(DriftRuntimeError):
    """Configuration error."""
    pass
```

These map to Drift's `catch` blocks:
- `catch network_error:` → `except DriftNetworkError`
- `catch ai_error:` → `except DriftAIError`

The transpiler should already be mapping these. Verify and fix if needed.

---

## Module: drift_runtime/__init__.py

```python
"""Drift Runtime — makes transpiled Drift programs execute."""

from drift_runtime.ai import DriftAI
from drift_runtime.data import fetch, read, save, query, merge
from drift_runtime.config import get_config
from drift_runtime.types import ConfidentValue
from drift_runtime.pipeline import deduplicate, group_by
from drift_runtime.exceptions import (
    DriftRuntimeError, DriftAIError, DriftNetworkError,
    DriftFileError, DriftConfigError,
)

# Singleton AI instance
ai = DriftAI()

# Log function (simple wrapper)
def log(message):
    print(f"[drift] {message}")

__all__ = [
    "ai", "fetch", "read", "save", "query", "merge", "log",
    "get_config", "ConfidentValue", "deduplicate", "group_by",
    "DriftRuntimeError", "DriftAIError", "DriftNetworkError",
    "DriftFileError", "DriftConfigError",
]
```

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    "anthropic>=0.40.0",
    "openai>=1.50.0",
    "httpx>=0.27.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23.0",
]
```

---

## Testing Strategy

### Unit Tests (mock AI calls)

All AI tests should mock the API calls so tests don't require API keys or make real requests.

```python
# tests/test_runtime_ai.py
from unittest.mock import patch, MagicMock
from drift_runtime.ai import DriftAI
from dataclasses import dataclass

@dataclass
class TestSchema:
    name: str
    score: float

def test_ai_ask_simple():
    ai = DriftAI()
    with patch.object(ai, '_call_model', return_value="42"):
        result = ai.ask("What is the meaning of life?")
    assert result == "42"

def test_ai_ask_with_schema():
    ai = DriftAI()
    response_json = '{"name": "test", "score": 95.0}'
    with patch.object(ai, '_call_model', return_value=response_json):
        result = ai.ask("Analyze this", schema=TestSchema)
    assert isinstance(result, TestSchema)
    assert result.name == "test"
    assert result.score == 95.0

def test_ai_classify():
    ai = DriftAI()
    with patch.object(ai, '_call_model', return_value="urgent"):
        result = ai.classify("Server is down!", categories=["urgent", "routine", "spam"])
    assert result == "urgent"

def test_ai_classify_validates_category():
    ai = DriftAI()
    # First call returns invalid, second returns valid
    with patch.object(ai, '_call_model', side_effect=["invalid_category", "urgent"]):
        result = ai.classify("Server is down!", categories=["urgent", "routine", "spam"])
    assert result == "urgent"

def test_ai_score_items():
    ai = DriftAI()
    items = [{"name": "A"}, {"name": "B"}]
    with patch.object(ai, '_call_model', side_effect=["85", "42"]):
        result = ai.score(items, "Rate quality 1-100")
    assert result[0]["score"] == 85
    assert result[1]["score"] == 42

def test_ai_enrich_items():
    ai = DriftAI()
    items = [{"name": "A"}]
    with patch.object(ai, '_call_model', return_value='{"summary": "Great item"}'):
        result = ai.enrich(items, "Add a summary")
    assert "summary" in result[0]
```

### Unit Tests (data operations)

```python
# tests/test_runtime_data.py
import os
import json
import tempfile
from drift_runtime.data import fetch, read, save, merge

def test_read_json():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({"key": "value"}, f)
        f.flush()
        result = read(f.name)
    os.unlink(f.name)
    assert result == {"key": "value"}

def test_read_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("name,age\nAlice,30\nBob,25\n")
        f.flush()
        result = read(f.name)
    os.unlink(f.name)
    assert len(result) == 2
    assert result[0]["name"] == "Alice"

def test_save_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "output.json")
        save({"key": "value"}, path)
        with open(path) as f:
            data = json.load(f)
        assert data == {"key": "value"}

def test_save_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "output.csv")
        save([{"name": "Alice", "age": 30}], path)
        assert os.path.exists(path)
        result = read(path)
        assert result[0]["name"] == "Alice"

def test_save_creates_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nested", "deep", "output.json")
        save({"key": "value"}, path)
        assert os.path.exists(path)

def test_merge():
    a = [{"id": 1}, {"id": 2}]
    b = [{"id": 3}]
    result = merge([a, b])
    assert len(result) == 3

def test_read_nonexistent_file():
    import pytest
    with pytest.raises(FileNotFoundError):
        read("nonexistent_file.json")
```

### Unit Tests (fetch with mocking)

```python
# tests/test_runtime_fetch.py
from unittest.mock import patch, MagicMock
from drift_runtime.data import fetch

def test_fetch_json():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = [{"id": 1}]
    mock_response.raise_for_status = MagicMock()
    
    with patch("drift_runtime.data.httpx.get", return_value=mock_response):
        result = fetch("https://api.example.com/data")
    assert result == [{"id": 1}]

def test_fetch_with_headers():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status = MagicMock()
    
    with patch("drift_runtime.data.httpx.get", return_value=mock_response) as mock_get:
        result = fetch("https://api.example.com", headers={"X-Key": "abc"})
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs.get("headers") == {"X-Key": "abc"} or call_kwargs[1].get("headers") == {"X-Key": "abc"}
```

### Unit Tests (pipeline helpers)

```python
# tests/test_runtime_pipeline.py
from drift_runtime.pipeline import deduplicate, group_by

def test_deduplicate_dicts():
    items = [
        {"address": "123 Main", "price": 100},
        {"address": "456 Oak", "price": 200},
        {"address": "123 Main", "price": 150},  # duplicate
    ]
    result = deduplicate(items, "address")
    assert len(result) == 2

def test_deduplicate_preserves_first():
    items = [
        {"id": "a", "value": 1},
        {"id": "a", "value": 2},
    ]
    result = deduplicate(items, "id")
    assert result[0]["value"] == 1

def test_group_by():
    items = [
        {"city": "Austin", "name": "A"},
        {"city": "Austin", "name": "B"},
        {"city": "Denver", "name": "C"},
    ]
    result = group_by(items, "city")
    assert len(result) == 2
    austin = [g for g in result if g["key"] == "Austin"][0]
    assert len(austin["items"]) == 2
```

### Unit Tests (config)

```python
# tests/test_runtime_config.py
import os
import tempfile
from unittest.mock import patch
from drift_runtime.config import get_config, _reset_config

def test_default_config():
    _reset_config()  # reset cached config
    with patch("os.path.exists", return_value=False):
        config = get_config()
    assert config["ai"]["provider"] == "anthropic"
    assert config["ai"]["default_model"] == "claude-sonnet-4-5-20250929"

def test_custom_config():
    _reset_config()
    config_content = """
ai:
  provider: openai
  default_model: gpt-4o
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        f.flush()
        with patch("drift_runtime.config.os.path.join", return_value=f.name):
            with patch("os.path.exists", return_value=True):
                config = get_config()
    os.unlink(f.name)
    assert config["ai"]["provider"] == "openai"
```

### Unit Tests (types)

```python
# tests/test_runtime_types.py
from drift_runtime.types import ConfidentValue, parse_ai_response_to_schema
from dataclasses import dataclass

def test_confident_value_comparison():
    cv = ConfidentValue(value=350000, confidence=0.85)
    assert cv > 300000
    assert cv < 400000
    assert cv.confidence == 0.85

def test_confident_value_repr():
    cv = ConfidentValue(value=100, confidence=0.92)
    assert "92%" in repr(cv)

def test_parse_ai_response():
    @dataclass
    class Score:
        name: str
        value: int
    
    result = parse_ai_response_to_schema('{"name": "test", "value": 42}', Score)
    assert result.name == "test"
    assert result.value == 42

def test_parse_ai_response_with_code_fences():
    @dataclass
    class Score:
        name: str
    
    result = parse_ai_response_to_schema('```json\n{"name": "test"}\n```', Score)
    assert result.name == "test"
```

### End-to-End Tests (require API key)

Mark these so they only run when `ANTHROPIC_API_KEY` is set:

```python
# tests/test_runtime_e2e.py
import os
import pytest
import subprocess
import sys
import tempfile

requires_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)

DRIFT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_drift(filepath):
    return subprocess.run(
        [sys.executable, "-m", "drift.cli", "run", filepath],
        capture_output=True, text=True, cwd=DRIFT_DIR, timeout=30,
    )

@requires_api_key
def test_hello_world_runs():
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False) as f:
        f.write('name = "Drift"\nprint "Hello from {name}!"')
        f.flush()
        result = run_drift(f.name)
    os.unlink(f.name)
    assert result.returncode == 0
    assert "Hello from Drift!" in result.stdout

@requires_api_key
def test_ai_ask_runs():
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False) as f:
        f.write('answer = ai.ask("What is 2 + 2? Reply with just the number.")\nprint answer')
        f.flush()
        result = run_drift(f.name)
    os.unlink(f.name)
    assert result.returncode == 0
    assert "4" in result.stdout

@requires_api_key  
def test_fetch_runs():
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False) as f:
        f.write('data = fetch "https://httpbin.org/json"\nprint data')
        f.flush()
        result = run_drift(f.name)
    os.unlink(f.name)
    assert result.returncode == 0

@requires_api_key
def test_save_and_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        drift_file = os.path.join(tmpdir, "test.drift")
        out_file = os.path.join(tmpdir, "out.json")
        with open(drift_file, "w") as f:
            f.write(f'save {{ "result": 42 }} to "{out_file}"\ndata = read "{out_file}"\nprint data')
        result = run_drift(drift_file)
        assert result.returncode == 0
        assert os.path.exists(out_file)
```

---

## Transpiler Adjustments

Phase 2 may require small fixes to the transpiler output. Known items to check:

1. **`drift_runtime` import** — Verify the transpiled Python starts with `import drift_runtime` and uses `drift_runtime.ai.ask(...)`, `drift_runtime.fetch(...)`, etc.

2. **Schema dataclasses** — The transpiler emits `@dataclass` classes. Verify they include `from dataclasses import dataclass` in the output.

3. **Exception mapping** — Verify `catch network_error:` transpiles to `except drift_runtime.DriftNetworkError:` (not generic `Exception`).

4. **`env.X` mapping** — Currently transpiles to `os.environ["X"]`. This is fine. Make sure `import os` is in the transpiled output.

5. **Pipeline helpers** — Verify `deduplicate by field` transpiles to `drift_runtime.deduplicate(data, "field")` and `group by field` transpiles to `drift_runtime.group_by(data, "field")`.

6. **`log` statement** — Verify it transpiles to `drift_runtime.log(...)`.

7. **`save X to Y`** — Verify it transpiles to `drift_runtime.save(X, Y)`.

Run the existing Phase 1 end-to-end tests after any transpiler changes to make sure nothing breaks.

---

## Implementation Order

### Task 1: Project setup
- Create `drift_runtime/` package structure
- Add dependencies to `pyproject.toml`
- Create `drift_runtime/exceptions.py`

### Task 2: Config module
- Implement `drift_runtime/config.py`
- Write config tests

### Task 3: Types module
- Implement `ConfidentValue`, `schema_to_json_description`, `parse_ai_response_to_schema`
- Write types tests

### Task 4: Data module (read, save, merge)
- Implement file I/O functions
- Write data tests (these don't need mocking)

### Task 5: Data module (fetch)
- Implement HTTP fetch with httpx
- Write fetch tests with mocking

### Task 6: Pipeline helpers
- Implement `deduplicate`, `group_by`
- Write pipeline tests

### Task 7: AI module — core infrastructure
- Implement `DriftAI` class with `_call_model` method
- Support Anthropic and OpenAI providers
- Handle retries, timeouts, error wrapping
- Write tests with mocked `_call_model`

### Task 8: AI module — all primitives
- Implement `ask`, `classify`, `embed`, `see`, `predict`, `enrich`, `score`
- Write tests for each with mocking

### Task 9: Runtime __init__.py and integration
- Wire everything together in `__init__.py`
- Verify `import drift_runtime` works and exposes `drift_runtime.ai.ask`, `drift_runtime.fetch`, etc.

### Task 10: Transpiler adjustments
- Review and fix transpiled output to match runtime API
- Re-run all Phase 1 tests to verify no regressions
- Run transpiler output through Python to verify it imports correctly

### Task 11: End-to-end tests
- Create e2e tests that run `.drift` files through the full pipeline
- `hello.drift` should produce real output
- AI tests should work with real API key (mark as skip if no key)

### Task 12: drift run smoke test
- `drift run examples/hello.drift` → prints "Hello from Drift!"
- `drift run examples/pipeline.drift` → executes (may need mock data)
- `drift run examples/deal_analyzer.drift` → executes with API key
- Fix any remaining issues

---

## Success Criteria

Phase 2 is complete when:

1. `drift run examples/hello.drift` prints "Hello from Drift!" 
2. All unit tests pass without API keys (using mocks)
3. E2E tests pass with a valid `ANTHROPIC_API_KEY`
4. `ai.ask`, `ai.classify`, `ai.embed`, `ai.see`, `ai.predict` all work
5. `ai.enrich` and `ai.score` work on lists (pipeline usage)
6. `fetch` makes real HTTP requests
7. `read` and `save` handle JSON and CSV files
8. `drift.config` is loaded and respected
9. Error types map correctly from Drift `catch` blocks
10. Zero regressions in Phase 1 tests
