# Phase 2: drift_runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the `drift_runtime` Python package so transpiled Drift programs actually execute — AI calls hit real LLMs, `fetch` makes HTTP requests, pipelines process real data, and `drift run examples/hello.drift` prints output.

**Architecture:** Single Python package `drift_runtime/` alongside the existing `drift/` compiler package. Seven modules organized by domain (ai, data, config, types, pipeline, exceptions, \_\_init\_\_). AI module supports both Anthropic and OpenAI providers via `drift.config` YAML. All tests mock AI calls via `unittest.mock.patch` on `DriftAI._call_model`.

**Tech Stack:** Python 3.11+, `anthropic` SDK, `openai` SDK, `httpx`, `pyyaml`, `pytest`

---

## Critical Warnings (READ BEFORE EACH TASK)

1. **AI response parsing is fragile** — LLMs return JSON wrapped in markdown code fences (` ```json ... ``` `), with extra whitespace, or with schema mismatches. `parse_ai_response_to_schema()` must strip fences and handle malformed responses. Test with fenced, unfenced, and malformed JSON.

2. **Mock testing is mandatory** — Every AI test MUST use `unittest.mock.patch` so tests pass without an API key. The mock target is `DriftAI._call_model`. Do NOT write tests that make real API calls unless explicitly marked with the `requires_api_key` decorator.

3. **Transpiler output must match runtime API exactly** — If the transpiler emits `drift_runtime.fetch(url)` but the runtime function signature is `fetch(url, headers=None, params=None)`, things break silently. Task 10 exists specifically to catch these mismatches.

4. **Config caching pollutes tests** — `drift_runtime.config` caches after first `get_config()` call. Tests MUST call `_reset_config()` before each test to clear the cache.

---

### Task 1: Project Setup + Exceptions

**Files:**
- Create: `drift_runtime/__init__.py` (empty placeholder)
- Create: `drift_runtime/exceptions.py`
- Modify: `pyproject.toml`
- Create: `tests/test_runtime_exceptions.py`

**Context:** The `drift/` compiler package already exists and is installed. We're creating a sibling `drift_runtime/` package in the same repo. The transpiled Python code starts with `import drift_runtime`, so this package must be importable.

**Step 1: Write the failing test**

Create `tests/test_runtime_exceptions.py`:

```python
"""Tests for drift_runtime exception hierarchy."""

from drift_runtime.exceptions import (
    DriftRuntimeError,
    DriftAIError,
    DriftNetworkError,
    DriftFileError,
    DriftConfigError,
)


def test_drift_runtime_error_is_exception():
    assert issubclass(DriftRuntimeError, Exception)


def test_drift_ai_error_inherits_runtime_error():
    assert issubclass(DriftAIError, DriftRuntimeError)


def test_drift_network_error_inherits_runtime_error():
    assert issubclass(DriftNetworkError, DriftRuntimeError)


def test_drift_file_error_inherits_runtime_error():
    assert issubclass(DriftFileError, DriftRuntimeError)


def test_drift_config_error_inherits_runtime_error():
    assert issubclass(DriftConfigError, DriftRuntimeError)


def test_errors_carry_message():
    err = DriftAIError("model timed out")
    assert str(err) == "model timed out"


def test_errors_are_catchable_as_runtime_error():
    try:
        raise DriftNetworkError("connection refused")
    except DriftRuntimeError as e:
        assert "connection refused" in str(e)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_exceptions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift_runtime'`

**Step 3: Create package structure and implement**

Create `drift_runtime/__init__.py`:
```python
"""Drift Runtime — makes transpiled Drift programs execute."""
```

Create `drift_runtime/exceptions.py`:
```python
"""Drift runtime exception types.

These map to Drift catch blocks:
  catch network_error:  ->  except DriftNetworkError
  catch ai_error:       ->  except DriftAIError
"""


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

Update `pyproject.toml` to add dependencies and include `drift_runtime` as a package:

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "drift-lang"
version = "0.1.0"
description = "The AI-native programming language"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
    "anthropic>=0.40.0",
    "openai>=1.50.0",
    "httpx>=0.27.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[project.scripts]
drift = "drift.cli:main"

[tool.setuptools.packages.find]
include = ["drift*", "drift_runtime*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ethansurfas/drift && pip3 install -e . && python3 -m pytest tests/test_runtime_exceptions.py -v`
Expected: 7 tests PASS

**Step 5: Commit**

```bash
git add drift_runtime/__init__.py drift_runtime/exceptions.py pyproject.toml tests/test_runtime_exceptions.py
git commit -m "feat(runtime): add drift_runtime package with exception hierarchy"
```

---

### Task 2: Config Module

**Files:**
- Create: `drift_runtime/config.py`
- Create: `tests/test_runtime_config.py`

**Context:** The config module loads `drift.config` (YAML) from cwd on first call to `get_config()`. It caches the result in a module-level global. Tests MUST call `_reset_config()` before each test to avoid cache pollution. If no config file exists, sensible defaults are used (anthropic provider, claude-sonnet-4-5-20250929, 2 retries, 30s timeout).

**Step 1: Write the failing tests**

Create `tests/test_runtime_config.py`:

```python
"""Tests for drift_runtime.config module."""

import os
import tempfile
from drift_runtime.config import get_config, _reset_config


def test_default_config_when_no_file():
    _reset_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        config = get_config(config_dir=tmpdir)
    assert config["ai"]["provider"] == "anthropic"
    assert config["ai"]["default_model"] == "claude-sonnet-4-5-20250929"
    assert config["ai"]["max_retries"] == 2
    assert config["ai"]["timeout"] == 30
    assert config["data"]["output_dir"] == "./output"


def test_custom_config_overrides_defaults():
    _reset_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "drift.config")
        with open(config_path, "w") as f:
            f.write("ai:\n  provider: openai\n  default_model: gpt-4o\n")
        config = get_config(config_dir=tmpdir)
    assert config["ai"]["provider"] == "openai"
    assert config["ai"]["default_model"] == "gpt-4o"
    # Non-overridden defaults should still be present
    assert config["ai"]["max_retries"] == 2
    assert config["ai"]["timeout"] == 30


def test_config_caching():
    _reset_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        c1 = get_config(config_dir=tmpdir)
        c2 = get_config(config_dir=tmpdir)
    assert c1 is c2  # Same object — cached


def test_reset_config_clears_cache():
    _reset_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        c1 = get_config(config_dir=tmpdir)
        _reset_config()
        c2 = get_config(config_dir=tmpdir)
    assert c1 is not c2  # Different objects after reset


def test_deep_merge_preserves_nested_defaults():
    _reset_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "drift.config")
        with open(config_path, "w") as f:
            f.write("ai:\n  timeout: 60\n")
        config = get_config(config_dir=tmpdir)
    assert config["ai"]["timeout"] == 60
    assert config["ai"]["provider"] == "anthropic"  # Default preserved
    assert config["ai"]["max_retries"] == 2  # Default preserved


def test_empty_config_file_uses_defaults():
    _reset_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "drift.config")
        with open(config_path, "w") as f:
            f.write("")
        config = get_config(config_dir=tmpdir)
    assert config["ai"]["provider"] == "anthropic"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift_runtime.config'`

**Step 3: Implement the config module**

Create `drift_runtime/config.py`:

```python
"""Drift config loader.

Reads drift.config (YAML) from the project directory.
Caches result after first load. Call _reset_config() in tests.
"""

import os
import yaml

_config = None

DEFAULTS = {
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


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def get_config(config_dir: str | None = None) -> dict:
    """Load and return the Drift config, caching after first call."""
    global _config
    if _config is not None:
        return _config

    if config_dir is None:
        config_dir = os.getcwd()

    config_path = os.path.join(config_dir, "drift.config")

    if os.path.exists(config_path):
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        if user_config and isinstance(user_config, dict):
            _config = _deep_merge(DEFAULTS, user_config)
        else:
            _config = dict(DEFAULTS)
    else:
        _config = dict(DEFAULTS)

    return _config


def _reset_config():
    """Clear cached config. Call this in tests."""
    global _config
    _config = None
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_config.py -v`
Expected: 6 tests PASS

**Step 5: Commit**

```bash
git add drift_runtime/config.py tests/test_runtime_config.py
git commit -m "feat(runtime): add config module with YAML loading and caching"
```

---

### Task 3: Types Module

**Files:**
- Create: `drift_runtime/types.py`
- Create: `tests/test_runtime_types.py`

**Context:** This module provides `ConfidentValue` (wraps a value + confidence score, supports `>` and `<` comparisons against numbers), `schema_to_json_description` (introspects dataclass fields for AI prompting), and `parse_ai_response_to_schema` (parses JSON from AI responses including stripping markdown code fences). The parse function is fragile — test with fenced JSON, unfenced JSON, JSON with extra whitespace, and malformed JSON.

**Step 1: Write the failing tests**

Create `tests/test_runtime_types.py`:

```python
"""Tests for drift_runtime.types module."""

import pytest
from dataclasses import dataclass
from drift_runtime.types import (
    ConfidentValue,
    schema_to_json_description,
    parse_ai_response_to_schema,
)


# -- ConfidentValue tests --

def test_confident_value_stores_value_and_confidence():
    cv = ConfidentValue(value=350000, confidence=0.85)
    assert cv.value == 350000
    assert cv.confidence == 0.85


def test_confident_value_gt_number():
    cv = ConfidentValue(value=350000, confidence=0.85)
    assert cv > 300000
    assert not (cv > 400000)


def test_confident_value_lt_number():
    cv = ConfidentValue(value=350000, confidence=0.85)
    assert cv < 400000
    assert not (cv < 300000)


def test_confident_value_repr():
    cv = ConfidentValue(value=100, confidence=0.92)
    r = repr(cv)
    assert "100" in r
    assert "92%" in r


def test_confident_value_ge_number():
    cv = ConfidentValue(value=100, confidence=0.5)
    assert cv >= 100
    assert cv >= 99
    assert not (cv >= 101)


def test_confident_value_le_number():
    cv = ConfidentValue(value=100, confidence=0.5)
    assert cv <= 100
    assert cv <= 101
    assert not (cv <= 99)


def test_confident_value_eq_number():
    cv = ConfidentValue(value=42, confidence=0.9)
    assert cv == 42
    assert not (cv == 43)


# -- schema_to_json_description tests --

def test_schema_to_json_description_simple():
    @dataclass
    class Score:
        name: str
        value: int

    desc = schema_to_json_description(Score)
    assert "name" in desc
    assert "value" in desc
    assert "str" in desc
    assert "int" in desc


def test_schema_to_json_description_multiple_types():
    @dataclass
    class Analysis:
        address: str
        arv: float
        photos: list

    desc = schema_to_json_description(Analysis)
    assert "address" in desc
    assert "arv" in desc
    assert "photos" in desc


# -- parse_ai_response_to_schema tests --

def test_parse_plain_json():
    @dataclass
    class Score:
        name: str
        value: int

    result = parse_ai_response_to_schema('{"name": "test", "value": 42}', Score)
    assert result.name == "test"
    assert result.value == 42


def test_parse_fenced_json():
    @dataclass
    class Score:
        name: str

    result = parse_ai_response_to_schema('```json\n{"name": "test"}\n```', Score)
    assert result.name == "test"


def test_parse_fenced_json_no_language_tag():
    @dataclass
    class Score:
        name: str

    result = parse_ai_response_to_schema('```\n{"name": "test"}\n```', Score)
    assert result.name == "test"


def test_parse_json_with_whitespace():
    @dataclass
    class Score:
        name: str

    result = parse_ai_response_to_schema('  \n  {"name": "test"}  \n  ', Score)
    assert result.name == "test"


def test_parse_malformed_json_raises():
    @dataclass
    class Score:
        name: str

    with pytest.raises(Exception):
        parse_ai_response_to_schema("not json at all", Score)


def test_parse_json_extra_fields_ignored():
    @dataclass
    class Score:
        name: str

    # Extra field "extra" should not crash — just pass what the dataclass accepts
    result = parse_ai_response_to_schema('{"name": "test", "extra": 99}', Score)
    assert result.name == "test"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift_runtime.types'`

**Step 3: Implement the types module**

Create `drift_runtime/types.py`:

```python
"""Drift runtime types.

ConfidentValue — wraps a value + confidence score, supports numeric comparisons.
schema_to_json_description — introspects dataclass fields for AI prompting.
parse_ai_response_to_schema — parses AI JSON responses (handles code fences).
"""

import dataclasses
import json
import re


@dataclasses.dataclass
class ConfidentValue:
    """A value with an associated confidence score (0.0 to 1.0)."""
    value: object
    confidence: float

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

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return self.value >= other
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, (int, float)):
            return self.value <= other
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.value == other
        if isinstance(other, ConfidentValue):
            return self.value == other.value and self.confidence == other.confidence
        return NotImplemented

    def __hash__(self):
        return hash((self.value, self.confidence))


def schema_to_json_description(cls) -> str:
    """Produce a JSON schema description string from a dataclass for AI prompting."""
    fields = {}
    for f in dataclasses.fields(cls):
        fields[f.name] = f.type.__name__ if hasattr(f.type, "__name__") else str(f.type)
    return json.dumps(fields, indent=2)


def parse_ai_response_to_schema(response: str, schema_class):
    """Parse an AI JSON response into a dataclass instance.

    Handles markdown code fences, extra whitespace, and extra fields.
    """
    text = response.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        # Remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    text = text.strip()
    data = json.loads(text)

    # Only pass fields the dataclass expects
    valid_fields = {f.name for f in dataclasses.fields(schema_class)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    return schema_class(**filtered)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_types.py -v`
Expected: 15 tests PASS

**Step 5: Commit**

```bash
git add drift_runtime/types.py tests/test_runtime_types.py
git commit -m "feat(runtime): add types module with ConfidentValue and schema parsing"
```

---

### Task 4: Data Module — read, save, merge, query

**Files:**
- Create: `drift_runtime/data.py`
- Create: `tests/test_runtime_data.py`

**Context:** File I/O functions for Drift. `read` handles CSV→list[dict], JSON→parsed, text→str. `save` auto-creates directories, writes JSON/CSV/text by extension, converts dataclasses to dicts. `merge` concatenates lists. `query` does SQLite only for v1. These functions don't need mocking — they use real temp files.

**Step 1: Write the failing tests**

Create `tests/test_runtime_data.py`:

```python
"""Tests for drift_runtime.data — file I/O functions."""

import os
import json
import tempfile
import sqlite3
import pytest
from dataclasses import dataclass


def test_read_json():
    from drift_runtime.data import read
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({"key": "value"}, f)
        path = f.name
    try:
        result = read(path)
        assert result == {"key": "value"}
    finally:
        os.unlink(path)


def test_read_csv():
    from drift_runtime.data import read
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("name,age\nAlice,30\nBob,25\n")
        path = f.name
    try:
        result = read(path)
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[0]["age"] == "30"
    finally:
        os.unlink(path)


def test_read_text():
    from drift_runtime.data import read
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("hello world")
        path = f.name
    try:
        result = read(path)
        assert result == "hello world"
    finally:
        os.unlink(path)


def test_read_markdown():
    from drift_runtime.data import read
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("# Title\nBody")
        path = f.name
    try:
        result = read(path)
        assert "# Title" in result
    finally:
        os.unlink(path)


def test_read_nonexistent_raises():
    from drift_runtime.data import read
    with pytest.raises(FileNotFoundError):
        read("/nonexistent/path/file.json")


def test_save_json():
    from drift_runtime.data import save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.json")
        save({"key": "value"}, path)
        with open(path) as f:
            data = json.load(f)
        assert data == {"key": "value"}


def test_save_csv():
    from drift_runtime.data import save, read
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.csv")
        save([{"name": "Alice", "age": 30}], path)
        assert os.path.exists(path)
        result = read(path)
        assert result[0]["name"] == "Alice"


def test_save_text():
    from drift_runtime.data import save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.txt")
        save("hello world", path)
        with open(path) as f:
            assert f.read() == "hello world"


def test_save_creates_directories():
    from drift_runtime.data import save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nested", "deep", "out.json")
        save({"key": "value"}, path)
        assert os.path.exists(path)


def test_save_dataclass():
    from drift_runtime.data import save
    @dataclass
    class Item:
        name: str
        value: int
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.json")
        save(Item(name="test", value=42), path)
        with open(path) as f:
            data = json.load(f)
        assert data == {"name": "test", "value": 42}


def test_merge():
    from drift_runtime.data import merge
    a = [{"id": 1}, {"id": 2}]
    b = [{"id": 3}]
    result = merge([a, b])
    assert len(result) == 3
    assert result[2]["id"] == 3


def test_merge_empty():
    from drift_runtime.data import merge
    assert merge([]) == []
    assert merge([[], []]) == []


def test_query_sqlite():
    from drift_runtime.data import query
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE users (name TEXT, age INTEGER)")
        conn.execute("INSERT INTO users VALUES ('Alice', 30)")
        conn.execute("INSERT INTO users VALUES ('Bob', 25)")
        conn.commit()
        conn.close()

        result = query("SELECT * FROM users", db_path)
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[1]["age"] == 25
    finally:
        os.unlink(db_path)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift_runtime.data'`

**Step 3: Implement read, save, merge, query**

Create `drift_runtime/data.py`:

```python
"""Drift runtime data operations — file I/O, HTTP, merge, query."""

import os
import csv
import json
import sqlite3
import dataclasses

import httpx

from drift_runtime.exceptions import DriftNetworkError, DriftFileError


def read(path: str):
    """Read a file from disk. CSV→list[dict], JSON→parsed, text→str."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path) as f:
            return json.load(f)
    elif ext == ".csv":
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        with open(path) as f:
            return f.read()


def save(data, path: str):
    """Save data to disk. Auto-creates directories. Format by extension."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Convert dataclass to dict
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        data = dataclasses.asdict(data)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    elif ext == ".csv":
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        else:
            with open(path, "w") as f:
                f.write(str(data))
    else:
        with open(path, "w") as f:
            f.write(str(data))

    print(f"Saved: {path}")


def merge(sources: list) -> list:
    """Combine multiple lists into one."""
    result = []
    for s in sources:
        result.extend(s)
    return result


def query(sql: str, source: str) -> list:
    """Execute a SQL query against a SQLite database. Returns list[dict]."""
    conn = sqlite3.connect(source)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(sql)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def fetch(url: str, headers: dict = None, params: dict = None):
    """Make an HTTP GET request. Returns parsed JSON or text."""
    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise DriftNetworkError(f"HTTP {e.response.status_code}: {url}") from e
    except httpx.RequestError as e:
        raise DriftNetworkError(f"Request failed: {url} — {e}") from e

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    return response.text
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_data.py -v`
Expected: 14 tests PASS

**Step 5: Commit**

```bash
git add drift_runtime/data.py tests/test_runtime_data.py
git commit -m "feat(runtime): add data module with read, save, merge, query, fetch"
```

---

### Task 5: Fetch Tests (Mocked HTTP)

**Files:**
- Create: `tests/test_runtime_fetch.py`

**Context:** The `fetch` function is already implemented in `drift_runtime/data.py` from Task 4. Here we add tests that mock `httpx.get` so they don't make real HTTP calls. We test JSON responses, headers, error handling, and non-JSON content types.

**Step 1: Write the tests**

Create `tests/test_runtime_fetch.py`:

```python
"""Tests for drift_runtime.data.fetch — mocked HTTP."""

from unittest.mock import patch, MagicMock
import pytest
from drift_runtime.data import fetch
from drift_runtime.exceptions import DriftNetworkError
import httpx


def _mock_response(status_code=200, json_data=None, text="", content_type="application/json"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    resp.json.return_value = json_data
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code}", request=MagicMock(), response=resp
        )
    return resp


def test_fetch_json():
    mock_resp = _mock_response(json_data=[{"id": 1}])
    with patch("drift_runtime.data.httpx.get", return_value=mock_resp):
        result = fetch("https://api.example.com/data")
    assert result == [{"id": 1}]


def test_fetch_with_headers():
    mock_resp = _mock_response(json_data={"ok": True})
    with patch("drift_runtime.data.httpx.get", return_value=mock_resp) as mock_get:
        fetch("https://api.example.com", headers={"X-Key": "abc"})
    mock_get.assert_called_once_with(
        "https://api.example.com",
        headers={"X-Key": "abc"},
        params=None,
        timeout=30,
    )


def test_fetch_with_params():
    mock_resp = _mock_response(json_data=[])
    with patch("drift_runtime.data.httpx.get", return_value=mock_resp) as mock_get:
        fetch("https://api.example.com", params={"limit": 50})
    mock_get.assert_called_once_with(
        "https://api.example.com",
        headers=None,
        params={"limit": 50},
        timeout=30,
    )


def test_fetch_non_json():
    mock_resp = _mock_response(text="hello world", content_type="text/plain")
    with patch("drift_runtime.data.httpx.get", return_value=mock_resp):
        result = fetch("https://example.com/text")
    assert result == "hello world"


def test_fetch_http_error_raises_drift_network_error():
    mock_resp = _mock_response(status_code=404)
    with patch("drift_runtime.data.httpx.get", return_value=mock_resp):
        with pytest.raises(DriftNetworkError, match="HTTP 404"):
            fetch("https://api.example.com/missing")


def test_fetch_connection_error_raises_drift_network_error():
    with patch("drift_runtime.data.httpx.get", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(DriftNetworkError, match="Request failed"):
            fetch("https://unreachable.example.com")
```

**Step 2: Run test to verify it passes**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_fetch.py -v`
Expected: 6 tests PASS

**Step 3: Commit**

```bash
git add tests/test_runtime_fetch.py
git commit -m "test(runtime): add mocked HTTP fetch tests"
```

---

### Task 6: Pipeline Helpers

**Files:**
- Create: `drift_runtime/pipeline.py`
- Create: `tests/test_runtime_pipeline.py`

**Context:** The pipeline module provides `deduplicate(items, key)` and `group_by(items, key)`. These are used by transpiled pipeline code (`|> deduplicate by field` and `|> group by field`). They work on both dicts and dataclass objects.

**Step 1: Write the failing tests**

Create `tests/test_runtime_pipeline.py`:

```python
"""Tests for drift_runtime.pipeline helpers."""

from dataclasses import dataclass
from drift_runtime.pipeline import deduplicate, group_by


# -- deduplicate tests --

def test_deduplicate_dicts():
    items = [
        {"address": "123 Main", "price": 100},
        {"address": "456 Oak", "price": 200},
        {"address": "123 Main", "price": 150},
    ]
    result = deduplicate(items, "address")
    assert len(result) == 2


def test_deduplicate_preserves_first():
    items = [
        {"id": "a", "value": 1},
        {"id": "a", "value": 2},
    ]
    result = deduplicate(items, "id")
    assert len(result) == 1
    assert result[0]["value"] == 1


def test_deduplicate_dataclasses():
    @dataclass
    class Item:
        id: str
        value: int
    items = [Item("a", 1), Item("b", 2), Item("a", 3)]
    result = deduplicate(items, "id")
    assert len(result) == 2
    assert result[0].value == 1


def test_deduplicate_empty():
    assert deduplicate([], "id") == []


# -- group_by tests --

def test_group_by_dicts():
    items = [
        {"city": "Austin", "name": "A"},
        {"city": "Austin", "name": "B"},
        {"city": "Denver", "name": "C"},
    ]
    result = group_by(items, "city")
    assert len(result) == 2
    austin = [g for g in result if g["key"] == "Austin"][0]
    assert len(austin["items"]) == 2
    denver = [g for g in result if g["key"] == "Denver"][0]
    assert len(denver["items"]) == 1


def test_group_by_dataclasses():
    @dataclass
    class Item:
        category: str
        name: str
    items = [Item("a", "X"), Item("b", "Y"), Item("a", "Z")]
    result = group_by(items, "category")
    assert len(result) == 2


def test_group_by_empty():
    assert group_by([], "id") == []


def test_group_by_single_group():
    items = [{"type": "x", "v": 1}, {"type": "x", "v": 2}]
    result = group_by(items, "type")
    assert len(result) == 1
    assert result[0]["key"] == "x"
    assert len(result[0]["items"]) == 2
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift_runtime.pipeline'`

**Step 3: Implement pipeline helpers**

Create `drift_runtime/pipeline.py`:

```python
"""Drift pipeline helper functions.

Used by transpiled pipeline code for deduplicate and group_by stages.
"""


def deduplicate(items: list, key: str) -> list:
    """Remove duplicates from a list by a key field. Keeps the first occurrence."""
    seen = {}
    for item in items:
        k = item[key] if isinstance(item, dict) else getattr(item, key)
        if k not in seen:
            seen[k] = item
    return list(seen.values())


def group_by(items: list, key: str) -> list:
    """Group a list of items by a key field. Returns list of {key, items} dicts."""
    groups = {}
    for item in items:
        k = item[key] if isinstance(item, dict) else getattr(item, key)
        groups.setdefault(k, []).append(item)
    return [{"key": k, "items": v} for k, v in groups.items()]
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_pipeline.py -v`
Expected: 8 tests PASS

**Step 5: Commit**

```bash
git add drift_runtime/pipeline.py tests/test_runtime_pipeline.py
git commit -m "feat(runtime): add pipeline helpers — deduplicate and group_by"
```

---

### Task 7: AI Module — Core Infrastructure

**Files:**
- Create: `drift_runtime/ai.py`
- Create: `tests/test_runtime_ai.py`

**Context:** The `DriftAI` class is the heart of the runtime. It has a single `_call_model(messages, model=None)` method that dispatches to either Anthropic or OpenAI based on the config. All other methods (`ask`, `classify`, etc.) call `_call_model`. API keys come from env vars (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`). All tests mock `_call_model` — never make real API calls.

**Step 1: Write the failing tests**

Create `tests/test_runtime_ai.py`:

```python
"""Tests for drift_runtime.ai module.

ALL tests mock DriftAI._call_model — no real API calls.
"""

from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from drift_runtime.ai import DriftAI


@dataclass
class TestSchema:
    name: str
    score: float


# -- _call_model infrastructure tests --

def test_drift_ai_creates():
    ai = DriftAI()
    assert ai is not None


# -- ask tests --

def test_ask_simple():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value="42"):
        result = ai.ask("What is the meaning of life?")
    assert result == "42"


def test_ask_with_context():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value="yes") as mock:
        result = ai.ask("Is this good?", context={"data": "test"})
    assert result == "yes"
    # Verify context was included in the prompt
    call_args = mock.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    user_msg = [m for m in messages if m["role"] == "user"][0]
    assert "test" in user_msg["content"]


def test_ask_with_schema():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value='{"name": "test", "score": 95.0}'):
        result = ai.ask("Analyze this", schema=TestSchema)
    assert isinstance(result, TestSchema)
    assert result.name == "test"
    assert result.score == 95.0


def test_ask_with_schema_fenced_response():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value='```json\n{"name": "test", "score": 80.0}\n```'):
        result = ai.ask("Analyze this", schema=TestSchema)
    assert isinstance(result, TestSchema)
    assert result.name == "test"


# -- classify tests --

def test_classify():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value="urgent"):
        result = ai.classify("Server is down!", categories=["urgent", "routine", "spam"])
    assert result == "urgent"


def test_classify_strips_whitespace():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value="  urgent  \n"):
        result = ai.classify("Server is down!", categories=["urgent", "routine", "spam"])
    assert result == "urgent"


def test_classify_retries_on_invalid():
    ai = DriftAI()
    with patch.object(ai, "_call_model", side_effect=["invalid_category", "urgent"]):
        result = ai.classify("Server is down!", categories=["urgent", "routine", "spam"])
    assert result == "urgent"


# -- embed tests --

def test_embed():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value="[0.1, 0.2, 0.3]"):
        result = ai.embed("hello world")
    assert isinstance(result, list)
    assert len(result) == 3


# -- see tests --

def test_see():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value="A red house with a green roof"):
        result = ai.see("photo.jpg", "Describe this image")
    assert "red house" in result


# -- predict tests --

def test_predict_returns_confident_value():
    ai = DriftAI()
    with patch.object(ai, "_call_model", return_value='{"value": 350000, "confidence": 0.85}'):
        result = ai.predict("Estimate the ARV")
    from drift_runtime.types import ConfidentValue
    assert isinstance(result, ConfidentValue)
    assert result.value == 350000
    assert result.confidence == 0.85


# -- enrich tests --

def test_enrich_items():
    ai = DriftAI()
    items = [{"name": "A"}, {"name": "B"}]
    with patch.object(ai, "_call_model", side_effect=[
        '{"summary": "Great A"}',
        '{"summary": "Great B"}',
    ]):
        result = ai.enrich(items, "Add a summary")
    assert result[0]["summary"] == "Great A"
    assert result[1]["summary"] == "Great B"


def test_enrich_empty_list():
    ai = DriftAI()
    result = ai.enrich([], "Add a summary")
    assert result == []


# -- score tests --

def test_score_items():
    ai = DriftAI()
    items = [{"name": "A"}, {"name": "B"}]
    with patch.object(ai, "_call_model", side_effect=["85", "42"]):
        result = ai.score(items, "Rate quality 1-100")
    assert result[0]["score"] == 85
    assert result[1]["score"] == 42


def test_score_empty_list():
    ai = DriftAI()
    result = ai.score([], "Rate quality")
    assert result == []
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_ai.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift_runtime.ai'`

**Step 3: Implement the AI module**

Create `drift_runtime/ai.py`:

```python
"""Drift AI module — ask, classify, embed, see, predict, enrich, score.

All AI primitives dispatch through _call_model(), which routes to
Anthropic or OpenAI based on drift.config.
"""

import json
import os
import base64

from drift_runtime.config import get_config
from drift_runtime.types import (
    ConfidentValue,
    schema_to_json_description,
    parse_ai_response_to_schema,
)
from drift_runtime.exceptions import DriftAIError


class DriftAI:
    """AI inference engine for Drift programs."""

    def _call_model(self, messages: list[dict], model: str | None = None) -> str:
        """Call the configured AI provider. Returns the text response."""
        config = get_config()
        provider = config["ai"]["provider"]
        model = model or config["ai"]["default_model"]
        timeout = config["ai"]["timeout"]

        try:
            if provider == "anthropic":
                return self._call_anthropic(messages, model, timeout)
            elif provider == "openai":
                return self._call_openai(messages, model, timeout)
            else:
                raise DriftAIError(f"Unknown AI provider: {provider}")
        except DriftAIError:
            raise
        except Exception as e:
            raise DriftAIError(f"AI call failed: {e}") from e

    def _call_anthropic(self, messages: list[dict], model: str, timeout: int) -> str:
        """Call Anthropic's API."""
        import anthropic

        client = anthropic.Anthropic()

        # Separate system message from user messages
        system_msg = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "messages": user_messages,
            "timeout": timeout,
        }
        if system_msg:
            kwargs["system"] = system_msg

        response = client.messages.create(**kwargs)
        return response.content[0].text

    def _call_openai(self, messages: list[dict], model: str, timeout: int) -> str:
        """Call OpenAI's API."""
        import openai

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
        )
        return response.choices[0].message.content

    def ask(self, prompt: str, schema=None, context: dict = None) -> object:
        """Ask the AI a question. Optionally parse response into a schema."""
        user_content = prompt
        if context:
            user_content += f"\n\nContext:\n{json.dumps(context, indent=2)}"

        messages = [{"role": "user", "content": user_content}]

        if schema:
            desc = schema_to_json_description(schema)
            messages.insert(0, {
                "role": "system",
                "content": f"Respond with valid JSON matching this schema:\n{desc}\n\nReturn ONLY the JSON object, no other text.",
            })

        response = self._call_model(messages)

        if schema:
            return parse_ai_response_to_schema(response, schema)
        return response

    def classify(self, input: str, categories: list[str]) -> str:
        """Classify input into one of the provided categories."""
        prompt = (
            f"Classify the following text into exactly one of these categories: "
            f"{categories}\n\nText: {input}\n\n"
            f"Respond with only the category name, nothing else."
        )
        messages = [{"role": "user", "content": prompt}]
        response = self._call_model(messages).strip()

        if response in categories:
            return response

        # Retry once if response didn't match
        retry_prompt = (
            f"Your response '{response}' was not one of the valid categories. "
            f"Choose exactly one of: {categories}\n\nRespond with only the category name."
        )
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": retry_prompt})
        return self._call_model(messages).strip()

    def embed(self, input: str) -> list[float]:
        """Generate an embedding vector for the input text."""
        config = get_config()
        provider = config["ai"]["provider"]

        if provider == "openai":
            import openai
            client = openai.OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=input,
            )
            return response.data[0].embedding

        # For Anthropic (and others), use a prompt-based approach
        messages = [{"role": "user", "content": (
            f"Generate a numerical embedding vector for this text as a JSON array of floats. "
            f"Return ONLY the JSON array.\n\nText: {input}"
        )}]
        response = self._call_model(messages)
        return json.loads(response.strip())

    def see(self, input, prompt: str) -> str:
        """Analyze an image with AI vision."""
        # For now, build a text-based message referencing the image
        if isinstance(input, str):
            # File path — read and encode
            with open(input, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(input, bytes):
            image_data = base64.b64encode(input).decode("utf-8")
        else:
            image_data = str(input)

        config = get_config()
        provider = config["ai"]["provider"]

        if provider == "anthropic":
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                    {"type": "text", "text": prompt},
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    {"type": "text", "text": prompt},
                ],
            }]

        return self._call_model(messages)

    def predict(self, prompt: str, schema=None) -> object:
        """Make a prediction with confidence score."""
        predict_prompt = (
            f"{prompt}\n\nProvide your prediction as JSON with "
            f'"value" (your prediction) and "confidence" (0.0 to 1.0).'
        )
        messages = [{"role": "user", "content": predict_prompt}]
        response = self._call_model(messages)

        # Parse the response
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        data = json.loads(text)
        return ConfidentValue(value=data["value"], confidence=data["confidence"])

    def enrich(self, items: list, prompt: str) -> list:
        """Enrich a list of items using AI. Used in pipelines."""
        if not items:
            return []

        result = []
        for item in items:
            msg = f"{prompt}\n\nItem: {json.dumps(item)}"
            messages = [{"role": "user", "content": msg}]
            response = self._call_model(messages)

            # Parse enrichment data and merge into item
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.rstrip().endswith("```"):
                    text = text.rstrip()[:-3]
                text = text.strip()

            try:
                enrichment = json.loads(text)
                if isinstance(item, dict):
                    merged = dict(item)
                    merged.update(enrichment)
                    result.append(merged)
                else:
                    result.append(item)
            except json.JSONDecodeError:
                # If response isn't JSON, add as "enrichment" field
                if isinstance(item, dict):
                    merged = dict(item)
                    merged["enrichment"] = response
                    result.append(merged)
                else:
                    result.append(item)

        return result

    def score(self, items: list, prompt: str) -> list:
        """Score a list of items using AI. Used in pipelines."""
        if not items:
            return []

        result = []
        for item in items:
            msg = f"{prompt}\n\nItem: {json.dumps(item)}\n\nRespond with only a number."
            messages = [{"role": "user", "content": msg}]
            response = self._call_model(messages)

            try:
                score_val = float(response.strip())
            except ValueError:
                score_val = 0

            if isinstance(item, dict):
                scored = dict(item)
                scored["score"] = score_val
                result.append(scored)
            else:
                result.append(item)

        return result
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_ai.py -v`
Expected: 17 tests PASS

**Step 5: Commit**

```bash
git add drift_runtime/ai.py tests/test_runtime_ai.py
git commit -m "feat(runtime): add AI module with ask, classify, embed, see, predict, enrich, score"
```

---

### Task 8: Runtime \_\_init\_\_.py — Wire Everything

**Files:**
- Modify: `drift_runtime/__init__.py`
- Create: `tests/test_runtime_init.py`

**Context:** The `__init__.py` wires all modules together and exposes the public API. Transpiled code uses `drift_runtime.ai.ask(...)`, `drift_runtime.fetch(...)`, `drift_runtime.save(...)`, `drift_runtime.log(...)`, etc. The `ai` export is a singleton `DriftAI()` instance, not the module. The `log` function wraps `print` with a `[drift]` prefix.

**Step 1: Write the failing tests**

Create `tests/test_runtime_init.py`:

```python
"""Tests for drift_runtime public API surface."""

import drift_runtime


def test_ai_is_drift_ai_instance():
    from drift_runtime.ai import DriftAI
    assert isinstance(drift_runtime.ai, DriftAI)


def test_fetch_is_callable():
    assert callable(drift_runtime.fetch)


def test_read_is_callable():
    assert callable(drift_runtime.read)


def test_save_is_callable():
    assert callable(drift_runtime.save)


def test_query_is_callable():
    assert callable(drift_runtime.query)


def test_merge_is_callable():
    assert callable(drift_runtime.merge)


def test_log_is_callable():
    assert callable(drift_runtime.log)


def test_log_output(capsys):
    drift_runtime.log("hello")
    captured = capsys.readouterr()
    assert "[drift]" in captured.out
    assert "hello" in captured.out


def test_deduplicate_is_callable():
    assert callable(drift_runtime.deduplicate)


def test_group_by_is_callable():
    assert callable(drift_runtime.group_by)


def test_exception_classes_accessible():
    assert drift_runtime.DriftRuntimeError is not None
    assert drift_runtime.DriftAIError is not None
    assert drift_runtime.DriftNetworkError is not None
    assert drift_runtime.DriftFileError is not None
    assert drift_runtime.DriftConfigError is not None


def test_confident_value_accessible():
    cv = drift_runtime.ConfidentValue(value=100, confidence=0.9)
    assert cv.value == 100
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_init.py -v`
Expected: FAIL — attributes not found on `drift_runtime`

**Step 3: Implement the \_\_init\_\_.py**

Update `drift_runtime/__init__.py`:

```python
"""Drift Runtime — makes transpiled Drift programs execute."""

from drift_runtime.ai import DriftAI
from drift_runtime.data import fetch, read, save, query, merge
from drift_runtime.config import get_config
from drift_runtime.types import ConfidentValue
from drift_runtime.pipeline import deduplicate, group_by
from drift_runtime.exceptions import (
    DriftRuntimeError,
    DriftAIError,
    DriftNetworkError,
    DriftFileError,
    DriftConfigError,
)

# Singleton AI instance — transpiled code uses drift_runtime.ai.ask(...)
ai = DriftAI()


def log(message):
    """Log a message with [drift] prefix."""
    print(f"[drift] {message}")


__all__ = [
    "ai", "fetch", "read", "save", "query", "merge", "log",
    "get_config", "ConfidentValue", "deduplicate", "group_by",
    "DriftRuntimeError", "DriftAIError", "DriftNetworkError",
    "DriftFileError", "DriftConfigError",
]
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_init.py -v`
Expected: 13 tests PASS

**Step 5: Commit**

```bash
git add drift_runtime/__init__.py tests/test_runtime_init.py
git commit -m "feat(runtime): wire __init__.py with full public API surface"
```

---

### Task 9: Transpiler Adjustments

**Files:**
- Modify: `drift/transpiler.py` (lines 131-134 for log, lines 270-273 for error types, lines 543-552 for deduplicate/group)

**Context:** The transpiler has three mismatches with the runtime API that must be fixed:

1. `log` emits `print(value)` but should emit `drift_runtime.log(value)`
2. `_ERROR_TYPE_MAP` maps `network_error` → `ConnectionError` and `ai_error` → `RuntimeError` but should map to `drift_runtime.DriftNetworkError` and `drift_runtime.DriftAIError`
3. Pipeline `DeduplicateStage` and `GroupStage` use inline Python but should use `drift_runtime.deduplicate()` and `drift_runtime.group_by()`

After fixing, ALL existing Phase 1 tests must still pass (zero regressions).

**Step 1: Write the failing tests for the correct output**

These tests go in the existing `tests/test_transpiler.py`. Add to the end of the file:

```python
def test_log_emits_drift_runtime_log():
    """log should emit drift_runtime.log(), not print()."""
    code = compile_drift('log "hello"')
    assert "drift_runtime.log" in code
    # Should NOT be bare print for log statements
    lines = code.strip().split("\n")
    log_lines = [l for l in lines if "log" in l.lower() and "import" not in l]
    for line in log_lines:
        assert "drift_runtime.log" in line


def test_catch_network_error_emits_drift_exception():
    """catch network_error should emit DriftNetworkError, not ConnectionError."""
    code = compile_drift('try:\n  x = 1\ncatch network_error:\n  print "fail"')
    assert "drift_runtime.DriftNetworkError" in code
    assert "ConnectionError" not in code


def test_catch_ai_error_emits_drift_exception():
    """catch ai_error should emit DriftAIError, not RuntimeError."""
    code = compile_drift('try:\n  x = 1\ncatch ai_error:\n  print "fail"')
    assert "drift_runtime.DriftAIError" in code
    # Make sure it's not bare RuntimeError (but RuntimeError could appear in the string)


def test_deduplicate_uses_runtime_helper():
    """deduplicate by field should use drift_runtime.deduplicate()."""
    source = 'results = fetch "https://api.example.com"\n  |> deduplicate by name'
    code = compile_drift(source)
    assert "drift_runtime.deduplicate" in code


def test_group_by_uses_runtime_helper():
    """group by field should use drift_runtime.group_by()."""
    source = 'results = fetch "https://api.example.com"\n  |> group by city'
    code = compile_drift(source)
    assert "drift_runtime.group_by" in code
```

**Step 2: Run test to verify the new tests fail**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_transpiler.py -v -k "log_emits or catch_network or catch_ai or deduplicate_uses or group_by_uses"`
Expected: FAIL — current transpiler emits `print()`, `ConnectionError`, `RuntimeError`, and inline code

**Step 3: Fix the transpiler**

In `drift/transpiler.py`, make these changes:

**Change 1:** Fix `_emit_log` (line ~131-134):
```python
# BEFORE:
    def _emit_log(self, node: LogStatement) -> list[str]:
        """Emit ``print(value)`` (log maps to print)."""
        value = self._emit_expr(node.value)
        return [f"{self._indent()}print({value})"]

# AFTER:
    def _emit_log(self, node: LogStatement) -> list[str]:
        """Emit ``drift_runtime.log(value)``."""
        value = self._emit_expr(node.value)
        return [f"{self._indent()}drift_runtime.log({value})"]
```

**Change 2:** Fix `_ERROR_TYPE_MAP` (line ~270-273):
```python
# BEFORE:
    _ERROR_TYPE_MAP = {
        "network_error": "ConnectionError",
        "ai_error": "RuntimeError",
    }

# AFTER:
    _ERROR_TYPE_MAP = {
        "network_error": "drift_runtime.DriftNetworkError",
        "ai_error": "drift_runtime.DriftAIError",
    }
```

**Change 3:** Fix `DeduplicateStage` pipeline emission (line ~551-552):
```python
# BEFORE:
        if isinstance(stage, DeduplicateStage):
            return [f'{self._indent()}_pipe = list({{_item["{stage.field_name}"]: _item for _item in _pipe}}.values())']

# AFTER:
        if isinstance(stage, DeduplicateStage):
            return [f'{self._indent()}_pipe = drift_runtime.deduplicate(_pipe, "{stage.field_name}")']
```

**Change 4:** Fix `GroupStage` pipeline emission (line ~542-549):
```python
# BEFORE:
        if isinstance(stage, GroupStage):
            ind = self._indent()
            return [
                f"{ind}_groups = {{}}",
                f"{ind}for _item in _pipe:",
                f"{ind}    _key = _item[\"{stage.field_name}\"]",
                f"{ind}    _groups.setdefault(_key, []).append(_item)",
                f'{ind}_pipe = [{{"key": k, "items": v}} for k, v in _groups.items()]',
            ]

# AFTER:
        if isinstance(stage, GroupStage):
            return [f'{self._indent()}_pipe = drift_runtime.group_by(_pipe, "{stage.field_name}")']
```

**Step 4: Run ALL tests to verify no regressions**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/ -v`
Expected: ALL tests PASS (Phase 1 + Phase 2 runtime tests). Some existing transpiler tests may need their expected output updated if they tested for the old `ConnectionError`/`RuntimeError`/inline code patterns. Fix those assertions to match the new output.

**Step 5: Commit**

```bash
git add drift/transpiler.py tests/test_transpiler.py
git commit -m "fix(transpiler): align output with drift_runtime API — log, exceptions, pipeline helpers"
```

---

### Task 10: End-to-End Runtime Tests

**Files:**
- Create: `tests/test_runtime_e2e.py`

**Context:** These tests run `.drift` files through the full pipeline: lex → parse → transpile → execute. The `hello.drift` example should work without an API key (it doesn't use AI). Tests that require AI or network calls are gated behind `requires_api_key`. We test that `drift run` works by exec'ing the transpiled Python code directly.

**Step 1: Write the tests**

Create `tests/test_runtime_e2e.py`:

```python
"""End-to-end runtime tests — execute transpiled Drift code.

Tests without AI calls run always. Tests that need an API key
are skipped unless ANTHROPIC_API_KEY is set.
"""

import os
import sys
import subprocess
import tempfile
import pytest

DRIFT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

requires_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


def run_drift(filepath):
    """Run a .drift file via the CLI."""
    return subprocess.run(
        [sys.executable, "-m", "drift.cli", "run", filepath],
        capture_output=True,
        text=True,
        cwd=DRIFT_DIR,
        timeout=30,
    )


def test_hello_drift_runs():
    """hello.drift should print 'Hello from Drift!' without any API key."""
    result = run_drift(os.path.join(DRIFT_DIR, "examples", "hello.drift"))
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Hello from Drift!" in result.stdout


def test_simple_variables_and_print():
    """Basic variable assignment and print should work."""
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False, dir=DRIFT_DIR) as f:
        f.write('x = 42\nprint x\nname = "world"\nprint "Hello {name}"')
        path = f.name
    try:
        result = run_drift(path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "42" in result.stdout
        assert "Hello world" in result.stdout
    finally:
        os.unlink(path)


def test_if_else():
    """If/else control flow should work."""
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False, dir=DRIFT_DIR) as f:
        f.write('x = 10\nif x > 5:\n  print "big"\nelse:\n  print "small"')
        path = f.name
    try:
        result = run_drift(path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "big" in result.stdout
    finally:
        os.unlink(path)


def test_for_each():
    """For each loop should work."""
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False, dir=DRIFT_DIR) as f:
        f.write('items = [1, 2, 3]\nfor each item in items:\n  print item')
        path = f.name
    try:
        result = run_drift(path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "1" in result.stdout
        assert "2" in result.stdout
        assert "3" in result.stdout
    finally:
        os.unlink(path)


def test_function_def_and_call():
    """Function definition and calling should work."""
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False, dir=DRIFT_DIR) as f:
        f.write('define greet(name: string) -> string:\n  return "Hi {name}"\nresult = greet("Drift")\nprint result')
        path = f.name
    try:
        result = run_drift(path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Hi Drift" in result.stdout
    finally:
        os.unlink(path)


def test_save_and_read_json():
    """save and read should work with JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_out.json")
        drift_source = f'save {{"result": 42}} to "{out_path}"\ndata = read "{out_path}"\nprint data'
        with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False, dir=DRIFT_DIR) as f:
            f.write(drift_source)
            drift_path = f.name
        try:
            result = run_drift(drift_path)
            assert result.returncode == 0, f"stderr: {result.stderr}"
            assert os.path.exists(out_path)
        finally:
            os.unlink(drift_path)


def test_log_uses_drift_prefix():
    """log should output with [drift] prefix."""
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False, dir=DRIFT_DIR) as f:
        f.write('log "test message"')
        path = f.name
    try:
        result = run_drift(path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "[drift]" in result.stdout
        assert "test message" in result.stdout
    finally:
        os.unlink(path)


@requires_api_key
def test_ai_ask_runs():
    """ai.ask should work with a real API key."""
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False, dir=DRIFT_DIR) as f:
        f.write('answer = ai.ask("What is 2 + 2? Reply with just the number.")\nprint answer')
        path = f.name
    try:
        result = run_drift(path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "4" in result.stdout
    finally:
        os.unlink(path)
```

**Step 2: Run tests to verify they pass**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/test_runtime_e2e.py -v`
Expected: All non-API tests PASS. API tests skip if no key set.

**Step 3: Commit**

```bash
git add tests/test_runtime_e2e.py
git commit -m "test(runtime): add end-to-end runtime execution tests"
```

---

### Task 11: Smoke Test — drift run examples

**Files:**
- No new files — verification only

**Context:** Final verification. Run all three example programs through `drift run`. `hello.drift` must print output. `pipeline.drift` and `deal_analyzer.drift` will fail on network/AI calls (no real API key in tests), but should fail with `DriftNetworkError` or `DriftAIError`, NOT with an import error or syntax error. This proves the runtime is wired up correctly.

**Step 1: Verify hello.drift runs end-to-end**

Run: `cd /Users/ethansurfas/drift && python3 -m drift.cli run examples/hello.drift`
Expected: Prints `Hello from Drift!`

**Step 2: Verify pipeline.drift fails gracefully (not on import)**

Run: `cd /Users/ethansurfas/drift && python3 -m drift.cli run examples/pipeline.drift 2>&1 || true`
Expected: Fails with a network error (not a ModuleNotFoundError or ImportError)

**Step 3: Verify deal_analyzer.drift fails gracefully (not on import)**

Run: `cd /Users/ethansurfas/drift && python3 -m drift.cli run examples/deal_analyzer.drift 2>&1 || true`
Expected: Fails with a key/network error (not a ModuleNotFoundError or ImportError)

**Step 4: Run the full test suite**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/ -v`
Expected: ALL tests PASS (Phase 1 + Phase 2, zero regressions)

**Step 5: Commit (no changes expected, but if any fixes were needed)**

```bash
git add -A
git commit -m "fix(runtime): resolve smoke test issues" # only if changes were made
```

---

### Task 12: Update CLAUDE.md and Final Cleanup

**Files:**
- Modify: `CLAUDE.md` (update test count, remove "Phase 3" comments from cli.py)
- Modify: `drift/cli.py` (remove outdated comment about drift_runtime not existing)

**Context:** The cli.py has a comment saying `# Note: drift_runtime doesn't exist yet (Phase 3)`. This is now wrong — drift_runtime exists and works. Update the comment and the CLAUDE.md test count.

**Step 1: Update cli.py**

In `drift/cli.py`, remove the outdated comment on line 48-49:

```python
# BEFORE:
            if command == "run":
                # Note: drift_runtime doesn't exist yet (Phase 3)
                # This will fail on import but the transpilation works
                exec(python_code, {"__name__": "__main__"})

# AFTER:
            if command == "run":
                exec(python_code, {"__name__": "__main__"})
```

**Step 2: Update CLAUDE.md test count**

Update the test count references in `CLAUDE.md` to reflect the new total (Phase 1 + Phase 2 tests). Find the line that says `354 tests` and update to the actual count. Also update the "Phase 2 next" status line to "Phase 2 complete".

**Step 3: Run full test suite one final time**

Run: `cd /Users/ethansurfas/drift && python3 -m pytest tests/ -v`
Expected: ALL tests PASS

**Step 4: Commit**

```bash
git add CLAUDE.md drift/cli.py
git commit -m "docs: update CLAUDE.md and cli.py for Phase 2 completion"
```

---

## Success Criteria

Phase 2 is complete when:

1. `drift run examples/hello.drift` prints "Hello from Drift!"
2. All unit tests pass without API keys (using mocks)
3. `ai.ask`, `ai.classify`, `ai.embed`, `ai.see`, `ai.predict` all work (mocked)
4. `ai.enrich` and `ai.score` work on lists (pipeline usage, mocked)
5. `fetch` handles JSON and text responses (mocked)
6. `read` and `save` handle JSON, CSV, and text files
7. `drift.config` is loaded and respected
8. Error types (`DriftNetworkError`, `DriftAIError`) map correctly from Drift `catch` blocks
9. `deduplicate by` and `group by` pipeline stages use runtime helpers
10. `log` emits `[drift]` prefix via `drift_runtime.log()`
11. Zero regressions in Phase 1 tests
