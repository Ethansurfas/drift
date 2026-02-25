# Drift MCP Server — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an MCP server that lets Claude Desktop and Claude Code write, check, and run Drift programs through 4 tools + a language guide prompt.

**Architecture:** Single Python module (`drift_mcp/server.py`) using FastMCP with stdio transport. Imports the Drift compiler directly — `Lexer`, `Parser`, `Transpiler` from the `drift` package. `drift_run` captures stdout via `io.StringIO` redirect. All errors are caught and returned as tool results, never raised.

**Tech Stack:** Python 3.11+, `mcp` package (FastMCP), existing Drift compiler (`drift/`), existing Drift runtime (`drift_runtime/`)

---

### Task 1: Install `mcp` dependency and create package

**Files:**
- Modify: `pyproject.toml`
- Create: `drift_mcp/__init__.py`
- Create: `drift_mcp/server.py` (skeleton only)

**Step 1: Add `mcp` to pyproject.toml optional dependencies**

Add an `mcp` optional dependency group in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]
mcp = [
    "mcp>=1.0.0",
]
```

Also add `drift_mcp*` to the setuptools packages find include:

```toml
[tool.setuptools.packages.find]
include = ["drift*", "drift_runtime*", "drift_mcp*"]
```

**Step 2: Install the mcp dependency**

Run: `pip install "mcp>=1.0.0"`

**Step 3: Create the package directory and skeleton**

Create `drift_mcp/__init__.py`:

```python
"""Drift MCP Server — lets Claude write and run Drift programs."""
```

Create `drift_mcp/server.py` with just the FastMCP setup:

```python
"""Drift MCP Server — exposes Drift compiler tools via MCP protocol."""

import sys
import io
import os

from mcp.server.fastmcp import FastMCP

from drift.lexer import Lexer
from drift.parser import Parser
from drift.transpiler import Transpiler
from drift.errors import DriftError

mcp = FastMCP("drift")


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

**Step 4: Verify the server starts**

Run: `python3 -m drift_mcp.server --help 2>&1 || echo "Check import works"`

Verify it doesn't crash on import. (It won't do much yet — no tools registered.)

**Step 5: Commit**

```bash
git add pyproject.toml drift_mcp/__init__.py drift_mcp/server.py
git commit -m "feat: scaffold drift_mcp package with FastMCP server"
```

---

### Task 2: Implement `drift_write` tool

**Files:**
- Modify: `drift_mcp/server.py`
- Create: `tests/test_mcp_server.py`

**Step 1: Write the failing test**

Create `tests/test_mcp_server.py`:

```python
"""Tests for the Drift MCP server tools."""

import os
import tempfile
from drift_mcp.server import write_drift_file


def test_write_drift_file_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.drift")
        result = write_drift_file(filepath, 'print "hello"')
        assert "Saved" in result
        assert os.path.exists(filepath)
        with open(filepath) as f:
            assert f.read() == 'print "hello"'


def test_write_drift_file_creates_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "sub", "dir", "test.drift")
        result = write_drift_file(filepath, 'print "hello"')
        assert "Saved" in result
        assert os.path.exists(filepath)


def test_write_drift_file_rejects_non_drift():
    result = write_drift_file("/tmp/test.py", 'print("hello")')
    assert "Error" in result
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_mcp_server.py -v`
Expected: FAIL — `write_drift_file` doesn't exist yet.

**Step 3: Implement `drift_write` in server.py**

Add to `drift_mcp/server.py`:

```python
@mcp.tool()
def drift_write(filepath: str, source: str) -> str:
    """Write Drift source code to a file. Use this to create new .drift programs.

    Args:
        filepath: Path to the .drift file to create (e.g. "my_program.drift")
        source: The Drift source code to write
    """
    return write_drift_file(filepath, source)


def write_drift_file(filepath: str, source: str) -> str:
    """Core logic for writing a drift file — testable without MCP."""
    if not filepath.endswith(".drift"):
        return "Error: filepath must end with .drift"
    try:
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(source)
        return f"Saved: {filepath}"
    except OSError as e:
        return f"Error writing file: {e}"
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_mcp_server.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add drift_mcp/server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add drift_write tool"
```

---

### Task 3: Implement `drift_check` tool

**Files:**
- Modify: `drift_mcp/server.py`
- Modify: `tests/test_mcp_server.py`

**Step 1: Write the failing tests**

Add to `tests/test_mcp_server.py`:

```python
from drift_mcp.server import check_drift_file


def test_check_valid_drift():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.drift")
        with open(filepath, "w") as f:
            f.write('name = "World"\nprint "Hello {name}!"')
        result = check_drift_file(filepath)
        assert "OK" in result


def test_check_invalid_drift():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.drift")
        with open(filepath, "w") as f:
            f.write('if if if')
        result = check_drift_file(filepath)
        assert "Error" in result


def test_check_missing_file():
    result = check_drift_file("/nonexistent/file.drift")
    assert "Error" in result
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_mcp_server.py::test_check_valid_drift -v`
Expected: FAIL — `check_drift_file` doesn't exist yet.

**Step 3: Implement `drift_check` in server.py**

Add to `drift_mcp/server.py`:

```python
@mcp.tool()
def drift_check(filepath: str) -> str:
    """Check Drift syntax without running. Validates that the file is valid Drift code.

    Args:
        filepath: Path to the .drift file to check
    """
    return check_drift_file(filepath)


def check_drift_file(filepath: str) -> str:
    """Core logic for checking a drift file — testable without MCP."""
    try:
        with open(filepath) as f:
            source = f.read()
    except FileNotFoundError:
        return f"Error: file not found: {filepath}"
    except OSError as e:
        return f"Error reading file: {e}"

    try:
        tokens = Lexer(source).tokenize()
        Parser(tokens).parse()
        return f"OK: {filepath}"
    except DriftError as e:
        return f"Error: {e}"
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_mcp_server.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add drift_mcp/server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add drift_check tool"
```

---

### Task 4: Implement `drift_build` tool

**Files:**
- Modify: `drift_mcp/server.py`
- Modify: `tests/test_mcp_server.py`

**Step 1: Write the failing tests**

Add to `tests/test_mcp_server.py`:

```python
from drift_mcp.server import build_drift_file


def test_build_valid_drift():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.drift")
        with open(filepath, "w") as f:
            f.write('name = "Drift"\nprint "Hello {name}!"')
        result = build_drift_file(filepath)
        assert "import drift_runtime" in result
        assert 'f"Hello {name}!"' in result


def test_build_invalid_drift():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.drift")
        with open(filepath, "w") as f:
            f.write('if if if')
        result = build_drift_file(filepath)
        assert "Error" in result


def test_build_missing_file():
    result = build_drift_file("/nonexistent/file.drift")
    assert "Error" in result
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_mcp_server.py::test_build_valid_drift -v`
Expected: FAIL — `build_drift_file` doesn't exist yet.

**Step 3: Implement `drift_build` in server.py**

Add to `drift_mcp/server.py`:

```python
@mcp.tool()
def drift_build(filepath: str) -> str:
    """Transpile a Drift file to Python. Returns the generated Python code.

    Args:
        filepath: Path to the .drift file to transpile
    """
    return build_drift_file(filepath)


def build_drift_file(filepath: str) -> str:
    """Core logic for building a drift file — testable without MCP."""
    try:
        with open(filepath) as f:
            source = f.read()
    except FileNotFoundError:
        return f"Error: file not found: {filepath}"
    except OSError as e:
        return f"Error reading file: {e}"

    try:
        tokens = Lexer(source).tokenize()
        tree = Parser(tokens).parse()
        python_code = Transpiler(tree).transpile()
        return python_code
    except DriftError as e:
        return f"Error: {e}"
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_mcp_server.py -v`
Expected: 9 passed

**Step 5: Commit**

```bash
git add drift_mcp/server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add drift_build tool"
```

---

### Task 5: Implement `drift_run` tool

This is the most important tool. It compiles a `.drift` file, executes it, and captures stdout/stderr to return as the result.

**Files:**
- Modify: `drift_mcp/server.py`
- Modify: `tests/test_mcp_server.py`

**Step 1: Write the failing tests**

Add to `tests/test_mcp_server.py`:

```python
from drift_mcp.server import run_drift_file


def test_run_hello_world():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.drift")
        with open(filepath, "w") as f:
            f.write('print "Hello from Drift!"')
        result = run_drift_file(filepath)
        assert "Hello from Drift!" in result


def test_run_with_variables():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.drift")
        with open(filepath, "w") as f:
            f.write('name = "World"\nprint "Hello {name}!"')
        result = run_drift_file(filepath)
        assert "Hello World!" in result


def test_run_invalid_drift():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.drift")
        with open(filepath, "w") as f:
            f.write('if if if')
        result = run_drift_file(filepath)
        assert "Error" in result


def test_run_missing_file():
    result = run_drift_file("/nonexistent/file.drift")
    assert "Error" in result


def test_run_runtime_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.drift")
        with open(filepath, "w") as f:
            f.write('x = 1 / 0')
        result = run_drift_file(filepath)
        assert "Error" in result or "ZeroDivision" in result
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_mcp_server.py::test_run_hello_world -v`
Expected: FAIL — `run_drift_file` doesn't exist yet.

**Step 3: Implement `drift_run` in server.py**

Add to `drift_mcp/server.py`:

```python
@mcp.tool()
def drift_run(filepath: str) -> str:
    """Run a Drift program. Transpiles to Python and executes it, returning the output.

    Args:
        filepath: Path to the .drift file to run
    """
    return run_drift_file(filepath)


def run_drift_file(filepath: str) -> str:
    """Core logic for running a drift file — testable without MCP."""
    try:
        with open(filepath) as f:
            source = f.read()
    except FileNotFoundError:
        return f"Error: file not found: {filepath}"
    except OSError as e:
        return f"Error reading file: {e}"

    try:
        tokens = Lexer(source).tokenize()
        tree = Parser(tokens).parse()
        python_code = Transpiler(tree).transpile()
    except DriftError as e:
        return f"Error: {e}"

    # Capture stdout during execution
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    try:
        sys.stdout = captured_out
        sys.stderr = captured_err
        exec(python_code, {"__name__": "__main__"})
    except Exception as e:
        return f"{captured_out.getvalue()}Error during execution: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    output = captured_out.getvalue()
    err_output = captured_err.getvalue()
    if err_output:
        output += f"\n[stderr]: {err_output}"
    return output if output else "(program produced no output)"
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_mcp_server.py -v`
Expected: 14 passed

**Step 5: Commit**

```bash
git add drift_mcp/server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add drift_run tool with stdout capture"
```

---

### Task 6: Add Drift language guide prompt

The MCP prompt teaches Claude how to write valid Drift code. This is what makes Claude *choose* to write Drift.

**Files:**
- Modify: `drift_mcp/server.py`

**Step 1: Add the prompt to server.py**

Add to `drift_mcp/server.py`, after the tool definitions:

```python
DRIFT_LANGUAGE_GUIDE = """# Writing Drift Programs

Drift is an AI-native programming language that transpiles to Python. It reads like English.

## Syntax Reference

### Variables (no let/var/const needed)
```
city = "Austin"
price = 450000
active = true
tags = ["one", "two"]
```

### Print (no parentheses needed)
```
print "Hello {name}!"
print "Price: ${amount}"
```

### Comments
```
-- This is a comment
```

### Schemas (structured data types)
```
schema DealScore:
  address: string
  arv: number
  verdict: string
  tags: list of string
```

### AI Primitives (built-in, no imports)
```
-- Simple question
answer = ai.ask("Summarize this text: {content}")

-- Structured output with schema
analysis = ai.ask("Analyze this data") -> MySchema using {
  context_key: context_value
}

-- Classify
category = ai.classify(text, into: ["urgent", "routine", "spam"])

-- Enrich a list (add AI-generated fields to each item)
enriched = items |> ai.enrich("Add a category field to each item")
```

### Data Operations
```
data = read "file.csv"
data = read "file.json"
content = fetch "https://example.com"
data = fetch "https://api.example.com" with {
  headers: { "X-Key": env.API_KEY }
  params: { limit: 50 }
}
save results to "output.json"
```

### Pipelines
```
results = data
  |> filter where price < 500000
  |> sort by price ascending
  |> take 10
  |> ai.enrich("Add analysis")
  |> save to "output.json"
```

### Control Flow
```
if score > 90:
  print "excellent"
else if score > 70:
  print "good"
else:
  print "needs work"

for each item in items:
  print "{item.name}: {item.value}"

match status:
  200 -> print "ok"
  404 -> print "not found"
  _ -> print "error"
```

### Functions
```
define greet(name: string) -> string:
  return "Hello {name}"
```

### Environment Variables
```
api_key = env.MY_API_KEY
```

### Error Handling
```
try:
  data = fetch url
catch network_error:
  log "API unreachable"
catch ai_error:
  log "AI call failed"
```

## Important Rules
1. Strings use double quotes only: "hello" (not 'hello')
2. String interpolation uses {}: "Hello {name}" (not f-strings)
3. Booleans are lowercase: true, false (not True, False)
4. No parentheses on print: print "hello" (not print("hello"))
5. Comments use --: -- comment (not # or //)
6. Loops use "for each": for each item in list: (not "for item in list:")
7. No imports needed — ai, fetch, read, save, print are all built-in
8. Indentation is 2 spaces
9. Map literals use { key: value } syntax
10. Files must end with .drift extension
"""


@mcp.prompt()
def drift_guide() -> str:
    """Complete guide to writing Drift programs. Use this when writing .drift files."""
    return DRIFT_LANGUAGE_GUIDE
```

**Step 2: Verify server still starts**

Run: `python3 -c "from drift_mcp.server import mcp; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add drift_mcp/server.py
git commit -m "feat(mcp): add drift-guide prompt with language reference"
```

---

### Task 7: Add `__main__.py` and verify end-to-end

**Files:**
- Create: `drift_mcp/__main__.py`
- Modify: `pyproject.toml`

**Step 1: Create `__main__.py`**

Create `drift_mcp/__main__.py`:

```python
"""Entry point for running the Drift MCP server: python3 -m drift_mcp"""
from drift_mcp.server import mcp

mcp.run(transport="stdio")
```

**Step 2: Update pyproject.toml**

Add `drift_mcp*` is already in the setuptools find include from Task 1. No additional changes needed. But verify it's there.

**Step 3: Install the mcp extras and verify**

Run: `pip install -e ".[mcp]"`

**Step 4: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: All tests pass (existing 457+ tests + new MCP tests).

**Step 5: Commit**

```bash
git add drift_mcp/__main__.py
git commit -m "feat(mcp): add __main__.py entry point for drift_mcp server"
```

---

### Task 8: Update README and push

**Files:**
- Modify: `README.md`

**Step 1: Add MCP section to README**

Add after the "Configuration" section and before "How It Works":

```markdown
## Claude Integration (MCP Server)

Use Drift directly from Claude Desktop or Claude Code. Claude can write, check, and run Drift programs.

### Setup

1. Install with MCP support:
```bash
git clone https://github.com/ethansurfas/drift.git
cd drift
pip install -e ".[mcp]"
```

2. Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "drift": {
      "command": "python3",
      "args": ["-m", "drift_mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

3. Restart Claude Desktop. Now just ask Claude to write and run Drift programs.
```

**Step 2: Commit and push**

```bash
git add README.md
git commit -m "docs: add MCP server setup instructions to README"
git push
```
