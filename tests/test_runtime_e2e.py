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
