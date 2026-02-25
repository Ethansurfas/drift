"""Tests for the Drift MCP server tools."""

import os
import tempfile
from drift_mcp.server import write_drift_file, check_drift_file, build_drift_file, run_drift_file


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
