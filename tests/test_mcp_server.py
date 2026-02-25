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
