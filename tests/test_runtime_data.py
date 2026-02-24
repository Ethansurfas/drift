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
