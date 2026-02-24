"""Drift runtime data operations — file I/O, HTTP, merge, query."""

import os
import csv
import json
import sqlite3
import dataclasses

import httpx

from drift_runtime.exceptions import DriftNetworkError, DriftFileError
from drift_runtime.types import _to_drift_dict


def read(path: str):
    """Read a file from disk. CSV->list[dict], JSON->parsed, text->str."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path) as f:
            return _to_drift_dict(json.load(f))
    elif ext == ".csv":
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return [_to_drift_dict(row) for row in reader]
    else:
        with open(path) as f:
            return f.read()


def save(data, path: str):
    """Save data to disk. Auto-creates directories. Format by extension."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

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
    return [_to_drift_dict(item) if isinstance(item, dict) else item for item in result]


def query(sql: str, source: str) -> list:
    """Execute a SQL query against a SQLite database. Returns list[dict]."""
    conn = sqlite3.connect(source)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(sql)
    rows = [_to_drift_dict(dict(row)) for row in cursor.fetchall()]
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
        return _to_drift_dict(response.json())
    return response.text
