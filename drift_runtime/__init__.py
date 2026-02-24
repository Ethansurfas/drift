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
