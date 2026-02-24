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
