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
    assert config["ai"]["max_retries"] == 2
    assert config["ai"]["timeout"] == 30


def test_config_caching():
    _reset_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        c1 = get_config(config_dir=tmpdir)
        c2 = get_config(config_dir=tmpdir)
    assert c1 is c2


def test_reset_config_clears_cache():
    _reset_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        c1 = get_config(config_dir=tmpdir)
        _reset_config()
        c2 = get_config(config_dir=tmpdir)
    assert c1 is not c2


def test_deep_merge_preserves_nested_defaults():
    _reset_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "drift.config")
        with open(config_path, "w") as f:
            f.write("ai:\n  timeout: 60\n")
        config = get_config(config_dir=tmpdir)
    assert config["ai"]["timeout"] == 60
    assert config["ai"]["provider"] == "anthropic"
    assert config["ai"]["max_retries"] == 2


def test_empty_config_file_uses_defaults():
    _reset_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "drift.config")
        with open(config_path, "w") as f:
            f.write("")
        config = get_config(config_dir=tmpdir)
    assert config["ai"]["provider"] == "anthropic"
