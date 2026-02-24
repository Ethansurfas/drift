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
