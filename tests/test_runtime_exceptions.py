"""Tests for drift_runtime exception hierarchy."""

from drift_runtime.exceptions import (
    DriftRuntimeError,
    DriftAIError,
    DriftNetworkError,
    DriftFileError,
    DriftConfigError,
)


def test_drift_runtime_error_is_exception():
    assert issubclass(DriftRuntimeError, Exception)


def test_drift_ai_error_inherits_runtime_error():
    assert issubclass(DriftAIError, DriftRuntimeError)


def test_drift_network_error_inherits_runtime_error():
    assert issubclass(DriftNetworkError, DriftRuntimeError)


def test_drift_file_error_inherits_runtime_error():
    assert issubclass(DriftFileError, DriftRuntimeError)


def test_drift_config_error_inherits_runtime_error():
    assert issubclass(DriftConfigError, DriftRuntimeError)


def test_errors_carry_message():
    err = DriftAIError("model timed out")
    assert str(err) == "model timed out"


def test_errors_are_catchable_as_runtime_error():
    try:
        raise DriftNetworkError("connection refused")
    except DriftRuntimeError as e:
        assert "connection refused" in str(e)
