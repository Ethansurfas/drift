"""Drift runtime exception types.

These map to Drift catch blocks:
  catch network_error:  ->  except DriftNetworkError
  catch ai_error:       ->  except DriftAIError
"""


class DriftRuntimeError(Exception):
    """Base runtime error."""
    pass


class DriftAIError(DriftRuntimeError):
    """AI inference failed."""
    pass


class DriftNetworkError(DriftRuntimeError):
    """HTTP request failed."""
    pass


class DriftFileError(DriftRuntimeError):
    """File operation failed."""
    pass


class DriftConfigError(DriftRuntimeError):
    """Configuration error."""
    pass
