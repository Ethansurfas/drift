"""Drift error types with source location info."""


class DriftError(Exception):
    def __init__(self, message: str, line: int = 0, column: int = 0):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Line {line}, Col {column}: {message}")


class LexerError(DriftError):
    pass


class ParseError(DriftError):
    pass


class TranspileError(DriftError):
    pass
