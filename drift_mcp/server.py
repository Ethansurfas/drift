"""Drift MCP Server — exposes Drift compiler tools via MCP protocol."""

import sys
import io
import os

from mcp.server.fastmcp import FastMCP

from drift.lexer import Lexer
from drift.parser import Parser
from drift.transpiler import Transpiler
from drift.errors import DriftError

mcp = FastMCP("drift")


@mcp.tool()
def drift_write(filepath: str, source: str) -> str:
    """Write Drift source code to a file. Use this to create new .drift programs.

    Args:
        filepath: Path to the .drift file to create (e.g. "my_program.drift")
        source: The Drift source code to write
    """
    return write_drift_file(filepath, source)


def write_drift_file(filepath: str, source: str) -> str:
    """Core logic for writing a drift file — testable without MCP."""
    if not filepath.endswith(".drift"):
        return "Error: filepath must end with .drift"
    try:
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(source)
        return f"Saved: {filepath}"
    except OSError as e:
        return f"Error writing file: {e}"


@mcp.tool()
def drift_check(filepath: str) -> str:
    """Check Drift syntax without running. Validates that the file is valid Drift code.

    Args:
        filepath: Path to the .drift file to check
    """
    return check_drift_file(filepath)


def check_drift_file(filepath: str) -> str:
    """Core logic for checking a drift file — testable without MCP."""
    try:
        with open(filepath) as f:
            source = f.read()
    except FileNotFoundError:
        return f"Error: file not found: {filepath}"
    except OSError as e:
        return f"Error reading file: {e}"

    try:
        tokens = Lexer(source).tokenize()
        Parser(tokens).parse()
        return f"OK: {filepath}"
    except DriftError as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
