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


@mcp.tool()
def drift_build(filepath: str) -> str:
    """Transpile a Drift file to Python. Returns the generated Python code.

    Args:
        filepath: Path to the .drift file to transpile
    """
    return build_drift_file(filepath)


def build_drift_file(filepath: str) -> str:
    """Core logic for building a drift file — testable without MCP."""
    try:
        with open(filepath) as f:
            source = f.read()
    except FileNotFoundError:
        return f"Error: file not found: {filepath}"
    except OSError as e:
        return f"Error reading file: {e}"

    try:
        tokens = Lexer(source).tokenize()
        tree = Parser(tokens).parse()
        python_code = Transpiler(tree).transpile()
        return python_code
    except DriftError as e:
        return f"Error: {e}"


@mcp.tool()
def drift_run(filepath: str) -> str:
    """Run a Drift program. Transpiles to Python and executes it, returning the output.

    Args:
        filepath: Path to the .drift file to run
    """
    return run_drift_file(filepath)


def run_drift_file(filepath: str) -> str:
    """Core logic for running a drift file — testable without MCP."""
    try:
        with open(filepath) as f:
            source = f.read()
    except FileNotFoundError:
        return f"Error: file not found: {filepath}"
    except OSError as e:
        return f"Error reading file: {e}"

    try:
        tokens = Lexer(source).tokenize()
        tree = Parser(tokens).parse()
        python_code = Transpiler(tree).transpile()
    except DriftError as e:
        return f"Error: {e}"

    # Capture stdout during execution
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    try:
        sys.stdout = captured_out
        sys.stderr = captured_err
        exec(python_code, {"__name__": "__main__"})
    except Exception as e:
        return f"{captured_out.getvalue()}Error during execution: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    output = captured_out.getvalue()
    err_output = captured_err.getvalue()
    if err_output:
        output += f"\n[stderr]: {err_output}"
    return output if output else "(program produced no output)"


if __name__ == "__main__":
    mcp.run(transport="stdio")
