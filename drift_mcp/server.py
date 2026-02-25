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


if __name__ == "__main__":
    mcp.run(transport="stdio")
