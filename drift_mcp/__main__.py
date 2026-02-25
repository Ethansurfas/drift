"""Entry point for running the Drift MCP server: python3 -m drift_mcp"""
from drift_mcp.server import mcp

mcp.run(transport="stdio")
