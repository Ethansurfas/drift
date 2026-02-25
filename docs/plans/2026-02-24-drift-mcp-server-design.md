# Drift MCP Server — Design Document

## Goal

An MCP server that lets Claude Desktop and Claude Code write, check, and run Drift programs. When a user says "analyze my spending," Claude writes a .drift file and executes it — producing readable code and real results.

## Architecture

Single Python module (`drift_mcp/server.py`) using FastMCP. Imports the Drift compiler directly — no subprocess calls. Runs over stdio transport.

```
User → Claude Desktop/Code → MCP (stdio) → drift_mcp/server.py → Drift compiler + runtime
```

## Tools

| Tool | Input | What it does | Returns |
|------|-------|-------------|---------|
| `drift_write` | `filepath`, `source` | Saves Drift source code to a file | Confirmation |
| `drift_check` | `filepath` | Lexes + parses, validates syntax | "OK" or error |
| `drift_run` | `filepath` | Compiles + executes, captures stdout | Program output |
| `drift_build` | `filepath` | Compiles, returns Python code | Transpiled Python |

## Prompt

One registered MCP prompt (`drift-guide`) containing the Drift language syntax reference. Teaches Claude how to write valid Drift code.

## File Structure

```
drift_mcp/
├── __init__.py
└── server.py      # ~150 lines
```

## Installation

```json
{
  "mcpServers": {
    "drift": {
      "command": "python3",
      "args": ["-m", "drift_mcp.server"],
      "env": {
        "ANTHROPIC_API_KEY": "user-key"
      }
    }
  }
}
```

## Dependencies

- `mcp` (Python MCP SDK with FastMCP)
- Drift compiler (already in repo)

## Key Constraints

- Never `print()` to stdout — stdio is reserved for MCP protocol. All logging goes to stderr.
- `drift_run` must capture stdout from `exec()` using `io.StringIO` redirect.
- Error messages from the compiler (`DriftError`) are caught and returned as tool results, not exceptions.
