# Phase 1 Design: Lexer + Parser + Transpiler

**Date:** 2026-02-23
**Status:** Approved

---

## Overview

Build the core Drift toolchain: a hand-written recursive descent lexer and parser in Python, plus a transpiler that emits runnable Python from the AST. Covers the core subset of Drift syntax (variables, types, print, comments, pipelines, control flow, functions, schemas, AI primitives). Agents (v2) and advanced error handling with retry/fallback are deferred.

## Architecture

```
file.drift → Lexer → [Tokens] → Parser → [AST] → Transpiler → file.py
```

### Modules

| Module | Responsibility |
|--------|---------------|
| `lexer.py` | Scans `.drift` source into tokens |
| `parser.py` | Consumes tokens, produces AST nodes |
| `ast_nodes.py` | Dataclass definitions for every AST node |
| `transpiler.py` | Walks the AST, emits Python source code |
| `cli.py` | Entry point (`drift run`, `drift build`, `drift check`) |
| `errors.py` | Custom error types with line/column info |

### Project Structure

```
drift/
├── drift/
│   ├── __init__.py
│   ├── lexer.py
│   ├── parser.py
│   ├── ast_nodes.py
│   ├── transpiler.py
│   ├── errors.py
│   └── cli.py
├── tests/
│   ├── test_lexer.py
│   ├── test_parser.py
│   └── test_transpiler.py
├── examples/
│   ├── hello.drift
│   ├── deal_analyzer.drift
│   └── pipeline.drift
├── docs/
│   └── plans/
├── pyproject.toml
└── DRIFT_LANGUAGE_SPEC.md
```

## Lexer Design

Scans source character by character, emits flat token list.

### Token Types

| Category | Tokens |
|----------|--------|
| Literals | `STRING`, `NUMBER`, `BOOLEAN`, `NONE` |
| Identifiers | `IDENTIFIER` |
| Keywords | `if`, `else`, `for`, `each`, `in`, `define`, `return`, `schema`, `try`, `catch`, `match`, `print`, `fetch`, `read`, `save`, `query`, `log`, `filter`, `sort`, `take`, `skip`, `group`, `merge`, `deduplicate`, `transform`, `where`, `by`, `to`, `on`, `with`, `using`, `as`, `and`, `or`, `not`, `ascending`, `descending`, `true`, `false`, `none`, `of`, `optional`, `retry`, `after`, `max`, `times`, `seconds`, `fallback`, `confident` |
| Operators | `PLUS`, `MINUS`, `STAR`, `SLASH`, `PERCENT`, `PIPE_ARROW` (`|>`), `ARROW` (`->`), `DOT`, `EQUALS`, `DOUBLE_EQUALS`, `NOT_EQUALS`, `LT`, `GT`, `LTE`, `GTE` |
| Delimiters | `COLON`, `COMMA`, `LPAREN`, `RPAREN`, `LBRACKET`, `RBRACKET`, `LBRACE`, `RBRACE` |
| Structure | `NEWLINE`, `INDENT`, `DEDENT`, `EOF` |

### Key Decisions

- Indentation-based scoping (like Python) — lexer tracks indent level, emits INDENT/DEDENT tokens
- String interpolation — `"Hello {name}"` lexed as single STRING token; parser extracts interpolations
- Comments — `--` to end of line, discarded
- `|>` is a single token

## AST Nodes

Every construct becomes a Python dataclass.

### Expression Nodes

- `NumberLiteral(value)`, `StringLiteral(value, interpolations)`, `BooleanLiteral(value)`, `NoneLiteral()`
- `ListLiteral(elements)`, `MapLiteral(pairs)`
- `Identifier(name)`, `DotAccess(object, field)`
- `BinaryOp(left, op, right)`, `UnaryOp(op, operand)`
- `FunctionCall(name, args, kwargs)`

### AI Primitive Nodes

- `AIAsk(prompt, schema?, using?)`, `AIClassify(input, categories)`
- `AIEmbed(input)`, `AISee(input, prompt)`, `AIPredict(prompt, schema?)`
- `AIEnrich(prompt)`, `AIScore(prompt, schema?)`

### Statement Nodes

- `Assignment(target, type_hint?, value)`, `PrintStatement(value)`, `LogStatement(value)`
- `IfStatement(condition, body, elseifs, else_body?)`, `ForEach(variable, iterable, body)`
- `MatchStatement(subject, arms)`, `FunctionDef(name, params, return_type?, body)`
- `ReturnStatement(value)`, `SchemaDefinition(name, fields)`

### Data Operation Nodes

- `FetchExpression(url, options?)`, `ReadExpression(path)`
- `SaveStatement(data, path)`, `QueryExpression(sql, source)`

### Pipeline Nodes

- `Pipeline(source, stages)`
- Stage types: `FilterStage`, `SortStage`, `TakeStage`, `SkipStage`, `GroupStage`, `MergeStage`, `DeduplicateStage`, `TransformStage`, `EachStage`

### Error Handling Nodes

- `TryCatch(try_body, catches)`, `CatchClause(error_type, body)`, `RetryStatement(delay, max_attempts)`

## Transpiler Design

Walks AST, emits Python source. Key translations:

| Drift | Python |
|-------|--------|
| `name = "Austin"` | `name = "Austin"` |
| `print "Hello {name}"` | `print(f"Hello {name}")` |
| `ai.ask("prompt")` | `drift_runtime.ai.ask("prompt")` |
| `ai.ask(...) -> Schema using {...}` | `drift_runtime.ai.ask(..., schema=Schema, context={...})` |
| `fetch url with {...}` | `drift_runtime.fetch(url, **{...})` |
| `read "file.csv"` | `drift_runtime.read("file.csv")` |
| `save data to "out.json"` | `drift_runtime.save(data, "out.json")` |
| `query "SQL" on db.main` | `drift_runtime.query("SQL", db.main)` |
| `list \|> filter where x > 5` | `[item for item in list if item.x > 5]` |
| `\|> sort by price ascending` | `sorted(..., key=lambda item: item["price"])` |
| `\|> take 10` | `[...][:10]` |
| `\|> ai.enrich("prompt")` | `drift_runtime.ai.enrich(..., "prompt")` |
| `schema X:` | `@dataclass class X:` |
| `define func(x: string):` | `def func(x: str):` |
| `for each x in list:` | `for x in list:` |
| `match x:` | `match x:` (Python 3.10+) |

Generated files import `drift_runtime` at the top. Runtime is Phase 3; we stub it for testing.

## Testing Strategy

### test_lexer.py
- Basic tokens, string interpolation, indentation, comment stripping, `|>`, edge cases

### test_parser.py
- Assignments, schemas, functions, pipelines, control flow, AI primitives, fetch/read/save/query

### test_transpiler.py
- Each node type produces valid Python
- String interpolation → f-strings
- Pipelines → list comprehensions / chained calls
- Schemas → dataclasses
- End-to-end: all example programs parse and transpile; output passes `ast.parse`

### Integration
- Example `.drift` files from spec serve as integration tests
- Each must lex → parse → transpile without errors
- Output validated with Python's `ast.parse`

## Decisions

- **Language:** Python (natural fit for Python-targeting transpiler)
- **Parser style:** Hand-written recursive descent (zero deps, full control)
- **Scope:** Core subset — variables, types, print, comments, pipelines, control flow, functions, schemas, AI primitives
- **Deferred:** Agents (v2), advanced retry/fallback error handling
