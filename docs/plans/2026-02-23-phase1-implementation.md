# Phase 1: Lexer + Parser + Transpiler — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working Drift toolchain that takes `.drift` files and produces runnable `.py` files.

**Architecture:** Hand-written recursive descent lexer and parser in Python, zero dependencies. Lexer emits tokens, parser builds an AST from dataclass nodes, transpiler walks the AST and emits Python source. CLI wraps it all.

**Tech Stack:** Python 3.11+, pytest, no external deps for core (just stdlib)

**Reference:** `docs/plans/2026-02-23-phase1-lexer-parser-transpiler-design.md` and `DRIFT_LANGUAGE_SPEC.md`

---

### Task 1: Project Scaffold

**Files:**
- Create: `drift/__init__.py`
- Create: `drift/errors.py`
- Create: `pyproject.toml`
- Create: `tests/__init__.py`
- Copy: `DRIFT_LANGUAGE_SPEC.md` into project root

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "drift-lang"
version = "0.1.0"
description = "The AI-native programming language"
requires-python = ">=3.11"
license = "MIT"

[project.scripts]
drift = "drift.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create drift/__init__.py**

```python
"""Drift — The AI-native programming language."""
__version__ = "0.1.0"
```

**Step 3: Create drift/errors.py**

```python
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
```

**Step 4: Create tests/__init__.py**

Empty file.

**Step 5: Copy spec into project**

```bash
cp ~/Downloads/DRIFT_LANGUAGE_SPEC.md /Users/ethansurfas/drift/
```

**Step 6: Create examples directory**

```bash
mkdir -p /Users/ethansurfas/drift/examples
```

**Step 7: Commit**

```bash
git add -A && git commit -m "feat: project scaffold with pyproject.toml and error types"
```

---

### Task 2: AST Node Definitions

**Files:**
- Create: `drift/ast_nodes.py`
- Create: `tests/test_ast_nodes.py`

**Step 1: Write test verifying AST nodes are constructable**

```python
# tests/test_ast_nodes.py
from drift.ast_nodes import (
    Program, NumberLiteral, StringLiteral, BooleanLiteral, NoneLiteral,
    ListLiteral, MapLiteral, Identifier, DotAccess, BinaryOp, UnaryOp,
    FunctionCall, Assignment, PrintStatement, LogStatement,
    IfStatement, ForEach, MatchStatement, MatchArm, FunctionDef,
    ReturnStatement, SchemaDefinition, SchemaField,
    FetchExpression, ReadExpression, SaveStatement, QueryExpression,
    Pipeline, FilterStage, SortStage, TakeStage, SkipStage,
    GroupStage, DeduplicateStage, TransformStage, EachStage, MergeExpression,
    AIAsk, AIClassify, AIEmbed, AISee, AIPredict, AIEnrich, AIScore,
    TryCatch, CatchClause,
)

def test_program_node():
    p = Program(body=[])
    assert p.body == []

def test_number_literal():
    n = NumberLiteral(value=42.0, line=1, col=1)
    assert n.value == 42.0

def test_string_literal_with_interpolations():
    s = StringLiteral(value="Hello {name}", parts=["Hello ", Identifier(name="name", line=1, col=8)], line=1, col=1)
    assert len(s.parts) == 2

def test_assignment():
    a = Assignment(
        target="x",
        type_hint=None,
        value=NumberLiteral(value=5.0, line=1, col=5),
        line=1, col=1,
    )
    assert a.target == "x"

def test_pipeline():
    p = Pipeline(
        source=Identifier(name="data", line=1, col=1),
        stages=[TakeStage(count=NumberLiteral(value=10.0, line=2, col=1), line=2, col=1)],
        line=1, col=1,
    )
    assert len(p.stages) == 1

def test_ai_ask():
    a = AIAsk(
        prompt=StringLiteral(value="hello", parts=["hello"], line=1, col=1),
        schema=None,
        using=None,
        line=1, col=1,
    )
    assert a.schema is None

def test_schema_definition():
    s = SchemaDefinition(
        name="Deal",
        fields=[SchemaField(name="price", type_name="number", optional=False, line=2, col=2)],
        line=1, col=1,
    )
    assert s.fields[0].type_name == "number"

def test_fetch_expression():
    f = FetchExpression(
        url=StringLiteral(value="http://example.com", parts=["http://example.com"], line=1, col=1),
        options=None,
        line=1, col=1,
    )
    assert f.options is None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_ast_nodes.py -v`
Expected: ImportError

**Step 3: Implement ast_nodes.py**

All nodes are dataclasses. Every node has `line` and `col` fields for error reporting.

```python
# drift/ast_nodes.py
"""AST node definitions for the Drift language."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# ---------- Base ----------

@dataclass
class Node:
    line: int = 0
    col: int = 0

# ---------- Program ----------

@dataclass
class Program(Node):
    body: list[Any] = field(default_factory=list)

# ---------- Expressions ----------

@dataclass
class NumberLiteral(Node):
    value: float = 0.0

@dataclass
class StringLiteral(Node):
    value: str = ""
    parts: list[Any] = field(default_factory=list)  # mix of str and Expression nodes

@dataclass
class BooleanLiteral(Node):
    value: bool = False

@dataclass
class NoneLiteral(Node):
    pass

@dataclass
class ListLiteral(Node):
    elements: list[Any] = field(default_factory=list)

@dataclass
class MapLiteral(Node):
    pairs: list[tuple[str, Any]] = field(default_factory=list)

@dataclass
class Identifier(Node):
    name: str = ""

@dataclass
class DotAccess(Node):
    object: Any = None
    field_name: str = ""

@dataclass
class BinaryOp(Node):
    left: Any = None
    op: str = ""
    right: Any = None

@dataclass
class UnaryOp(Node):
    op: str = ""
    operand: Any = None

@dataclass
class FunctionCall(Node):
    callee: Any = None
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)

# ---------- AI Primitives ----------

@dataclass
class AIAsk(Node):
    prompt: Any = None
    schema: str | None = None
    using: dict[str, Any] | None = None

@dataclass
class AIClassify(Node):
    input: Any = None
    categories: list[Any] = field(default_factory=list)

@dataclass
class AIEmbed(Node):
    input: Any = None

@dataclass
class AISee(Node):
    input: Any = None
    prompt: Any = None

@dataclass
class AIPredict(Node):
    prompt: Any = None
    schema: str | None = None

@dataclass
class AIEnrich(Node):
    prompt: Any = None

@dataclass
class AIScore(Node):
    prompt: Any = None
    schema: str | None = None

# ---------- Statements ----------

@dataclass
class Assignment(Node):
    target: str = ""
    type_hint: str | None = None
    value: Any = None

@dataclass
class PrintStatement(Node):
    value: Any = None

@dataclass
class LogStatement(Node):
    value: Any = None

@dataclass
class ReturnStatement(Node):
    value: Any = None

@dataclass
class IfStatement(Node):
    condition: Any = None
    body: list[Any] = field(default_factory=list)
    elseifs: list[tuple[Any, list[Any]]] = field(default_factory=list)
    else_body: list[Any] | None = None

@dataclass
class ForEach(Node):
    variable: str = ""
    iterable: Any = None
    body: list[Any] = field(default_factory=list)

@dataclass
class MatchStatement(Node):
    subject: Any = None
    arms: list[MatchArm] = field(default_factory=list)

@dataclass
class MatchArm(Node):
    pattern: Any = None  # expression or "_" wildcard
    body: Any = None

@dataclass
class FunctionDef(Node):
    name: str = ""
    params: list[tuple[str, str | None]] = field(default_factory=list)  # (name, type_hint)
    return_type: str | None = None
    body: list[Any] = field(default_factory=list)

@dataclass
class SchemaDefinition(Node):
    name: str = ""
    fields: list[SchemaField] = field(default_factory=list)

@dataclass
class SchemaField(Node):
    name: str = ""
    type_name: str = ""
    optional: bool = False

# ---------- Data Operations ----------

@dataclass
class FetchExpression(Node):
    url: Any = None
    options: dict[str, Any] | None = None

@dataclass
class ReadExpression(Node):
    path: Any = None

@dataclass
class SaveStatement(Node):
    data: Any = None
    path: Any = None

@dataclass
class QueryExpression(Node):
    sql: Any = None
    source: Any = None

@dataclass
class MergeExpression(Node):
    sources: list[Any] = field(default_factory=list)

# ---------- Pipelines ----------

@dataclass
class Pipeline(Node):
    source: Any = None
    stages: list[Any] = field(default_factory=list)

@dataclass
class FilterStage(Node):
    condition: Any = None

@dataclass
class SortStage(Node):
    field_name: str = ""
    direction: str = "ascending"

@dataclass
class TakeStage(Node):
    count: Any = None

@dataclass
class SkipStage(Node):
    count: Any = None

@dataclass
class GroupStage(Node):
    field_name: str = ""

@dataclass
class DeduplicateStage(Node):
    field_name: str = ""

@dataclass
class TransformStage(Node):
    variable: str = ""
    body: list[Any] = field(default_factory=list)

@dataclass
class EachStage(Node):
    variable: str = ""
    body: list[Any] = field(default_factory=list)

# ---------- Error Handling ----------

@dataclass
class TryCatch(Node):
    try_body: list[Any] = field(default_factory=list)
    catches: list[CatchClause] = field(default_factory=list)

@dataclass
class CatchClause(Node):
    error_type: str = ""
    body: list[Any] = field(default_factory=list)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_ast_nodes.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/ast_nodes.py tests/test_ast_nodes.py && git commit -m "feat: AST node definitions"
```

---

### Task 3: Lexer — Token Types and Basic Tokenization

**Files:**
- Create: `drift/lexer.py`
- Create: `tests/test_lexer.py`

**Step 1: Write failing tests for basic tokenization**

```python
# tests/test_lexer.py
from drift.lexer import Lexer, TokenType

def test_empty_input():
    tokens = Lexer("").tokenize()
    assert tokens[-1].type == TokenType.EOF

def test_number_integer():
    tokens = Lexer("42").tokenize()
    assert tokens[0].type == TokenType.NUMBER
    assert tokens[0].value == "42"

def test_number_float():
    tokens = Lexer("3.14").tokenize()
    assert tokens[0].type == TokenType.NUMBER
    assert tokens[0].value == "3.14"

def test_string_double_quotes():
    tokens = Lexer('"hello world"').tokenize()
    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == "hello world"

def test_identifier():
    tokens = Lexer("my_var").tokenize()
    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[0].value == "my_var"

def test_keyword_if():
    tokens = Lexer("if").tokenize()
    assert tokens[0].type == TokenType.IF

def test_keyword_true():
    tokens = Lexer("true").tokenize()
    assert tokens[0].type == TokenType.TRUE

def test_keyword_false():
    tokens = Lexer("false").tokenize()
    assert tokens[0].type == TokenType.FALSE

def test_operators():
    tokens = Lexer("+ - * / %").tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert TokenType.PLUS in types
    assert TokenType.MINUS in types
    assert TokenType.STAR in types
    assert TokenType.SLASH in types
    assert TokenType.PERCENT in types

def test_pipe_arrow():
    tokens = Lexer("|>").tokenize()
    assert tokens[0].type == TokenType.PIPE_ARROW

def test_arrow():
    tokens = Lexer("->").tokenize()
    assert tokens[0].type == TokenType.ARROW

def test_comparison_operators():
    tokens = Lexer("== != < > <= >=").tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert TokenType.DOUBLE_EQUALS in types
    assert TokenType.NOT_EQUALS in types
    assert TokenType.LT in types
    assert TokenType.GT in types
    assert TokenType.LTE in types
    assert TokenType.GTE in types

def test_delimiters():
    tokens = Lexer(": , ( ) [ ] { }").tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert TokenType.COLON in types
    assert TokenType.COMMA in types
    assert TokenType.LPAREN in types
    assert TokenType.RPAREN in types
    assert TokenType.LBRACKET in types
    assert TokenType.RBRACKET in types
    assert TokenType.LBRACE in types
    assert TokenType.RBRACE in types

def test_dot():
    tokens = Lexer("ai.ask").tokenize()
    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[1].type == TokenType.DOT
    assert tokens[2].type == TokenType.IDENTIFIER

def test_comment_stripped():
    tokens = Lexer("x = 5 -- this is a comment").tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF and t.type != TokenType.NEWLINE]
    assert TokenType.IDENTIFIER in types
    assert TokenType.EQUALS in types
    assert TokenType.NUMBER in types
    # no comment token should appear
    for t in tokens:
        assert "comment" not in str(t.type).lower()

def test_comment_only_line():
    tokens = Lexer("-- just a comment").tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF and t.type != TokenType.NEWLINE]
    assert len(types) == 0

def test_assignment_tokens():
    tokens = Lexer('name = "Drift"').tokenize()
    types = [t.type for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)]
    assert types == [TokenType.IDENTIFIER, TokenType.EQUALS, TokenType.STRING]

def test_equals_vs_double_equals():
    tokens = Lexer("x = 5\ny == 3").tokenize()
    equals_tokens = [t for t in tokens if t.type in (TokenType.EQUALS, TokenType.DOUBLE_EQUALS)]
    assert equals_tokens[0].type == TokenType.EQUALS
    assert equals_tokens[1].type == TokenType.DOUBLE_EQUALS

def test_token_line_numbers():
    tokens = Lexer("x\ny\nz").tokenize()
    idents = [t for t in tokens if t.type == TokenType.IDENTIFIER]
    assert idents[0].line == 1
    assert idents[1].line == 2
    assert idents[2].line == 3

def test_string_with_interpolation_braces():
    tokens = Lexer('"Hello {name}!"').tokenize()
    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == "Hello {name}!"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_lexer.py -v`
Expected: ImportError

**Step 3: Implement lexer.py**

Create `drift/lexer.py` with:
- `TokenType` enum with all token types from the design doc
- `Token` dataclass with `type`, `value`, `line`, `column`
- `Lexer` class with `tokenize()` method
- Character-by-character scanning
- Keywords dict mapping strings to TokenTypes
- Comment stripping (`--` to EOL)
- String scanning (handle `{...}` as part of string value)
- Number scanning (integers and floats)
- Multi-char operator scanning (`|>`, `->`, `==`, `!=`, `<=`, `>=`)
- Do NOT implement indentation tracking yet (Task 4)

The `Lexer.__init__` takes source string. `tokenize()` returns `list[Token]`.

Key implementation details:
- Scan one char at a time with `self.pos`, `self.line`, `self.col`
- `advance()` moves forward, returns current char
- `peek()` looks ahead without advancing
- Skip whitespace (spaces/tabs on same line) but NOT newlines
- Emit `NEWLINE` token on `\n`
- Identifiers: `[a-zA-Z_][a-zA-Z0-9_]*`, then check keywords dict
- Numbers: `[0-9]+(\.[0-9]+)?`
- Strings: `"` to `"`, include `{...}` content as-is in value
- At end, emit `EOF`

**Step 4: Run test to verify it passes**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_lexer.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/lexer.py tests/test_lexer.py && git commit -m "feat: lexer with basic tokenization"
```

---

### Task 4: Lexer — Indentation Tracking

**Files:**
- Modify: `drift/lexer.py`
- Modify: `tests/test_lexer.py`

**Step 1: Add indentation tests**

```python
# append to tests/test_lexer.py

def test_indent_simple():
    src = "if x:\n  y = 1"
    tokens = Lexer(src).tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert TokenType.INDENT in types

def test_dedent_simple():
    src = "if x:\n  y = 1\nz = 2"
    tokens = Lexer(src).tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert TokenType.DEDENT in types

def test_nested_indent():
    src = "if x:\n  if y:\n    z = 1\n  w = 2\na = 3"
    tokens = Lexer(src).tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    indent_count = types.count(TokenType.INDENT)
    dedent_count = types.count(TokenType.DEDENT)
    assert indent_count == 2
    assert dedent_count == 2

def test_no_indent_flat():
    src = "x = 1\ny = 2\nz = 3"
    tokens = Lexer(src).tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert TokenType.INDENT not in types
    assert TokenType.DEDENT not in types

def test_dedent_at_eof():
    src = "if x:\n  y = 1"
    tokens = Lexer(src).tokenize()
    # Should have a DEDENT before EOF
    non_eof = [t for t in tokens if t.type != TokenType.EOF]
    assert non_eof[-1].type == TokenType.DEDENT or non_eof[-2].type == TokenType.DEDENT

def test_blank_lines_ignored():
    src = "x = 1\n\n\ny = 2"
    tokens = Lexer(src).tokenize()
    idents = [t for t in tokens if t.type == TokenType.IDENTIFIER]
    assert len(idents) == 2
```

**Step 2: Run tests — new ones should fail**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_lexer.py -v`
Expected: New indent/dedent tests FAIL

**Step 3: Add indentation tracking to lexer**

Modify `Lexer.tokenize()`:
- Maintain an `indent_stack = [0]` (stack of indent levels)
- At the start of each line (after NEWLINE), count leading spaces
- If indent > top of stack: push new level, emit INDENT
- If indent < top of stack: pop levels and emit DEDENT for each
- If indent == top of stack: no token
- Skip blank lines (lines that are only whitespace or comments)
- Before EOF: emit DEDENT for any remaining indent levels > 0

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_lexer.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/lexer.py tests/test_lexer.py && git commit -m "feat: lexer indentation tracking (INDENT/DEDENT)"
```

---

### Task 5: Parser — Expressions

**Files:**
- Create: `drift/parser.py`
- Create: `tests/test_parser.py`

**Step 1: Write failing tests for expression parsing**

```python
# tests/test_parser.py
from drift.lexer import Lexer
from drift.parser import Parser
from drift.ast_nodes import *

def parse(src: str):
    tokens = Lexer(src).tokenize()
    return Parser(tokens).parse()

def test_number_literal():
    prog = parse("x = 42")
    stmt = prog.body[0]
    assert isinstance(stmt, Assignment)
    assert isinstance(stmt.value, NumberLiteral)
    assert stmt.value.value == 42.0

def test_string_literal():
    prog = parse('x = "hello"')
    stmt = prog.body[0]
    assert isinstance(stmt.value, StringLiteral)
    assert stmt.value.value == "hello"

def test_boolean_literal():
    prog = parse("x = true")
    stmt = prog.body[0]
    assert isinstance(stmt.value, BooleanLiteral)
    assert stmt.value.value is True

def test_list_literal():
    prog = parse('x = [1, 2, 3]')
    stmt = prog.body[0]
    assert isinstance(stmt.value, ListLiteral)
    assert len(stmt.value.elements) == 3

def test_map_literal():
    prog = parse('x = { name: "test", value: 42 }')
    stmt = prog.body[0]
    assert isinstance(stmt.value, MapLiteral)
    assert len(stmt.value.pairs) == 2

def test_identifier():
    prog = parse("x = y")
    stmt = prog.body[0]
    assert isinstance(stmt.value, Identifier)
    assert stmt.value.name == "y"

def test_dot_access():
    prog = parse("x = obj.field")
    stmt = prog.body[0]
    assert isinstance(stmt.value, DotAccess)

def test_chained_dot_access():
    prog = parse("x = a.b.c")
    stmt = prog.body[0]
    val = stmt.value
    assert isinstance(val, DotAccess)
    assert isinstance(val.object, DotAccess)

def test_binary_op_arithmetic():
    prog = parse("x = 1 + 2")
    stmt = prog.body[0]
    assert isinstance(stmt.value, BinaryOp)
    assert stmt.value.op == "+"

def test_binary_op_comparison():
    prog = parse("x = a > 5")
    stmt = prog.body[0]
    assert isinstance(stmt.value, BinaryOp)
    assert stmt.value.op == ">"

def test_binary_op_and_or():
    prog = parse("x = a and b or c")
    stmt = prog.body[0]
    assert isinstance(stmt.value, BinaryOp)

def test_parenthesized_expression():
    prog = parse("x = (1 + 2) * 3")
    stmt = prog.body[0]
    assert isinstance(stmt.value, BinaryOp)
    assert stmt.value.op == "*"

def test_unary_not():
    prog = parse("x = not y")
    stmt = prog.body[0]
    assert isinstance(stmt.value, UnaryOp)
    assert stmt.value.op == "not"

def test_unary_minus():
    prog = parse("x = -5")
    stmt = prog.body[0]
    assert isinstance(stmt.value, UnaryOp)
    assert stmt.value.op == "-"
```

**Step 2: Run tests to verify failure**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_parser.py -v`
Expected: ImportError

**Step 3: Implement parser.py — expression parsing**

Create `drift/parser.py` with:
- `Parser.__init__(self, tokens: list[Token])` — stores tokens, `self.pos = 0`
- `parse() -> Program` — parse top-level statements until EOF
- Helper methods: `current()`, `peek()`, `advance()`, `expect(type)`, `match(*types)`
- `skip_newlines()` — skip NEWLINE tokens between statements

Expression parsing (recursive descent, precedence climbing):
- `parse_expression()` → entry point, calls `parse_or()`
- `parse_or()` → `parse_and()` (`or`)
- `parse_and()` → `parse_not()` (`and`)
- `parse_not()` → `parse_comparison()` (`not` prefix)
- `parse_comparison()` → `parse_addition()` (`==`, `!=`, `<`, `>`, `<=`, `>=`)
- `parse_addition()` → `parse_multiplication()` (`+`, `-`)
- `parse_multiplication()` → `parse_unary()` (`*`, `/`, `%`)
- `parse_unary()` → `parse_postfix()` (`-` prefix)
- `parse_postfix()` → `parse_primary()` (then handle `.field` chains)
- `parse_primary()` → literals, identifiers, `(expr)`, `[list]`, `{map}`

Statement parsing (just `parse_assignment` for now — more in later tasks):
- If current is IDENTIFIER and next is EQUALS: parse assignment
- If current is IDENTIFIER and next is COLON then IDENTIFIER then EQUALS: typed assignment

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/parser.py tests/test_parser.py && git commit -m "feat: parser with expression and assignment parsing"
```

---

### Task 6: Parser — Print, Log, Schemas

**Files:**
- Modify: `drift/parser.py`
- Modify: `tests/test_parser.py`

**Step 1: Add tests**

```python
# append to tests/test_parser.py

def test_print_string():
    prog = parse('print "Hello, Drift!"')
    assert isinstance(prog.body[0], PrintStatement)

def test_print_interpolated():
    prog = parse('print "Hello {name}!"')
    stmt = prog.body[0]
    assert isinstance(stmt, PrintStatement)
    assert isinstance(stmt.value, StringLiteral)

def test_log_statement():
    prog = parse('log "Processing item"')
    assert isinstance(prog.body[0], LogStatement)

def test_schema_simple():
    src = "schema Deal:\n  address: string\n  price: number"
    prog = parse(src)
    schema = prog.body[0]
    assert isinstance(schema, SchemaDefinition)
    assert schema.name == "Deal"
    assert len(schema.fields) == 2
    assert schema.fields[0].name == "address"
    assert schema.fields[0].type_name == "string"

def test_schema_optional_field():
    src = "schema X:\n  data: map (optional)"
    prog = parse(src)
    schema = prog.body[0]
    assert schema.fields[0].optional is True

def test_schema_list_of_type():
    src = "schema X:\n  items: list of string"
    prog = parse(src)
    schema = prog.body[0]
    assert "list of string" in schema.fields[0].type_name or schema.fields[0].type_name == "list of string"

def test_multiple_statements():
    src = 'x = 1\nprint "hello"\ny = 2'
    prog = parse(src)
    assert len(prog.body) == 3
```

**Step 2: Run tests — new ones fail**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_parser.py -v`

**Step 3: Add print, log, schema parsing**

In `Parser`:
- `parse_statement()`: check current token type
  - `PRINT` → `parse_print()`
  - `LOG` → `parse_log()`
  - `SCHEMA` → `parse_schema()`
- `parse_print()`: advance past PRINT, parse expression → `PrintStatement`
- `parse_log()`: advance past LOG, parse expression → `LogStatement`
- `parse_schema()`: advance past SCHEMA, expect IDENTIFIER (name), expect COLON, expect NEWLINE+INDENT, parse fields until DEDENT. Each field: IDENTIFIER, COLON, type name. Handle `list of <type>` and `(optional)` suffix.

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/parser.py tests/test_parser.py && git commit -m "feat: parse print, log, schema"
```

---

### Task 7: Parser — Control Flow (if/else, for each, match)

**Files:**
- Modify: `drift/parser.py`
- Modify: `tests/test_parser.py`

**Step 1: Add tests**

```python
# append to tests/test_parser.py

def test_if_simple():
    src = "if x > 5:\n  print \"big\""
    prog = parse(src)
    stmt = prog.body[0]
    assert isinstance(stmt, IfStatement)
    assert isinstance(stmt.condition, BinaryOp)
    assert len(stmt.body) == 1

def test_if_else():
    src = "if x > 5:\n  print \"big\"\nelse:\n  print \"small\""
    prog = parse(src)
    stmt = prog.body[0]
    assert isinstance(stmt, IfStatement)
    assert stmt.else_body is not None

def test_if_elseif_else():
    src = "if x > 10:\n  print \"a\"\nelse if x > 5:\n  print \"b\"\nelse:\n  print \"c\""
    prog = parse(src)
    stmt = prog.body[0]
    assert isinstance(stmt, IfStatement)
    assert len(stmt.elseifs) == 1
    assert stmt.else_body is not None

def test_for_each():
    src = "for each item in items:\n  print item"
    prog = parse(src)
    stmt = prog.body[0]
    assert isinstance(stmt, ForEach)
    assert stmt.variable == "item"

def test_match_statement():
    src = 'match status:\n  200 -> print "ok"\n  404 -> print "not found"\n  _ -> print "error"'
    prog = parse(src)
    stmt = prog.body[0]
    assert isinstance(stmt, MatchStatement)
    assert len(stmt.arms) == 3
```

**Step 2: Run tests — new ones fail**

**Step 3: Implement control flow parsing**

- `parse_if()`: parse condition, expect COLON, parse indented body. Then check for `else if` or `else` at same indent level. `else if` is two tokens: ELSE then IF.
- `parse_for_each()`: expect FOR, EACH, IDENTIFIER, IN, expression, COLON, indented body.
- `parse_match()`: expect MATCH, expression, COLON, NEWLINE+INDENT. Each arm: pattern (expression or `_` wildcard), ARROW, statement(s). DEDENT ends.

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/parser.py tests/test_parser.py && git commit -m "feat: parse if/else, for each, match"
```

---

### Task 8: Parser — Functions and Return

**Files:**
- Modify: `drift/parser.py`
- Modify: `tests/test_parser.py`

**Step 1: Add tests**

```python
# append to tests/test_parser.py

def test_function_simple():
    src = "define greet(name: string):\n  print name"
    prog = parse(src)
    func = prog.body[0]
    assert isinstance(func, FunctionDef)
    assert func.name == "greet"
    assert len(func.params) == 1
    assert func.params[0] == ("name", "string")

def test_function_with_return_type():
    src = "define add(a: number, b: number) -> number:\n  return a + b"
    prog = parse(src)
    func = prog.body[0]
    assert func.return_type == "number"
    assert isinstance(func.body[0], ReturnStatement)

def test_function_multiple_params():
    src = "define f(x: string, y: number, z: boolean):\n  print x"
    prog = parse(src)
    func = prog.body[0]
    assert len(func.params) == 3

def test_return_expression():
    src = "define f():\n  return 42"
    prog = parse(src)
    ret = prog.body[0].body[0]
    assert isinstance(ret, ReturnStatement)
    assert isinstance(ret.value, NumberLiteral)
```

**Step 2: Run tests — new ones fail**

**Step 3: Implement function parsing**

- `parse_function_def()`: expect DEFINE, IDENTIFIER (name), LPAREN, param list, RPAREN, optional `-> type`, COLON, indented body.
- Param list: IDENTIFIER, COLON, type_name. Comma-separated.
- `parse_return()`: expect RETURN, parse expression.

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/parser.py tests/test_parser.py && git commit -m "feat: parse function definitions and return"
```

---

### Task 9: Parser — AI Primitives

**Files:**
- Modify: `drift/parser.py`
- Modify: `tests/test_parser.py`

**Step 1: Add tests**

```python
# append to tests/test_parser.py

def test_ai_ask_simple():
    src = 'x = ai.ask("What is 2+2?")'
    prog = parse(src)
    assert isinstance(prog.body[0].value, AIAsk)

def test_ai_ask_with_schema():
    src = 'x = ai.ask("Analyze this") -> DealScore'
    prog = parse(src)
    ask = prog.body[0].value
    assert isinstance(ask, AIAsk)
    assert ask.schema == "DealScore"

def test_ai_ask_with_using():
    src = 'x = ai.ask("Analyze") -> Deal using {\n  address: addr\n  price: 100\n}'
    prog = parse(src)
    ask = prog.body[0].value
    assert ask.schema == "Deal"
    assert ask.using is not None

def test_ai_classify():
    src = 'x = ai.classify(text, into: ["a", "b", "c"])'
    prog = parse(src)
    assert isinstance(prog.body[0].value, AIClassify)

def test_ai_embed():
    src = "x = ai.embed(doc.text)"
    prog = parse(src)
    assert isinstance(prog.body[0].value, AIEmbed)

def test_ai_see():
    src = 'x = ai.see(photo, "Describe this")'
    prog = parse(src)
    assert isinstance(prog.body[0].value, AISee)

def test_ai_predict():
    src = 'x = ai.predict("Estimate value") -> confident number'
    prog = parse(src)
    assert isinstance(prog.body[0].value, AIPredict)
```

**Step 2: Run tests — new ones fail**

**Step 3: Implement AI primitive parsing**

When the parser sees `ai` (IDENTIFIER with value "ai") followed by DOT:
- `ai.ask(prompt)` — parse prompt expression. Then optionally `-> SchemaName` and `using { ... }`.
- `ai.classify(input, into: [list])` — parse input, then `into:` keyword arg with list.
- `ai.embed(input)` — simple single-arg call.
- `ai.see(input, prompt)` — two args.
- `ai.predict(prompt)` — like ask, supports `-> confident type` or `-> type`.
- `ai.enrich(prompt)` — single arg (used in pipelines, parsed as expression).
- `ai.score(prompt)` — single arg, optional `-> type`.

These are handled in `parse_postfix()` or `parse_primary()` — when we see `ai.xxx(`, dispatch to the appropriate AI node constructor.

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/parser.py tests/test_parser.py && git commit -m "feat: parse AI primitives (ask, classify, embed, see, predict)"
```

---

### Task 10: Parser — Data Operations (fetch, read, save, query)

**Files:**
- Modify: `drift/parser.py`
- Modify: `tests/test_parser.py`

**Step 1: Add tests**

```python
# append to tests/test_parser.py

def test_fetch_simple():
    src = 'data = fetch "https://api.example.com/data"'
    prog = parse(src)
    assert isinstance(prog.body[0].value, FetchExpression)

def test_fetch_with_options():
    src = 'data = fetch "https://api.example.com" with {\n  headers: { "X-Key": "abc" }\n  params: { limit: 50 }\n}'
    prog = parse(src)
    fetch = prog.body[0].value
    assert isinstance(fetch, FetchExpression)
    assert fetch.options is not None

def test_read_file():
    src = 'data = read "deals.csv"'
    prog = parse(src)
    assert isinstance(prog.body[0].value, ReadExpression)

def test_save_statement():
    src = 'save data to "output.json"'
    prog = parse(src)
    assert isinstance(prog.body[0], SaveStatement)

def test_query_expression():
    src = 'records = query "SELECT * FROM users" on db.main'
    prog = parse(src)
    assert isinstance(prog.body[0].value, QueryExpression)

def test_merge_expression():
    src = "combined = merge [source_a, source_b]"
    prog = parse(src)
    assert isinstance(prog.body[0].value, MergeExpression)
```

**Step 2: Run tests — new ones fail**

**Step 3: Implement data operation parsing**

- `FETCH` → `parse_fetch()`: expect string/expression (url). If followed by `with`, parse `{ key: value }` map for options.
- `READ` → `parse_read()`: expect string/expression (path).
- `SAVE` → `parse_save()`: expect expression (data), `to`, expression (path).
- `QUERY` → `parse_query()`: expect string (SQL), `on`, expression (source).
- `MERGE` → `parse_merge()`: expect `[list of expressions]`.

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/parser.py tests/test_parser.py && git commit -m "feat: parse fetch, read, save, query, merge"
```

---

### Task 11: Parser — Pipelines

**Files:**
- Modify: `drift/parser.py`
- Modify: `tests/test_parser.py`

**Step 1: Add tests**

```python
# append to tests/test_parser.py

def test_pipeline_filter():
    src = 'results = data\n  |> filter where price < 500000'
    prog = parse(src)
    stmt = prog.body[0]
    assert isinstance(stmt.value, Pipeline)
    assert len(stmt.value.stages) == 1
    assert isinstance(stmt.value.stages[0], FilterStage)

def test_pipeline_sort():
    src = "results = data\n  |> sort by price ascending"
    prog = parse(src)
    pipe = prog.body[0].value
    assert isinstance(pipe.stages[0], SortStage)
    assert pipe.stages[0].direction == "ascending"

def test_pipeline_take():
    src = "results = data\n  |> take 10"
    prog = parse(src)
    pipe = prog.body[0].value
    assert isinstance(pipe.stages[0], TakeStage)

def test_pipeline_skip():
    src = "results = data\n  |> skip 5"
    prog = parse(src)
    pipe = prog.body[0].value
    assert isinstance(pipe.stages[0], SkipStage)

def test_pipeline_chain():
    src = 'results = data\n  |> filter where x > 5\n  |> sort by x descending\n  |> take 10'
    prog = parse(src)
    pipe = prog.body[0].value
    assert isinstance(pipe, Pipeline)
    assert len(pipe.stages) == 3

def test_pipeline_deduplicate():
    src = "results = data\n  |> deduplicate by address"
    prog = parse(src)
    pipe = prog.body[0].value
    assert isinstance(pipe.stages[0], DeduplicateStage)

def test_pipeline_group():
    src = "results = data\n  |> group by city"
    prog = parse(src)
    pipe = prog.body[0].value
    assert isinstance(pipe.stages[0], GroupStage)

def test_pipeline_ai_enrich():
    src = 'results = data\n  |> ai.enrich("Add summary")'
    prog = parse(src)
    pipe = prog.body[0].value
    assert isinstance(pipe.stages[0], AIEnrich)

def test_pipeline_ai_score():
    src = 'results = data\n  |> ai.score("Rate 1-100") -> number'
    prog = parse(src)
    pipe = prog.body[0].value
    assert isinstance(pipe.stages[0], AIScore)

def test_pipeline_each():
    src = 'results = data\n  |> each { |item|\n    print item\n  }'
    prog = parse(src)
    pipe = prog.body[0].value
    assert isinstance(pipe.stages[0], EachStage)
    assert pipe.stages[0].variable == "item"

def test_pipeline_save():
    src = 'data\n  |> filter where x > 5\n  |> save to "out.csv"'
    prog = parse(src)
    # This should be a pipeline that ends with a save
    # The save is the last stage
```

**Step 2: Run tests — new ones fail**

**Step 3: Implement pipeline parsing**

After parsing an expression, check if the next non-newline token is `|>`. If so, enter pipeline parsing mode:
- Collect `source` expression
- Loop: consume `|>`, then parse the pipeline stage:
  - `filter where <condition>` → FilterStage
  - `sort by <field> ascending|descending` → SortStage
  - `take <expr>` → TakeStage
  - `skip <expr>` → SkipStage
  - `group by <field>` → GroupStage
  - `deduplicate by <field>` → DeduplicateStage
  - `transform { |var| ... }` → TransformStage
  - `each { |var| ... }` → EachStage
  - `ai.enrich(...)` → AIEnrich
  - `ai.score(...)` → AIScore
  - `save to <path>` → SaveStatement (terminal)
- Continue until no more `|>` tokens

Key: `|>` may appear at the start of the next line (after NEWLINE), so peek past newlines to check.

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/parser.py tests/test_parser.py && git commit -m "feat: parse pipelines with all stage types"
```

---

### Task 12: Parser — Error Handling (try/catch)

**Files:**
- Modify: `drift/parser.py`
- Modify: `tests/test_parser.py`

**Step 1: Add tests**

```python
# append to tests/test_parser.py

def test_try_catch():
    src = 'try:\n  x = fetch "http://example.com"\ncatch network_error:\n  log "failed"'
    prog = parse(src)
    tc = prog.body[0]
    assert isinstance(tc, TryCatch)
    assert len(tc.catches) == 1
    assert tc.catches[0].error_type == "network_error"

def test_try_multiple_catches():
    src = 'try:\n  x = 1\ncatch network_error:\n  log "net"\ncatch ai_error:\n  log "ai"'
    prog = parse(src)
    tc = prog.body[0]
    assert len(tc.catches) == 2
```

**Step 2: Run tests — new ones fail**

**Step 3: Implement try/catch parsing**

- `parse_try()`: expect TRY, COLON, indented body. Then one or more `catch <error_type>:` blocks with indented bodies.

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/parser.py tests/test_parser.py && git commit -m "feat: parse try/catch"
```

---

### Task 13: Transpiler — Literals, Expressions, Assignments

**Files:**
- Create: `drift/transpiler.py`
- Create: `tests/test_transpiler.py`

**Step 1: Write failing tests**

```python
# tests/test_transpiler.py
import ast as python_ast
from drift.lexer import Lexer
from drift.parser import Parser
from drift.transpiler import Transpiler

def transpile(src: str) -> str:
    tokens = Lexer(src).tokenize()
    tree = Parser(tokens).parse()
    return Transpiler(tree).transpile()

def test_valid_python(src: str):
    """Helper: verify output is valid Python."""
    code = transpile(src)
    python_ast.parse(code)  # raises SyntaxError if invalid
    return code

def test_number_assignment():
    code = transpile("x = 42")
    assert "x = 42" in code

def test_string_assignment():
    code = transpile('name = "Drift"')
    assert 'name = "Drift"' in code

def test_string_interpolation():
    code = transpile('x = "Hello {name}!"')
    assert 'f"Hello {name}!"' in code

def test_boolean_assignment():
    code = transpile("x = true")
    assert "x = True" in code

def test_list_assignment():
    code = transpile("x = [1, 2, 3]")
    assert "[1, 2, 3]" in code

def test_arithmetic():
    code = transpile("x = 1 + 2 * 3")
    assert "1 + 2 * 3" in code or "1 + (2 * 3)" in code

def test_comparison():
    code = transpile("x = a > 5")
    assert "a > 5" in code

def test_print_simple():
    code = transpile('print "Hello, Drift!"')
    assert 'print("Hello, Drift!")' in code or "print(f\"Hello, Drift!\")" in code

def test_print_interpolated():
    code = transpile('print "Hello {name}!"')
    assert 'print(f"Hello {name}!")' in code

def test_log_statement():
    code = transpile('log "Processing"')
    # log maps to print or drift_runtime.log
    assert "Processing" in code

def test_output_has_import():
    code = transpile("x = 42")
    assert "import drift_runtime" in code

def test_output_is_valid_python():
    code = transpile('x = 42\nname = "hello"\nprint "test"')
    python_ast.parse(code)
```

**Step 2: Run tests — fail (no transpiler module)**

**Step 3: Implement transpiler.py**

```python
# drift/transpiler.py
"""Transpiles a Drift AST to Python source code."""
from drift.ast_nodes import *

class Transpiler:
    def __init__(self, program: Program):
        self.program = program
        self.indent = 0
        self.lines: list[str] = []

    def transpile(self) -> str:
        self.lines = ["import drift_runtime", ""]
        for stmt in self.program.body:
            self.emit_statement(stmt)
        return "\n".join(self.lines) + "\n"

    def emit_statement(self, node):
        prefix = "    " * self.indent
        # dispatch based on node type
        ...

    def emit_expression(self, node) -> str:
        # dispatch based on node type, return Python expression string
        ...
```

Key translations in `emit_expression`:
- `NumberLiteral` → `str(int(v))` if whole number, else `str(v)`
- `StringLiteral` → `f"..."` if has interpolations, else `"..."`
- `BooleanLiteral` → `True`/`False`
- `NoneLiteral` → `None`
- `ListLiteral` → `[el1, el2, ...]`
- `MapLiteral` → `{"key": val, ...}`
- `Identifier` → `name`
- `DotAccess` → `obj.field` (but `env.X` → `os.environ["X"]`)
- `BinaryOp` → `left op right`
- `UnaryOp` → `op operand`

Key translations in `emit_statement`:
- `Assignment` → `target = expr`
- `PrintStatement` → `print(expr)`
- `LogStatement` → `print(expr)` (or `drift_runtime.log(expr)`)

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_transpiler.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/transpiler.py tests/test_transpiler.py && git commit -m "feat: transpiler for literals, expressions, assignments, print"
```

---

### Task 14: Transpiler — Schemas, Functions, Control Flow

**Files:**
- Modify: `drift/transpiler.py`
- Modify: `tests/test_transpiler.py`

**Step 1: Add tests**

```python
# append to tests/test_transpiler.py

def test_schema_to_dataclass():
    src = "schema Deal:\n  address: string\n  price: number"
    code = transpile(src)
    assert "@dataclass" in code
    assert "class Deal:" in code
    assert "address: str" in code
    assert "price: float" in code
    python_ast.parse(code)

def test_function_def():
    src = "define greet(name: string):\n  print name"
    code = transpile(src)
    assert "def greet(name: str):" in code
    python_ast.parse(code)

def test_function_with_return_type():
    src = "define add(a: number, b: number) -> number:\n  return a + b"
    code = transpile(src)
    assert "def add(a: float, b: float) -> float:" in code
    assert "return" in code
    python_ast.parse(code)

def test_if_else():
    src = "if x > 5:\n  print \"big\"\nelse:\n  print \"small\""
    code = transpile(src)
    assert "if x > 5:" in code
    assert "else:" in code
    python_ast.parse(code)

def test_for_each():
    src = "for each item in items:\n  print item"
    code = transpile(src)
    assert "for item in items:" in code
    python_ast.parse(code)

def test_match_statement():
    src = 'match status:\n  200 -> print "ok"\n  _ -> print "error"'
    code = transpile(src)
    assert "match status:" in code
    python_ast.parse(code)

def test_try_catch():
    src = 'try:\n  x = 1\ncatch network_error:\n  log "fail"'
    code = transpile(src)
    assert "try:" in code
    assert "except" in code
    python_ast.parse(code)
```

**Step 2: Run tests — new ones fail**

**Step 3: Implement schema, function, control flow transpilation**

Type mapping: `string→str`, `number→float`, `boolean→bool`, `list→list`, `map→dict`, `date→str`, `none→None`, `list of X→list[X]`

- `SchemaDefinition` → `from dataclasses import dataclass` (add to imports if needed), `@dataclass\nclass Name:\n  field: type`
- `FunctionDef` → `def name(params) -> return_type:\n  body`
- `ReturnStatement` → `return expr`
- `IfStatement` → `if condition:\n  body\nelif:\n  body\nelse:\n  body`
- `ForEach` → `for var in iterable:\n  body`
- `MatchStatement` → `match subject:\n  case pattern:\n    body` (Python 3.10+)
- `TryCatch` → `try:\n  body\nexcept Exception as e:\n  body` (map `network_error`→`ConnectionError`, `ai_error`→`RuntimeError`)

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_transpiler.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/transpiler.py tests/test_transpiler.py && git commit -m "feat: transpile schemas, functions, control flow"
```

---

### Task 15: Transpiler — AI Primitives and Data Operations

**Files:**
- Modify: `drift/transpiler.py`
- Modify: `tests/test_transpiler.py`

**Step 1: Add tests**

```python
# append to tests/test_transpiler.py

def test_ai_ask():
    src = 'x = ai.ask("What is AI?")'
    code = transpile(src)
    assert "drift_runtime.ai.ask" in code
    python_ast.parse(code)

def test_ai_ask_with_schema():
    src = 'x = ai.ask("Analyze") -> Deal'
    code = transpile(src)
    assert "drift_runtime.ai.ask" in code
    assert "schema=Deal" in code or "Deal" in code
    python_ast.parse(code)

def test_ai_classify():
    src = 'x = ai.classify(text, into: ["a", "b"])'
    code = transpile(src)
    assert "drift_runtime.ai.classify" in code
    python_ast.parse(code)

def test_ai_embed():
    src = "x = ai.embed(doc)"
    code = transpile(src)
    assert "drift_runtime.ai.embed" in code
    python_ast.parse(code)

def test_fetch_simple():
    src = 'data = fetch "https://example.com/api"'
    code = transpile(src)
    assert "drift_runtime.fetch" in code
    python_ast.parse(code)

def test_fetch_with_options():
    src = 'data = fetch "https://example.com" with {\n  headers: { "Key": "val" }\n}'
    code = transpile(src)
    assert "drift_runtime.fetch" in code
    python_ast.parse(code)

def test_read_file():
    src = 'data = read "file.csv"'
    code = transpile(src)
    assert "drift_runtime.read" in code
    python_ast.parse(code)

def test_save_statement():
    src = 'save data to "output.json"'
    code = transpile(src)
    assert "drift_runtime.save" in code
    python_ast.parse(code)

def test_query_expression():
    src = 'records = query "SELECT * FROM users" on db.main'
    code = transpile(src)
    assert "drift_runtime.query" in code
    python_ast.parse(code)

def test_env_access():
    src = "key = env.API_KEY"
    code = transpile(src)
    assert 'os.environ["API_KEY"]' in code or "drift_runtime.env" in code
    python_ast.parse(code)
```

**Step 2: Run tests — new ones fail**

**Step 3: Implement AI and data operation transpilation**

- `AIAsk` → `drift_runtime.ai.ask(prompt, schema=Schema, context={...})`
- `AIClassify` → `drift_runtime.ai.classify(input, categories=[...])`
- `AIEmbed` → `drift_runtime.ai.embed(input)`
- `AISee` → `drift_runtime.ai.see(input, prompt)`
- `AIPredict` → `drift_runtime.ai.predict(prompt, schema=Schema)`
- `FetchExpression` → `drift_runtime.fetch(url, headers={...}, params={...})`
- `ReadExpression` → `drift_runtime.read(path)`
- `SaveStatement` → `drift_runtime.save(data, path)`
- `QueryExpression` → `drift_runtime.query(sql, source)`
- `DotAccess` on `env` → `os.environ["FIELD"]` (add `import os` to preamble)

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_transpiler.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/transpiler.py tests/test_transpiler.py && git commit -m "feat: transpile AI primitives and data operations"
```

---

### Task 16: Transpiler — Pipelines

**Files:**
- Modify: `drift/transpiler.py`
- Modify: `tests/test_transpiler.py`

**Step 1: Add tests**

```python
# append to tests/test_transpiler.py

def test_pipeline_filter():
    src = 'results = data\n  |> filter where price < 500000'
    code = transpile(src)
    assert "item" in code  # list comprehension
    assert "500000" in code
    python_ast.parse(code)

def test_pipeline_sort():
    src = "results = data\n  |> sort by price ascending"
    code = transpile(src)
    assert "sorted" in code
    python_ast.parse(code)

def test_pipeline_take():
    src = "results = data\n  |> take 10"
    code = transpile(src)
    assert "[:10]" in code
    python_ast.parse(code)

def test_pipeline_chain():
    src = 'results = data\n  |> filter where x > 5\n  |> sort by x descending\n  |> take 10'
    code = transpile(src)
    python_ast.parse(code)

def test_pipeline_ai_enrich():
    src = 'results = data\n  |> ai.enrich("Add summary")'
    code = transpile(src)
    assert "drift_runtime.ai.enrich" in code
    python_ast.parse(code)

def test_pipeline_deduplicate():
    src = "results = data\n  |> deduplicate by address"
    code = transpile(src)
    python_ast.parse(code)

def test_pipeline_each():
    src = 'results = data\n  |> each { |item|\n    print item\n  }'
    code = transpile(src)
    assert "for item in" in code
    python_ast.parse(code)
```

**Step 2: Run tests — new ones fail**

**Step 3: Implement pipeline transpilation**

Pipeline transpilation works by building up a variable through successive reassignments:

```python
# Drift: data |> filter where x > 5 |> sort by x |> take 10
# Python:
_pipe = data
_pipe = [item for item in _pipe if item["x"] > 5]
_pipe = sorted(_pipe, key=lambda item: item["x"])
_pipe = _pipe[:10]
results = _pipe
```

Stage translations:
- `FilterStage` → `[item for item in _pipe if <condition>]` (replace bare identifiers in condition with `item["field"]`)
- `SortStage` → `sorted(_pipe, key=lambda item: item["field"], reverse=<desc>)`
- `TakeStage` → `_pipe[:N]`
- `SkipStage` → `_pipe[N:]`
- `GroupStage` → helper using `itertools.groupby` or dict comprehension
- `DeduplicateStage` → `list({item["field"]: item for item in _pipe}.values())`
- `TransformStage` → `[<body>(item) for item in _pipe]`
- `EachStage` → `for item in _pipe:\n  <body>`
- `AIEnrich` → `drift_runtime.ai.enrich(_pipe, prompt)`
- `AIScore` → `drift_runtime.ai.score(_pipe, prompt)`

If the pipeline is assigned (`results = data |> ...`), the final value assigns to `results`. If standalone, use a temp var.

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_transpiler.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/transpiler.py tests/test_transpiler.py && git commit -m "feat: transpile pipelines"
```

---

### Task 17: CLI

**Files:**
- Create: `drift/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_cli.py
import subprocess
import sys
import os
import tempfile

DRIFT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_drift(args: list[str], input_text: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "drift.cli"] + args,
        capture_output=True, text=True, cwd=DRIFT_DIR,
    )

def test_drift_check_valid():
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False) as f:
        f.write('print "Hello, Drift!"')
        f.flush()
        result = run_drift(["check", f.name])
    os.unlink(f.name)
    assert result.returncode == 0

def test_drift_check_invalid():
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False) as f:
        f.write('}}}}invalid{{{{')
        f.flush()
        result = run_drift(["check", f.name])
    os.unlink(f.name)
    assert result.returncode != 0

def test_drift_build():
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False) as f:
        f.write('x = 42\nprint "hello"')
        f.flush()
        result = run_drift(["build", f.name])
        py_path = f.name.replace(".drift", ".py")
    os.unlink(f.name)
    assert result.returncode == 0
    assert os.path.exists(py_path)
    os.unlink(py_path)

def test_drift_no_args():
    result = run_drift([])
    assert result.returncode != 0 or "usage" in result.stdout.lower() or "usage" in result.stderr.lower()
```

**Step 2: Run tests — fail**

**Step 3: Implement cli.py**

```python
# drift/cli.py
"""Drift CLI — drift run, drift build, drift check."""
import sys
import os

from drift.lexer import Lexer
from drift.parser import Parser
from drift.transpiler import Transpiler
from drift.errors import DriftError

def main():
    if len(sys.argv) < 2:
        print("Usage: drift <command> [file.drift]", file=sys.stderr)
        print("Commands: run, build, check", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command in ("run", "build", "check"):
        if len(sys.argv) < 3:
            print(f"Usage: drift {command} <file.drift>", file=sys.stderr)
            sys.exit(1)
        filepath = sys.argv[2]
        if not os.path.exists(filepath):
            print(f"Error: file not found: {filepath}", file=sys.stderr)
            sys.exit(1)
        with open(filepath) as f:
            source = f.read()

        try:
            tokens = Lexer(source).tokenize()
            tree = Parser(tokens).parse()

            if command == "check":
                print(f"OK: {filepath}")
                sys.exit(0)

            python_code = Transpiler(tree).transpile()

            if command == "build":
                out_path = filepath.replace(".drift", ".py")
                with open(out_path, "w") as out:
                    out.write(python_code)
                print(f"Built: {out_path}")
                sys.exit(0)

            if command == "run":
                exec(python_code, {"__name__": "__main__"})

        except DriftError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Also create `drift/__main__.py`:
```python
from drift.cli import main
main()
```

**Step 4: Run tests**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_cli.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add drift/cli.py drift/__main__.py tests/test_cli.py && git commit -m "feat: CLI with run, build, check commands"
```

---

### Task 18: Example Programs and End-to-End Tests

**Files:**
- Create: `examples/hello.drift`
- Create: `examples/pipeline.drift`
- Create: `examples/deal_analyzer.drift`
- Create: `tests/test_end_to_end.py`

**Step 1: Create example .drift files**

`examples/hello.drift`:
```drift
-- Hello World in Drift
name = "Drift"
print "Hello from {name}!"
```

`examples/pipeline.drift`:
```drift
-- Pipeline example
schema Lead:
  name: string
  email: string
  score: number

data = fetch "https://crm.example.com/leads" with {
  headers: { "Authorization": "Bearer {env.CRM_TOKEN}" }
}

qualified = data
  |> filter where score > 60
  |> sort by score descending
  |> take 20
  |> save to "qualified.csv"

print "Done"
```

`examples/deal_analyzer.drift`:
```drift
-- Deal Analyzer
schema DealScore:
  address: string
  arv: number
  rehab_cost: number
  roi: number
  verdict: string

address = "742 Evergreen Terrace, Springfield"

comps = fetch "https://api.rentcast.io/v1/comps" with {
  headers: { "X-Api-Key": env.RENTCAST_KEY }
  params: { address: address, radius: 0.5 }
}

score = ai.ask("Analyze this investment property") -> DealScore using {
  address: address
  purchase_price: 285000
  comparable_sales: comps
}

if score.roi > 15:
  print "Hot deal: {score.verdict}"
  save score to "deals/result.json"
else:
  print "Pass: {score.verdict}"
```

**Step 2: Write end-to-end tests**

```python
# tests/test_end_to_end.py
import ast as python_ast
import os
import glob
from drift.lexer import Lexer
from drift.parser import Parser
from drift.transpiler import Transpiler

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")

def get_example_files():
    return glob.glob(os.path.join(EXAMPLES_DIR, "*.drift"))

def test_hello_world_end_to_end():
    with open(os.path.join(EXAMPLES_DIR, "hello.drift")) as f:
        source = f.read()
    tokens = Lexer(source).tokenize()
    tree = Parser(tokens).parse()
    python_code = Transpiler(tree).transpile()
    # Verify it's valid Python
    python_ast.parse(python_code)

def test_pipeline_end_to_end():
    with open(os.path.join(EXAMPLES_DIR, "pipeline.drift")) as f:
        source = f.read()
    tokens = Lexer(source).tokenize()
    tree = Parser(tokens).parse()
    python_code = Transpiler(tree).transpile()
    python_ast.parse(python_code)

def test_deal_analyzer_end_to_end():
    with open(os.path.join(EXAMPLES_DIR, "deal_analyzer.drift")) as f:
        source = f.read()
    tokens = Lexer(source).tokenize()
    tree = Parser(tokens).parse()
    python_code = Transpiler(tree).transpile()
    python_ast.parse(python_code)

def test_all_examples_produce_valid_python():
    """Every .drift file in examples/ must lex, parse, transpile to valid Python."""
    files = get_example_files()
    assert len(files) > 0, "No example files found"
    for filepath in files:
        with open(filepath) as f:
            source = f.read()
        tokens = Lexer(source).tokenize()
        tree = Parser(tokens).parse()
        python_code = Transpiler(tree).transpile()
        try:
            python_ast.parse(python_code)
        except SyntaxError as e:
            raise AssertionError(f"{filepath} produced invalid Python: {e}\n\n{python_code}")
```

**Step 3: Run tests — they should fail initially if transpiler doesn't handle all constructs**

Run: `cd /Users/ethansurfas/drift && python -m pytest tests/test_end_to_end.py -v`

**Step 4: Fix any issues until all pass**

These are integration tests. If any fail, fix the lexer/parser/transpiler as needed.

**Step 5: Run full test suite**

Run: `cd /Users/ethansurfas/drift && python -m pytest -v`
Expected: ALL tests pass

**Step 6: Commit**

```bash
git add examples/ tests/test_end_to_end.py && git commit -m "feat: example programs and end-to-end tests"
```

---

### Task 19: Install and Smoke Test

**Step 1: Install in dev mode**

```bash
cd /Users/ethansurfas/drift && pip install -e .
```

**Step 2: Smoke test CLI commands**

```bash
drift check examples/hello.drift
drift build examples/hello.drift
cat examples/hello.py  # verify output looks right
drift run examples/hello.drift  # should fail on drift_runtime import, that's expected (Phase 3)
```

**Step 3: Clean up build artifacts**

```bash
rm -f examples/hello.py
```

**Step 4: Final commit**

```bash
git add -A && git commit -m "chore: pyproject.toml installable, Phase 1 complete"
```
