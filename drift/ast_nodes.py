"""Drift AST node definitions.

Every node is a Python dataclass carrying ``line`` and ``col`` for
source-location tracking.  A single ``Node`` base class provides
these fields so concrete nodes only declare domain-specific data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Base ────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    """Base class for every AST node."""
    line: int = 0
    col: int = 0


# ── Program ─────────────────────────────────────────────────────────────────

@dataclass
class Program(Node):
    body: list = field(default_factory=list)


# ── Expressions ─────────────────────────────────────────────────────────────

@dataclass
class NumberLiteral(Node):
    value: float = 0.0


@dataclass
class StringLiteral(Node):
    value: str = ""
    parts: list = field(default_factory=list)


@dataclass
class BooleanLiteral(Node):
    value: bool = False


@dataclass
class NoneLiteral(Node):
    pass


@dataclass
class ListLiteral(Node):
    elements: list = field(default_factory=list)


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
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


# ── AI Primitives ───────────────────────────────────────────────────────────

@dataclass
class AIAsk(Node):
    prompt: Any = None
    schema: str | None = None
    using: dict | None = None


@dataclass
class AIClassify(Node):
    input: Any = None
    categories: list = field(default_factory=list)


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


# ── Statements ──────────────────────────────────────────────────────────────

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
    body: list = field(default_factory=list)
    elseifs: list[tuple] = field(default_factory=list)
    else_body: list | None = None


@dataclass
class ForEach(Node):
    variable: str = ""
    iterable: Any = None
    body: list = field(default_factory=list)


@dataclass
class MatchArm(Node):
    pattern: Any = None
    body: Any = None


@dataclass
class MatchStatement(Node):
    subject: Any = None
    arms: list[MatchArm] = field(default_factory=list)


@dataclass
class FunctionDef(Node):
    name: str = ""
    params: list[tuple[str, str | None]] = field(default_factory=list)
    return_type: str | None = None
    body: list = field(default_factory=list)


@dataclass
class SchemaField(Node):
    name: str = ""
    type_name: str = ""
    optional: bool = False


@dataclass
class SchemaDefinition(Node):
    name: str = ""
    fields: list[SchemaField] = field(default_factory=list)


# ── Data Operations ─────────────────────────────────────────────────────────

@dataclass
class FetchExpression(Node):
    url: Any = None
    options: dict | None = None


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
    sources: list = field(default_factory=list)


# ── Pipeline Stages ─────────────────────────────────────────────────────────

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
    body: list = field(default_factory=list)


@dataclass
class EachStage(Node):
    variable: str = ""
    body: list = field(default_factory=list)


@dataclass
class Pipeline(Node):
    source: Any = None
    stages: list = field(default_factory=list)


# ── Error Handling ──────────────────────────────────────────────────────────

@dataclass
class CatchClause(Node):
    error_type: str = ""
    body: list = field(default_factory=list)


@dataclass
class TryCatch(Node):
    try_body: list = field(default_factory=list)
    catches: list[CatchClause] = field(default_factory=list)
