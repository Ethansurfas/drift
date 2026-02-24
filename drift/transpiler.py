"""Drift transpiler — walks the AST and emits Python source code."""

from __future__ import annotations

from drift.ast_nodes import (
    Program,
    NumberLiteral,
    StringLiteral,
    BooleanLiteral,
    NoneLiteral,
    ListLiteral,
    MapLiteral,
    Identifier,
    DotAccess,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    Assignment,
    PrintStatement,
    LogStatement,
)
from drift.errors import TranspileError


class Transpiler:
    """Transpile a Drift AST into Python source code."""

    def __init__(self, program: Program) -> None:
        self.program = program
        self.indent_level: int = 0
        self.lines: list[str] = []
        self.needs_dataclass_import: bool = False
        self.needs_os_import: bool = False

    def transpile(self) -> str:
        """Transpile the entire program to Python source."""
        # First pass: emit body lines (this also sets import flags)
        body_lines: list[str] = []
        for stmt in self.program.body:
            body_lines.extend(self._emit_statement(stmt))

        # Build header
        header = ["import drift_runtime"]
        if self.needs_os_import:
            header.append("import os")
        if self.needs_dataclass_import:
            header.append("from dataclasses import dataclass")
        header.append("")  # blank line after imports

        return "\n".join(header + body_lines) + "\n"

    def _indent(self) -> str:
        """Return the current indentation string."""
        return "    " * self.indent_level

    # -- Statement emission ------------------------------------------------

    def _emit_statement(self, node) -> list[str]:
        """Emit a statement, returning a list of lines."""
        if isinstance(node, Assignment):
            return self._emit_assignment(node)
        if isinstance(node, PrintStatement):
            return self._emit_print(node)
        if isinstance(node, LogStatement):
            return self._emit_log(node)

        raise TranspileError(
            f"Unsupported statement type: {type(node).__name__}",
            getattr(node, "line", 0),
            getattr(node, "col", 0),
        )

    def _emit_assignment(self, node: Assignment) -> list[str]:
        """Emit ``target = value``."""
        value = self._emit_expr(node.value)
        return [f"{self._indent()}{node.target} = {value}"]

    def _emit_print(self, node: PrintStatement) -> list[str]:
        """Emit ``print(value)``."""
        value = self._emit_expr(node.value)
        return [f"{self._indent()}print({value})"]

    def _emit_log(self, node: LogStatement) -> list[str]:
        """Emit ``print(value)`` (log maps to print)."""
        value = self._emit_expr(node.value)
        return [f"{self._indent()}print({value})"]

    # -- Expression emission -----------------------------------------------

    def _emit_expr(self, node) -> str:
        """Emit an expression, returning a Python expression string."""
        if isinstance(node, NumberLiteral):
            return self._emit_number(node)
        if isinstance(node, StringLiteral):
            return self._emit_string(node)
        if isinstance(node, BooleanLiteral):
            return "True" if node.value else "False"
        if isinstance(node, NoneLiteral):
            return "None"
        if isinstance(node, Identifier):
            return node.name
        if isinstance(node, ListLiteral):
            return self._emit_list(node)
        if isinstance(node, MapLiteral):
            return self._emit_map(node)
        if isinstance(node, DotAccess):
            return self._emit_dot_access(node)
        if isinstance(node, BinaryOp):
            return self._emit_binary_op(node)
        if isinstance(node, UnaryOp):
            return self._emit_unary_op(node)
        if isinstance(node, FunctionCall):
            return self._emit_function_call(node)

        raise TranspileError(
            f"Unsupported expression type: {type(node).__name__}",
            getattr(node, "line", 0),
            getattr(node, "col", 0),
        )

    def _emit_number(self, node: NumberLiteral) -> str:
        """Emit a number: use int form for whole numbers, float otherwise."""
        if node.value == int(node.value):
            return str(int(node.value))
        return str(node.value)

    def _emit_string(self, node: StringLiteral) -> str:
        """Emit a string, using f-string syntax when interpolation is present."""
        has_interpolation = any(not isinstance(p, str) for p in node.parts)
        if has_interpolation:
            result = 'f"'
            for part in node.parts:
                if isinstance(part, str):
                    result += part
                else:
                    result += "{" + self._emit_expr(part) + "}"
            result += '"'
            return result
        else:
            return f'"{node.value}"'

    def _emit_list(self, node: ListLiteral) -> str:
        """Emit ``[el1, el2, ...]``."""
        elements = ", ".join(self._emit_expr(el) for el in node.elements)
        return f"[{elements}]"

    def _emit_map(self, node: MapLiteral) -> str:
        """Emit ``{"key": value, ...}``."""
        pairs = ", ".join(
            f'"{key}": {self._emit_expr(value)}'
            for key, value in node.pairs
        )
        return "{" + pairs + "}"

    def _emit_dot_access(self, node: DotAccess) -> str:
        """Emit ``obj.field``, or ``os.environ["FIELD"]`` for env access."""
        if isinstance(node.object, Identifier) and node.object.name == "env":
            self.needs_os_import = True
            return f'os.environ["{node.field_name}"]'
        obj = self._emit_expr(node.object)
        return f"{obj}.{node.field_name}"

    def _emit_binary_op(self, node: BinaryOp) -> str:
        """Emit ``(left op right)``."""
        left = self._emit_expr(node.left)
        right = self._emit_expr(node.right)
        return f"({left} {node.op} {right})"

    def _emit_unary_op(self, node: UnaryOp) -> str:
        """Emit ``(-x)`` or ``(not x)``."""
        operand = self._emit_expr(node.operand)
        if node.op == "not":
            return f"(not {operand})"
        return f"({node.op}{operand})"

    def _emit_function_call(self, node: FunctionCall) -> str:
        """Emit ``callee(arg1, arg2, key=val)``."""
        callee = self._emit_expr(node.callee)
        parts: list[str] = []
        for arg in node.args:
            parts.append(self._emit_expr(arg))
        for key, val in node.kwargs.items():
            parts.append(f"{key}={self._emit_expr(val)}")
        return f"{callee}({', '.join(parts)})"
