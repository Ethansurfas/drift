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
    SchemaDefinition,
    FunctionDef,
    ReturnStatement,
    IfStatement,
    ForEach,
    MatchStatement,
    TryCatch,
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
        if isinstance(node, SchemaDefinition):
            return self._emit_schema(node)
        if isinstance(node, FunctionDef):
            return self._emit_function(node)
        if isinstance(node, ReturnStatement):
            return self._emit_return(node)
        if isinstance(node, IfStatement):
            return self._emit_if(node)
        if isinstance(node, ForEach):
            return self._emit_for_each(node)
        if isinstance(node, MatchStatement):
            return self._emit_match(node)
        if isinstance(node, TryCatch):
            return self._emit_try_catch(node)

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

    # -- Type mapping --------------------------------------------------------

    _TYPE_MAP = {
        "string": "str",
        "number": "float",
        "boolean": "bool",
        "list": "list",
        "map": "dict",
        "date": "str",
        "none": "None",
    }

    _ERROR_TYPE_MAP = {
        "network_error": "ConnectionError",
        "ai_error": "RuntimeError",
    }

    def _map_type(self, drift_type: str) -> str:
        """Map a Drift type annotation to a Python type string."""
        # Handle compound "list of <element_type>"
        if drift_type.startswith("list of "):
            element = drift_type[len("list of "):]
            inner = self._TYPE_MAP.get(element, element)
            return f"list[{inner}]"
        return self._TYPE_MAP.get(drift_type, drift_type)

    def _map_error_type(self, drift_error: str) -> str:
        """Map a Drift error type to a Python exception class name."""
        return self._ERROR_TYPE_MAP.get(drift_error, "Exception")

    # -- Schema emission -----------------------------------------------------

    def _emit_schema(self, node: SchemaDefinition) -> list[str]:
        """Emit a ``@dataclass`` class from a Drift schema."""
        self.needs_dataclass_import = True
        lines: list[str] = []
        lines.append(f"{self._indent()}@dataclass")
        lines.append(f"{self._indent()}class {node.name}:")
        self.indent_level += 1
        for fld in node.fields:
            py_type = self._map_type(fld.type_name)
            if fld.optional:
                lines.append(f"{self._indent()}{fld.name}: {py_type} = None")
            else:
                lines.append(f"{self._indent()}{fld.name}: {py_type}")
        self.indent_level -= 1
        return lines

    # -- Function emission ---------------------------------------------------

    def _emit_function(self, node: FunctionDef) -> list[str]:
        """Emit a Python ``def`` from a Drift ``define``."""
        # Build parameter list
        param_parts: list[str] = []
        for pname, ptype in node.params:
            if ptype:
                param_parts.append(f"{pname}: {self._map_type(ptype)}")
            else:
                param_parts.append(pname)
        params_str = ", ".join(param_parts)

        # Build return type annotation
        ret = ""
        if node.return_type:
            ret = f" -> {self._map_type(node.return_type)}"

        lines: list[str] = []
        lines.append(f"{self._indent()}def {node.name}({params_str}){ret}:")
        self.indent_level += 1
        for stmt in node.body:
            lines.extend(self._emit_statement(stmt))
        self.indent_level -= 1
        return lines

    # -- Return emission -----------------------------------------------------

    def _emit_return(self, node: ReturnStatement) -> list[str]:
        """Emit ``return <expr>``."""
        value = self._emit_expr(node.value)
        return [f"{self._indent()}return {value}"]

    # -- If / Else If / Else emission ----------------------------------------

    def _emit_if(self, node: IfStatement) -> list[str]:
        """Emit ``if / elif / else`` chain."""
        lines: list[str] = []
        cond = self._emit_expr(node.condition)
        lines.append(f"{self._indent()}if {cond}:")
        self.indent_level += 1
        for stmt in node.body:
            lines.extend(self._emit_statement(stmt))
        self.indent_level -= 1

        for ei_cond, ei_body in node.elseifs:
            ei_cond_str = self._emit_expr(ei_cond)
            lines.append(f"{self._indent()}elif {ei_cond_str}:")
            self.indent_level += 1
            for stmt in ei_body:
                lines.extend(self._emit_statement(stmt))
            self.indent_level -= 1

        if node.else_body:
            lines.append(f"{self._indent()}else:")
            self.indent_level += 1
            for stmt in node.else_body:
                lines.extend(self._emit_statement(stmt))
            self.indent_level -= 1

        return lines

    # -- For Each emission ---------------------------------------------------

    def _emit_for_each(self, node: ForEach) -> list[str]:
        """Emit ``for <var> in <iterable>:``."""
        iterable = self._emit_expr(node.iterable)
        lines: list[str] = []
        lines.append(f"{self._indent()}for {node.variable} in {iterable}:")
        self.indent_level += 1
        for stmt in node.body:
            lines.extend(self._emit_statement(stmt))
        self.indent_level -= 1
        return lines

    # -- Match emission ------------------------------------------------------

    def _emit_match(self, node: MatchStatement) -> list[str]:
        """Emit Python 3.10+ ``match / case`` from Drift ``match``."""
        subject = self._emit_expr(node.subject)
        lines: list[str] = []
        lines.append(f"{self._indent()}match {subject}:")
        self.indent_level += 1
        for arm in node.arms:
            pattern = self._emit_expr(arm.pattern)
            lines.append(f"{self._indent()}case {pattern}:")
            self.indent_level += 1
            # arm.body is a single statement, not a list
            lines.extend(self._emit_statement(arm.body))
            self.indent_level -= 1
        self.indent_level -= 1
        return lines

    # -- Try / Catch emission ------------------------------------------------

    def _emit_try_catch(self, node: TryCatch) -> list[str]:
        """Emit ``try / except`` from Drift ``try / catch``."""
        lines: list[str] = []
        lines.append(f"{self._indent()}try:")
        self.indent_level += 1
        for stmt in node.try_body:
            lines.extend(self._emit_statement(stmt))
        self.indent_level -= 1

        for catch in node.catches:
            py_exc = self._map_error_type(catch.error_type)
            lines.append(f"{self._indent()}except {py_exc}:")
            self.indent_level += 1
            for stmt in catch.body:
                lines.extend(self._emit_statement(stmt))
            self.indent_level -= 1

        return lines
