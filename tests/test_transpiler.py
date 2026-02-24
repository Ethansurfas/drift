"""Tests for Drift transpiler — literals, expressions, assignments, schemas, functions, control flow."""

import ast as python_ast

from drift.lexer import Lexer
from drift.parser import Parser
from drift.transpiler import Transpiler


def transpile(src: str) -> str:
    tokens = Lexer(src).tokenize()
    tree = Parser(tokens).parse()
    return Transpiler(tree).transpile()


def test_number_assignment():
    code = transpile("x = 42")
    assert "x = 42" in code
    python_ast.parse(code)


def test_float_assignment():
    code = transpile("x = 3.14")
    assert "x = 3.14" in code
    python_ast.parse(code)


def test_string_assignment():
    code = transpile('name = "Drift"')
    assert 'name = "Drift"' in code
    python_ast.parse(code)


def test_string_interpolation():
    code = transpile('x = "Hello {name}!"')
    assert 'f"Hello {name}!"' in code
    python_ast.parse(code)


def test_boolean_true():
    code = transpile("x = true")
    assert "x = True" in code
    python_ast.parse(code)


def test_boolean_false():
    code = transpile("x = false")
    assert "x = False" in code
    python_ast.parse(code)


def test_none_value():
    code = transpile("x = none")
    assert "x = None" in code
    python_ast.parse(code)


def test_list_assignment():
    code = transpile('x = [1, 2, 3]')
    assert "[1, 2, 3]" in code
    python_ast.parse(code)


def test_map_assignment():
    code = transpile('x = { name: "test", value: 42 }')
    assert '"name"' in code
    assert '"test"' in code
    python_ast.parse(code)


def test_arithmetic():
    code = transpile("x = 1 + 2 * 3")
    python_ast.parse(code)


def test_comparison():
    code = transpile("x = a > 5")
    python_ast.parse(code)


def test_and_or():
    code = transpile("x = a and b or c")
    python_ast.parse(code)


def test_print_simple():
    code = transpile('print "Hello, Drift!"')
    assert 'print("Hello, Drift!")' in code or 'print(f"Hello, Drift!")' in code
    python_ast.parse(code)


def test_print_interpolated():
    code = transpile('print "Hello {name}!"')
    assert 'print(f"Hello {name}!")' in code
    python_ast.parse(code)


def test_log_statement():
    code = transpile('log "Processing"')
    assert "Processing" in code
    python_ast.parse(code)


def test_output_has_import():
    code = transpile("x = 42")
    assert "import drift_runtime" in code


def test_env_access():
    code = transpile("key = env.API_KEY")
    assert 'os.environ["API_KEY"]' in code
    assert "import os" in code
    python_ast.parse(code)


def test_dot_access():
    code = transpile("x = obj.field")
    assert "obj.field" in code
    python_ast.parse(code)


def test_unary_minus():
    code = transpile("x = -5")
    python_ast.parse(code)


def test_unary_not():
    code = transpile("x = not y")
    python_ast.parse(code)


def test_multiple_statements():
    code = transpile('x = 42\nname = "hello"\nprint "test"')
    python_ast.parse(code)


def test_output_is_valid_python():
    code = transpile('x = 42\nname = "hello"\nprint "test"')
    python_ast.parse(code)  # This is the key validation


# ── Schema → @dataclass ─────────────────────────────────────────────────────


def test_schema_to_dataclass():
    src = "schema Deal:\n  address: string\n  price: number"
    code = transpile(src)
    assert "@dataclass" in code
    assert "class Deal:" in code
    assert "address: str" in code
    assert "price: float" in code
    assert "from dataclasses import dataclass" in code
    python_ast.parse(code)


def test_schema_optional_field():
    src = "schema X:\n  data: map (optional)"
    code = transpile(src)
    assert "data: dict = None" in code
    python_ast.parse(code)


def test_schema_list_of_type():
    src = "schema X:\n  items: list of string"
    code = transpile(src)
    assert "list[str]" in code
    python_ast.parse(code)


# ── Function Definition ─────────────────────────────────────────────────────


def test_function_def():
    src = 'define greet(name: string):\n  print name'
    code = transpile(src)
    assert "def greet(name: str):" in code
    python_ast.parse(code)


def test_function_with_return_type():
    src = "define add(a: number, b: number) -> number:\n  return a + b"
    code = transpile(src)
    assert "def add(a: float, b: float) -> float:" in code
    assert "return" in code
    python_ast.parse(code)


def test_function_no_params():
    src = 'define hello():\n  print "hi"'
    code = transpile(src)
    assert "def hello():" in code
    python_ast.parse(code)


# ── Return Statement ────────────────────────────────────────────────────────


def test_return_expression():
    src = "define f():\n  return 42"
    code = transpile(src)
    assert "return 42" in code
    python_ast.parse(code)


# ── If / Else If / Else ─────────────────────────────────────────────────────


def test_if_simple():
    src = 'if x > 5:\n  print "big"'
    code = transpile(src)
    assert "if " in code
    python_ast.parse(code)


def test_if_else():
    src = 'if x > 5:\n  print "big"\nelse:\n  print "small"'
    code = transpile(src)
    assert "if " in code
    assert "else:" in code
    python_ast.parse(code)


def test_if_elseif_else():
    src = 'if x > 10:\n  print "a"\nelse if x > 5:\n  print "b"\nelse:\n  print "c"'
    code = transpile(src)
    assert "elif " in code
    python_ast.parse(code)


# ── For Each ────────────────────────────────────────────────────────────────


def test_for_each():
    src = "for each item in items:\n  print item"
    code = transpile(src)
    assert "for item in items:" in code
    python_ast.parse(code)


# ── Match Statement ─────────────────────────────────────────────────────────


def test_match_statement():
    src = 'match status:\n  200 -> print "ok"\n  _ -> print "error"'
    code = transpile(src)
    assert "match " in code
    assert "case " in code
    python_ast.parse(code)


# ── Try / Catch ─────────────────────────────────────────────────────────────


def test_try_catch():
    src = 'try:\n  x = 1\ncatch network_error:\n  log "fail"'
    code = transpile(src)
    assert "try:" in code
    assert "except " in code
    python_ast.parse(code)
