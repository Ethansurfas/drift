"""Tests for Drift transpiler — literals, expressions, assignments."""

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
