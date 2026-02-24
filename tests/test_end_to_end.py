"""End-to-end tests for the Drift pipeline: lex -> parse -> transpile -> valid Python."""

import ast as python_ast
import os
import glob
from drift.lexer import Lexer
from drift.parser import Parser
from drift.transpiler import Transpiler

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def compile_drift(source: str) -> str:
    """Full pipeline: source -> tokens -> AST -> Python."""
    tokens = Lexer(source).tokenize()
    tree = Parser(tokens).parse()
    return Transpiler(tree).transpile()


def test_hello_world_e2e():
    with open(os.path.join(EXAMPLES_DIR, "hello.drift")) as f:
        source = f.read()
    python_code = compile_drift(source)
    python_ast.parse(python_code)  # Must be valid Python
    assert "Hello from" in python_code
    assert 'f"' in python_code  # String interpolation used


def test_pipeline_e2e():
    with open(os.path.join(EXAMPLES_DIR, "pipeline.drift")) as f:
        source = f.read()
    python_code = compile_drift(source)
    python_ast.parse(python_code)
    assert "drift_runtime" in python_code
    assert "sorted" in python_code or "sort" in python_code
    assert "@dataclass" in python_code


def test_deal_analyzer_e2e():
    with open(os.path.join(EXAMPLES_DIR, "deal_analyzer.drift")) as f:
        source = f.read()
    python_code = compile_drift(source)
    python_ast.parse(python_code)
    assert "drift_runtime.ai.ask" in python_code
    assert "drift_runtime.read" in python_code
    assert "@dataclass" in python_code
    assert "class DealScore" in python_code


def test_all_examples_produce_valid_python():
    """Every .drift file in examples/ must compile to valid Python."""
    files = glob.glob(os.path.join(EXAMPLES_DIR, "*.drift"))
    assert len(files) >= 3, f"Expected at least 3 example files, found {len(files)}"
    for filepath in files:
        with open(filepath) as f:
            source = f.read()
        python_code = compile_drift(source)
        try:
            python_ast.parse(python_code)
        except SyntaxError as e:
            raise AssertionError(
                f"{filepath} produced invalid Python:\n{e}\n\nGenerated:\n{python_code}"
            )


def test_hello_world_output_structure():
    """Verify the hello world output has the expected structure."""
    code = compile_drift('name = "Drift"\nprint "Hello from {name}!"')
    python_ast.parse(code)
    assert "import drift_runtime" in code
    assert 'name = "Drift"' in code
    assert "print(" in code
