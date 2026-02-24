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


# ── AI Primitives ───────────────────────────────────────────────────────────


def test_ai_ask_simple():
    src = 'x = ai.ask("What is AI?")'
    code = transpile(src)
    assert "drift_runtime.ai.ask" in code
    python_ast.parse(code)


def test_ai_ask_with_schema():
    src = 'x = ai.ask("Analyze") -> DealScore'
    code = transpile(src)
    assert "drift_runtime.ai.ask" in code
    assert "schema=DealScore" in code
    python_ast.parse(code)


def test_ai_ask_with_using():
    src = 'x = ai.ask("Analyze") -> Deal using {\n  address: addr\n  price: 100\n}'
    code = transpile(src)
    assert "drift_runtime.ai.ask" in code
    assert "context=" in code
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


def test_ai_see():
    src = 'x = ai.see(photo, "Describe this")'
    code = transpile(src)
    assert "drift_runtime.ai.see" in code
    python_ast.parse(code)


def test_ai_predict():
    src = 'x = ai.predict("Estimate value") -> confident number'
    code = transpile(src)
    assert "drift_runtime.ai.predict" in code
    python_ast.parse(code)


def test_ai_enrich():
    src = 'x = ai.enrich("Add summary")'
    code = transpile(src)
    assert "drift_runtime.ai.enrich" in code
    python_ast.parse(code)


def test_ai_score():
    src = 'x = ai.score("Rate 1-100") -> number'
    code = transpile(src)
    assert "drift_runtime.ai.score" in code
    python_ast.parse(code)


# ── Data Operations ─────────────────────────────────────────────────────────


def test_fetch_simple():
    src = 'data = fetch "https://example.com/api"'
    code = transpile(src)
    assert "drift_runtime.fetch" in code
    python_ast.parse(code)


def test_fetch_with_options():
    src = 'data = fetch "https://example.com" with {\n  headers: { "Key": "val" }\n}'
    code = transpile(src)
    assert "drift_runtime.fetch" in code
    assert "headers=" in code
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


def test_merge_expression():
    src = 'combined = merge [a, b, c]'
    code = transpile(src)
    assert "drift_runtime.merge" in code
    python_ast.parse(code)


# ── Pipelines ───────────────────────────────────────────────────────────────


def test_pipeline_filter():
    src = 'results = data\n  |> filter where price < 500000'
    code = transpile(src)
    assert "_pipe" in code or "results" in code
    assert "500000" in code
    python_ast.parse(code)


def test_pipeline_sort_ascending():
    src = "results = data\n  |> sort by price ascending"
    code = transpile(src)
    assert "sorted" in code
    python_ast.parse(code)


def test_pipeline_sort_descending():
    src = "results = data\n  |> sort by score descending"
    code = transpile(src)
    assert "reverse=True" in code
    python_ast.parse(code)


def test_pipeline_take():
    src = "results = data\n  |> take 10"
    code = transpile(src)
    assert "[:10]" in code
    python_ast.parse(code)


def test_pipeline_skip():
    src = "results = data\n  |> skip 5"
    code = transpile(src)
    assert "[5:]" in code
    python_ast.parse(code)


def test_pipeline_chain():
    src = 'results = data\n  |> filter where x > 5\n  |> sort by x descending\n  |> take 10'
    code = transpile(src)
    python_ast.parse(code)
    assert "results = _pipe" in code or "results" in code


def test_pipeline_deduplicate():
    src = "results = data\n  |> deduplicate by address"
    code = transpile(src)
    python_ast.parse(code)


def test_pipeline_group():
    src = "results = data\n  |> group by city"
    code = transpile(src)
    python_ast.parse(code)


def test_pipeline_ai_enrich():
    src = 'results = data\n  |> ai.enrich("Add summary")'
    code = transpile(src)
    assert "drift_runtime.ai.enrich" in code
    python_ast.parse(code)


def test_pipeline_ai_score():
    src = 'results = data\n  |> ai.score("Rate 1-100") -> number'
    code = transpile(src)
    assert "drift_runtime.ai.score" in code
    python_ast.parse(code)


def test_pipeline_each():
    src = 'results = data\n  |> each { |item|\n    print item\n  }'
    code = transpile(src)
    assert "for item in" in code
    python_ast.parse(code)


def test_pipeline_save():
    src = 'results = data\n  |> filter where x > 5\n  |> save to "out.csv"'
    code = transpile(src)
    assert "drift_runtime.save" in code
    python_ast.parse(code)


def test_pipeline_filter_with_and():
    src = 'results = data\n  |> filter where price < 500000 and beds >= 3'
    code = transpile(src)
    python_ast.parse(code)
