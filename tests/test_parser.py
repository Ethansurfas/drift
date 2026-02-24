"""Tests for Drift parser — expression parsing and basic assignments."""

import pytest

from drift.lexer import Lexer
from drift.parser import Parser
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
    SchemaField,
    IfStatement,
    ForEach,
    MatchStatement,
    MatchArm,
    FunctionDef,
    ReturnStatement,
    AIAsk,
    AIClassify,
    AIEmbed,
    AISee,
    AIPredict,
    AIEnrich,
    AIScore,
    FetchExpression,
    ReadExpression,
    SaveStatement,
    QueryExpression,
    MergeExpression,
    Pipeline,
    FilterStage,
    SortStage,
    TakeStage,
    SkipStage,
    GroupStage,
    DeduplicateStage,
    TransformStage,
    EachStage,
    TryCatch,
    CatchClause,
)
from drift.errors import ParseError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse(src: str) -> Program:
    """Convenience: lex + parse source and return the Program AST."""
    tokens = Lexer(src).tokenize()
    return Parser(tokens).parse()


def expr(src: str):
    """Parse a single expression statement and return the expression node."""
    program = parse(src)
    assert len(program.body) == 1
    return program.body[0]


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------

class TestLiterals:
    def test_number_integer(self):
        node = expr("42")
        assert isinstance(node, NumberLiteral)
        assert node.value == 42.0

    def test_number_float(self):
        node = expr("3.14")
        assert isinstance(node, NumberLiteral)
        assert node.value == 3.14

    def test_string_plain(self):
        node = expr('"hello"')
        assert isinstance(node, StringLiteral)
        assert node.value == "hello"
        assert node.parts == ["hello"]

    def test_boolean_true(self):
        node = expr("true")
        assert isinstance(node, BooleanLiteral)
        assert node.value is True

    def test_boolean_false(self):
        node = expr("false")
        assert isinstance(node, BooleanLiteral)
        assert node.value is False

    def test_none_literal(self):
        node = expr("none")
        assert isinstance(node, NoneLiteral)

    def test_list_literal(self):
        node = expr("[1, 2, 3]")
        assert isinstance(node, ListLiteral)
        assert len(node.elements) == 3
        assert all(isinstance(e, NumberLiteral) for e in node.elements)
        assert node.elements[0].value == 1.0
        assert node.elements[1].value == 2.0
        assert node.elements[2].value == 3.0

    def test_list_literal_empty(self):
        node = expr("[]")
        assert isinstance(node, ListLiteral)
        assert node.elements == []

    def test_map_literal(self):
        node = expr('{ name: "test", value: 42 }')
        assert isinstance(node, MapLiteral)
        assert len(node.pairs) == 2
        assert node.pairs[0][0] == "name"
        assert isinstance(node.pairs[0][1], StringLiteral)
        assert node.pairs[0][1].value == "test"
        assert node.pairs[1][0] == "value"
        assert isinstance(node.pairs[1][1], NumberLiteral)
        assert node.pairs[1][1].value == 42.0

    def test_map_literal_empty(self):
        node = expr("{}")
        assert isinstance(node, MapLiteral)
        assert node.pairs == []


# ---------------------------------------------------------------------------
# Identifiers and Dot Access
# ---------------------------------------------------------------------------

class TestIdentifiersAndAccess:
    def test_identifier(self):
        node = expr("foo")
        assert isinstance(node, Identifier)
        assert node.name == "foo"

    def test_dot_access(self):
        node = expr("obj.field")
        assert isinstance(node, DotAccess)
        assert isinstance(node.object, Identifier)
        assert node.object.name == "obj"
        assert node.field_name == "field"

    def test_chained_dot_access(self):
        node = expr("a.b.c")
        assert isinstance(node, DotAccess)
        assert node.field_name == "c"
        inner = node.object
        assert isinstance(inner, DotAccess)
        assert inner.field_name == "b"
        assert isinstance(inner.object, Identifier)
        assert inner.object.name == "a"


# ---------------------------------------------------------------------------
# Binary Operations
# ---------------------------------------------------------------------------

class TestBinaryOps:
    def test_addition(self):
        node = expr("1 + 2")
        assert isinstance(node, BinaryOp)
        assert node.op == "+"
        assert isinstance(node.left, NumberLiteral)
        assert node.left.value == 1.0
        assert isinstance(node.right, NumberLiteral)
        assert node.right.value == 2.0

    def test_subtraction(self):
        node = expr("5 - 3")
        assert isinstance(node, BinaryOp)
        assert node.op == "-"

    def test_multiplication(self):
        node = expr("2 * 3")
        assert isinstance(node, BinaryOp)
        assert node.op == "*"

    def test_division(self):
        node = expr("10 / 2")
        assert isinstance(node, BinaryOp)
        assert node.op == "/"

    def test_modulo(self):
        node = expr("10 % 3")
        assert isinstance(node, BinaryOp)
        assert node.op == "%"

    def test_comparison_gt(self):
        node = expr("a > 5")
        assert isinstance(node, BinaryOp)
        assert node.op == ">"
        assert isinstance(node.left, Identifier)
        assert node.left.name == "a"
        assert isinstance(node.right, NumberLiteral)

    def test_comparison_lt(self):
        node = expr("a < 5")
        assert isinstance(node, BinaryOp)
        assert node.op == "<"

    def test_comparison_eq(self):
        node = expr("a == 5")
        assert isinstance(node, BinaryOp)
        assert node.op == "=="

    def test_comparison_neq(self):
        node = expr("a != 5")
        assert isinstance(node, BinaryOp)
        assert node.op == "!="

    def test_comparison_gte(self):
        node = expr("a >= 5")
        assert isinstance(node, BinaryOp)
        assert node.op == ">="

    def test_comparison_lte(self):
        node = expr("a <= 5")
        assert isinstance(node, BinaryOp)
        assert node.op == "<="

    def test_and_operator(self):
        node = expr("a and b")
        assert isinstance(node, BinaryOp)
        assert node.op == "and"
        assert isinstance(node.left, Identifier)
        assert isinstance(node.right, Identifier)

    def test_or_operator(self):
        node = expr("a or b")
        assert isinstance(node, BinaryOp)
        assert node.op == "or"

    def test_and_or_precedence(self):
        """'a and b or c' should parse as '(a and b) or c'."""
        node = expr("a and b or c")
        assert isinstance(node, BinaryOp)
        assert node.op == "or"
        assert isinstance(node.left, BinaryOp)
        assert node.left.op == "and"
        assert isinstance(node.right, Identifier)
        assert node.right.name == "c"

    def test_arithmetic_precedence(self):
        """'1 + 2 * 3' should parse as '1 + (2 * 3)'."""
        node = expr("1 + 2 * 3")
        assert isinstance(node, BinaryOp)
        assert node.op == "+"
        assert isinstance(node.left, NumberLiteral)
        assert node.left.value == 1.0
        assert isinstance(node.right, BinaryOp)
        assert node.right.op == "*"


# ---------------------------------------------------------------------------
# Unary Operations
# ---------------------------------------------------------------------------

class TestUnaryOps:
    def test_unary_not(self):
        node = expr("not true")
        assert isinstance(node, UnaryOp)
        assert node.op == "not"
        assert isinstance(node.operand, BooleanLiteral)
        assert node.operand.value is True

    def test_unary_minus(self):
        node = expr("-5")
        assert isinstance(node, UnaryOp)
        assert node.op == "-"
        assert isinstance(node.operand, NumberLiteral)
        assert node.operand.value == 5.0


# ---------------------------------------------------------------------------
# Parenthesized Expressions
# ---------------------------------------------------------------------------

class TestParenthesized:
    def test_parenthesized_expression(self):
        """'(1 + 2) * 3' should parse as multiplication at top level."""
        node = expr("(1 + 2) * 3")
        assert isinstance(node, BinaryOp)
        assert node.op == "*"
        assert isinstance(node.left, BinaryOp)
        assert node.left.op == "+"
        assert isinstance(node.right, NumberLiteral)
        assert node.right.value == 3.0


# ---------------------------------------------------------------------------
# Function Calls
# ---------------------------------------------------------------------------

class TestFunctionCalls:
    def test_function_call_no_args(self):
        node = expr("foo()")
        assert isinstance(node, FunctionCall)
        assert isinstance(node.callee, Identifier)
        assert node.callee.name == "foo"
        assert node.args == []

    def test_function_call_with_args(self):
        node = expr("foo(1, 2)")
        assert isinstance(node, FunctionCall)
        assert isinstance(node.callee, Identifier)
        assert node.callee.name == "foo"
        assert len(node.args) == 2
        assert isinstance(node.args[0], NumberLiteral)
        assert node.args[0].value == 1.0
        assert isinstance(node.args[1], NumberLiteral)
        assert node.args[1].value == 2.0

    def test_method_call(self):
        """obj.method(arg) should parse as FunctionCall on DotAccess."""
        node = expr("obj.method(1)")
        assert isinstance(node, FunctionCall)
        assert isinstance(node.callee, DotAccess)
        assert node.callee.field_name == "method"
        assert len(node.args) == 1


# ---------------------------------------------------------------------------
# Assignments
# ---------------------------------------------------------------------------

class TestAssignments:
    def test_simple_assignment(self):
        node = expr("x = 42")
        assert isinstance(node, Assignment)
        assert node.target == "x"
        assert node.type_hint is None
        assert isinstance(node.value, NumberLiteral)
        assert node.value.value == 42.0

    def test_typed_assignment(self):
        node = expr("x: number = 42")
        assert isinstance(node, Assignment)
        assert node.target == "x"
        assert node.type_hint == "number"
        assert isinstance(node.value, NumberLiteral)
        assert node.value.value == 42.0

    def test_string_assignment(self):
        node = expr('name = "drift"')
        assert isinstance(node, Assignment)
        assert node.target == "name"
        assert isinstance(node.value, StringLiteral)
        assert node.value.value == "drift"


# ---------------------------------------------------------------------------
# String Interpolation
# ---------------------------------------------------------------------------

class TestStringInterpolation:
    def test_plain_string_parts(self):
        node = expr('"hello"')
        assert isinstance(node, StringLiteral)
        assert node.parts == ["hello"]

    def test_interpolation_simple(self):
        node = expr('"Hello {name}!"')
        assert isinstance(node, StringLiteral)
        assert node.value == "Hello {name}!"
        assert len(node.parts) == 3
        assert node.parts[0] == "Hello "
        assert isinstance(node.parts[1], Identifier)
        assert node.parts[1].name == "name"
        assert node.parts[2] == "!"

    def test_interpolation_dot_access(self):
        node = expr('"Score: {score.verdict}"')
        assert isinstance(node, StringLiteral)
        assert len(node.parts) == 2
        assert node.parts[0] == "Score: "
        assert isinstance(node.parts[1], DotAccess)
        assert node.parts[1].field_name == "verdict"
        assert isinstance(node.parts[1].object, Identifier)
        assert node.parts[1].object.name == "score"

    def test_interpolation_only(self):
        node = expr('"{x}"')
        assert isinstance(node, StringLiteral)
        assert len(node.parts) == 1
        assert isinstance(node.parts[0], Identifier)
        assert node.parts[0].name == "x"


# ---------------------------------------------------------------------------
# Multiple Statements
# ---------------------------------------------------------------------------

class TestMultipleStatements:
    def test_two_assignments(self):
        program = parse("x = 1\ny = 2")
        assert len(program.body) == 2
        assert isinstance(program.body[0], Assignment)
        assert program.body[0].target == "x"
        assert isinstance(program.body[1], Assignment)
        assert program.body[1].target == "y"

    def test_mixed_statements(self):
        program = parse("x = 1\n42")
        assert len(program.body) == 2
        assert isinstance(program.body[0], Assignment)
        assert isinstance(program.body[1], NumberLiteral)


# ---------------------------------------------------------------------------
# Program Structure
# ---------------------------------------------------------------------------

class TestProgramStructure:
    def test_empty_program(self):
        program = parse("")
        assert isinstance(program, Program)
        assert program.body == []

    def test_program_returns_program_node(self):
        program = parse("42")
        assert isinstance(program, Program)


# ---------------------------------------------------------------------------
# Error Cases
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_unclosed_paren(self):
        with pytest.raises(ParseError):
            parse("(1 + 2")

    def test_unclosed_bracket(self):
        with pytest.raises(ParseError):
            parse("[1, 2")

    def test_unclosed_brace(self):
        with pytest.raises(ParseError):
            parse("{ name: 1")


# ---------------------------------------------------------------------------
# Print Statement
# ---------------------------------------------------------------------------

class TestPrintStatement:
    def test_print_string(self):
        prog = parse('print "Hello, Drift!"')
        assert isinstance(prog.body[0], PrintStatement)
        assert isinstance(prog.body[0].value, StringLiteral)

    def test_print_identifier(self):
        prog = parse('print name')
        assert isinstance(prog.body[0], PrintStatement)
        assert isinstance(prog.body[0].value, Identifier)

    def test_print_interpolated(self):
        prog = parse('print "Hello {name}!"')
        stmt = prog.body[0]
        assert isinstance(stmt, PrintStatement)
        assert isinstance(stmt.value, StringLiteral)


# ---------------------------------------------------------------------------
# Log Statement
# ---------------------------------------------------------------------------

class TestLogStatement:
    def test_log_statement(self):
        prog = parse('log "Processing item"')
        assert isinstance(prog.body[0], LogStatement)


# ---------------------------------------------------------------------------
# Schema Definition
# ---------------------------------------------------------------------------

class TestSchemaDefinition:
    def test_schema_simple(self):
        src = "schema Deal:\n  address: string\n  price: number"
        prog = parse(src)
        schema = prog.body[0]
        assert isinstance(schema, SchemaDefinition)
        assert schema.name == "Deal"
        assert len(schema.fields) == 2
        assert schema.fields[0].name == "address"
        assert schema.fields[0].type_name == "string"
        assert schema.fields[1].name == "price"
        assert schema.fields[1].type_name == "number"

    def test_schema_optional_field(self):
        src = "schema X:\n  data: map (optional)"
        prog = parse(src)
        schema = prog.body[0]
        assert schema.fields[0].optional is True
        assert schema.fields[0].type_name == "map"

    def test_schema_list_of_type(self):
        src = "schema X:\n  items: list of string"
        prog = parse(src)
        schema = prog.body[0]
        assert schema.fields[0].type_name == "list of string"


# ---------------------------------------------------------------------------
# Mixed Statement Sequences
# ---------------------------------------------------------------------------

class TestMixedStatements:
    def test_multiple_statements(self):
        src = 'x = 1\nprint "hello"\ny = 2'
        prog = parse(src)
        assert len(prog.body) == 3
        assert isinstance(prog.body[0], Assignment)
        assert isinstance(prog.body[1], PrintStatement)
        assert isinstance(prog.body[2], Assignment)

    def test_schema_then_code(self):
        src = 'schema D:\n  x: number\nprint "done"'
        prog = parse(src)
        assert len(prog.body) == 2
        assert isinstance(prog.body[0], SchemaDefinition)
        assert isinstance(prog.body[1], PrintStatement)


# ---------------------------------------------------------------------------
# Control Flow — If / Else / Else-If
# ---------------------------------------------------------------------------

class TestIfStatement:
    def test_if_simple(self):
        src = 'if x > 5:\n  print "big"'
        prog = parse(src)
        stmt = prog.body[0]
        assert isinstance(stmt, IfStatement)
        assert isinstance(stmt.condition, BinaryOp)
        assert len(stmt.body) == 1
        assert stmt.else_body is None
        assert len(stmt.elseifs) == 0

    def test_if_else(self):
        src = 'if x > 5:\n  print "big"\nelse:\n  print "small"'
        prog = parse(src)
        stmt = prog.body[0]
        assert isinstance(stmt, IfStatement)
        assert stmt.else_body is not None
        assert len(stmt.else_body) == 1

    def test_if_elseif_else(self):
        src = 'if x > 10:\n  print "a"\nelse if x > 5:\n  print "b"\nelse:\n  print "c"'
        prog = parse(src)
        stmt = prog.body[0]
        assert isinstance(stmt, IfStatement)
        assert len(stmt.elseifs) == 1
        assert stmt.else_body is not None

    def test_if_with_and(self):
        src = 'if x > 5 and y < 10:\n  print "range"'
        prog = parse(src)
        stmt = prog.body[0]
        assert isinstance(stmt.condition, BinaryOp)
        assert stmt.condition.op == "and"

    def test_code_after_if(self):
        src = 'if x > 5:\n  print "big"\ny = 2'
        prog = parse(src)
        assert len(prog.body) == 2
        assert isinstance(prog.body[0], IfStatement)
        assert isinstance(prog.body[1], Assignment)


# ---------------------------------------------------------------------------
# Control Flow — For Each
# ---------------------------------------------------------------------------

class TestForEach:
    def test_for_each(self):
        src = "for each item in items:\n  print item"
        prog = parse(src)
        stmt = prog.body[0]
        assert isinstance(stmt, ForEach)
        assert stmt.variable == "item"
        assert isinstance(stmt.iterable, Identifier)
        assert len(stmt.body) == 1

    def test_for_each_dot_access(self):
        src = "for each prop in listings:\n  print prop.name"
        prog = parse(src)
        stmt = prog.body[0]
        assert isinstance(stmt, ForEach)


# ---------------------------------------------------------------------------
# Control Flow — Match Statement
# ---------------------------------------------------------------------------

class TestMatchStatement:
    def test_match_statement(self):
        src = 'match status:\n  200 -> print "ok"\n  404 -> print "not found"\n  _ -> print "error"'
        prog = parse(src)
        stmt = prog.body[0]
        assert isinstance(stmt, MatchStatement)
        assert len(stmt.arms) == 3

    def test_match_wildcard(self):
        src = 'match x:\n  1 -> print "one"\n  _ -> print "other"'
        prog = parse(src)
        stmt = prog.body[0]
        last_arm = stmt.arms[-1]
        assert isinstance(last_arm.pattern, Identifier)
        assert last_arm.pattern.name == "_"


# ---------------------------------------------------------------------------
# Function Definitions and Return Statements
# ---------------------------------------------------------------------------

class TestFunctionDef:
    def test_function_simple(self):
        src = 'define greet(name: string):\n  print name'
        prog = parse(src)
        func = prog.body[0]
        assert isinstance(func, FunctionDef)
        assert func.name == "greet"
        assert len(func.params) == 1
        assert func.params[0] == ("name", "string")
        assert func.return_type is None

    def test_function_with_return_type(self):
        src = "define add(a: number, b: number) -> number:\n  return a + b"
        prog = parse(src)
        func = prog.body[0]
        assert func.return_type == "number"
        assert isinstance(func.body[0], ReturnStatement)

    def test_function_multiple_params(self):
        src = "define f(x: string, y: number, z: boolean):\n  print x"
        prog = parse(src)
        func = prog.body[0]
        assert len(func.params) == 3
        assert func.params[0] == ("x", "string")
        assert func.params[1] == ("y", "number")
        assert func.params[2] == ("z", "boolean")

    def test_function_no_params(self):
        src = 'define hello():\n  print "hi"'
        prog = parse(src)
        func = prog.body[0]
        assert len(func.params) == 0

    def test_return_expression(self):
        src = "define f():\n  return 42"
        prog = parse(src)
        ret = prog.body[0].body[0]
        assert isinstance(ret, ReturnStatement)
        assert isinstance(ret.value, NumberLiteral)

    def test_return_binary(self):
        src = "define f():\n  return a + b"
        prog = parse(src)
        ret = prog.body[0].body[0]
        assert isinstance(ret, ReturnStatement)
        assert isinstance(ret.value, BinaryOp)

    def test_function_then_code(self):
        src = 'define f():\n  return 1\nprint "done"'
        prog = parse(src)
        assert len(prog.body) == 2
        assert isinstance(prog.body[0], FunctionDef)
        assert isinstance(prog.body[1], PrintStatement)


# ---------------------------------------------------------------------------
# AI Primitives
# ---------------------------------------------------------------------------

class TestAIPrimitives:
    def test_ai_ask_simple(self):
        src = 'x = ai.ask("What is 2+2?")'
        prog = parse(src)
        assert isinstance(prog.body[0].value, AIAsk)
        assert isinstance(prog.body[0].value.prompt, StringLiteral)

    def test_ai_ask_with_schema(self):
        src = 'x = ai.ask("Analyze this") -> DealScore'
        prog = parse(src)
        ask = prog.body[0].value
        assert isinstance(ask, AIAsk)
        assert ask.schema == "DealScore"

    def test_ai_ask_with_using(self):
        src = 'x = ai.ask("Analyze") -> Deal using {\n  address: addr\n  price: 100\n}'
        prog = parse(src)
        ask = prog.body[0].value
        assert isinstance(ask, AIAsk)
        assert ask.schema == "Deal"
        assert ask.using is not None

    def test_ai_classify(self):
        src = 'x = ai.classify(text, into: ["a", "b", "c"])'
        prog = parse(src)
        assert isinstance(prog.body[0].value, AIClassify)
        assert len(prog.body[0].value.categories) == 3

    def test_ai_embed(self):
        src = "x = ai.embed(doc.text)"
        prog = parse(src)
        assert isinstance(prog.body[0].value, AIEmbed)

    def test_ai_see(self):
        src = 'x = ai.see(photo, "Describe this")'
        prog = parse(src)
        node = prog.body[0].value
        assert isinstance(node, AISee)

    def test_ai_predict(self):
        src = 'x = ai.predict("Estimate value") -> confident number'
        prog = parse(src)
        node = prog.body[0].value
        assert isinstance(node, AIPredict)

    def test_ai_enrich(self):
        src = 'x = ai.enrich("Add summary")'
        prog = parse(src)
        assert isinstance(prog.body[0].value, AIEnrich)

    def test_ai_score(self):
        src = 'x = ai.score("Rate 1-100") -> number'
        prog = parse(src)
        assert isinstance(prog.body[0].value, AIScore)


# ---------------------------------------------------------------------------
# Data Operations — fetch, read, save, query, merge
# ---------------------------------------------------------------------------

class TestDataOperations:
    def test_fetch_simple(self):
        src = 'data = fetch "https://api.example.com/data"'
        prog = parse(src)
        assert isinstance(prog.body[0].value, FetchExpression)
        assert isinstance(prog.body[0].value.url, StringLiteral)

    def test_fetch_variable(self):
        src = 'data = fetch api_url'
        prog = parse(src)
        assert isinstance(prog.body[0].value, FetchExpression)
        assert isinstance(prog.body[0].value.url, Identifier)

    def test_fetch_with_options(self):
        src = 'data = fetch "https://api.example.com" with {\n  headers: { "X-Key": "abc" }\n  params: { limit: 50 }\n}'
        prog = parse(src)
        fetch = prog.body[0].value
        assert isinstance(fetch, FetchExpression)
        assert fetch.options is not None

    def test_read_file(self):
        src = 'data = read "deals.csv"'
        prog = parse(src)
        assert isinstance(prog.body[0].value, ReadExpression)

    def test_save_statement(self):
        src = 'save data to "output.json"'
        prog = parse(src)
        assert isinstance(prog.body[0], SaveStatement)
        assert isinstance(prog.body[0].data, Identifier)
        assert isinstance(prog.body[0].path, StringLiteral)

    def test_query_expression(self):
        src = 'records = query "SELECT * FROM users" on db.main'
        prog = parse(src)
        q = prog.body[0].value
        assert isinstance(q, QueryExpression)
        assert isinstance(q.sql, StringLiteral)
        assert isinstance(q.source, DotAccess)

    def test_merge_expression(self):
        src = "combined = merge [source_a, source_b]"
        prog = parse(src)
        m = prog.body[0].value
        assert isinstance(m, MergeExpression)
        assert len(m.sources) == 2

    def test_fetch_then_code(self):
        src = 'data = fetch "http://example.com"\nprint data'
        prog = parse(src)
        assert len(prog.body) == 2


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

class TestPipelines:
    def test_pipeline_filter(self):
        src = 'results = data\n  |> filter where price < 500000'
        prog = parse(src)
        stmt = prog.body[0]
        assert isinstance(stmt.value, Pipeline)
        assert len(stmt.value.stages) == 1
        assert isinstance(stmt.value.stages[0], FilterStage)

    def test_pipeline_sort(self):
        src = "results = data\n  |> sort by price ascending"
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe.stages[0], SortStage)
        assert pipe.stages[0].field_name == "price"
        assert pipe.stages[0].direction == "ascending"

    def test_pipeline_sort_descending(self):
        src = "results = data\n  |> sort by score descending"
        prog = parse(src)
        pipe = prog.body[0].value
        assert pipe.stages[0].direction == "descending"

    def test_pipeline_take(self):
        src = "results = data\n  |> take 10"
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe.stages[0], TakeStage)

    def test_pipeline_skip(self):
        src = "results = data\n  |> skip 5"
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe.stages[0], SkipStage)

    def test_pipeline_chain(self):
        src = 'results = data\n  |> filter where x > 5\n  |> sort by x descending\n  |> take 10'
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe, Pipeline)
        assert len(pipe.stages) == 3

    def test_pipeline_deduplicate(self):
        src = "results = data\n  |> deduplicate by address"
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe.stages[0], DeduplicateStage)
        assert pipe.stages[0].field_name == "address"

    def test_pipeline_group(self):
        src = "results = data\n  |> group by city"
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe.stages[0], GroupStage)
        assert pipe.stages[0].field_name == "city"

    def test_pipeline_ai_enrich(self):
        src = 'results = data\n  |> ai.enrich("Add summary")'
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe.stages[0], AIEnrich)

    def test_pipeline_ai_score(self):
        src = 'results = data\n  |> ai.score("Rate 1-100") -> number'
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe.stages[0], AIScore)

    def test_pipeline_each(self):
        src = 'results = data\n  |> each { |item|\n    print item\n  }'
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe.stages[0], EachStage)
        assert pipe.stages[0].variable == "item"
        assert len(pipe.stages[0].body) >= 1

    def test_pipeline_save(self):
        src = 'data\n  |> filter where x > 5\n  |> save to "out.csv"'
        prog = parse(src)
        # This is a standalone pipeline (not assigned)
        # The result should still be parseable
        pipe = prog.body[0]
        assert isinstance(pipe, Pipeline)
        assert len(pipe.stages) == 2
        assert isinstance(pipe.stages[0], FilterStage)
        assert isinstance(pipe.stages[1], SaveStatement)

    def test_pipeline_with_fetch_source(self):
        src = 'results = fetch "https://api.example.com/listings"\n  |> filter where price < 500000\n  |> take 10'
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe, Pipeline)
        assert isinstance(pipe.source, FetchExpression)
        assert len(pipe.stages) == 2

    def test_pipeline_then_more_code(self):
        src = 'results = data\n  |> take 5\nprint "done"'
        prog = parse(src)
        assert len(prog.body) == 2
        assert isinstance(prog.body[0].value, Pipeline)
        assert isinstance(prog.body[1], PrintStatement)

    def test_pipeline_filter_with_and(self):
        src = 'results = data\n  |> filter where price < 500000 and beds >= 3'
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe, Pipeline)
        stage = pipe.stages[0]
        assert isinstance(stage, FilterStage)
        assert isinstance(stage.condition, BinaryOp)
        assert stage.condition.op == "and"

    def test_pipeline_sort_default_direction(self):
        src = "results = data\n  |> sort by price"
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe.stages[0], SortStage)
        assert pipe.stages[0].direction == "ascending"

    def test_pipeline_transform(self):
        src = 'results = data\n  |> transform { |item| item.price * 1.1 }'
        prog = parse(src)
        pipe = prog.body[0].value
        assert isinstance(pipe.stages[0], TransformStage)
        assert pipe.stages[0].variable == "item"


# ---------------------------------------------------------------------------
# Try / Catch
# ---------------------------------------------------------------------------

class TestTryCatch:
    def test_try_catch_simple(self):
        src = 'try:\n  x = 1\ncatch network_error:\n  log "failed"'
        prog = parse(src)
        tc = prog.body[0]
        assert isinstance(tc, TryCatch)
        assert len(tc.try_body) == 1
        assert len(tc.catches) == 1
        assert tc.catches[0].error_type == "network_error"

    def test_try_multiple_catches(self):
        src = 'try:\n  x = 1\ncatch network_error:\n  log "net"\ncatch ai_error:\n  log "ai"'
        prog = parse(src)
        tc = prog.body[0]
        assert len(tc.catches) == 2
        assert tc.catches[0].error_type == "network_error"
        assert tc.catches[1].error_type == "ai_error"

    def test_try_catch_then_code(self):
        src = 'try:\n  x = 1\ncatch err:\n  log "fail"\nprint "done"'
        prog = parse(src)
        assert len(prog.body) == 2
        assert isinstance(prog.body[0], TryCatch)
        assert isinstance(prog.body[1], PrintStatement)
