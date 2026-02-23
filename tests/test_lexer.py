"""Tests for Drift lexer — basic tokenization."""

import pytest

from drift.lexer import Lexer, Token, TokenType
from drift.errors import LexerError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lex(source: str) -> list[Token]:
    """Convenience: tokenize source and return the token list."""
    return Lexer(source).tokenize()


def types(source: str) -> list[TokenType]:
    """Return just the token types (excluding EOF) for quick assertions."""
    return [t.type for t in lex(source) if t.type != TokenType.EOF]


def values(source: str) -> list[str]:
    """Return just the token values (excluding EOF)."""
    return [t.value for t in lex(source) if t.type != TokenType.EOF]


# ---------------------------------------------------------------------------
# Empty / Minimal Input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_string(self):
        tokens = lex("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_whitespace_only(self):
        tokens = lex("   \t  ")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_newline_only(self):
        tokens = lex("\n")
        assert types("\n") == [TokenType.NEWLINE]

    def test_multiple_newlines(self):
        tokens = lex("\n\n\n")
        assert all(t == TokenType.NEWLINE for t in types("\n\n\n"))


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

class TestNumbers:
    def test_integer(self):
        tokens = lex("42")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "42"

    def test_float(self):
        tokens = lex("3.14")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "3.14"

    def test_zero(self):
        tokens = lex("0")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "0"

    def test_float_with_leading_zero(self):
        tokens = lex("0.5")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "0.5"

    def test_multiple_numbers(self):
        assert types("1 2 3") == [TokenType.NUMBER, TokenType.NUMBER, TokenType.NUMBER]
        assert values("1 2 3") == ["1", "2", "3"]

    def test_number_position(self):
        tokens = lex("  42")
        assert tokens[0].line == 1
        assert tokens[0].column == 3


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------

class TestStrings:
    def test_basic_string(self):
        tokens = lex('"hello"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"

    def test_empty_string(self):
        tokens = lex('""')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == ""

    def test_string_with_spaces(self):
        tokens = lex('"hello world"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello world"

    def test_string_with_interpolation_braces(self):
        """Interpolation braces are included as-is in the string value."""
        tokens = lex('"hello {name}"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello {name}"

    def test_string_with_nested_braces(self):
        tokens = lex('"result: {compute(x)}"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "result: {compute(x)}"

    def test_string_position(self):
        tokens = lex('"hi"')
        assert tokens[0].line == 1
        assert tokens[0].column == 1

    def test_unterminated_string_raises(self):
        with pytest.raises(LexerError):
            lex('"unterminated')


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

class TestIdentifiers:
    def test_simple_identifier(self):
        tokens = lex("name")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "name"

    def test_identifier_with_underscore(self):
        tokens = lex("my_var")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "my_var"

    def test_identifier_starting_with_underscore(self):
        tokens = lex("_private")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "_private"

    def test_identifier_with_digits(self):
        tokens = lex("var2")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "var2"

    def test_multiple_identifiers(self):
        assert types("a b c") == [TokenType.IDENTIFIER] * 3


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

class TestKeywords:
    def test_if_keyword(self):
        assert types("if") == [TokenType.IF]

    def test_else_keyword(self):
        assert types("else") == [TokenType.ELSE]

    def test_for_keyword(self):
        assert types("for") == [TokenType.FOR]

    def test_each_keyword(self):
        assert types("each") == [TokenType.EACH]

    def test_in_keyword(self):
        assert types("in") == [TokenType.IN]

    def test_define_keyword(self):
        assert types("define") == [TokenType.DEFINE]

    def test_return_keyword(self):
        assert types("return") == [TokenType.RETURN]

    def test_schema_keyword(self):
        assert types("schema") == [TokenType.SCHEMA]

    def test_try_catch_keywords(self):
        assert types("try") == [TokenType.TRY]
        assert types("catch") == [TokenType.CATCH]

    def test_match_keyword(self):
        assert types("match") == [TokenType.MATCH]

    def test_print_keyword(self):
        assert types("print") == [TokenType.PRINT]

    def test_fetch_keyword(self):
        assert types("fetch") == [TokenType.FETCH]

    def test_read_keyword(self):
        assert types("read") == [TokenType.READ]

    def test_save_keyword(self):
        assert types("save") == [TokenType.SAVE]

    def test_query_keyword(self):
        assert types("query") == [TokenType.QUERY]

    def test_log_keyword(self):
        assert types("log") == [TokenType.LOG]

    def test_pipeline_keywords(self):
        assert types("filter") == [TokenType.FILTER]
        assert types("sort") == [TokenType.SORT]
        assert types("take") == [TokenType.TAKE]
        assert types("skip") == [TokenType.SKIP]
        assert types("group") == [TokenType.GROUP]
        assert types("merge") == [TokenType.MERGE]
        assert types("deduplicate") == [TokenType.DEDUPLICATE]
        assert types("transform") == [TokenType.TRANSFORM]

    def test_connector_keywords(self):
        assert types("where") == [TokenType.WHERE]
        assert types("by") == [TokenType.BY]
        assert types("to") == [TokenType.TO]
        assert types("on") == [TokenType.ON]
        assert types("with") == [TokenType.WITH]
        assert types("using") == [TokenType.USING]
        assert types("as") == [TokenType.AS]

    def test_logical_keywords(self):
        assert types("and") == [TokenType.AND]
        assert types("or") == [TokenType.OR]
        assert types("not") == [TokenType.NOT]

    def test_direction_keywords(self):
        assert types("ascending") == [TokenType.ASCENDING]
        assert types("descending") == [TokenType.DESCENDING]

    def test_boolean_keywords(self):
        assert types("true") == [TokenType.TRUE]
        assert types("false") == [TokenType.FALSE]

    def test_none_keyword(self):
        assert types("none") == [TokenType.NONE]

    def test_of_keyword(self):
        assert types("of") == [TokenType.OF]

    def test_optional_keyword(self):
        assert types("optional") == [TokenType.OPTIONAL]

    def test_error_handling_keywords(self):
        assert types("retry") == [TokenType.RETRY]
        assert types("after") == [TokenType.AFTER]
        assert types("max") == [TokenType.MAX]
        assert types("times") == [TokenType.TIMES]
        assert types("seconds") == [TokenType.SECONDS]
        assert types("fallback") == [TokenType.FALLBACK]

    def test_confident_keyword(self):
        assert types("confident") == [TokenType.CONFIDENT]

    def test_keyword_is_case_sensitive(self):
        """Keywords are lowercase only; uppercase is an identifier."""
        assert types("If") == [TokenType.IDENTIFIER]
        assert types("TRUE") == [TokenType.IDENTIFIER]
        assert types("None") == [TokenType.IDENTIFIER]


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class TestOperators:
    def test_plus(self):
        assert types("+") == [TokenType.PLUS]

    def test_minus(self):
        assert types("-") == [TokenType.MINUS]

    def test_star(self):
        assert types("*") == [TokenType.STAR]

    def test_slash(self):
        assert types("/") == [TokenType.SLASH]

    def test_percent(self):
        assert types("%") == [TokenType.PERCENT]

    def test_pipe_arrow(self):
        assert types("|>") == [TokenType.PIPE_ARROW]

    def test_arrow(self):
        assert types("->") == [TokenType.ARROW]

    def test_dot(self):
        assert types(".") == [TokenType.DOT]

    def test_equals(self):
        assert types("=") == [TokenType.EQUALS]

    def test_double_equals(self):
        assert types("==") == [TokenType.DOUBLE_EQUALS]

    def test_not_equals(self):
        assert types("!=") == [TokenType.NOT_EQUALS]

    def test_less_than(self):
        assert types("<") == [TokenType.LT]

    def test_greater_than(self):
        assert types(">") == [TokenType.GT]

    def test_less_than_equal(self):
        assert types("<=") == [TokenType.LTE]

    def test_greater_than_equal(self):
        assert types(">=") == [TokenType.GTE]


# ---------------------------------------------------------------------------
# Delimiters
# ---------------------------------------------------------------------------

class TestDelimiters:
    def test_colon(self):
        assert types(":") == [TokenType.COLON]

    def test_comma(self):
        assert types(",") == [TokenType.COMMA]

    def test_parens(self):
        assert types("()") == [TokenType.LPAREN, TokenType.RPAREN]

    def test_brackets(self):
        assert types("[]") == [TokenType.LBRACKET, TokenType.RBRACKET]

    def test_braces(self):
        assert types("{}") == [TokenType.LBRACE, TokenType.RBRACE]

    def test_pipe(self):
        assert types("|") == [TokenType.PIPE]


# ---------------------------------------------------------------------------
# Dot Access
# ---------------------------------------------------------------------------

class TestDotAccess:
    def test_dot_access(self):
        """ai.ask should produce IDENTIFIER DOT IDENTIFIER."""
        result = types("ai.ask")
        assert result == [TokenType.IDENTIFIER, TokenType.DOT, TokenType.IDENTIFIER]

    def test_chained_dot_access(self):
        result = types("a.b.c")
        assert result == [
            TokenType.IDENTIFIER, TokenType.DOT,
            TokenType.IDENTIFIER, TokenType.DOT,
            TokenType.IDENTIFIER,
        ]

    def test_dot_access_values(self):
        result = values("person.name")
        assert result == ["person", ".", "name"]


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------

class TestComments:
    def test_comment_stripped(self):
        """A comment-only line produces no tokens (besides EOF)."""
        tokens = lex("-- this is a comment")
        assert types("-- this is a comment") == []

    def test_inline_comment(self):
        """Code before a comment should tokenize; comment is discarded."""
        result = types("42 -- the answer")
        assert result == [TokenType.NUMBER]

    def test_comment_does_not_eat_next_line(self):
        result = types("-- comment\n42")
        assert TokenType.NUMBER in result

    def test_multiple_comment_lines(self):
        src = "-- line 1\n-- line 2\n42"
        result = types(src)
        assert TokenType.NUMBER in result


# ---------------------------------------------------------------------------
# Assignment Tokens
# ---------------------------------------------------------------------------

class TestAssignment:
    def test_simple_assignment(self):
        result = types('name = "Drift"')
        assert result == [TokenType.IDENTIFIER, TokenType.EQUALS, TokenType.STRING]

    def test_assignment_values(self):
        result = values('name = "Drift"')
        assert result == ["name", "=", "Drift"]


# ---------------------------------------------------------------------------
# = vs == Distinction
# ---------------------------------------------------------------------------

class TestEqualsDistinction:
    def test_single_equals(self):
        result = types("x = 5")
        assert result == [TokenType.IDENTIFIER, TokenType.EQUALS, TokenType.NUMBER]

    def test_double_equals(self):
        result = types("x == 5")
        assert result == [TokenType.IDENTIFIER, TokenType.DOUBLE_EQUALS, TokenType.NUMBER]

    def test_equals_followed_by_equals(self):
        """== is one token, not two = tokens."""
        tokens = [t for t in lex("==") if t.type != TokenType.EOF]
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.DOUBLE_EQUALS


# ---------------------------------------------------------------------------
# Token Line Numbers
# ---------------------------------------------------------------------------

class TestLineNumbers:
    def test_first_line(self):
        tokens = lex("hello")
        assert tokens[0].line == 1
        assert tokens[0].column == 1

    def test_second_line(self):
        tokens = lex("a\nb")
        # Filter to identifiers
        idents = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        assert idents[0].line == 1
        assert idents[1].line == 2

    def test_column_tracking(self):
        tokens = lex("  x")
        assert tokens[0].column == 3  # x starts at col 3 (1-indexed)

    def test_multiline_positions(self):
        src = "x = 1\ny = 2"
        tokens = [t for t in lex(src) if t.type == TokenType.IDENTIFIER]
        assert tokens[0].line == 1
        assert tokens[0].column == 1
        assert tokens[1].line == 2
        assert tokens[1].column == 1

    def test_eof_position(self):
        tokens = lex("abc")
        eof = tokens[-1]
        assert eof.type == TokenType.EOF


# ---------------------------------------------------------------------------
# Pipe as Standalone Token
# ---------------------------------------------------------------------------

class TestPipeToken:
    def test_pipe_standalone(self):
        assert types("|") == [TokenType.PIPE]

    def test_pipe_in_block_syntax(self):
        """|item| in a pipeline block should lex as PIPE IDENTIFIER PIPE."""
        result = types("|item|")
        assert result == [TokenType.PIPE, TokenType.IDENTIFIER, TokenType.PIPE]

    def test_pipe_vs_pipe_arrow(self):
        """|> should be PIPE_ARROW, not PIPE GT."""
        assert types("|>") == [TokenType.PIPE_ARROW]


# ---------------------------------------------------------------------------
# Error Cases
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unexpected_character(self):
        with pytest.raises(LexerError):
            lex("@")

    def test_unexpected_character_with_position(self):
        try:
            lex("  @")
            assert False, "Should have raised LexerError"
        except LexerError as e:
            assert e.line == 1
            assert e.column == 3


# ---------------------------------------------------------------------------
# Integration: Realistic Snippets
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_function_definition(self):
        src = 'define greet(name: string) -> string:'
        result = types(src)
        assert result == [
            TokenType.DEFINE, TokenType.IDENTIFIER,
            TokenType.LPAREN, TokenType.IDENTIFIER, TokenType.COLON,
            TokenType.IDENTIFIER, TokenType.RPAREN,
            TokenType.ARROW, TokenType.IDENTIFIER, TokenType.COLON,
        ]

    def test_pipeline(self):
        src = 'data |> filter where age > 18 |> sort by name ascending |> take 10'
        result = types(src)
        expected = [
            TokenType.IDENTIFIER, TokenType.PIPE_ARROW,
            TokenType.FILTER, TokenType.WHERE, TokenType.IDENTIFIER,
            TokenType.GT, TokenType.NUMBER,
            TokenType.PIPE_ARROW, TokenType.SORT, TokenType.BY,
            TokenType.IDENTIFIER, TokenType.ASCENDING,
            TokenType.PIPE_ARROW, TokenType.TAKE, TokenType.NUMBER,
        ]
        assert result == expected

    def test_if_statement(self):
        src = 'if x == 10:\n    print "hello"'
        result = types(src)
        assert TokenType.IF in result
        assert TokenType.DOUBLE_EQUALS in result
        assert TokenType.PRINT in result
        assert TokenType.STRING in result

    def test_list_literal(self):
        src = '[1, 2, 3]'
        result = types(src)
        assert result == [
            TokenType.LBRACKET,
            TokenType.NUMBER, TokenType.COMMA,
            TokenType.NUMBER, TokenType.COMMA,
            TokenType.NUMBER,
            TokenType.RBRACKET,
        ]

    def test_schema_definition(self):
        src = 'schema User:\n    name: string\n    age: number'
        result = types(src)
        assert result[0] == TokenType.SCHEMA
        assert result[1] == TokenType.IDENTIFIER  # User

    def test_for_each_loop(self):
        src = 'for each item in items:'
        result = types(src)
        assert result == [
            TokenType.FOR, TokenType.EACH,
            TokenType.IDENTIFIER, TokenType.IN,
            TokenType.IDENTIFIER, TokenType.COLON,
        ]

    def test_match_expression(self):
        src = 'match status:\n    "active" -> print "go"'
        result = types(src)
        assert TokenType.MATCH in result
        assert TokenType.ARROW in result

    def test_not_equals_in_expression(self):
        src = 'x != 5'
        result = types(src)
        assert result == [
            TokenType.IDENTIFIER, TokenType.NOT_EQUALS, TokenType.NUMBER,
        ]


# ---------------------------------------------------------------------------
# Indentation Tracking
# ---------------------------------------------------------------------------

class TestIndentation:
    def test_indent_simple(self):
        src = "if x:\n  y = 1"
        tokens = Lexer(src).tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.INDENT in types

    def test_dedent_simple(self):
        src = "if x:\n  y = 1\nz = 2"
        tokens = Lexer(src).tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.DEDENT in types

    def test_nested_indent(self):
        src = "if x:\n  if y:\n    z = 1\n  w = 2\na = 3"
        tokens = Lexer(src).tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        indent_count = types.count(TokenType.INDENT)
        dedent_count = types.count(TokenType.DEDENT)
        assert indent_count == 2
        assert dedent_count == 2

    def test_no_indent_flat(self):
        src = "x = 1\ny = 2\nz = 3"
        tokens = Lexer(src).tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.INDENT not in types
        assert TokenType.DEDENT not in types

    def test_dedent_at_eof(self):
        src = "if x:\n  y = 1"
        tokens = Lexer(src).tokenize()
        # Should have DEDENT before EOF
        types = [t.type for t in tokens]
        eof_idx = types.index(TokenType.EOF)
        pre_eof = types[:eof_idx]
        assert TokenType.DEDENT in pre_eof

    def test_blank_lines_ignored(self):
        src = "x = 1\n\n\ny = 2"
        tokens = Lexer(src).tokenize()
        idents = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        assert len(idents) == 2
        # No INDENT/DEDENT from blank lines
        types = [t.type for t in tokens]
        assert TokenType.INDENT not in types
        assert TokenType.DEDENT not in types
