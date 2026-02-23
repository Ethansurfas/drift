"""Drift lexer — scans source text into a flat list of tokens."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from drift.errors import LexerError


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TokenType(Enum):
    # Literals
    STRING = auto()
    NUMBER = auto()

    # Identifiers
    IDENTIFIER = auto()

    # Keywords
    IF = auto()
    ELSE = auto()
    FOR = auto()
    EACH = auto()
    IN = auto()
    DEFINE = auto()
    RETURN = auto()
    SCHEMA = auto()
    TRY = auto()
    CATCH = auto()
    MATCH = auto()
    PRINT = auto()
    FETCH = auto()
    READ = auto()
    SAVE = auto()
    QUERY = auto()
    LOG = auto()
    FILTER = auto()
    SORT = auto()
    TAKE = auto()
    SKIP = auto()
    GROUP = auto()
    MERGE = auto()
    DEDUPLICATE = auto()
    TRANSFORM = auto()
    WHERE = auto()
    BY = auto()
    TO = auto()
    ON = auto()
    WITH = auto()
    USING = auto()
    AS = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    ASCENDING = auto()
    DESCENDING = auto()
    TRUE = auto()
    FALSE = auto()
    NONE = auto()
    OF = auto()
    OPTIONAL = auto()
    RETRY = auto()
    AFTER = auto()
    MAX = auto()
    TIMES = auto()
    SECONDS = auto()
    FALLBACK = auto()
    CONFIDENT = auto()

    # Operators
    PLUS = auto()          # +
    MINUS = auto()         # -
    STAR = auto()          # *
    SLASH = auto()         # /
    PERCENT = auto()       # %
    PIPE_ARROW = auto()    # |>
    ARROW = auto()         # ->
    DOT = auto()           # .
    EQUALS = auto()        # =
    DOUBLE_EQUALS = auto() # ==
    NOT_EQUALS = auto()    # !=
    LT = auto()            # <
    GT = auto()            # >
    LTE = auto()           # <=
    GTE = auto()           # >=

    # Delimiters
    COLON = auto()         # :
    COMMA = auto()         # ,
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    LBRACKET = auto()      # [
    RBRACKET = auto()      # ]
    LBRACE = auto()        # {
    RBRACE = auto()        # }
    PIPE = auto()          # |

    # Structure
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()


# ---------------------------------------------------------------------------
# Keyword lookup
# ---------------------------------------------------------------------------

KEYWORDS: dict[str, TokenType] = {
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "for": TokenType.FOR,
    "each": TokenType.EACH,
    "in": TokenType.IN,
    "define": TokenType.DEFINE,
    "return": TokenType.RETURN,
    "schema": TokenType.SCHEMA,
    "try": TokenType.TRY,
    "catch": TokenType.CATCH,
    "match": TokenType.MATCH,
    "print": TokenType.PRINT,
    "fetch": TokenType.FETCH,
    "read": TokenType.READ,
    "save": TokenType.SAVE,
    "query": TokenType.QUERY,
    "log": TokenType.LOG,
    "filter": TokenType.FILTER,
    "sort": TokenType.SORT,
    "take": TokenType.TAKE,
    "skip": TokenType.SKIP,
    "group": TokenType.GROUP,
    "merge": TokenType.MERGE,
    "deduplicate": TokenType.DEDUPLICATE,
    "transform": TokenType.TRANSFORM,
    "where": TokenType.WHERE,
    "by": TokenType.BY,
    "to": TokenType.TO,
    "on": TokenType.ON,
    "with": TokenType.WITH,
    "using": TokenType.USING,
    "as": TokenType.AS,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "ascending": TokenType.ASCENDING,
    "descending": TokenType.DESCENDING,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "none": TokenType.NONE,
    "of": TokenType.OF,
    "optional": TokenType.OPTIONAL,
    "retry": TokenType.RETRY,
    "after": TokenType.AFTER,
    "max": TokenType.MAX,
    "times": TokenType.TIMES,
    "seconds": TokenType.SECONDS,
    "fallback": TokenType.FALLBACK,
    "confident": TokenType.CONFIDENT,
}


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.column})"


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class Lexer:
    """Scans Drift source text and produces a flat list of Token objects."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.pos: int = 0
        self.line: int = 1
        self.col: int = 1

    # -- Character-level helpers -------------------------------------------

    def _current(self) -> str:
        """Return the character at the current position, or '' at EOF."""
        if self.pos < len(self.source):
            return self.source[self.pos]
        return ""

    def peek(self) -> str:
        """Look ahead one character without consuming it."""
        next_pos = self.pos + 1
        if next_pos < len(self.source):
            return self.source[next_pos]
        return ""

    def advance(self) -> str:
        """Consume and return the current character, advancing position."""
        ch = self._current()
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    # -- Main entry point --------------------------------------------------

    def tokenize(self) -> list[Token]:
        """Scan the entire source and return a list of tokens ending with EOF."""
        tokens: list[Token] = []

        while self.pos < len(self.source):
            ch = self._current()

            # Skip spaces and tabs (but NOT newlines)
            if ch in (" ", "\t"):
                self.advance()
                continue

            # Newlines
            if ch == "\n":
                tokens.append(Token(TokenType.NEWLINE, "\n", self.line, self.col))
                self.advance()
                continue

            # Comments: -- to end of line
            if ch == "-" and self.peek() == "-":
                self._skip_comment()
                continue

            # Strings
            if ch == '"':
                tokens.append(self._read_string())
                continue

            # Numbers
            if ch.isdigit():
                tokens.append(self._read_number())
                continue

            # Identifiers and keywords
            if ch.isalpha() or ch == "_":
                tokens.append(self._read_identifier())
                continue

            # Multi-character operators (must check before single-char)
            if ch == "|" and self.peek() == ">":
                tokens.append(Token(TokenType.PIPE_ARROW, "|>", self.line, self.col))
                self.advance()
                self.advance()
                continue

            if ch == "-" and self.peek() == ">":
                tokens.append(Token(TokenType.ARROW, "->", self.line, self.col))
                self.advance()
                self.advance()
                continue

            if ch == "=" and self.peek() == "=":
                tokens.append(Token(TokenType.DOUBLE_EQUALS, "==", self.line, self.col))
                self.advance()
                self.advance()
                continue

            if ch == "!" and self.peek() == "=":
                tokens.append(Token(TokenType.NOT_EQUALS, "!=", self.line, self.col))
                self.advance()
                self.advance()
                continue

            if ch == "<" and self.peek() == "=":
                tokens.append(Token(TokenType.LTE, "<=", self.line, self.col))
                self.advance()
                self.advance()
                continue

            if ch == ">" and self.peek() == "=":
                tokens.append(Token(TokenType.GTE, ">=", self.line, self.col))
                self.advance()
                self.advance()
                continue

            # Single-character operators
            if ch == "+":
                tokens.append(Token(TokenType.PLUS, "+", self.line, self.col))
                self.advance()
                continue

            if ch == "-":
                tokens.append(Token(TokenType.MINUS, "-", self.line, self.col))
                self.advance()
                continue

            if ch == "*":
                tokens.append(Token(TokenType.STAR, "*", self.line, self.col))
                self.advance()
                continue

            if ch == "/":
                tokens.append(Token(TokenType.SLASH, "/", self.line, self.col))
                self.advance()
                continue

            if ch == "%":
                tokens.append(Token(TokenType.PERCENT, "%", self.line, self.col))
                self.advance()
                continue

            if ch == ".":
                tokens.append(Token(TokenType.DOT, ".", self.line, self.col))
                self.advance()
                continue

            if ch == "=":
                tokens.append(Token(TokenType.EQUALS, "=", self.line, self.col))
                self.advance()
                continue

            if ch == "<":
                tokens.append(Token(TokenType.LT, "<", self.line, self.col))
                self.advance()
                continue

            if ch == ">":
                tokens.append(Token(TokenType.GT, ">", self.line, self.col))
                self.advance()
                continue

            # Delimiters
            if ch == ":":
                tokens.append(Token(TokenType.COLON, ":", self.line, self.col))
                self.advance()
                continue

            if ch == ",":
                tokens.append(Token(TokenType.COMMA, ",", self.line, self.col))
                self.advance()
                continue

            if ch == "(":
                tokens.append(Token(TokenType.LPAREN, "(", self.line, self.col))
                self.advance()
                continue

            if ch == ")":
                tokens.append(Token(TokenType.RPAREN, ")", self.line, self.col))
                self.advance()
                continue

            if ch == "[":
                tokens.append(Token(TokenType.LBRACKET, "[", self.line, self.col))
                self.advance()
                continue

            if ch == "]":
                tokens.append(Token(TokenType.RBRACKET, "]", self.line, self.col))
                self.advance()
                continue

            if ch == "{":
                tokens.append(Token(TokenType.LBRACE, "{", self.line, self.col))
                self.advance()
                continue

            if ch == "}":
                tokens.append(Token(TokenType.RBRACE, "}", self.line, self.col))
                self.advance()
                continue

            if ch == "|":
                tokens.append(Token(TokenType.PIPE, "|", self.line, self.col))
                self.advance()
                continue

            # Unknown character
            raise LexerError(f"Unexpected character: {ch!r}", self.line, self.col)

        # End of input
        tokens.append(Token(TokenType.EOF, "", self.line, self.col))
        return tokens

    # -- Token readers -----------------------------------------------------

    def _skip_comment(self) -> None:
        """Consume from -- to end of line (or end of source)."""
        while self.pos < len(self.source) and self._current() != "\n":
            self.advance()
        # Do NOT consume the newline — let the main loop handle it.

    def _read_string(self) -> Token:
        """Read a double-quoted string, preserving {interpolation} as-is."""
        start_line = self.line
        start_col = self.col
        self.advance()  # consume opening "

        value_chars: list[str] = []

        while self.pos < len(self.source):
            ch = self._current()
            if ch == '"':
                self.advance()  # consume closing "
                return Token(TokenType.STRING, "".join(value_chars), start_line, start_col)
            if ch == "\n":
                # Unterminated string (hit newline before closing quote)
                raise LexerError("Unterminated string literal", start_line, start_col)
            value_chars.append(ch)
            self.advance()

        # Reached end of source without closing quote
        raise LexerError("Unterminated string literal", start_line, start_col)

    def _read_number(self) -> Token:
        """Read an integer or float literal: [0-9]+(\\.[0-9]+)?"""
        start_line = self.line
        start_col = self.col
        chars: list[str] = []

        while self.pos < len(self.source) and self._current().isdigit():
            chars.append(self.advance())

        # Optional decimal part
        if self.pos < len(self.source) and self._current() == "." and self.peek().isdigit():
            chars.append(self.advance())  # consume '.'
            while self.pos < len(self.source) and self._current().isdigit():
                chars.append(self.advance())

        return Token(TokenType.NUMBER, "".join(chars), start_line, start_col)

    def _read_identifier(self) -> Token:
        """Read an identifier or keyword: [a-zA-Z_][a-zA-Z0-9_]*"""
        start_line = self.line
        start_col = self.col
        chars: list[str] = []

        while self.pos < len(self.source) and (self._current().isalnum() or self._current() == "_"):
            chars.append(self.advance())

        word = "".join(chars)
        token_type = KEYWORDS.get(word, TokenType.IDENTIFIER)
        return Token(token_type, word, start_line, start_col)
