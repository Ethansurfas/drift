"""Drift parser — recursive-descent parser producing an AST from tokens."""

from __future__ import annotations

import re

from drift.lexer import Token, TokenType
from drift.errors import ParseError
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
)


class Parser:
    """Recursive-descent parser for the Drift language.

    Consumes a flat list of tokens (from the Lexer) and produces an AST
    rooted at a ``Program`` node.
    """

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos: int = 0

    # -- Navigation helpers ------------------------------------------------

    def current(self) -> Token:
        """Return the token at the current position, or an EOF token if past end."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        # Synthesise an EOF token so callers never crash
        last = self.tokens[-1] if self.tokens else Token(TokenType.EOF, "", 1, 1)
        return Token(TokenType.EOF, "", last.line, last.column)

    def peek(self, offset: int = 1) -> Token:
        """Look ahead *offset* tokens without consuming."""
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        last = self.tokens[-1] if self.tokens else Token(TokenType.EOF, "", 1, 1)
        return Token(TokenType.EOF, "", last.line, last.column)

    def advance(self) -> Token:
        """Consume and return the current token, then increment pos."""
        tok = self.current()
        self.pos += 1
        return tok

    def expect(self, token_type: TokenType) -> Token:
        """Consume the current token if it matches *token_type*, else raise ParseError."""
        tok = self.current()
        if tok.type != token_type:
            raise ParseError(
                f"Expected {token_type.name} but got {tok.type.name} ({tok.value!r})",
                tok.line,
                tok.column,
            )
        return self.advance()

    def match(self, *types: TokenType) -> Token | None:
        """If the current token matches any of *types*, consume and return it; else None."""
        if self.current().type in types:
            return self.advance()
        return None

    def skip_newlines(self) -> None:
        """Skip any NEWLINE tokens at the current position."""
        while self.current().type == TokenType.NEWLINE:
            self.advance()

    def at_end(self) -> bool:
        """Check whether the current token is EOF."""
        return self.current().type == TokenType.EOF

    # -- Top-level ---------------------------------------------------------

    def parse(self) -> Program:
        """Parse the full token stream into a ``Program`` AST node."""
        body: list = []
        self.skip_newlines()
        while not self.at_end():
            stmt = self.parse_statement()
            body.append(stmt)
            self.skip_newlines()
        return Program(body=body)

    # -- Statement parsing -------------------------------------------------

    def parse_statement(self):
        """Parse a single statement.

        Currently handles assignments and expression statements.
        """
        self.skip_newlines()

        if self.current().type == TokenType.PRINT:
            return self.parse_print()
        if self.current().type == TokenType.LOG:
            return self.parse_log()
        if self.current().type == TokenType.SCHEMA:
            return self.parse_schema()
        if self.current().type == TokenType.IF:
            return self.parse_if()
        if self.current().type == TokenType.FOR:
            return self.parse_for_each()
        if self.current().type == TokenType.MATCH:
            return self.parse_match()
        if self.current().type == TokenType.DEFINE:
            return self.parse_function_def()
        if self.current().type == TokenType.RETURN:
            return self.parse_return()

        if self.current().type == TokenType.IDENTIFIER:
            # Check for assignment: IDENTIFIER EQUALS ...
            if self.peek().type == TokenType.EQUALS:
                return self.parse_assignment()
            # Check for typed assignment: IDENTIFIER COLON TYPE EQUALS ...
            if (
                self.peek().type == TokenType.COLON
                and self.peek(2).type == TokenType.IDENTIFIER
                and self.peek(3).type == TokenType.EQUALS
            ):
                return self.parse_assignment()

        # Otherwise, expression statement
        return self.parse_expression()

    def parse_assignment(self):
        """Parse an assignment statement: ``target [: type] = value``."""
        tok = self.expect(TokenType.IDENTIFIER)
        target = tok.value
        line, col = tok.line, tok.column

        # Optional type hint
        type_hint: str | None = None
        if self.match(TokenType.COLON):
            type_tok = self.expect(TokenType.IDENTIFIER)
            type_hint = type_tok.value

        self.expect(TokenType.EQUALS)
        value = self.parse_expression()
        return Assignment(target=target, type_hint=type_hint, value=value, line=line, col=col)

    def parse_print(self):
        """Parse a print statement: ``print <expression>``."""
        tok = self.expect(TokenType.PRINT)
        value = self.parse_expression()
        return PrintStatement(value=value, line=tok.line, col=tok.column)

    def parse_log(self):
        """Parse a log statement: ``log <expression>``."""
        tok = self.expect(TokenType.LOG)
        value = self.parse_expression()
        return LogStatement(value=value, line=tok.line, col=tok.column)

    def parse_schema(self):
        """Parse a schema definition.

        ::

            schema Name:
              field1: type
              field2: list of type
              field3: type (optional)
        """
        tok = self.expect(TokenType.SCHEMA)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.COLON)
        self.expect(TokenType.NEWLINE)
        self.expect(TokenType.INDENT)

        fields: list[SchemaField] = []

        while self.current().type != TokenType.DEDENT and not self.at_end():
            self.skip_newlines()
            if self.current().type == TokenType.DEDENT or self.at_end():
                break

            # Field name
            field_tok = self.expect(TokenType.IDENTIFIER)
            self.expect(TokenType.COLON)

            # Type spec — first word (could be a keyword like "string" or "list")
            type_tok = self.advance()
            type_name = type_tok.value

            # Handle "list of <type>"
            if type_name == "list" and self.current().type == TokenType.OF:
                self.advance()  # consume OF
                element_type_tok = self.advance()
                type_name = f"list of {element_type_tok.value}"

            # Check for (optional) suffix
            optional = False
            if self.current().type == TokenType.LPAREN:
                self.advance()  # consume (
                self.expect(TokenType.OPTIONAL)
                self.expect(TokenType.RPAREN)
                optional = True

            fields.append(SchemaField(
                name=field_tok.value,
                type_name=type_name,
                optional=optional,
                line=field_tok.line,
                col=field_tok.column,
            ))

        # Consume DEDENT if present
        if self.current().type == TokenType.DEDENT:
            self.advance()

        return SchemaDefinition(
            name=name_tok.value,
            fields=fields,
            line=tok.line,
            col=tok.column,
        )

    # -- Block and control flow parsing ------------------------------------

    def parse_block(self) -> list:
        """Parse NEWLINE INDENT <statements> DEDENT and return the statement list."""
        self.skip_newlines()
        self.expect(TokenType.INDENT)
        stmts = []
        while not self.at_end() and self.current().type != TokenType.DEDENT:
            self.skip_newlines()
            if self.current().type == TokenType.DEDENT:
                break
            stmts.append(self.parse_statement())
            self.skip_newlines()
        if self.current().type == TokenType.DEDENT:
            self.advance()  # consume DEDENT
        return stmts

    def parse_if(self):
        """Parse an if / else-if / else statement."""
        tok = self.expect(TokenType.IF)
        condition = self.parse_expression()
        self.expect(TokenType.COLON)
        body = self.parse_block()

        elseifs: list[tuple] = []
        else_body: list | None = None

        # After the block, check for else-if or else chains
        while True:
            self.skip_newlines()
            if self.current().type == TokenType.ELSE:
                if self.peek().type == TokenType.IF:
                    # else if branch
                    self.advance()  # consume ELSE
                    self.advance()  # consume IF
                    ei_condition = self.parse_expression()
                    self.expect(TokenType.COLON)
                    ei_body = self.parse_block()
                    elseifs.append((ei_condition, ei_body))
                else:
                    # plain else branch
                    self.advance()  # consume ELSE
                    self.expect(TokenType.COLON)
                    else_body = self.parse_block()
                    break
            else:
                break

        return IfStatement(
            condition=condition,
            body=body,
            elseifs=elseifs,
            else_body=else_body,
            line=tok.line,
            col=tok.column,
        )

    def parse_for_each(self):
        """Parse a for-each loop: ``for each <var> in <iterable>: <body>``."""
        tok = self.expect(TokenType.FOR)
        self.expect(TokenType.EACH)
        var_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.IN)
        iterable = self.parse_expression()
        self.expect(TokenType.COLON)
        body = self.parse_block()

        return ForEach(
            variable=var_tok.value,
            iterable=iterable,
            body=body,
            line=tok.line,
            col=tok.column,
        )

    def parse_match(self):
        """Parse a match statement with pattern arms."""
        tok = self.expect(TokenType.MATCH)
        subject = self.parse_expression()
        self.expect(TokenType.COLON)

        # Parse indented block of match arms
        self.skip_newlines()
        self.expect(TokenType.INDENT)

        arms: list[MatchArm] = []
        while not self.at_end() and self.current().type != TokenType.DEDENT:
            self.skip_newlines()
            if self.current().type == TokenType.DEDENT:
                break

            # Parse pattern (expression or _ wildcard)
            pattern = self.parse_expression()

            # Expect ->
            self.expect(TokenType.ARROW)

            # Parse the arm body (a single statement)
            arm_body = self.parse_statement()

            arms.append(MatchArm(
                pattern=pattern,
                body=arm_body,
                line=pattern.line,
                col=pattern.col,
            ))
            self.skip_newlines()

        if self.current().type == TokenType.DEDENT:
            self.advance()  # consume DEDENT

        return MatchStatement(
            subject=subject,
            arms=arms,
            line=tok.line,
            col=tok.column,
        )

    def parse_function_def(self):
        """Parse a function definition.

        ::

            define name(param: type, ...) [-> return_type]:
              <body>
        """
        tok = self.expect(TokenType.DEFINE)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LPAREN)

        # Parse parameter list
        params: list[tuple[str, str | None]] = []
        if self.current().type != TokenType.RPAREN:
            # First parameter
            param_tok = self.expect(TokenType.IDENTIFIER)
            self.expect(TokenType.COLON)
            type_tok = self.expect(TokenType.IDENTIFIER)
            params.append((param_tok.value, type_tok.value))

            while self.match(TokenType.COMMA):
                param_tok = self.expect(TokenType.IDENTIFIER)
                self.expect(TokenType.COLON)
                type_tok = self.expect(TokenType.IDENTIFIER)
                params.append((param_tok.value, type_tok.value))

        self.expect(TokenType.RPAREN)

        # Optional return type: -> type
        return_type: str | None = None
        if self.match(TokenType.ARROW):
            rt_tok = self.expect(TokenType.IDENTIFIER)
            return_type = rt_tok.value

        self.expect(TokenType.COLON)
        body = self.parse_block()

        return FunctionDef(
            name=name_tok.value,
            params=params,
            return_type=return_type,
            body=body,
            line=tok.line,
            col=tok.column,
        )

    def parse_return(self):
        """Parse a return statement: ``return <expression>``."""
        tok = self.expect(TokenType.RETURN)
        value = self.parse_expression()
        return ReturnStatement(value=value, line=tok.line, col=tok.column)

    # -- Expression parsing (recursive descent by precedence) --------------

    def parse_expression(self):
        """Entry point for expression parsing — lowest precedence."""
        return self.parse_or()

    def parse_or(self):
        """Parse ``or`` expressions (lowest precedence binary)."""
        left = self.parse_and()
        while self.current().type == TokenType.OR:
            op_tok = self.advance()
            right = self.parse_and()
            left = BinaryOp(left=left, op="or", right=right, line=op_tok.line, col=op_tok.column)
        return left

    def parse_and(self):
        """Parse ``and`` expressions."""
        left = self.parse_not()
        while self.current().type == TokenType.AND:
            op_tok = self.advance()
            right = self.parse_not()
            left = BinaryOp(left=left, op="and", right=right, line=op_tok.line, col=op_tok.column)
        return left

    def parse_not(self):
        """Parse ``not`` prefix (unary)."""
        if self.current().type == TokenType.NOT:
            op_tok = self.advance()
            operand = self.parse_comparison()
            return UnaryOp(op="not", operand=operand, line=op_tok.line, col=op_tok.column)
        return self.parse_comparison()

    def parse_comparison(self):
        """Parse comparison operators: ``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``."""
        left = self.parse_addition()

        comparison_types = {
            TokenType.DOUBLE_EQUALS: "==",
            TokenType.NOT_EQUALS: "!=",
            TokenType.LT: "<",
            TokenType.GT: ">",
            TokenType.LTE: "<=",
            TokenType.GTE: ">=",
        }

        while self.current().type in comparison_types:
            op_tok = self.advance()
            op_str = comparison_types[op_tok.type]
            right = self.parse_addition()
            left = BinaryOp(left=left, op=op_str, right=right, line=op_tok.line, col=op_tok.column)
        return left

    def parse_addition(self):
        """Parse ``+`` and ``-`` (addition-level precedence)."""
        left = self.parse_multiplication()

        while self.current().type in (TokenType.PLUS, TokenType.MINUS):
            op_tok = self.advance()
            op_str = "+" if op_tok.type == TokenType.PLUS else "-"
            right = self.parse_multiplication()
            left = BinaryOp(left=left, op=op_str, right=right, line=op_tok.line, col=op_tok.column)
        return left

    def parse_multiplication(self):
        """Parse ``*``, ``/``, ``%`` (multiplication-level precedence)."""
        left = self.parse_unary()

        mul_ops = {
            TokenType.STAR: "*",
            TokenType.SLASH: "/",
            TokenType.PERCENT: "%",
        }

        while self.current().type in mul_ops:
            op_tok = self.advance()
            op_str = mul_ops[op_tok.type]
            right = self.parse_unary()
            left = BinaryOp(left=left, op=op_str, right=right, line=op_tok.line, col=op_tok.column)
        return left

    def parse_unary(self):
        """Parse unary ``-`` prefix."""
        if self.current().type == TokenType.MINUS:
            op_tok = self.advance()
            operand = self.parse_postfix()
            return UnaryOp(op="-", operand=operand, line=op_tok.line, col=op_tok.column)
        return self.parse_postfix()

    def parse_postfix(self):
        """Parse postfix operations: ``.field`` chains and ``(args)`` calls."""
        node = self.parse_primary()

        while True:
            if self.current().type == TokenType.DOT:
                self.advance()  # consume '.'
                field_tok = self.expect(TokenType.IDENTIFIER)
                node = DotAccess(
                    object=node,
                    field_name=field_tok.value,
                    line=field_tok.line,
                    col=field_tok.column,
                )
            elif self.current().type == TokenType.LPAREN:
                node = self._parse_call(node)
            else:
                break

        return node

    def _parse_call(self, callee):
        """Parse a function call ``(arg1, arg2, ...)``."""
        lparen = self.expect(TokenType.LPAREN)
        args: list = []

        if self.current().type != TokenType.RPAREN:
            args.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                args.append(self.parse_expression())

        self.expect(TokenType.RPAREN)
        return FunctionCall(callee=callee, args=args, line=lparen.line, col=lparen.column)

    def parse_primary(self):
        """Parse primary (atomic) expressions."""
        tok = self.current()

        # Number
        if tok.type == TokenType.NUMBER:
            self.advance()
            return NumberLiteral(value=float(tok.value), line=tok.line, col=tok.column)

        # String
        if tok.type == TokenType.STRING:
            self.advance()
            parts = self._parse_interpolation(tok.value)
            return StringLiteral(value=tok.value, parts=parts, line=tok.line, col=tok.column)

        # Booleans
        if tok.type == TokenType.TRUE:
            self.advance()
            return BooleanLiteral(value=True, line=tok.line, col=tok.column)
        if tok.type == TokenType.FALSE:
            self.advance()
            return BooleanLiteral(value=False, line=tok.line, col=tok.column)

        # None
        if tok.type == TokenType.NONE:
            self.advance()
            return NoneLiteral(line=tok.line, col=tok.column)

        # Identifier
        if tok.type == TokenType.IDENTIFIER:
            self.advance()
            return Identifier(name=tok.value, line=tok.line, col=tok.column)

        # Grouped expression: ( expr )
        if tok.type == TokenType.LPAREN:
            self.advance()  # consume '('
            node = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return node

        # List literal: [ expr, ... ]
        if tok.type == TokenType.LBRACKET:
            return self._parse_list_literal()

        # Map literal: { key: value, ... }
        if tok.type == TokenType.LBRACE:
            return self._parse_map_literal()

        raise ParseError(
            f"Unexpected token {tok.type.name} ({tok.value!r})",
            tok.line,
            tok.column,
        )

    def _parse_list_literal(self):
        """Parse ``[expr, expr, ...]``."""
        tok = self.expect(TokenType.LBRACKET)
        elements: list = []

        if self.current().type != TokenType.RBRACKET:
            elements.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                elements.append(self.parse_expression())

        self.expect(TokenType.RBRACKET)
        return ListLiteral(elements=elements, line=tok.line, col=tok.column)

    def _parse_map_literal(self):
        """Parse ``{ key: value, key: value, ... }``."""
        tok = self.expect(TokenType.LBRACE)
        pairs: list[tuple[str, object]] = []

        if self.current().type != TokenType.RBRACE:
            key_tok = self.expect(TokenType.IDENTIFIER)
            self.expect(TokenType.COLON)
            value = self.parse_expression()
            pairs.append((key_tok.value, value))

            while self.match(TokenType.COMMA):
                key_tok = self.expect(TokenType.IDENTIFIER)
                self.expect(TokenType.COLON)
                value = self.parse_expression()
                pairs.append((key_tok.value, value))

        self.expect(TokenType.RBRACE)
        return MapLiteral(pairs=pairs, line=tok.line, col=tok.column)

    # -- String interpolation ----------------------------------------------

    @staticmethod
    def _parse_interpolation(value: str) -> list:
        """Split a string value into parts: plain text strings and AST nodes.

        Handles simple identifiers like ``{name}`` and dot-access like
        ``{score.verdict}`` inside ``{...}`` delimiters.
        """
        parts: list = []
        # Match {identifier} or {identifier.identifier...} patterns
        pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\}")

        last_end = 0
        for m in pattern.finditer(value):
            # Add any plain text before this match
            if m.start() > last_end:
                parts.append(value[last_end:m.start()])

            # Parse the identifier (possibly with dots)
            inner = m.group(1)
            segments = inner.split(".")
            node = Identifier(name=segments[0])
            for seg in segments[1:]:
                node = DotAccess(object=node, field_name=seg)
            parts.append(node)

            last_end = m.end()

        # Add any trailing plain text
        if last_end < len(value):
            parts.append(value[last_end:])

        # If no interpolation was found, parts is just the whole string
        if not parts:
            parts = [value]

        return parts
