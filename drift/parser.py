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
        if self.current().type == TokenType.SAVE:
            return self.parse_save()
        if self.current().type == TokenType.TRY:
            return self.parse_try()

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

        # Otherwise, expression statement (possibly the source of a pipeline)
        expr = self.parse_expression()
        return self._maybe_parse_pipeline(expr)

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
        value = self._maybe_parse_pipeline(value)
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

    def parse_save(self):
        """Parse a save statement: ``save <expr> to <expr>``."""
        tok = self.expect(TokenType.SAVE)
        data = self.parse_expression()
        self.expect(TokenType.TO)
        path = self.parse_expression()
        return SaveStatement(data=data, path=path, line=tok.line, col=tok.column)

    def parse_try(self):
        """Parse a try/catch statement.

        ::

            try:
              <body>
            catch <error_type>:
              <body>
            [catch <error_type>:
              <body> ...]
        """
        tok = self.expect(TokenType.TRY)
        self.expect(TokenType.COLON)
        try_body = self.parse_block()

        catches: list[CatchClause] = []
        while True:
            self.skip_newlines()
            if self.current().type != TokenType.CATCH:
                break
            self.advance()  # consume CATCH
            error_tok = self.expect(TokenType.IDENTIFIER)
            self.expect(TokenType.COLON)
            catch_body = self.parse_block()
            catches.append(CatchClause(
                error_type=error_tok.value,
                body=catch_body,
                line=error_tok.line,
                col=error_tok.column,
            ))

        return TryCatch(
            try_body=try_body,
            catches=catches,
            line=tok.line,
            col=tok.column,
        )

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
        """Parse postfix operations: ``.field`` chains, ``(args)`` calls, and AI primitives."""
        node = self.parse_primary()

        while True:
            if self.current().type == TokenType.DOT:
                # Check for ai.method() dispatch
                if isinstance(node, Identifier) and node.name == "ai":
                    ai_node = self._try_parse_ai_primitive()
                    if ai_node is not None:
                        node = ai_node
                        break  # AI primitives are terminal — no further postfix chaining

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

    # -- AI Primitive parsing -----------------------------------------------

    _AI_DISPATCH = {
        "ask": "_parse_ai_ask",
        "classify": "_parse_ai_classify",
        "embed": "_parse_ai_embed",
        "see": "_parse_ai_see",
        "predict": "_parse_ai_predict",
        "enrich": "_parse_ai_enrich",
        "score": "_parse_ai_score",
    }

    def _try_parse_ai_primitive(self):
        """If the current position is DOT followed by a known AI method name, parse it.

        Returns the AI AST node, or None if this is not an AI primitive
        (allowing normal dot-access to proceed).
        """
        # We are at DOT and the primary was Identifier("ai")
        # Peek at the identifier after DOT
        if self.peek().type != TokenType.IDENTIFIER:
            return None
        method_name = self.peek().value
        dispatch_method = self._AI_DISPATCH.get(method_name)
        if dispatch_method is None:
            return None

        # Commit: consume DOT and method name
        dot_tok = self.advance()  # consume DOT
        self.advance()  # consume method name IDENTIFIER
        return getattr(self, dispatch_method)(dot_tok)

    def _parse_ai_ask(self, tok):
        """Parse ``ai.ask(<prompt>) [-> <Schema>] [using { ... }]``."""
        self.expect(TokenType.LPAREN)
        prompt = self.parse_expression()
        self.expect(TokenType.RPAREN)

        # Optional schema: -> SchemaName
        schema = None
        if self.match(TokenType.ARROW):
            schema_tok = self.expect(TokenType.IDENTIFIER)
            schema = schema_tok.value

        # Optional using block: using { key: value ... }
        using = None
        if self.current().type == TokenType.USING:
            self.advance()  # consume USING
            using = self._parse_using_map()

        return AIAsk(prompt=prompt, schema=schema, using=using, line=tok.line, col=tok.column)

    def _parse_ai_classify(self, tok):
        """Parse ``ai.classify(<input>, into: [<categories>])``."""
        self.expect(TokenType.LPAREN)
        input_expr = self.parse_expression()
        self.expect(TokenType.COMMA)

        # Expect "into" identifier, COLON, then list literal
        into_tok = self.current()
        if into_tok.type != TokenType.IDENTIFIER or into_tok.value != "into":
            raise ParseError(
                f"Expected 'into' keyword but got {into_tok.type.name} ({into_tok.value!r})",
                into_tok.line,
                into_tok.column,
            )
        self.advance()  # consume "into"
        self.expect(TokenType.COLON)
        categories_node = self._parse_list_literal()
        categories = categories_node.elements

        self.expect(TokenType.RPAREN)
        return AIClassify(input=input_expr, categories=categories, line=tok.line, col=tok.column)

    def _parse_ai_embed(self, tok):
        """Parse ``ai.embed(<input>)``."""
        self.expect(TokenType.LPAREN)
        input_expr = self.parse_expression()
        self.expect(TokenType.RPAREN)
        return AIEmbed(input=input_expr, line=tok.line, col=tok.column)

    def _parse_ai_see(self, tok):
        """Parse ``ai.see(<image>, <prompt>)``."""
        self.expect(TokenType.LPAREN)
        image = self.parse_expression()
        self.expect(TokenType.COMMA)
        prompt = self.parse_expression()
        self.expect(TokenType.RPAREN)
        return AISee(input=image, prompt=prompt, line=tok.line, col=tok.column)

    def _parse_ai_predict(self, tok):
        """Parse ``ai.predict(<prompt>) [-> [confident] <type>]``."""
        self.expect(TokenType.LPAREN)
        prompt = self.parse_expression()
        self.expect(TokenType.RPAREN)

        # Optional type/schema: -> [confident] type
        schema = None
        if self.match(TokenType.ARROW):
            schema = self._parse_schema_type()

        return AIPredict(prompt=prompt, schema=schema, line=tok.line, col=tok.column)

    def _parse_ai_enrich(self, tok):
        """Parse ``ai.enrich(<prompt>)``."""
        self.expect(TokenType.LPAREN)
        prompt = self.parse_expression()
        self.expect(TokenType.RPAREN)
        return AIEnrich(prompt=prompt, line=tok.line, col=tok.column)

    def _parse_ai_score(self, tok):
        """Parse ``ai.score(<prompt>) [-> <type>]``."""
        self.expect(TokenType.LPAREN)
        prompt = self.parse_expression()
        self.expect(TokenType.RPAREN)

        # Optional type/schema: -> type
        schema = None
        if self.match(TokenType.ARROW):
            schema = self._parse_schema_type()

        return AIScore(prompt=prompt, schema=schema, line=tok.line, col=tok.column)

    def _parse_schema_type(self) -> str:
        """Parse an optional schema type after ``->``.

        Handles ``confident <type>`` (two tokens) or just ``<type>`` (one token).
        """
        if self.current().type == TokenType.CONFIDENT:
            self.advance()  # consume "confident"
            type_tok = self.expect(TokenType.IDENTIFIER)
            return f"confident {type_tok.value}"
        type_tok = self.expect(TokenType.IDENTIFIER)
        return type_tok.value

    def _skip_whitespace_tokens(self) -> None:
        """Skip NEWLINE, INDENT, and DEDENT tokens (for brace-delimited blocks)."""
        while self.current().type in (TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT):
            self.advance()

    def _parse_using_map(self) -> dict:
        """Parse a using block: ``{ key: value [\\n key: value ...] }``

        Entries may be separated by commas or newlines.  Because the lexer
        may insert INDENT/DEDENT tokens for the indented content inside
        braces, we skip those structure tokens here.
        """
        self.expect(TokenType.LBRACE)
        self._skip_whitespace_tokens()
        pairs: dict = {}

        while self.current().type != TokenType.RBRACE and not self.at_end():
            self._skip_whitespace_tokens()
            if self.current().type == TokenType.RBRACE:
                break
            # Accept IDENTIFIER or STRING as map key
            if self.current().type == TokenType.STRING:
                key_tok = self.advance()
            else:
                key_tok = self.expect(TokenType.IDENTIFIER)
            self.expect(TokenType.COLON)
            value = self.parse_expression()
            pairs[key_tok.value] = value

            # Skip optional comma and/or whitespace tokens between entries
            self.match(TokenType.COMMA)
            self._skip_whitespace_tokens()

        self.expect(TokenType.RBRACE)
        return pairs

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

        # Fetch expression: fetch <expr> [with { ... }]
        if tok.type == TokenType.FETCH:
            return self._parse_fetch()

        # Read expression: read <expr>
        if tok.type == TokenType.READ:
            return self._parse_read()

        # Query expression: query <expr> on <expr>
        if tok.type == TokenType.QUERY:
            return self._parse_query()

        # Merge expression: merge [<expr>, ...]
        if tok.type == TokenType.MERGE:
            return self._parse_merge()

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

    def _parse_map_key(self) -> Token:
        """Parse a map key — accepts IDENTIFIER or STRING tokens."""
        if self.current().type == TokenType.STRING:
            return self.advance()
        return self.expect(TokenType.IDENTIFIER)

    def _parse_map_literal(self):
        """Parse ``{ key: value, key: value, ... }``."""
        tok = self.expect(TokenType.LBRACE)
        pairs: list[tuple[str, object]] = []

        if self.current().type != TokenType.RBRACE:
            key_tok = self._parse_map_key()
            self.expect(TokenType.COLON)
            value = self.parse_expression()
            pairs.append((key_tok.value, value))

            while self.match(TokenType.COMMA):
                key_tok = self._parse_map_key()
                self.expect(TokenType.COLON)
                value = self.parse_expression()
                pairs.append((key_tok.value, value))

        self.expect(TokenType.RBRACE)
        return MapLiteral(pairs=pairs, line=tok.line, col=tok.column)

    # -- Data operation parsing --------------------------------------------

    def _parse_fetch(self):
        """Parse ``fetch <expr> [with { key: value, ... }]``."""
        tok = self.advance()  # consume FETCH
        url = self.parse_expression()

        # Optional: with { ... }
        options = None
        if self.current().type == TokenType.WITH:
            self.advance()  # consume WITH
            options = self._parse_using_map()

        return FetchExpression(url=url, options=options, line=tok.line, col=tok.column)

    def _parse_read(self):
        """Parse ``read <expr>``."""
        tok = self.advance()  # consume READ
        path = self.parse_expression()
        return ReadExpression(path=path, line=tok.line, col=tok.column)

    def _parse_query(self):
        """Parse ``query <expr> on <expr>``."""
        tok = self.advance()  # consume QUERY
        sql = self.parse_expression()
        self.expect(TokenType.ON)
        source = self.parse_expression()
        return QueryExpression(sql=sql, source=source, line=tok.line, col=tok.column)

    def _parse_merge(self):
        """Parse ``merge [<expr>, ...]``."""
        tok = self.advance()  # consume MERGE
        list_node = self._parse_list_literal()
        return MergeExpression(sources=list_node.elements, line=tok.line, col=tok.column)

    # -- Pipeline parsing ---------------------------------------------------

    def _maybe_parse_pipeline(self, source):
        """If ``|>`` follows (possibly after newlines/indent), parse a pipeline.

        Otherwise return *source* unchanged.  This method saves and restores
        the parser position when no pipeline is found so that the caller is
        unaffected.
        """
        saved_pos = self.pos
        # Skip NEWLINE, INDENT, DEDENT tokens that the lexer injects between
        # the source expression and the first |> stage.
        self._skip_whitespace_tokens()
        if not self.at_end() and self.current().type == TokenType.PIPE_ARROW:
            stages = []
            while not self.at_end() and self.current().type == TokenType.PIPE_ARROW:
                self.advance()  # consume |>
                stage = self._parse_pipeline_stage()
                stages.append(stage)
                # Between stages we may see NEWLINE tokens (same indent level).
                # Skip them to find the next |> or the end of the pipeline.
                self._skip_whitespace_tokens()
            return Pipeline(source=source, stages=stages, line=source.line, col=source.col)
        else:
            self.pos = saved_pos  # restore — no pipeline found
            return source

    def _parse_pipeline_stage(self):
        """Parse a single pipeline stage after ``|>`` has been consumed."""
        tok = self.current()

        # filter where <condition>
        if tok.type == TokenType.FILTER:
            return self._parse_filter_stage()

        # sort by <field> [ascending|descending]
        if tok.type == TokenType.SORT:
            return self._parse_sort_stage()

        # take <n>
        if tok.type == TokenType.TAKE:
            return self._parse_take_stage()

        # skip <n>
        if tok.type == TokenType.SKIP:
            return self._parse_skip_stage()

        # group by <field>
        if tok.type == TokenType.GROUP:
            return self._parse_group_stage()

        # deduplicate by <field>
        if tok.type == TokenType.DEDUPLICATE:
            return self._parse_deduplicate_stage()

        # each { |var| body }
        if tok.type == TokenType.EACH:
            return self._parse_each_stage()

        # transform { |var| body }
        if tok.type == TokenType.TRANSFORM:
            return self._parse_transform_stage()

        # ai.enrich(...) or ai.score(...)
        if tok.type == TokenType.IDENTIFIER and tok.value == "ai":
            return self._parse_ai_pipeline_stage()

        # save to <path>
        if tok.type == TokenType.SAVE:
            return self._parse_save_stage()

        raise ParseError(
            f"Unknown pipeline stage: {tok.type.name} ({tok.value!r})",
            tok.line,
            tok.column,
        )

    def _parse_filter_stage(self):
        """Parse ``filter where <condition>``."""
        tok = self.expect(TokenType.FILTER)
        self.expect(TokenType.WHERE)
        condition = self.parse_expression()
        return FilterStage(condition=condition, line=tok.line, col=tok.column)

    def _parse_sort_stage(self):
        """Parse ``sort by <field> [ascending|descending]``."""
        tok = self.expect(TokenType.SORT)
        self.expect(TokenType.BY)
        field_tok = self.expect(TokenType.IDENTIFIER)
        direction = "ascending"
        if self.current().type == TokenType.ASCENDING:
            self.advance()
            direction = "ascending"
        elif self.current().type == TokenType.DESCENDING:
            self.advance()
            direction = "descending"
        return SortStage(field_name=field_tok.value, direction=direction, line=tok.line, col=tok.column)

    def _parse_take_stage(self):
        """Parse ``take <expression>``."""
        tok = self.expect(TokenType.TAKE)
        count = self.parse_expression()
        return TakeStage(count=count, line=tok.line, col=tok.column)

    def _parse_skip_stage(self):
        """Parse ``skip <expression>``."""
        tok = self.expect(TokenType.SKIP)
        count = self.parse_expression()
        return SkipStage(count=count, line=tok.line, col=tok.column)

    def _parse_group_stage(self):
        """Parse ``group by <field>``."""
        tok = self.expect(TokenType.GROUP)
        self.expect(TokenType.BY)
        field_tok = self.expect(TokenType.IDENTIFIER)
        return GroupStage(field_name=field_tok.value, line=tok.line, col=tok.column)

    def _parse_deduplicate_stage(self):
        """Parse ``deduplicate by <field>``."""
        tok = self.expect(TokenType.DEDUPLICATE)
        self.expect(TokenType.BY)
        field_tok = self.expect(TokenType.IDENTIFIER)
        return DeduplicateStage(field_name=field_tok.value, line=tok.line, col=tok.column)

    def _parse_each_stage(self):
        """Parse ``each { |var| body }``."""
        tok = self.expect(TokenType.EACH)
        self.expect(TokenType.LBRACE)
        self._skip_whitespace_tokens()
        self.expect(TokenType.PIPE)
        var_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.PIPE)
        self._skip_whitespace_tokens()

        body = []
        while self.current().type != TokenType.RBRACE and not self.at_end():
            self._skip_whitespace_tokens()
            if self.current().type == TokenType.RBRACE:
                break
            body.append(self.parse_statement())
            self._skip_whitespace_tokens()

        self.expect(TokenType.RBRACE)
        return EachStage(variable=var_tok.value, body=body, line=tok.line, col=tok.column)

    def _parse_transform_stage(self):
        """Parse ``transform { |var| body }``."""
        tok = self.expect(TokenType.TRANSFORM)
        self.expect(TokenType.LBRACE)
        self._skip_whitespace_tokens()
        self.expect(TokenType.PIPE)
        var_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.PIPE)
        self._skip_whitespace_tokens()

        body = []
        while self.current().type != TokenType.RBRACE and not self.at_end():
            self._skip_whitespace_tokens()
            if self.current().type == TokenType.RBRACE:
                break
            body.append(self.parse_statement())
            self._skip_whitespace_tokens()

        self.expect(TokenType.RBRACE)
        return TransformStage(variable=var_tok.value, body=body, line=tok.line, col=tok.column)

    def _parse_ai_pipeline_stage(self):
        """Parse an AI primitive (``ai.enrich(...)`` or ``ai.score(...)``) as a pipeline stage."""
        # Parse as a normal expression — the postfix handler recognizes ai.method()
        return self.parse_expression()

    def _parse_save_stage(self):
        """Parse ``save to <path>`` as a pipeline terminal stage."""
        tok = self.expect(TokenType.SAVE)
        self.expect(TokenType.TO)
        path = self.parse_expression()
        return SaveStatement(data=None, path=path, line=tok.line, col=tok.column)

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
