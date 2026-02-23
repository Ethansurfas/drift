"""Tests for Drift AST node definitions."""

from drift.ast_nodes import (
    # Base
    Node,
    # Program
    Program,
    # Expressions
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
    # AI Primitives
    AIAsk,
    AIClassify,
    AIEmbed,
    AISee,
    AIPredict,
    AIEnrich,
    AIScore,
    # Statements
    Assignment,
    PrintStatement,
    LogStatement,
    ReturnStatement,
    IfStatement,
    ForEach,
    MatchStatement,
    MatchArm,
    FunctionDef,
    SchemaDefinition,
    SchemaField,
    # Data Operations
    FetchExpression,
    ReadExpression,
    SaveStatement,
    QueryExpression,
    MergeExpression,
    # Pipelines
    Pipeline,
    FilterStage,
    SortStage,
    TakeStage,
    SkipStage,
    GroupStage,
    DeduplicateStage,
    TransformStage,
    EachStage,
    # Error Handling
    TryCatch,
    CatchClause,
)


# ---------------------------------------------------------------------------
# Base Node
# ---------------------------------------------------------------------------

class TestNodeBase:
    def test_node_has_line_and_col(self):
        """Every node should carry source location info."""
        node = NumberLiteral(value=1.0, line=5, col=10)
        assert node.line == 5
        assert node.col == 10

    def test_default_line_col_zero(self):
        node = NumberLiteral(value=0.0)
        assert node.line == 0
        assert node.col == 0

    def test_node_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(NumberLiteral)

    def test_all_nodes_inherit_from_node(self):
        all_node_classes = [
            Program,
            NumberLiteral, StringLiteral, BooleanLiteral, NoneLiteral,
            ListLiteral, MapLiteral, Identifier, DotAccess, BinaryOp,
            UnaryOp, FunctionCall,
            AIAsk, AIClassify, AIEmbed, AISee, AIPredict, AIEnrich, AIScore,
            Assignment, PrintStatement, LogStatement, ReturnStatement,
            IfStatement, ForEach, MatchStatement, MatchArm, FunctionDef,
            SchemaDefinition, SchemaField,
            FetchExpression, ReadExpression, SaveStatement, QueryExpression,
            MergeExpression,
            Pipeline, FilterStage, SortStage, TakeStage, SkipStage,
            GroupStage, DeduplicateStage, TransformStage, EachStage,
            TryCatch, CatchClause,
        ]
        for cls in all_node_classes:
            assert issubclass(cls, Node), f"{cls.__name__} must inherit from Node"


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

class TestProgram:
    def test_program_empty(self):
        p = Program()
        assert p.body == []

    def test_program_with_body(self):
        stmt = PrintStatement(value=NumberLiteral(value=1.0))
        p = Program(body=[stmt])
        assert len(p.body) == 1
        assert p.body[0] is stmt


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

class TestExpressions:
    def test_number_literal(self):
        n = NumberLiteral(value=42.5, line=1, col=0)
        assert n.value == 42.5

    def test_string_literal_plain(self):
        s = StringLiteral(value="hello")
        assert s.value == "hello"
        assert s.parts == []

    def test_string_literal_interpolation(self):
        ident = Identifier(name="name")
        s = StringLiteral(value="hello {name}", parts=["hello ", ident])
        assert len(s.parts) == 2
        assert isinstance(s.parts[0], str)
        assert isinstance(s.parts[1], Identifier)

    def test_boolean_literal(self):
        t = BooleanLiteral(value=True)
        f = BooleanLiteral(value=False)
        assert t.value is True
        assert f.value is False

    def test_none_literal(self):
        n = NoneLiteral()
        assert isinstance(n, Node)

    def test_list_literal_empty(self):
        ll = ListLiteral()
        assert ll.elements == []

    def test_list_literal_with_elements(self):
        elems = [NumberLiteral(value=1.0), NumberLiteral(value=2.0)]
        ll = ListLiteral(elements=elems)
        assert len(ll.elements) == 2

    def test_map_literal(self):
        pairs = [("key", StringLiteral(value="val"))]
        m = MapLiteral(pairs=pairs)
        assert m.pairs[0][0] == "key"

    def test_identifier(self):
        i = Identifier(name="x")
        assert i.name == "x"

    def test_dot_access(self):
        obj = Identifier(name="person")
        d = DotAccess(object=obj, field_name="name")
        assert d.field_name == "name"
        assert isinstance(d.object, Identifier)

    def test_binary_op(self):
        left = NumberLiteral(value=1.0)
        right = NumberLiteral(value=2.0)
        b = BinaryOp(left=left, op="+", right=right)
        assert b.op == "+"
        assert b.left is left
        assert b.right is right

    def test_unary_op(self):
        operand = NumberLiteral(value=5.0)
        u = UnaryOp(op="-", operand=operand)
        assert u.op == "-"
        assert u.operand is operand

    def test_function_call_no_kwargs(self):
        callee = Identifier(name="print")
        arg = StringLiteral(value="hi")
        fc = FunctionCall(callee=callee, args=[arg])
        assert fc.callee is callee
        assert len(fc.args) == 1
        assert fc.kwargs == {}

    def test_function_call_with_kwargs(self):
        callee = Identifier(name="fetch")
        fc = FunctionCall(
            callee=callee,
            args=[],
            kwargs={"timeout": NumberLiteral(value=30.0)},
        )
        assert "timeout" in fc.kwargs


# ---------------------------------------------------------------------------
# AI Primitives
# ---------------------------------------------------------------------------

class TestAIPrimitives:
    def test_ai_ask(self):
        node = AIAsk(prompt=StringLiteral(value="What is 2+2?"))
        assert node.schema is None
        assert node.using is None

    def test_ai_ask_with_options(self):
        node = AIAsk(
            prompt=StringLiteral(value="Summarize"),
            schema="Summary",
            using={"model": StringLiteral(value="gpt-4")},
        )
        assert node.schema == "Summary"
        assert "model" in node.using

    def test_ai_classify(self):
        cats = [StringLiteral(value="pos"), StringLiteral(value="neg")]
        node = AIClassify(
            input=StringLiteral(value="great product"),
            categories=cats,
        )
        assert len(node.categories) == 2

    def test_ai_embed(self):
        node = AIEmbed(input=StringLiteral(value="hello world"))
        assert isinstance(node.input, StringLiteral)

    def test_ai_see(self):
        node = AISee(
            input=StringLiteral(value="image.png"),
            prompt=StringLiteral(value="describe"),
        )
        assert isinstance(node.prompt, StringLiteral)

    def test_ai_predict(self):
        node = AIPredict(prompt=StringLiteral(value="forecast"))
        assert node.schema is None

    def test_ai_enrich(self):
        node = AIEnrich(prompt=StringLiteral(value="enrich this"))
        assert isinstance(node.prompt, StringLiteral)

    def test_ai_score(self):
        node = AIScore(prompt=StringLiteral(value="rate this"))
        assert node.schema is None


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

class TestStatements:
    def test_assignment_simple(self):
        a = Assignment(target="x", value=NumberLiteral(value=10.0))
        assert a.target == "x"
        assert a.type_hint is None

    def test_assignment_with_type(self):
        a = Assignment(
            target="name",
            type_hint="string",
            value=StringLiteral(value="drift"),
        )
        assert a.type_hint == "string"

    def test_print_statement(self):
        p = PrintStatement(value=StringLiteral(value="hello"))
        assert isinstance(p.value, StringLiteral)

    def test_log_statement(self):
        l = LogStatement(value=StringLiteral(value="debug info"))
        assert isinstance(l.value, StringLiteral)

    def test_return_statement(self):
        r = ReturnStatement(value=NumberLiteral(value=42.0))
        assert isinstance(r.value, NumberLiteral)

    def test_if_statement_simple(self):
        cond = BooleanLiteral(value=True)
        body = [PrintStatement(value=StringLiteral(value="yes"))]
        i = IfStatement(condition=cond, body=body)
        assert i.condition is cond
        assert len(i.body) == 1
        assert i.elseifs == []
        assert i.else_body is None

    def test_if_statement_with_else(self):
        i = IfStatement(
            condition=BooleanLiteral(value=True),
            body=[PrintStatement(value=StringLiteral(value="yes"))],
            else_body=[PrintStatement(value=StringLiteral(value="no"))],
        )
        assert i.else_body is not None
        assert len(i.else_body) == 1

    def test_if_statement_with_elseifs(self):
        elseif = (
            BooleanLiteral(value=False),
            [PrintStatement(value=StringLiteral(value="maybe"))],
        )
        i = IfStatement(
            condition=BooleanLiteral(value=True),
            body=[],
            elseifs=[elseif],
        )
        assert len(i.elseifs) == 1

    def test_for_each(self):
        f = ForEach(
            variable="item",
            iterable=Identifier(name="items"),
            body=[PrintStatement(value=Identifier(name="item"))],
        )
        assert f.variable == "item"
        assert len(f.body) == 1

    def test_match_statement(self):
        arm1 = MatchArm(
            pattern=StringLiteral(value="a"),
            body=PrintStatement(value=StringLiteral(value="matched a")),
        )
        arm2 = MatchArm(
            pattern=StringLiteral(value="b"),
            body=PrintStatement(value=StringLiteral(value="matched b")),
        )
        m = MatchStatement(
            subject=Identifier(name="x"),
            arms=[arm1, arm2],
        )
        assert len(m.arms) == 2
        assert isinstance(m.arms[0], MatchArm)

    def test_function_def(self):
        fd = FunctionDef(
            name="greet",
            params=[("name", "string"), ("age", None)],
            return_type="string",
            body=[ReturnStatement(value=Identifier(name="name"))],
        )
        assert fd.name == "greet"
        assert fd.params[0] == ("name", "string")
        assert fd.return_type == "string"

    def test_function_def_defaults(self):
        fd = FunctionDef(name="noop", body=[])
        assert fd.params == []
        assert fd.return_type is None

    def test_schema_definition(self):
        fields = [
            SchemaField(name="title", type_name="string", optional=False),
            SchemaField(name="rating", type_name="number", optional=True),
        ]
        sd = SchemaDefinition(name="Review", fields=fields)
        assert sd.name == "Review"
        assert len(sd.fields) == 2
        assert sd.fields[1].optional is True


# ---------------------------------------------------------------------------
# Data Operations
# ---------------------------------------------------------------------------

class TestDataOperations:
    def test_fetch_expression(self):
        f = FetchExpression(url=StringLiteral(value="https://api.example.com"))
        assert f.options is None

    def test_fetch_expression_with_options(self):
        f = FetchExpression(
            url=StringLiteral(value="https://api.example.com"),
            options={"method": StringLiteral(value="POST")},
        )
        assert "method" in f.options

    def test_read_expression(self):
        r = ReadExpression(path=StringLiteral(value="data.csv"))
        assert isinstance(r.path, StringLiteral)

    def test_save_statement(self):
        s = SaveStatement(
            data=Identifier(name="results"),
            path=StringLiteral(value="out.json"),
        )
        assert isinstance(s.data, Identifier)

    def test_query_expression(self):
        q = QueryExpression(
            sql=StringLiteral(value="SELECT * FROM users"),
            source=StringLiteral(value="db.sqlite"),
        )
        assert isinstance(q.sql, StringLiteral)

    def test_merge_expression(self):
        sources = [Identifier(name="a"), Identifier(name="b")]
        m = MergeExpression(sources=sources)
        assert len(m.sources) == 2


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

class TestPipelines:
    def test_pipeline_basic(self):
        source = Identifier(name="data")
        stages = [
            FilterStage(condition=BinaryOp(
                left=Identifier(name="age"),
                op=">",
                right=NumberLiteral(value=18.0),
            )),
            SortStage(field_name="name"),
            TakeStage(count=NumberLiteral(value=10.0)),
        ]
        p = Pipeline(source=source, stages=stages)
        assert p.source is source
        assert len(p.stages) == 3

    def test_filter_stage(self):
        f = FilterStage(condition=BooleanLiteral(value=True))
        assert isinstance(f.condition, BooleanLiteral)

    def test_sort_stage_default_direction(self):
        s = SortStage(field_name="created_at")
        assert s.direction == "ascending"

    def test_sort_stage_descending(self):
        s = SortStage(field_name="score", direction="descending")
        assert s.direction == "descending"

    def test_take_stage(self):
        t = TakeStage(count=NumberLiteral(value=5.0))
        assert isinstance(t.count, NumberLiteral)

    def test_skip_stage(self):
        s = SkipStage(count=NumberLiteral(value=3.0))
        assert isinstance(s.count, NumberLiteral)

    def test_group_stage(self):
        g = GroupStage(field_name="category")
        assert g.field_name == "category"

    def test_deduplicate_stage(self):
        d = DeduplicateStage(field_name="email")
        assert d.field_name == "email"

    def test_transform_stage(self):
        t = TransformStage(
            variable="item",
            body=[
                Assignment(
                    target="item.upper_name",
                    value=FunctionCall(
                        callee=Identifier(name="upper"),
                        args=[DotAccess(
                            object=Identifier(name="item"),
                            field_name="name",
                        )],
                    ),
                ),
            ],
        )
        assert t.variable == "item"
        assert len(t.body) == 1

    def test_each_stage(self):
        e = EachStage(
            variable="row",
            body=[PrintStatement(value=Identifier(name="row"))],
        )
        assert e.variable == "row"


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_try_catch(self):
        tc = TryCatch(
            try_body=[
                FetchExpression(url=StringLiteral(value="https://api.example.com")),
            ],
            catches=[
                CatchClause(
                    error_type="NetworkError",
                    body=[PrintStatement(value=StringLiteral(value="failed"))],
                ),
            ],
        )
        assert len(tc.try_body) == 1
        assert len(tc.catches) == 1
        assert tc.catches[0].error_type == "NetworkError"

    def test_multiple_catch_clauses(self):
        tc = TryCatch(
            try_body=[],
            catches=[
                CatchClause(error_type="NetworkError", body=[]),
                CatchClause(error_type="ParseError", body=[]),
            ],
        )
        assert len(tc.catches) == 2


# ---------------------------------------------------------------------------
# Nested / Complex Structures
# ---------------------------------------------------------------------------

class TestNestedStructures:
    def test_pipeline_with_all_stages(self):
        """A pipeline using every stage type."""
        p = Pipeline(
            source=FetchExpression(url=StringLiteral(value="https://data.example.com")),
            stages=[
                FilterStage(condition=BinaryOp(
                    left=Identifier(name="status"),
                    op="==",
                    right=StringLiteral(value="active"),
                )),
                DeduplicateStage(field_name="id"),
                GroupStage(field_name="region"),
                SortStage(field_name="revenue", direction="descending"),
                SkipStage(count=NumberLiteral(value=5.0)),
                TakeStage(count=NumberLiteral(value=20.0)),
                TransformStage(variable="item", body=[
                    Assignment(target="item.label", value=StringLiteral(value="processed")),
                ]),
                EachStage(variable="item", body=[
                    PrintStatement(value=Identifier(name="item")),
                ]),
            ],
        )
        assert len(p.stages) == 8

    def test_nested_if_in_function(self):
        """A function containing an if/elseif/else."""
        fd = FunctionDef(
            name="classify_age",
            params=[("age", "number")],
            return_type="string",
            body=[
                IfStatement(
                    condition=BinaryOp(
                        left=Identifier(name="age"),
                        op="<",
                        right=NumberLiteral(value=18.0),
                    ),
                    body=[ReturnStatement(value=StringLiteral(value="minor"))],
                    elseifs=[(
                        BinaryOp(
                            left=Identifier(name="age"),
                            op="<",
                            right=NumberLiteral(value=65.0),
                        ),
                        [ReturnStatement(value=StringLiteral(value="adult"))],
                    )],
                    else_body=[ReturnStatement(value=StringLiteral(value="senior"))],
                ),
            ],
        )
        assert fd.name == "classify_age"
        if_stmt = fd.body[0]
        assert isinstance(if_stmt, IfStatement)
        assert len(if_stmt.elseifs) == 1
        assert if_stmt.else_body is not None

    def test_try_catch_with_pipeline(self):
        """Error handling around a data pipeline."""
        tc = TryCatch(
            try_body=[
                Assignment(
                    target="results",
                    value=Pipeline(
                        source=ReadExpression(path=StringLiteral(value="data.csv")),
                        stages=[
                            FilterStage(condition=BinaryOp(
                                left=Identifier(name="value"),
                                op=">",
                                right=NumberLiteral(value=0.0),
                            )),
                        ],
                    ),
                ),
                SaveStatement(
                    data=Identifier(name="results"),
                    path=StringLiteral(value="clean.csv"),
                ),
            ],
            catches=[
                CatchClause(error_type="FileError", body=[
                    LogStatement(value=StringLiteral(value="file not found")),
                ]),
            ],
        )
        assert len(tc.try_body) == 2

    def test_mutable_defaults_are_independent(self):
        """Two instances must not share mutable default lists."""
        p1 = Program()
        p2 = Program()
        p1.body.append(NoneLiteral())
        assert len(p2.body) == 0, "Mutable default should not be shared"

        fc1 = FunctionCall(callee=Identifier(name="f"), args=[])
        fc2 = FunctionCall(callee=Identifier(name="g"), args=[])
        fc1.kwargs["x"] = NumberLiteral(value=1.0)
        assert "x" not in fc2.kwargs, "Mutable default dict should not be shared"
