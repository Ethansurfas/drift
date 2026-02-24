# Drift — Claude Code Context

## What is Drift?

An AI-native programming language that transpiles to Python. See `DRIFT_LANGUAGE_SPEC.md` for the full language spec.

## Quick Reference: Writing Drift

### Comments
```drift
-- This is a comment
```

### Variables (type inference, optional type hints)
```drift
city = "Austin"
price = 450000
active = true
tags = ["investment", "fix"]
score: number = 92.5
```

### Types
`string`, `number`, `boolean`, `list`, `map`, `date`, `none`
AI-native: `confident number` (wraps value + confidence score)

### String Interpolation
```drift
print "Hello {name}!"
print "Score: {deal.roi}"
```

### Schemas (structured data types)
```drift
schema DealScore:
  address: string
  arv: number
  photos: list of string
  data: map (optional)
```

### Functions
```drift
define analyze(address: string, budget: number) -> DealScore:
  return DealScore { address: address, arv: 0, photos: [], data: none }
```

### Control Flow
```drift
if score > 90:
  print "excellent"
else if score > 70:
  print "good"
else:
  print "needs work"

for each item in items:
  print item

match status:
  200 -> print "ok"
  404 -> print "not found"
  _ -> print "error"
```

### AI Primitives (built-in, no imports)
```drift
summary = ai.ask("Summarize: {text}")
analysis = ai.ask("Analyze this") -> DealScore using { data: input }
category = ai.classify(email.body, into: ["urgent", "routine", "spam"])
vector = ai.embed(document.text)
description = ai.see(photo, "Describe condition")
estimate = ai.predict("Estimate ARV") -> confident number
```

### Data Operations
```drift
data = fetch "https://api.example.com" with {
  headers: { "X-Key": env.API_KEY }
  params: { limit: 50 }
}
spreadsheet = read "data.csv"
save results to "output.json"
records = query "SELECT * FROM users" on db.main
combined = merge [source_a, source_b]
```

### Pipelines (|> operator)
```drift
results = fetch "https://api.example.com/listings"
  |> filter where price < 500000 and beds >= 3
  |> sort by price ascending
  |> take 10
  |> ai.enrich("Add investment thesis")
  |> save to "deals.csv"
```

Pipeline stages: `filter where`, `sort by ... ascending|descending`, `take`, `skip`, `group by`, `deduplicate by`, `transform { |item| ... }`, `each { |item| ... }`, `ai.enrich(...)`, `ai.score(...)`, `save to`

### Error Handling
```drift
try:
  data = fetch api_endpoint
catch network_error:
  log "API unreachable"
catch ai_error:
  log "AI failed"
```

### Environment (secrets never in code)
```drift
api_key = env.RENTCAST_KEY
model = config.default_model
```

## Project Structure

```
drift/
├── drift/           # Core package
│   ├── lexer.py     # Tokenizer (INDENT/DEDENT tracking)
│   ├── parser.py    # Recursive descent parser
│   ├── ast_nodes.py # 48 dataclass AST node types
│   ├── transpiler.py # AST → Python code generator
│   ├── cli.py       # drift check/build/run
│   └── errors.py    # LexerError, ParseError, TranspileError
├── tests/           # 354 tests
├── examples/        # hello.drift, pipeline.drift, deal_analyzer.drift
└── docs/plans/      # Design docs
```

## CLI Usage

```bash
python3 -m drift.cli check file.drift   # Validate syntax
python3 -m drift.cli build file.drift   # Transpile to .py
python3 -m drift.cli run file.drift     # Transpile + execute (needs drift_runtime)
```

## Running Tests

```bash
cd /Users/ethansurfas/drift && python3 -m pytest -v
```

## Key Transpilation Rules

| Drift | Python |
|-------|--------|
| `"Hello {name}"` | `f"Hello {name}"` |
| `true` / `false` | `True` / `False` |
| `schema X:` | `@dataclass class X:` |
| `define f():` | `def f():` |
| `for each x in y:` | `for x in y:` |
| `else if` | `elif` |
| `ai.ask(...)` | `drift_runtime.ai.ask(...)` |
| `fetch url with {...}` | `drift_runtime.fetch(url, **kwargs)` |
| `env.API_KEY` | `os.environ["API_KEY"]` |
| `\|> filter where x > 5` | `[_item for _item in _pipe if _item["x"] > 5]` |
| `\|> sort by x desc` | `sorted(_pipe, key=lambda _item: _item["x"], reverse=True)` |
| `\|> take 10` | `_pipe[:10]` |

## Current Status

- **Phase 1 complete:** Lexer, parser, transpiler, CLI, 354 tests
- **Phase 3 next:** `drift_runtime` module (makes `drift run` work end-to-end)
- **Deferred:** Agent syntax (v2), retry/fallback error handling, REPL, VS Code extension
