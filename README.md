# Drift

> **Warning** Drift is v0.1.0 — early and evolving. The compiler and runtime work, but expect rough edges.

A programming language anyone can understand.

An AI-native programming language that transpiles to Python. Drift treats AI as a first-class primitive — no imports, no SDKs, no boilerplate. Just write `ai.ask(...)` and it works.

```drift
-- Analyze a property deal in 5 lines
comps = fetch "https://api.rentcast.io/v1/comps" with {
  headers: { "X-Api-Key": env.RENTCAST_KEY }
  params: { address: "742 Evergreen Terrace", radius: 0.5 }
}

score = ai.ask("Analyze this investment property") -> DealScore using {
  comparable_sales: comps
}

if score.roi > 15:
  print "Hot deal: {score.verdict}"
```

## Quick Start

```bash
# Install
git clone https://github.com/ethansurfas/drift.git
cd drift
pip install -e .

# Run a program
drift run examples/hello.drift
# => Hello from Drift!

# Check syntax without running
drift check examples/hello.drift

# Transpile to Python
drift build examples/hello.drift
```

For AI features, set your API key:

```bash
export ANTHROPIC_API_KEY=your-key-here
# or
export OPENAI_API_KEY=your-key-here
```

## Language Overview

### Variables and Types

```drift
city = "Austin"
price = 450000
active = true
tags = ["investment", "fix"]
```

Types: `string`, `number`, `boolean`, `list`, `map`, `date`, `none`

### String Interpolation

```drift
print "Hello {name}!"
print "ROI: {score.roi}%"
```

### Schemas

```drift
schema DealScore:
  address: string
  arv: number
  roi: number
  verdict: string
  photos: list of string
  data: map (optional)
```

### Functions

```drift
define analyze(address: string, budget: number) -> DealScore:
  comps = fetch "https://api.example.com/comps" with {
    params: { address: address }
  }
  return ai.ask("Analyze this deal") -> DealScore using { comps: comps }
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

### AI Primitives

No imports needed. AI is built into the language.

```drift
-- Ask a question
summary = ai.ask("Summarize: {text}")

-- Get structured output
analysis = ai.ask("Analyze this") -> DealScore using { data: input }

-- Classify text
category = ai.classify(email.body, into: ["urgent", "routine", "spam"])

-- Generate embeddings
vector = ai.embed(document.text)

-- Analyze images
description = ai.see(photo, "Describe the property condition")

-- Make predictions with confidence
estimate = ai.predict("Estimate ARV") -> confident number
if estimate > 300000:
  print "High value property"
```

### Data Operations

```drift
-- HTTP requests
data = fetch "https://api.example.com" with {
  headers: { "X-Key": env.API_KEY }
  params: { limit: 50 }
}

-- File I/O
spreadsheet = read "data.csv"
save results to "output.json"

-- SQL queries
records = query "SELECT * FROM users" on db.main

-- Merge datasets
combined = merge [source_a, source_b]
```

### Pipelines

Chain operations with the `|>` operator:

```drift
results = fetch "https://api.example.com/listings"
  |> filter where price < 500000 and beds >= 3
  |> sort by price ascending
  |> take 10
  |> ai.enrich("Add investment thesis")
  |> save to "deals.csv"
```

### Error Handling

```drift
try:
  data = fetch api_endpoint
catch network_error:
  log "API unreachable"
catch ai_error:
  log "AI call failed"
```

### Environment Variables

Secrets stay out of code:

```drift
api_key = env.RENTCAST_KEY
```

## Configuration

Create a `drift.config` file in your project root (YAML):

```yaml
ai:
  provider: anthropic       # or "openai"
  default_model: claude-sonnet-4-5-20250929
  max_retries: 3
  timeout: 30
```

If no config file exists, Drift uses sensible defaults (Anthropic, Claude Sonnet, 2 retries, 30s timeout).

## How It Works

Drift programs transpile to Python:

| Drift | Python |
|-------|--------|
| `"Hello {name}"` | `f"Hello {name}"` |
| `true` / `false` | `True` / `False` |
| `schema X:` | `@dataclass class X:` |
| `define f():` | `def f():` |
| `for each x in y:` | `for x in y:` |
| `ai.ask(...)` | `drift_runtime.ai.ask(...)` |
| `fetch url` | `drift_runtime.fetch(url)` |
| `env.API_KEY` | `os.environ["API_KEY"]` |
| `\|> filter where x > 5` | list comprehension |
| `\|> sort by x desc` | `sorted(...)` |
| `catch network_error:` | `except DriftNetworkError:` |

## Project Structure

```
drift/
├── drift/              # Compiler (lexer, parser, transpiler, CLI)
├── drift_runtime/      # Runtime (AI, data I/O, config, pipeline helpers)
├── tests/              # 447+ tests
├── examples/           # hello.drift, pipeline.drift, deal_analyzer.drift
└── docs/plans/         # Design documents
```

## Running Tests

```bash
python3 -m pytest -v
```

## Requirements

- Python 3.11+
- For AI features: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

## Story

Built with the help of Claude Code. Drift was designed and spec'd by a human who doesn't write code — and built by AI. That's the whole point.

## License

MIT
