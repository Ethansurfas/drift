# DRIFT — Language Design Document v0.1

## "The language AI writes. The language that writes AI."

---

## 1. What is Drift?

Drift is an open-source, AI-native programming language designed from the ground up to be written by AI and to orchestrate AI. It transpiles to Python under the hood, giving it access to the entire Python ecosystem while presenting a radically simpler, intent-driven syntax.

Drift exists because every programming language in use today was designed for humans to write by hand. But the world has changed. AI generates most new code. Drift is the first language that embraces this reality — optimized for AI fluency, human readability, and AI-powered execution.

**Core belief:** The developer of the future describes *what* they want. The language handles *how*.

---

## 2. Design Philosophy

### 2.1 Five Principles

1. **AI-first authorship** — The syntax is designed so that LLMs (Claude, GPT, etc.) can write flawless Drift programs with minimal context. Every construct is unambiguous, consistent, and predictable.

2. **Human-readable, not human-written** — A non-programmer should be able to read any Drift program and understand what it does. But they never need to write it by hand — they describe what they want, and AI writes the Drift.

3. **Intent over implementation** — Drift programs describe outcomes, not steps. The runtime figures out the optimal execution path.

4. **AI is a primitive, not a plugin** — Model inference, embeddings, classification, and generation are built into the language syntax. No imports, no API keys in code, no boilerplate.

5. **Pipelines are the backbone** — Data flows through Drift programs like water through pipes: fetch → transform → enrich → output. This is the core mental model.

### 2.2 What Drift Is NOT

- Not a no-code tool — it produces real, executable code
- Not a prompt wrapper — it has deterministic logic, types, and control flow
- Not a toy language — it transpiles to production Python
- Not framework-specific — it's general purpose with AI superpowers

---

## 3. Syntax Specification

### 3.1 Programs and Entry Points

Every Drift program is a `.drift` file. Execution starts at the top and flows downward. No `main()` function required.

```drift
-- This is a comment
-- Drift programs read like a story

name = "Drift"
print "Hello from {name}!"
```

### 3.2 Variables and Types

Drift uses type inference. You never need to declare types, but you can.

```drift
city = "Austin"              -- string (inferred)
price = 450000               -- number (inferred)
active = true                -- boolean (inferred)
tags = ["investment", "fix"] -- list (inferred)

-- Explicit typing (optional)
score: number = 92.5
```

**Built-in types:** `string`, `number`, `boolean`, `list`, `map`, `date`, `none`

**AI-native types (unique to Drift):**
```drift
-- Confidence-wrapped values carry certainty metadata
estimate: confident number = ai.predict("ARV for 123 Main St")
-- estimate.value = 385000
-- estimate.confidence = 0.82

-- Schema types define structured AI outputs
schema DealScore:
  address: string
  arv: number
  rehab_cost: number
  roi: number
  recommendation: string
```

### 3.3 AI Inference (The Core Primitive)

AI calls are native syntax — no imports, no setup, no API configuration.

```drift
-- Simple inference (returns string)
summary = ai.ask("Summarize this article: {article_text}")

-- Structured inference (returns typed object)
analysis = ai.ask("Analyze this property deal") -> DealScore using {
  address: property.address
  comps: recent_sales
  photos: property.images
}

-- Classification
category = ai.classify(email.body, into: ["urgent", "routine", "spam"])

-- Embedding
vector = ai.embed(document.text)

-- Image analysis
description = ai.see(photo, "Describe the condition of this property")
```

**How it works under the hood:**
- The Drift runtime manages model selection, API keys, retries, and caching
- Configuration lives in a `drift.config` file, not in code
- Model provider is swappable without changing any code

### 3.4 Pipelines

Pipelines are Drift's signature feature. The `|>` operator chains operations.

```drift
-- Fetch → Transform → Output
results = fetch "https://api.example.com/listings"
  |> filter where price < 500000 and beds >= 3
  |> sort by price ascending
  |> take 10
  |> ai.enrich("Add a one-line investment thesis for each property")
  |> save to "top_deals.csv"
```

```drift
-- Multi-source pipeline
deals = merge [mls_listings, auction_data, fsbo_leads]
  |> deduplicate by address
  |> ai.score("Rate investment potential 1-100") -> number
  |> filter where score > 75
  |> group by zip_code
  |> each { |group|
    report = ai.ask("Write a market summary for {group.key}") 
    save report to "reports/{group.key}.md"
  }
```

### 3.5 Data Fetching

```drift
-- HTTP requests
data = fetch "https://api.rentcast.io/v1/properties" with {
  headers: { "X-Api-Key": env.RENTCAST_KEY }
  params: { zipcode: "78701", limit: 50 }
}

-- File reading
spreadsheet = read "deals.csv"
document = read "report.pdf"

-- Database queries
records = query "SELECT * FROM properties WHERE city = {city}" on db.main
```

### 3.6 Control Flow

```drift
-- Conditionals
if deal.roi > 20%:
  flag deal as "hot"
else if deal.roi > 10%:
  flag deal as "moderate"
else:
  skip deal

-- Loops
for each property in listings:
  score = ai.ask("Rate this deal") -> DealScore using property
  save score to results

-- Pattern matching
match response.status:
  200 -> process response.body
  404 -> log "Not found: {response.url}"
  _   -> log "Error: {response.status}"
```

### 3.7 Functions

```drift
define analyze_deal(address: string, budget: number) -> DealScore:
  comps = fetch comps_api with { address: address, radius: "0.5mi" }
  arv = ai.predict("Estimate ARV from these comps: {comps}") -> confident number
  rehab = ai.see(photos, "Estimate rehab cost") -> confident number
  roi = (arv.value - budget - rehab.value) / budget * 100
  
  return DealScore {
    address: address
    arv: arv.value
    rehab_cost: rehab.value
    roi: roi
    recommendation: ai.ask("Should I buy? ROI: {roi}%, Confidence: {arv.confidence}")
  }
```

### 3.8 Schemas (Structured Data)

```drift
schema Property:
  address: string
  price: number
  beds: number
  baths: number
  sqft: number
  year_built: number
  photos: list of string

schema Report:
  title: string
  sections: list of Section
  
schema Section:
  heading: string
  body: string
  data: map (optional)
```

### 3.9 Error Handling

```drift
try:
  data = fetch api_endpoint
  result = ai.ask("Process this: {data}") -> OutputSchema
catch network_error:
  log "API unreachable, retrying..."
  retry after 5 seconds, max 3 times
catch ai_error:
  log "AI inference failed: {error.message}"
  fallback = default_value
```

### 3.10 Environment and Configuration

Code never contains secrets. Everything is configured externally.

```drift
-- Access environment variables
api_key = env.RENTCAST_KEY
db_url = env.DATABASE_URL

-- Access Drift config
model = config.default_model    -- e.g., "claude-sonnet-4-5-20250929"
```

**drift.config (project-level settings):**
```yaml
name: my-project
version: 0.1.0

ai:
  provider: anthropic
  default_model: claude-sonnet-4-5-20250929
  fallback_model: claude-haiku-4-5-20251001
  cache: true
  max_retries: 3

data:
  output_dir: ./output
  
secrets:
  source: env  # or "vault", "aws-secrets", etc.
```

---

## 4. Agents (v2 — Future)

Agents are long-running autonomous workflows. They will be introduced after the core language is stable.

```drift
-- PREVIEW: Not in v1
agent DealScout:
  watches: [MLS_feed, auction_feed]
  every: 1 hour
  
  when new_listing matches (price < 500000 and beds >= 3):
    score = analyze_deal(listing.address, listing.price)
    if score.roi > 20%:
      notify user with score
      generate_memo(score, template: "investment_committee")
```

---

## 5. Standard Library

### 5.1 Core Modules

| Module       | Purpose                              | Example                                      |
|-------------|--------------------------------------|----------------------------------------------|
| `ai`        | Inference, classification, embedding | `ai.ask(...)`, `ai.classify(...)`, `ai.embed(...)` |
| `fetch`     | HTTP requests                        | `fetch url with { headers, params }`          |
| `read`      | File I/O                             | `read "data.csv"`, `read "report.pdf"`        |
| `save`      | Output files                         | `save data to "output.json"`                  |
| `query`     | Database access                      | `query "SELECT ..." on db.main`               |
| `log`       | Logging and debugging                | `log "Processing {item}"`                     |
| `env`       | Environment variables                | `env.API_KEY`                                 |
| `config`    | Project configuration                | `config.default_model`                        |
| `date`      | Date and time                        | `date.today`, `date.parse("2025-01-01")`      |
| `math`      | Mathematical operations              | `math.round(x, 2)`, `math.percent(a, b)`     |

### 5.2 Pipeline Operators

| Operator       | Purpose                  | Example                                  |
|---------------|--------------------------|------------------------------------------|
| `filter`      | Keep matching items      | `filter where price < 500000`            |
| `sort`        | Order items              | `sort by score descending`               |
| `take`        | Limit results            | `take 10`                                |
| `skip`        | Offset results           | `skip 5`                                 |
| `group`       | Group by field           | `group by city`                          |
| `merge`       | Combine sources          | `merge [source_a, source_b]`             |
| `deduplicate` | Remove duplicates        | `deduplicate by address`                 |
| `transform`   | Map/modify items         | `transform { |item| item.price * 1.1 }`  |
| `each`        | Iterate with side effects| `each { |item| save item to file }`      |
| `ai.enrich`   | AI-augment each item     | `ai.enrich("Add summary for each item")` |
| `ai.score`    | AI-rate each item        | `ai.score("Rate quality 1-100")`         |

---

## 6. Transpilation

Drift compiles to standard Python 3.11+. Every Drift program produces a readable `.py` file.

**Drift source:**
```drift
listings = fetch "https://api.example.com/homes" with {
  params: { city: "Austin", max_price: 500000 }
}
  |> filter where beds >= 3
  |> ai.enrich("Write a one-line investment thesis")
  |> save to "deals.json"
```

**Generated Python:**
```python
import drift_runtime as drift

listings = drift.fetch(
    "https://api.example.com/homes",
    params={"city": "Austin", "max_price": 500000}
)
listings = [item for item in listings if item["beds"] >= 3]
listings = drift.ai.enrich(
    listings,
    prompt="Write a one-line investment thesis"
)
drift.save(listings, "deals.json")
```

---

## 7. Toolchain

| Tool              | Command             | Purpose                          |
|-------------------|----------------------|----------------------------------|
| Run               | `drift run file.drift`      | Execute a Drift program     |
| Compile           | `drift build file.drift`    | Transpile to Python         |
| REPL              | `drift shell`               | Interactive Drift console   |
| Init              | `drift init my-project`     | Create new project scaffold |
| Check             | `drift check file.drift`    | Validate syntax without running |
| Format            | `drift fmt file.drift`      | Auto-format Drift code      |

---

## 8. Why "Drift"?

- **Data flows** drift through pipelines
- **AI outputs** drift between certainty and uncertainty (confidence scores)
- **The name** is short, memorable, searchable, and doesn't conflict with existing tools
- **The vibe** — effortless, fluid, natural

---

## 9. Roadmap

| Phase | Milestone                                    | Status    |
|-------|----------------------------------------------|-----------|
| 0     | Language spec and design doc                 | ✅ Done    |
| 1     | Lexer + Parser (tokenize and parse .drift)   | 🔲 Next   |
| 2     | Transpiler (Drift → Python)                  | 🔲        |
| 3     | Runtime library (drift_runtime)              | 🔲        |
| 4     | CLI toolchain (drift run, build, shell)      | 🔲        |
| 5     | Claude Code integration (CLAUDE.md spec)     | 🔲        |
| 6     | 20+ example programs                         | 🔲        |
| 7     | Public launch (GitHub, docs site)            | 🔲        |
| 8     | Agent syntax (v2)                            | 🔲        |
| 9     | VS Code extension / LSP                      | 🔲        |

---

## 10. Example Programs

### 10.1 — Hello World
```drift
print "Hello, Drift!"
```

### 10.2 — AI-Powered Deal Analyzer
```drift
schema DealScore:
  address: string
  arv: number
  rehab_cost: number
  monthly_rent: number
  roi: number
  verdict: string

address = "742 Evergreen Terrace, Springfield"

comps = fetch "https://api.rentcast.io/v1/comps" with {
  headers: { "X-Api-Key": env.RENTCAST_KEY }
  params: { address: address, radius: 0.5 }
}

score = ai.ask("Analyze this investment property") -> DealScore using {
  address: address
  purchase_price: 285000
  comparable_sales: comps
}

if score.roi > 15%:
  print "🔥 Hot deal: {score.verdict}"
  save score to "deals/{address}.json"
else:
  print "Pass: {score.verdict}"
```

### 10.3 — Bulk Content Generator
```drift
topics = read "blog_topics.csv"

articles = topics
  |> each { |topic|
    draft = ai.ask("Write a 500-word blog post about: {topic.title}")
    
    return {
      title: topic.title
      body: draft
      seo_keywords: ai.ask("List 5 SEO keywords for: {topic.title}") -> list of string
      published: date.today
    }
  }
  |> save to "articles.json"

print "Generated {articles.length} articles"
```

### 10.4 — Data Pipeline with AI Enrichment
```drift
schema Lead:
  name: string
  email: string
  company: string
  score: number
  reason: string

raw_leads = fetch "https://crm.example.com/api/leads" with {
  headers: { "Authorization": "Bearer {env.CRM_TOKEN}" }
}

qualified = raw_leads
  |> filter where status == "new"
  |> ai.score("Rate lead quality 1-100 based on company size and industry") -> number
  |> filter where score > 60
  |> sort by score descending
  |> take 20
  |> ai.enrich("Write a personalized outreach reason for each lead")
  |> save to "qualified_leads.csv"

print "Found {qualified.length} qualified leads"
```

### 10.5 — Report Generator
```drift
schema MarketReport:
  title: string
  executive_summary: string
  sections: list of Section

schema Section:
  heading: string
  body: string

data = query "SELECT city, avg_price, inventory, days_on_market 
              FROM market_stats 
              WHERE quarter = 'Q1-2026'" on db.analytics

report = ai.ask("Generate a real estate market report for Q1 2026") -> MarketReport using {
  market_data: data
  format: "professional, data-driven, with specific numbers"
}

save report to "Q1_2026_Market_Report.md"
print "Report generated: {report.title}"
```

---

## License

Drift is open source under the MIT License.

---

*This is a living document. Version 0.1 — February 2026.*
*Created by Ethan. Built with AI, for AI, about AI.*
