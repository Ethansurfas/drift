# Phase 3: Example Programs — Implementation Spec

## Overview

Four real-world Drift programs that demonstrate the language's power. Each one must:

1. Fit on a single screen (under 30 lines of Drift code)
2. Use a real, free API (no authentication required, OR use an API key the user provides via `env`)
3. Make at least one real `ai.ask` or `ai.classify` call
4. Produce visible, impressive output in the terminal
5. Actually run end-to-end with `drift run examples/<name>.drift`

These examples are the demo. They need to work flawlessly and look incredible.

**Dependencies:** Phase 1 (compiler) and Phase 2 (runtime) must be complete and passing.

---

## Example 1: news_analyzer.drift

**What it does:** Fetches real news headlines, classifies each by sentiment and topic, outputs a formatted summary.

**Free API:** https://newsdata.io/api/1/latest (free tier: 200 requests/day, requires free API key from newsdata.io)

**Alternative if newsdata is unreliable:** Use a hardcoded list of 5-6 recent headlines as fallback so the demo always works even without an API key.

### Drift Code

```drift
-- News Headline Analyzer
-- Fetches today's headlines and analyzes sentiment + topic

schema HeadlineAnalysis:
  title: string
  sentiment: string
  confidence: number
  topic: string
  summary: string

-- Fetch headlines (uses free newsdata.io API, or fallback to sample data)
headlines = fetch "https://newsdata.io/api/1/latest" with {
  params: { apikey: env.NEWSDATA_KEY, language: "en", size: 5 }
}

results = headlines.results
  |> ai.enrich("For each headline, analyze sentiment (positive/negative/neutral with confidence 0-1) and categorize the topic (politics/tech/business/sports/science/other). Return as JSON with fields: sentiment, confidence, topic, summary")
  |> take 5

print "============================="
print "  NEWS SENTIMENT ANALYZER"
print "============================="
print ""

for each article in results:
  print "Headline:   {article.title}"
  print "Sentiment:  {article.sentiment} ({article.confidence})"
  print "Topic:      {article.topic}"
  print "Summary:    {article.summary}"
  print "-----------------------------"

save results to "news_analysis.json"
print ""
print "Full analysis saved to news_analysis.json"
```

### Fallback Version (no API key needed)

Also create `news_analyzer_offline.drift` that uses hardcoded headlines:

```drift
-- News Headline Analyzer (Offline Demo)
-- Analyzes sample headlines with AI

headlines = [
  { "title": "Federal Reserve Holds Interest Rates Steady Amid Inflation Concerns" },
  { "title": "SpaceX Successfully Launches Starship on Sixth Test Flight" },
  { "title": "Apple Announces Record Q1 Revenue Driven by AI Features" },
  { "title": "Scientists Discover New Species in Deep Ocean Trench" },
  { "title": "NBA Playoffs: Knicks Advance to Eastern Conference Finals" }
]

results = headlines
  |> ai.enrich("For each headline, add: sentiment (positive/negative/neutral), confidence (0.0-1.0), topic (politics/tech/business/sports/science), and a one-sentence summary")

print "============================="
print "  NEWS SENTIMENT ANALYZER"
print "============================="
print ""

for each article in results:
  print "Headline:   {article.title}"
  print "Sentiment:  {article.sentiment} ({article.confidence})"
  print "Topic:      {article.topic}"
  print "Summary:    {article.summary}"
  print "-----------------------------"

save results to "news_analysis.json"
print "Saved to news_analysis.json"
```

### Implementation Notes
- The offline version is the safe demo — always works with just an Anthropic key
- The online version shows `fetch` working with a real API
- Both should produce identical output format
- `ai.enrich` is doing the heavy lifting — it takes raw data and adds AI-generated fields

---

## Example 2: budget.drift

**What it does:** Reads a CSV of transactions, uses AI to categorize each one, calculates spending by category, gives a personalized budget summary.

**No external API needed.** Just a CSV file and AI. This is powerful because it shows Drift doing something immediately useful for a normal person.

### Sample Data File: examples/transactions.csv

```csv
date,description,amount
2026-01-15,Starbucks Coffee,6.50
2026-01-15,Uber Ride to Office,24.00
2026-01-16,Whole Foods Market,87.32
2026-01-16,Netflix Subscription,15.99
2026-01-17,Shell Gas Station,45.00
2026-01-17,Amazon - Wireless Mouse,29.99
2026-01-18,Chipotle Mexican Grill,12.50
2026-01-18,Planet Fitness,25.00
2026-01-19,Con Edison Electric Bill,142.00
2026-01-19,Venmo - Jake (dinner split),35.00
2026-01-20,Apple App Store,4.99
2026-01-20,Target - Household Items,63.45
2026-01-21,Spotify Premium,10.99
2026-01-21,CVS Pharmacy,18.75
2026-01-22,MTA MetroCard Refill,33.00
```

### Drift Code

```drift
-- Personal Budget Categorizer
-- Reads transactions and uses AI to categorize spending

schema CategorySummary:
  category: string
  total: number
  count: number
  percentage: number

transactions = read "examples/transactions.csv"

categorized = transactions
  |> ai.enrich("Categorize each transaction into exactly one of: Food & Dining, Transportation, Shopping, Entertainment, Health & Fitness, Utilities, Subscriptions. Add a 'category' field.")

print "================================"
print "  BUDGET ANALYZER"
print "================================"
print ""

for each t in categorized:
  print "{t.description}: ${t.amount} -> [{t.category}]"

print ""
print "--------------------------------"

summary = ai.ask("Analyze these categorized transactions and provide: 1) Total spending 2) Spending by category with percentages 3) The single biggest expense 4) One specific suggestion to save money. Be concise and use actual numbers from the data.") using {
  transactions: categorized
}

print summary

save categorized to "budget_categorized.json"
print ""
print "Detailed breakdown saved to budget_categorized.json"
```

### Implementation Notes
- `read "examples/transactions.csv"` returns a list of dicts — this is the Phase 2 CSV reader
- `ai.enrich` adds a `category` field to each transaction
- The final `ai.ask` with `using` passes all categorized data as context for the summary
- Output should be visually clean and immediately useful
- This is the most "normal person" example — anyone with a bank statement gets it instantly

---

## Example 3: summarizer.drift

**What it does:** Takes a URL, fetches the page content, and uses AI to generate a summary, key points, and reading time estimate.

**No external API needed beyond the target URL.** Just `fetch` and `ai.ask`.

### Drift Code

```drift
-- Website Content Summarizer
-- Give it a URL, get a smart summary

schema PageSummary:
  title: string
  summary: string
  key_points: list of string
  word_count: number
  reading_time: string
  tone: string

url = "https://paulgraham.com/writes.html"

print "Fetching: {url}"
print ""

content = fetch url

analysis = ai.ask("Analyze this web page content. Provide a clear summary, 3-5 key points, estimate the word count, calculate reading time, and describe the overall tone.") -> PageSummary using {
  page_content: content
}

print "================================"
print "  PAGE SUMMARY"
print "================================"
print ""
print "Title:        {analysis.title}"
print "Reading Time: {analysis.reading_time}"
print "Tone:         {analysis.tone}"
print "Word Count:   ~{analysis.word_count}"
print ""
print "SUMMARY:"
print "{analysis.summary}"
print ""
print "KEY POINTS:"
for each point in analysis.key_points:
  print "  - {point}"

save analysis to "page_summary.json"
print ""
print "Full analysis saved to page_summary.json"
```

### Implementation Notes
- Uses Paul Graham's essay as default — short, text-heavy, publicly accessible, no auth required
- `fetch url` needs to handle HTML content — the runtime should return the text content, stripping HTML tags if possible. If drift_runtime.fetch doesn't strip HTML yet, add a simple tag stripper or return raw HTML and let the AI handle it (LLMs are good at reading through HTML)
- The `-> PageSummary` structured output is the star here — showing that AI returns typed data, not just strings
- `list of string` for key_points tests that the schema system handles lists properly
- Users can easily swap in any URL they want

### Potential Issue
- Some websites block non-browser requests. Paul Graham's site is simple and should work. If not, fall back to a known-accessible URL or include a `summarizer_offline.drift` with hardcoded content.

---

## Example 4: deal_analyzer.drift (Enhanced)

**What it does:** The flagship example. A real estate deal analyzer that takes an address, fetches comparable sales, runs AI analysis, and produces an investment recommendation with confidence scores.

**API:** This version should work with OR without a RentCast API key. With the key, it fetches real comps. Without it, it uses sample data and still demonstrates the full pipeline.

### Sample Data File: examples/sample_comps.json

```json
[
  {
    "address": "738 Evergreen Terrace",
    "price": 285000,
    "sqft": 1450,
    "beds": 3,
    "baths": 2,
    "year_built": 1998,
    "days_on_market": 12
  },
  {
    "address": "751 Maple Drive",
    "price": 312000,
    "sqft": 1620,
    "beds": 3,
    "baths": 2.5,
    "year_built": 2004,
    "days_on_market": 28
  },
  {
    "address": "804 Oak Avenue",
    "price": 265000,
    "sqft": 1380,
    "beds": 3,
    "baths": 1.5,
    "year_built": 1992,
    "days_on_market": 45
  },
  {
    "address": "689 Pine Street",
    "price": 299000,
    "sqft": 1510,
    "beds": 4,
    "baths": 2,
    "year_built": 2001,
    "days_on_market": 8
  },
  {
    "address": "720 Birch Lane",
    "price": 275000,
    "sqft": 1400,
    "beds": 3,
    "baths": 2,
    "year_built": 1996,
    "days_on_market": 33
  }
]
```

### Drift Code

```drift
-- Real Estate Deal Analyzer
-- Analyze a property using comparable sales and AI

schema DealScore:
  address: string
  estimated_arv: number
  repair_estimate: number
  max_offer: number
  roi_percent: number
  risk_level: string
  verdict: string
  reasoning: string

-- Target property
target = {
  address: "742 Evergreen Terrace",
  asking_price: 250000,
  sqft: 1450,
  beds: 3,
  baths: 2,
  condition: "Needs kitchen and bathroom renovation"
}

-- Load comparable sales
comps = read "examples/sample_comps.json"

print "================================"
print "  DEAL ANALYZER"
print "================================"
print ""
print "Target: {target.address}"
print "Asking: ${target.asking_price}"
print "Condition: {target.condition}"
print ""
print "Analyzing {comps} comparable sales..."
print ""

-- Run AI analysis
score = ai.ask("You are an experienced real estate investor. Analyze this potential flip deal. The target property details and comparable recent sales are provided. Calculate: 1) Estimated After-Repair Value (ARV) based on comps 2) Estimated repair costs given the condition 3) Maximum offer price using the 70% rule (MAO = ARV * 0.70 - repairs) 4) Expected ROI if purchased at asking price 5) Risk level (low/medium/high) 6) Clear verdict (strong buy / buy / pass / hard pass) 7) Brief reasoning. Use the actual numbers from the comps to justify your ARV estimate.") -> DealScore using {
  target_property: target,
  comparable_sales: comps
}

print "RESULTS:"
print "--------------------------------"
print "Estimated ARV:     ${score.estimated_arv}"
print "Repair Estimate:   ${score.repair_estimate}"
print "Max Offer (70%):   ${score.max_offer}"
print "Expected ROI:      {score.roi_percent}%"
print "Risk Level:        {score.risk_level}"
print ""
print "VERDICT: {score.verdict}"
print ""
print "REASONING:"
print "{score.reasoning}"
print "--------------------------------"

save score to "deal_score.json"
print ""
print "Full analysis saved to deal_score.json"
```

### Implementation Notes
- This is YOUR example. It's the one that tells your story — real estate guy builds a language, the first killer demo is a deal analyzer
- Uses `read` for the sample data (no API key needed for the demo)
- The prompt is long and specific — that's intentional. It shows that Drift can handle real, complex AI prompts
- `-> DealScore` structured output means the AI returns typed data, not a wall of text
- The 70% rule is real — investors actually use this. Shows domain expertise in the example
- Users who have a RentCast key could swap `read` for `fetch` to get real comps

---

## Implementation Tasks

### Task 1: Create sample data files
- Create `examples/transactions.csv` with the budget data
- Create `examples/sample_comps.json` with the real estate comps
- Verify both files parse correctly with `drift_runtime.read()`

### Task 2: Write news_analyzer_offline.drift
- Create the offline version first (no external API needed)
- Test with `drift run examples/news_analyzer_offline.drift`
- Verify `ai.enrich` works on a list of dicts
- Verify output formatting looks clean

### Task 3: Write budget.drift
- Create the program
- Test with `drift run examples/budget.drift`
- Verify CSV reading works
- Verify `ai.enrich` adds categories
- Verify `ai.ask` with `using` context works
- Verify `save` outputs valid JSON

### Task 4: Write summarizer.drift
- Create the program
- Test with `drift run examples/summarizer.drift`
- Verify `fetch` handles HTML content
- Verify `-> PageSummary` schema parsing works with `list of string` field
- If HTML stripping is needed, add a basic utility to drift_runtime.data.fetch
- Test with paulgraham.com, fall back to another URL if blocked

### Task 5: Write deal_analyzer.drift
- Create the program
- Test with `drift run examples/deal_analyzer.drift`
- Verify JSON reading works for comps
- Verify complex schema with many fields parses correctly
- Verify long prompts work without truncation

### Task 6: Write news_analyzer.drift (online version)
- Create the API version
- Test with a free newsdata.io key if available
- If API is unreliable, document in README and point users to offline version

### Task 7: Fix any runtime issues discovered
- During testing, any runtime bugs will surface. Fix them.
- Common issues to watch for:
  - `ai.enrich` may not handle list-of-dicts properly
  - Schema parsing may fail on complex nested types
  - `fetch` may need HTML handling for the summarizer
  - String interpolation with nested dict access (`{t.category}`) may need transpiler fixes
  - `save` may not handle dataclass-to-dict conversion for schema outputs
  - Map literals in Drift (`{ key: value }`) may need transpiler verification

### Task 8: Update README with new examples
- Add a "Examples" section showing each program with expected output
- Add instructions for running each example
- Note which ones need API keys and which work out of the box

### Task 9: End-to-end smoke test
- Run all four examples in sequence
- Verify all produce clean output
- Verify all save their output files correctly
- Fix any remaining issues

---

## Success Criteria

Phase 3 is complete when:

1. `drift run examples/news_analyzer_offline.drift` → fetches/analyzes headlines, prints formatted results
2. `drift run examples/budget.drift` → categorizes transactions, prints summary with totals
3. `drift run examples/summarizer.drift` → fetches a web page, prints structured summary
4. `drift run examples/deal_analyzer.drift` → analyzes deal with comps, prints investment recommendation
5. All four save output to JSON files
6. All four run without errors (given ANTHROPIC_API_KEY is set)
7. All four produce output that looks impressive in a terminal screenshot
8. README is updated with example descriptions and instructions
9. Zero regressions in Phase 1 and Phase 2 tests

---

## Important Notes for Claude Code

1. **These programs will be the first thing people run.** They need to work perfectly. Test each one multiple times.

2. **The output formatting matters.** Clean terminal output with clear headers and spacing. This is what gets screenshotted and shared.

3. **If a runtime feature doesn't work as expected** (e.g., `ai.enrich` on a list, schema parsing with `list of string`, dict access in string interpolation), fix the runtime. Don't hack around it in the example code. The examples should be idiomatic Drift.

4. **Long AI prompts are intentional.** Don't shorten them. The specificity is what makes the AI output good. Drift should handle multi-line strings naturally.

5. **The deal_analyzer.drift is the signature example.** Give it extra attention. It should feel like a real tool, not a toy demo.
