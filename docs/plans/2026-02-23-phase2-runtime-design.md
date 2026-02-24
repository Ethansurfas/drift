# Phase 2 Design: drift_runtime

**Date:** 2026-02-23
**Status:** Approved

---

## Overview

Build the `drift_runtime` Python package that makes transpiled Drift programs execute. After Phase 2, `drift run examples/hello.drift` produces real output, AI calls hit real LLMs, and pipelines process real data.

## Architecture

Single Python package `drift_runtime/` alongside `drift/`. Seven modules:

| Module | Responsibility |
|--------|---------------|
| `ai.py` | DriftAI class with ask, classify, embed, see, predict, enrich, score |
| `data.py` | fetch (httpx), read, save, query (SQLite), merge |
| `config.py` | Load drift.config YAML, cache globally, _reset_config() for tests |
| `types.py` | ConfidentValue, schema_to_json_description, parse_ai_response_to_schema |
| `pipeline.py` | deduplicate, group_by helpers |
| `exceptions.py` | DriftRuntimeError, DriftAIError, DriftNetworkError, DriftFileError, DriftConfigError |
| `__init__.py` | Wire everything, expose drift_runtime.ai, drift_runtime.fetch, etc. |

Dependencies: `anthropic`, `openai`, `httpx`, `pyyaml`.

## AI Module

- `DriftAI` class with `_call_model(messages, model=None)` as single dispatch point
- Provider selection via `config.ai.provider` — `"anthropic"` or `"openai"`
- Anthropic: `anthropic.Anthropic().messages.create()`
- OpenAI: `openai.OpenAI().chat.completions.create()`
- API keys from env: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
- Retries per `config.ai.max_retries`, timeout per `config.ai.timeout`
- Schema responses: append JSON schema instruction to system prompt, parse response with fence stripping
- `enrich` and `score` operate on lists (pipeline usage), per-item or batched

## Data Module

- `fetch(url, headers=None, params=None)` — httpx.get, retry on 429/5xx, parse JSON if content-type matches
- `read(path)` — CSV→list[dict], JSON→parsed, text/md→str, FileNotFoundError if missing
- `save(data, path)` — auto-create dirs, JSON/CSV/text by extension, dataclass→dict conversion
- `query(sql, source)` — SQLite only for v1, return list[dict]
- `merge(sources)` — concatenate lists

## Config Module

- Load `drift.config` YAML from cwd on first `get_config()` call
- Cache in module-level `_config` global
- `_reset_config()` clears cache (critical for tests)
- Deep merge user config into defaults
- Defaults: anthropic provider, claude-sonnet-4-5, 2 retries, 30s timeout

## Types Module

- `ConfidentValue(value, confidence)` — supports `>`, `<` comparisons against numbers
- `parse_ai_response_to_schema(response, schema_class)` — strips markdown code fences, parses JSON, constructs dataclass
- `schema_to_json_description(cls)` — introspects dataclass fields for AI prompting

## Transpiler Fixups

- `catch network_error:` → `except drift_runtime.DriftNetworkError:`
- `catch ai_error:` → `except drift_runtime.DriftAIError:`
- `log "msg"` → `drift_runtime.log(...)`
- Pipeline `deduplicate by field` → `drift_runtime.deduplicate(data, "field")`
- Pipeline `group by field` → `drift_runtime.group_by(data, "field")`

## Testing Strategy

- All AI tests mock `DriftAI._call_model` via `unittest.mock.patch`
- File I/O tests use `tempfile`
- Fetch tests mock `httpx.get`
- Config tests call `_reset_config()` before each test
- E2E tests gated behind `requires_api_key = pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), ...)`

## Warnings

1. AI response parsing must handle markdown fences, extra whitespace, schema mismatches
2. Never make real API calls in tests unless `requires_api_key` decorated
3. Transpiler output must match runtime function signatures exactly (Task 10)
4. Config caching pollutes tests without `_reset_config()`
