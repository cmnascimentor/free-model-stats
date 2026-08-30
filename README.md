# FreeModelStats

FreeModelStats is a reproducible dashboard and benchmark runner for free-tier LLM endpoints. It currently tracks:

- **NVIDIA NIM** hosted chat models discovered from NVIDIA's live catalog and filtered through a verified chat-capable allowlist.
- **OpenRouter `:free`** models discovered dynamically from OpenRouter's catalog.
- **`openrouter/free` router behavior**, including which concrete model handled each request.

The project deliberately separates three questions that are often conflated in LLM dashboards:

1. **Availability** — did the provider/model transport succeed?
2. **Task quality** — did the answer satisfy a deterministic task rubric?
3. **Performance** — how long did the operation take and what end-to-end token throughput was observed?

The dashboard is static (`index.html` + SQLite via sql.js) and GitHub Pages compatible. The optional `runner_server.py` adds local live execution and WebSocket progress.

## Benchmark v2 methodology

Benchmark v2 (`2.0.0`) replaces the old "HTTP success + non-empty response" definition with explicit evidence fields:

```text
transport_success  provider/model returned successfully
format_success     response had the required broad format
 task_success      deterministic task validator passed
quality_score      deterministic 0-100 rubric score
```

The legacy `model_results.success` column is retained for dashboard compatibility, but for v2 rows it means **task-valid success**.

Every v2 run also records benchmark/probe versions, temperature, completion-token budget, HTTP status, finish reason, resolved model where available, retry count, final-attempt latency, total elapsed latency, and an end-to-end tokens/second measurement.

### Shared cross-provider baseline

The scheduled NIM and OpenRouter jobs now use the same default baseline:

```text
probe:                 hermes_triage
temperature:           0.0
max completion tokens: 512
benchmark version:     2.0.0
```

This fixes the previous methodological mismatch where NIM ran a prime-function prompt while OpenRouter ran a Web3 triage prompt and both were still placed on one leaderboard.

Additional probes can be run, but provider-specific or extra probes do **not** alter the global cross-provider ranking. The overall v2 leaderboard uses only the shared `hermes_triage` v2 baseline.

### Deterministic probe validators

Probe definitions and validators live in `scripts/common/probe_suite.py` and are shared by providers.

- `hermes_triage` — requires an allowed verdict, numeric confidence, and at least two concrete missing checks from the capsule.
- `hermes_evidence_summary` — requires the requested five evidence sections and a maximum of 180 words.
- `hermes_code_reasoning` — requires explicit treatment of denominator state, external-token behavior, burn-before-transfer behavior, and uncertainty/assumptions.
- `hermes_json_schema` — validates the **exact** requested keys, verdict enum, confidence range, list types, and rejects extra keys.
- `hermes_tool_probe` — validates the exact function name, exact argument keys, verdict enum, confidence range, and non-empty reason. It remains OpenRouter/provider-specific rather than part of the global score.

No LLM judge is used. That keeps the benchmark cheap, deterministic, and resistant to evaluator-model drift.

## Overall score

The deployed dashboard loads `vendor/benchmark-v2.js` after the legacy dashboard code. For models with v2 baseline evidence, the old cohort-relative min/max score is replaced with:

```text
Quality                50%
Transport reliability  25%
Latency                 15%
End-to-end throughput   10%
```

Latency and throughput use fixed logarithmic scales rather than min/max normalization against whatever models happen to be present. Adding one extreme outlier therefore cannot rescale every other model's score.

The score cell is marked `v2`; hovering it shows the component scores and sample count.

**Performance caveat:** current throughput is completion tokens divided by total operation duration. That duration includes provider overhead and, for OpenRouter, retry/backoff time where applicable. It is therefore an **end-to-end throughput proxy**, not pure post-TTFT generation speed. The schema already reserves `ttft_ms` and `generation_ms` for a future streaming measurement implementation.

## OpenRouter quota-aware rotation

Scheduled OpenRouter runs no longer select the alphabetically first N model IDs forever.

`test_models.py` orders active models by:

1. never benchmarked before;
2. oldest `last_benchmarked_at`;
3. lowest benchmark count;
4. model ID only as a stable tie-breaker.

The CI job downloads the previous canonical `history.db`, copies only OpenRouter scheduling state into the scratch database, discovers the current catalog, and then runs the least-recently-tested cohort. Historical benchmark rows are **not** copied into the scratch artifact, so ingestion cannot duplicate old runs.

The default remains 20 OpenRouter models per cycle to respect free-tier quota constraints.

## Circuit breaker

The circuit breaker remains transport-only:

- 3 consecutive HTTP/network failures for a `(model, probe)` pair trip it;
- tripped pairs are skipped for 7 days;
- validation/task failures never trip the breaker;
- **any transport success clears the model's breaker entries**, even if the answer fails the semantic validator.

CI now uses separate cache namespaces for NIM group 1, NIM group 2, and OpenRouter instead of having parallel jobs compete over one cache key.

## Dashboard

The existing no-build dashboard remains intact and provides:

- Overview KPIs and trends
- Leaderboard
- Per-model Explorer
- Timeline
- Head-to-head Compare
- OpenRouter capability metadata
- `openrouter/free` router behavior
- Manual/live benchmark controls

GitHub Pages deployment injects the small `vendor/benchmark-v2.js` adapter into the assembled site. The source `index.html` is not rewritten by CI.

## Local static dashboard

```bash
python3 -m http.server 8000
# http://localhost:8000
```

This is read-only. It loads the local `history.db` in the browser.

## Running benchmarks locally

```bash
export NIM_API_KEY=your_nim_key
export OPENROUTER_API_KEY=your_openrouter_key
./scripts/run_all.sh
```

Either key may be omitted; `run_all.sh` skips that provider.

Individual examples:

```bash
# Shared cross-provider baseline
python3 scripts/nim/test_models.py --probe hermes_triage
python3 scripts/openrouter/discover_models.py --mark-missing-inactive
python3 scripts/openrouter/test_models.py --probe hermes_triage

# Additional OpenRouter probes
python3 scripts/openrouter/test_models.py --probe hermes_code_reasoning
python3 scripts/openrouter/test_models.py --probe hermes_json_schema
python3 scripts/openrouter/test_models.py --probe hermes_tool_probe

# openrouter/free router behavior
python3 scripts/openrouter/test_router.py --probe hermes_triage --runs 2
```

### Offline smoke runs

No API key or network is required:

```bash
python3 scripts/nim/test_models.py --dry-run --probe hermes_triage
python3 scripts/openrouter/discover_models.py --dry-run
python3 scripts/openrouter/test_models.py --dry-run --probe hermes_triage --limit 3
python3 scripts/openrouter/test_router.py --dry-run --probe hermes_triage --runs 1
```

## Live runner

Install the optional runner dependencies:

```bash
pip install -r requirements-runner.txt
python3 runner_server.py
# http://127.0.0.1:8420
```

The runner:

- serves the dashboard itself;
- starts benchmark subprocesses without shell interpolation;
- serializes live benchmark runs to avoid SQLite write races;
- streams stdout over WebSocket;
- persists runner jobs and events in SQLite;
- replays stored events to reconnecting clients;
- keeps the most recent 100 runner jobs;
- supports stop/terminate from the browser.

### Runner authentication

By default the runner binds to `127.0.0.1`.

Set a secret when exposing it beyond loopback:

```bash
RUNNER_TOKEN='a-long-random-secret' python3 runner_server.py
```

When `RUNNER_TOKEN` is configured, visit the dashboard served by the runner (`http://host:8420/`). A local login page validates the token server-side and creates an **HttpOnly, SameSite=Strict session cookie**. The configured token is never embedded in the dashboard HTML or JavaScript.

Mutating/status endpoints accept that browser session or an explicit `X-Runner-Token` header. WebSockets accept the browser session cookie or the legacy `?token=` mechanism for non-browser clients.

If you expose the runner over an untrusted network, use a TLS reverse proxy. `history.db` itself remains a public/read-only runner endpoint by design.

## Data model

Everything remains in one SQLite database:

```sql
runs(
  id, timestamp, platform, run_kind, probe_name, prompt,
  success_count, total_models, fastest_model, fastest_time,
  benchmark_version, probe_version, temperature, max_completion_tokens
)

model_results(
  id, run_id, model, success, error, response_time,
  tokens_generated, total_tokens, response, validation_error,
  transport_success, format_success, task_success, quality_score,
  http_status, finish_reason, resolved_model, attempt_count,
  final_attempt_ms, total_elapsed_ms, ttft_ms, generation_ms,
  tokens_per_second, benchmark_version, probe_version
)

or_models(
  openrouter_id, name, context_length, max_completion_tokens,
  input_modalities, output_modalities, supported_parameters,
  pricing_prompt, pricing_completion, expiration_date, knowledge_cutoff,
  active, last_seen_at, last_benchmarked_at, benchmark_count
)

router_results(
  id, run_id, requested_model, resolved_model, probe_name,
  success, http_status, error_type, latency_ms, tokens_per_second,
  score, created_at, validation_error, attempt_count, total_elapsed_ms,
  benchmark_version, probe_version
)

runner_jobs(...)
runner_events(...)
```

Schema migration is additive. Existing `history.db` files are upgraded in place by `db.init_schema()`; legacy rows are backfilled with their historical `success` semantics so the existing history remains readable.

## Retry semantics

OpenRouter retries only transient HTTP `502`, `503`, and `504` responses. `429` is not retried because failed retries consume free-tier quota.

For retried operations the database distinguishes:

- `final_attempt_ms` — duration of the final HTTP attempt;
- `attempt_count` — total HTTP attempts;
- `total_elapsed_ms` / legacy `response_time` — user-observed duration including retries and backoff.

This avoids reporting a retried request as if only the final fast attempt had occurred.

## CI

There are now two distinct workflows.

### `.github/workflows/ci.yml`

Runs on pull requests and `master` pushes:

- Python 3.11 / 3.12 / 3.13 syntax checks;
- complete offline `unittest` suite;
- NIM dry-run benchmark smoke test;
- OpenRouter discovery/model/router dry-run smoke tests;
- optional FastAPI runner import and authentication tests.

### `.github/workflows/benchmark.yml`

Runs every six hours or manually:

- two parallel NIM groups on the same shared baseline probe;
- quota-rotated OpenRouter cohort;
- `openrouter/free` router probes;
- artifact merge into canonical history;
- SQLite `PRAGMA integrity_check` and orphan-row validation;
- GitHub Pages deployment.

The benchmark workflow remains serialized with a workflow-level concurrency group.

## Tests

Run locally:

```bash
python3 -m unittest discover -s tests -v
```

The suite covers discovery filtering, OpenRouter retry behavior, circuit-breaker behavior, schema migration/cascades, deterministic probe validation, v2 evidence semantics, OpenRouter least-recently-tested rotation, and optional runner session authentication.

## Repository layout

```text
free-model-stats/
  index.html
  runner_server.py
  prompts/
  scripts/
    common/
      db.py
      breaker.py
      probe_suite.py
    nim/
      test_models.py
      merge_results.py
    openrouter/
      discover_models.py
      seed_scheduler.py
      openrouter_client.py
      probes.py              # compatibility shim -> common/probe_suite.py
      test_models.py
      test_router.py
      export_scratch.py
    ingest_openrouter_artifact.py
    migrate_legacy.py
    run_all.sh
  tests/
  vendor/
    chart.umd.min.js
    sql-wasm.js
    sql-wasm.wasm
    benchmark-v2.js
  .github/workflows/
    ci.yml
    benchmark.yml
```

## Design principles

- Benchmark configuration must be versioned.
- Cross-provider rankings must compare the same task/configuration.
- Transport availability and answer correctness are different metrics.
- Provider-specific capability probes must not silently alter the global score.
- Validators should be deterministic when practical.
- Quota-limited providers need fair model rotation rather than alphabetical sampling.
- Retry cost and elapsed time are evidence, not implementation details to hide.
- The static dashboard should remain usable without Node/npm or a build tool.
