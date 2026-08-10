# FreeModelStats

One dashboard for both free-tier LLM benchmark suites that used to live in separate projects:

- **NVIDIA NIM** — every chat-capable model dynamically discovered from NVIDIA's API catalog (`GET /v1/models`, all of them by default; `NIM_MODEL_LIMIT` is an opt-in cap), benchmarked hourly-ish with a single fixed coding prompt.
- **OpenRouter `:free`** — dynamically discovered `:free`-suffixed OpenRouter models, benchmarked with a small set of Hermes-style probes, plus separate tracking of how the `openrouter/free` auto-router resolves.

This project merges both into one static dashboard (`index.html`, no build step, no server — SQLite-in-the-browser via [sql.js](https://sql.js.org/)) and one benchmark schedule, so you don't have to run or check two separate tools.

It was assembled from two prior projects (`openrouter-free-stats` and `NIMStats`), starting from NIMStats' tab structure (Overview/Leaderboard/Explorer/Timeline/Compare) and porting over OpenRouter's better features (dynamic model discovery, capability/pricing metadata, router-behavior tracking, CSV export, extra filters), then re-skinned with its own visual identity — a warm copper/teal palette, Manrope/IBM Plex Mono type, and a left sidebar nav — instead of reusing NIMStats' look.

## Dashboard tabs

| Tab | What you get |
|-----|---------------|
| 📊 **Overview** | KPI cards, success trend charts, top-10 speed/throughput bars, model reliability pills, aggregate error breakdown, runs-by-platform split |
| 🏆 **Leaderboard** | Composite score ranking (uptime 40% + speed 30% + throughput 30%) across **both** platforms, sortable columns, sparklines, min-uptime/min-runs filters, CSV export |
| 🔬 **Explorer** | Per-model deep dive — response time history, error breakdown donut, availability heatmap, full response viewer |
| ⏱ **Timeline** | Filterable run history (All / 24h / 48h / 7d), expandable run cards with per-model detail, platform + probe badge per run |
| ⚔️ **Compare** | Head-to-head overlay chart, win-rate stats, side-by-side metrics — works across platforms (e.g. NIM's Qwen3 Coder vs OpenRouter's) |
| 🧩 **Capabilities** | Context length, max completion tokens, modalities, supported params, free pricing — **OpenRouter-only** for now (NIM has no equivalent catalog API) |
| 🔀 **Router** | Resolved-model distribution + recent calls for `openrouter/free` — **OpenRouter-only** |

A platform toggle (**All / NIM / OpenRouter**) in the nav bar filters Overview, Leaderboard, Timeline, Explorer, and Compare to one source at a time.

## Data model

Everything lives in one `history.db`, read entirely client-side:

```sql
runs          (id, timestamp, platform, probe_name, prompt, success_count, total_models, fastest_model, fastest_time)
model_results (id, run_id, model, success, error, response_time, tokens_generated, total_tokens, response)
or_models     (openrouter_id, name, context_length, max_completion_tokens, input_modalities, output_modalities,
               supported_parameters, pricing_prompt, pricing_completion, expiration_date, knowledge_cutoff, active, last_seen_at)
router_results(id, run_id, requested_model, resolved_model, probe_name, success, http_status, error_type,
               latency_ms, tokens_per_second, score, created_at)
```

`platform` is `'nim'` or `'openrouter'`. Model IDs already disambiguate platform on their own (e.g. `qwen/qwen3-coder:free` vs `qwen/qwen3-coder-480b-a35b-instruct`), and `model.split('/')[0]` gives the creator (Qwen, Meta, NVIDIA...) for provider-colored chips — `platform` is only needed to separate "which backend ran this" from "who made the model."

The unified leaderboard score is NIM's original composite formula, applied the same way regardless of platform — that's what makes "best free model overall" possible. OpenRouter's separate probe-correctness scoring isn't part of ranking (the semantics don't reconcile with NIM's uptime/speed model), but full response text is still stored and viewable in the Explorer/Timeline response modal.

## Local quick start

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

`history.db` in this repo was seeded once via `scripts/migrate_legacy.py` from both legacy projects' history, so charts aren't empty on day one.

## Running benchmarks locally

```bash
export NIM_API_KEY=your_nim_key
export OPENROUTER_API_KEY=your_openrouter_key
./scripts/run_all.sh
```

Either key can be omitted — `run_all.sh` skips whichever half you don't have a key for. This is also what the cron job (see below) runs every 6 hours.

Individual scripts:

```bash
# NIM (fixed prompt, dynamically discovered chat models, one run)
python3 scripts/nim/test_models.py

# OpenRouter (dynamic discovery + multi-probe + router tracking)
python3 scripts/openrouter/discover_models.py --mark-missing-inactive
python3 scripts/openrouter/test_models.py --probe hermes_triage
python3 scripts/openrouter/test_router.py --probe hermes_triage --runs 2

# Offline dry run (no API key or network needed) for either half
python3 scripts/nim/test_models.py --dry-run
python3 scripts/openrouter/discover_models.py --dry-run
python3 scripts/openrouter/test_models.py --dry-run --probe hermes_triage
python3 scripts/openrouter/test_router.py --dry-run --runs 2
```

## Tests

Offline, stdlib-only `unittest` coverage for the model discovery/filtering logic (no API keys or network needed):

```bash
python3 -m unittest discover -s tests -v
```

## GitHub Actions

Runs every 6 hours (`0 */6 * * *`) or on manual dispatch. Add these repository secrets:

```text
NIM_API_KEY
OPENROUTER_API_KEY
```

Four jobs: `nim_group1` + `nim_group2` (parallel, ~50% faster NIM benchmarks, same as before), `openrouter_benchmark` (discovery + probes + router, writes to a scratch db so it can't race the NIM jobs), then `merge_and_update` merges all three and uploads `history.db` as a CI artifact. A final `deploy_pages` job publishes `index.html` + `history.db` to GitHub Pages — no git commits required.

## Model discovery

Both halves discover their currently-available free models at run time instead of relying on a stale hardcoded list, with a safe fallback if the live catalog can't be reached:

- **OpenRouter** — `scripts/openrouter/discover_models.py` calls `GET /models` on the OpenRouter API and keeps entries whose ID ends in `:free` and whose `prompt`/`completion`/`request` pricing is all zero (`or_models` table, `active` flag). If the fetch fails, the script exits before writing anything, so the previously discovered `or_models` rows are left untouched rather than being wiped.
- **NIM** — `scripts/nim/test_models.py` calls `GET /v1/models` on NVIDIA's API catalog (no separate free/paid tier exists there — the hosted catalog itself is the free tier) and keeps only IDs present in a curated **chat-model allowlist** (`CHAT_MODEL_ALLOWLIST` in `scripts/nim/test_models.py`). The allowlist was built by probing every catalog entry with a minimal `max_tokens=1` chat/completions POST and keeping only those returning HTTP 200 — NVIDIA's catalog lists many models that 404 on chat/completions (base/completion-only, vision-only, embedding, deprecated, etc.) and the allowlist eliminates them upfront. Every surviving model is benchmarked by default (split evenly across the `group1`/`group2` CI jobs); `NIM_MODEL_LIMIT` is an opt-in cap — unset or `0` means "all discovered," any positive integer truncates the sorted list to that many models. If discovery fails, returns no `data` list, or no entries match the allowlist, it falls back to a pinned snapshot list (`FALLBACK_MODELS` in `scripts/nim/test_models.py`) so a benchmark run can still proceed.

## Quota policy (OpenRouter half)

OpenRouter's free plan lists 50 requests/day and 20 requests/minute (1000/day at $10+ credits, same RPM cap). Defaults stay conservative: concurrency 1, 3.5s delay between live requests, no retries, default probe `hermes_triage` only. Failed attempts still count against quota, so failures are recorded as signal instead of retried automatically.

## Files

```text
free-model-stats/
  index.html                     # dashboard (sql.js + Chart.js, no build step)
  history.db                     # SQLite dashboard data — CI uploads it as an artifact and Pages deploys it (not committed as source of truth)
  prompts/                       # OpenRouter probe prompts
  scripts/
    common/db.py                 # unified schema + write helpers
    nim/test_models.py           # NIM benchmark: dynamic /v1/models catalog discovery + fixed prompt
    nim/merge_results.py         # CI: merges parallel NIM group artifacts
    openrouter/discover_models.py  # dynamic :free model discovery -> or_models
    openrouter/test_models.py      # multi-probe benchmark -> runs/model_results
    openrouter/test_router.py      # openrouter/free router tracking -> router_results
    openrouter/export_scratch.py   # CI: dumps a scratch db to a JSON artifact
    ingest_openrouter_artifact.py  # CI: merges that artifact into the real history.db
    migrate_legacy.py            # one-off backfill from the two legacy projects
    run_all.sh                   # local cron entry point
  tests/                          # offline unittest coverage for model discovery/filtering
  .github/workflows/benchmark.yml
```

## Provenance

This project's `history.db` was seeded from:
- `~/Documents/NIMStats` (720 runs / ~14.4k model results)
- `~/Documents/openrouter-free-stats` (9 runs / 153 model results, 5 router calls, 24 model capability records)

Both original projects are left untouched on disk; nothing here writes back to them.
