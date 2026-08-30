#!/usr/bin/env python3
"""Benchmark pinned OpenRouter :free models, one run per probe (NIM-shaped: one prompt vs N models).

Adapted from openrouter-free-stats' scripts/test_models.py to write into the
unified history.db (runs/model_results) instead of its own richer per-probe
schema, so both benchmark suites share one dashboard.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "common"))

import db  # noqa: E402
import breaker  # noqa: E402
from openrouter_client import OpenRouterClient  # noqa: E402
from probes import (  # noqa: E402
    extract_text,
    load_probes,
    messages_for,
    response_format_for,
    tools_for,
    usage_from,
    validate_probe_response,
)
from sample_data import SAMPLE_JSON, SAMPLE_TEXT  # noqa: E402


def dry_response(probe_name: str) -> tuple[dict[str, Any], int]:
    content = SAMPLE_JSON if "json" in probe_name else SAMPLE_TEXT
    payload = {
        "model": "dry-run/sample-resolved-model",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 54, "total_tokens": 174},
    }
    return payload, 37


def get_active_models(conn, only: list[str] | None, limit: int | None) -> list[str]:
    sql = "SELECT openrouter_id FROM or_models WHERE active = 1"
    args: list[Any] = []
    if only:
        placeholders = ",".join("?" for _ in only)
        sql += f" AND openrouter_id IN ({placeholders})"
        args.extend(only)
    sql += " ORDER BY openrouter_id"
    if limit:
        sql += " LIMIT ?"
        args.append(limit)
    return [row["openrouter_id"] for row in conn.execute(sql, args).fetchall()]


def compile_run(timestamp: str, prompt: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [r for r in results if r.get("success")]
    if successful:
        fastest = min(successful, key=lambda r: r.get("responseTime") or float("inf"))
        fastest_model, fastest_time = fastest.get("model", "N/A"), fastest.get("responseTime", 0) or 0
    else:
        fastest_model, fastest_time = "N/A", 0
    return {
        "timestamp": timestamp,
        "prompt": prompt,
        "models": results,
        "summary": {
            "successCount": len(successful),
            "totalModels": len(results),
            "fastestModel": fastest_model,
            "fastestTime": fastest_time,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark pinned OpenRouter free models")
    parser.add_argument("--db", default=str(db.HISTORY_DB), help="SQLite DB path")
    parser.add_argument("--model", action="append", help="Specific OpenRouter model ID to test. Can be repeated.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of active models tested")
    parser.add_argument("--probe", action="append", help="Probe name from prompts/. Can be repeated. Defaults to hermes_triage.")
    parser.add_argument("--delay", type=float, default=float(os.getenv("OPENROUTER_STATS_DELAY", "3.5")), help="Delay between requests")
    parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout seconds")
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Do not call OpenRouter; insert deterministic sample results")
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required unless --dry-run is used")

    conn = db.connect(Path(args.db))
    db.init_schema(conn)

    probes = load_probes(args.probe or ["hermes_triage"])
    limit = args.limit if args.limit > 0 else None
    model_ids = get_active_models(conn, args.model, limit)
    conn.close()
    if not model_ids:
        raise SystemExit("No active or_models found. Run discover_models.py first, or use --dry-run discovery.")

    client = OpenRouterClient(timeout=args.timeout)
    total_inserted = 0

    for probe in probes:
        # Per-probe breaker filter: a model tripped for probe A must still run
        # probe B (keys are "model::probe"; legacy model-level keys skip all).
        run_ids = model_ids
        if not args.dry_run:
            runnable, skipped = breaker.tripped_models(Path(args.db), model_ids, probe=probe.name)
            for model in skipped:
                print(f"Skipping: {model} (circuit breaker: repeated failures on {probe.name})")
            run_ids = runnable
        if not run_ids:
            raise SystemExit("All candidate models are circuit-broken; nothing to benchmark this cycle.")

        results: list[dict[str, Any]] = []
        for idx, model_id in enumerate(run_ids):
            if idx > 0 and args.delay > 0 and not args.dry_run:
                time.sleep(args.delay)
            if args.dry_run:
                response_data, latency_ms = dry_response(probe.name)
                ok, status, err_msg = True, 200, None
            else:
                tools, tool_choice = tools_for(probe)
                result = client.chat_completion(
                    model=model_id,
                    messages=messages_for(probe),
                    temperature=args.temperature,
                    max_completion_tokens=args.max_completion_tokens,
                    response_format=response_format_for(probe),
                    tools=tools,
                    tool_choice=tool_choice,
                )
                ok, status, response_data, err_msg = result.ok, result.status, result.data, result.error_message
                latency_ms = result.latency_ms

            text, _resolved_model, _finish_reason, tool_call_valid = extract_text(response_data)
            usage = usage_from(response_data)

            error_text = None
            if not ok:
                error_text = f"HTTP {status}: {err_msg}" if err_msg else f"HTTP {status}"
            validation_error = validate_probe_response(
                probe=probe,
                http_ok=ok,
                text=text,
                tool_call_valid=tool_call_valid,
            )
            success = bool(ok and validation_error is None)

            if not args.dry_run:
                if success:
                    breaker.record_success(Path(args.db), model_id)
                elif error_text:  # transport/HTTP failure only — validation-only misfires never trip
                    breaker.record_failure(Path(args.db), model_id, error_text, probe=probe.name)

            results.append(
                {
                    "model": model_id,
                    "success": success,
                    "error": error_text,
                    "validationError": validation_error,
                    "responseTime": latency_ms,
                    "tokensGenerated": usage["completion_tokens"],
                    "totalTokens": usage["total_tokens"],
                    "response": text or None,
                }
            )
            total_inserted += 1
            detail = f" validation={validation_error}" if validation_error else ""
            print(f"[{total_inserted}] {model_id} probe={probe.name} success={int(success)} latency_ms={latency_ms}{detail}")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        run = compile_run(timestamp, probe.prompt, results)
        run["probe_name"] = probe.name
        run_id = db.write_run(run, db_path=Path(args.db), platform="openrouter")
        print(f"probe={probe.name} run_id={run_id} success={run['summary']['successCount']}/{run['summary']['totalModels']}")

    print(f"model_results_inserted={total_inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
