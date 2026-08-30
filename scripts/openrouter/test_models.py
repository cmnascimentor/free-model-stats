#!/usr/bin/env python3
"""Benchmark OpenRouter :free models with the shared FreeModelStats probe suite."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "common"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(COMMON_DIR))

import breaker  # noqa: E402
import db  # noqa: E402
from openrouter_client import OpenRouterClient  # noqa: E402
from probe_suite import (  # noqa: E402
    BENCHMARK_VERSION,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_PROBE,
    DEFAULT_TEMPERATURE,
    extract_text,
    load_probes,
    messages_for,
    quality_score,
    response_format_for,
    tools_for,
    usage_from,
    validate_probe_response,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def dry_response(probe_name: str) -> tuple[dict[str, Any], int]:
    if probe_name == "hermes_json_schema":
        content = json.dumps(
            {
                "verdict": "needs_more_evidence",
                "confidence": 0.82,
                "reasons": ["setter is privileged", "downstream arithmetic may revert"],
                "missing_evidence": ["governance reachability", "parameter bounds elsewhere"],
            }
        )
        message: dict[str, Any] = {"role": "assistant", "content": content}
    elif probe_name == "hermes_evidence_summary":
        content = (
            "Claim: a bad privileged fee update may cause swap denial of service.\n"
            "Supporting evidence: onlyOwner sets the fee; downstream arithmetic can revert.\n"
            "Assumptions: no other bounds or normalization exist.\n"
            "Missing evidence: no unprivileged setter path is confirmed.\n"
            "Next deterministic check: trace every fee setter and downstream bound."
        )
        message = {"role": "assistant", "content": content}
    elif probe_name == "hermes_code_reasoning":
        content = (
            "The denominator assumption is that totalSupply() is non-zero whenever previewRedeem is reached. "
            "External token behavior matters: token.transfer may return false, revert, charge fees, or reenter depending on the token. "
            "Burning before transfer is not automatically a loss because an EVM revert is atomic, but a non-reverting false-return token could matter. "
            "These assumptions need deterministic confirmation before calling this a vulnerability."
        )
        message = {"role": "assistant", "content": content}
    elif probe_name == "hermes_tool_probe":
        content = ""
        message = {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_model_probe_verdict",
                        "arguments": json.dumps(
                            {"verdict": "unclear", "confidence": 0.75, "reason": "missing deterministic evidence"}
                        ),
                    },
                }
            ],
        }
    else:
        content = (
            "Verdict: needs_more_evidence. Confidence: 0.84. "
            "Check whether accumulatedRewardDebt is reset during closePosition and whether claimReward is gated by active position status. "
            "Also test whether a repeated claim can succeed."
        )
        message = {"role": "assistant", "content": content}

    payload = {
        "model": "dry-run/sample-resolved-model",
        "choices": [{"finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 54, "total_tokens": 174},
    }
    return payload, 37


def get_active_models(conn, only: list[str] | None, limit: int | None) -> list[str]:
    """Choose active models using least-recently-benchmarked rotation.

    New models are tested first; after that, the oldest benchmark timestamp wins.
    Alphabetical ID is only a stable tie-breaker, removing the old LIMIT bias.
    """
    sql = "SELECT openrouter_id FROM or_models WHERE active = 1"
    args: list[Any] = []
    if only:
        placeholders = ",".join("?" for _ in only)
        sql += f" AND openrouter_id IN ({placeholders})"
        args.extend(only)
    sql += (
        " ORDER BY CASE WHEN last_benchmarked_at IS NULL THEN 0 ELSE 1 END, "
        "last_benchmarked_at ASC, benchmark_count ASC, openrouter_id ASC"
    )
    if limit:
        sql += " LIMIT ?"
        args.append(limit)
    return [row["openrouter_id"] for row in conn.execute(sql, args).fetchall()]


def compile_run(timestamp: str, prompt: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [r for r in results if r.get("taskSuccess")]
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


def _format_success(probe_name: str, text: str, tool_call_valid: bool | None) -> bool:
    if probe_name == "hermes_tool_probe":
        return tool_call_valid is True
    if probe_name == "hermes_json_schema":
        try:
            return isinstance(json.loads(text), dict)
        except (TypeError, json.JSONDecodeError):
            return False
    return bool(text.strip())


def _tokens_per_second(tokens: int | None, elapsed_ms: int | None) -> float | None:
    if not tokens or not elapsed_ms or elapsed_ms <= 0:
        return None
    return round(tokens / (elapsed_ms / 1000.0), 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OpenRouter free models")
    parser.add_argument("--db", default=str(db.HISTORY_DB), help="SQLite DB path")
    parser.add_argument("--model", action="append", help="Specific OpenRouter model ID to test. Can be repeated.")
    parser.add_argument("--limit", type=int, default=0, help="Limit active models using least-recently-tested rotation")
    parser.add_argument("--probe", action="append", help=f"Probe name from prompts/. Defaults to {DEFAULT_PROBE}.")
    parser.add_argument("--delay", type=float, default=float(os.getenv("OPENROUTER_STATS_DELAY", "3.5")))
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required unless --dry-run is used")

    db_path = Path(args.db)
    conn = db.connect(db_path)
    db.init_schema(conn)
    probes = load_probes(args.probe or [DEFAULT_PROBE])
    limit = args.limit if args.limit > 0 else None
    model_ids = get_active_models(conn, args.model, limit)
    conn.commit()
    conn.close()
    if not model_ids:
        raise SystemExit("No active or_models found. Run discover_models.py first.")

    client = OpenRouterClient(timeout=args.timeout)
    total_inserted = 0

    for probe in probes:
        run_ids = model_ids
        if not args.dry_run:
            runnable, skipped = breaker.tripped_models(db_path, model_ids, probe=probe.name)
            for model in skipped:
                print(f"Skipping: {model} (circuit breaker: repeated transport failures on {probe.name})")
            run_ids = runnable
        if not run_ids:
            print(f"probe={probe.name} no runnable models; skipping probe")
            continue

        results: list[dict[str, Any]] = []
        for idx, model_id in enumerate(run_ids):
            if idx > 0 and args.delay > 0 and not args.dry_run:
                time.sleep(args.delay)

            if args.dry_run:
                response_data, elapsed_ms = dry_response(probe.name)
                ok, status, err_msg = True, 200, None
                final_attempt_ms, attempt_count = elapsed_ms, 1
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
                final_attempt_ms = result.latency_ms
                elapsed_ms = result.total_elapsed_ms or result.latency_ms
                attempt_count = result.attempt_count

            text, resolved_model, finish_reason, tool_call_valid = extract_text(response_data)
            usage = usage_from(response_data)
            error_text = None if ok else (f"HTTP {status}: {err_msg}" if err_msg else f"HTTP {status}")
            validation_error = validate_probe_response(
                probe=probe,
                http_ok=ok,
                text=text,
                tool_call_valid=tool_call_valid,
            )
            task_success = bool(ok and validation_error is None)
            format_success = bool(ok and _format_success(probe.name, text, tool_call_valid))
            q_score = quality_score(probe=probe, http_ok=ok, text=text, tool_call_valid=tool_call_valid)

            if not args.dry_run:
                # The breaker models transport availability only. A semantically bad
                # answer still proves the endpoint/model is alive and clears it.
                if ok:
                    breaker.record_success(db_path, model_id)
                elif error_text:
                    breaker.record_failure(db_path, model_id, error_text, probe=probe.name)

            result_row = {
                "model": model_id,
                "success": task_success,
                "transportSuccess": ok,
                "formatSuccess": format_success,
                "taskSuccess": task_success,
                "qualityScore": q_score,
                "error": error_text,
                "validationError": validation_error,
                "responseTime": elapsed_ms,
                "finalAttemptMs": final_attempt_ms,
                "totalElapsedMs": elapsed_ms,
                "attemptCount": attempt_count,
                "httpStatus": status,
                "finishReason": finish_reason,
                "resolvedModel": resolved_model,
                "tokensGenerated": usage["completion_tokens"],
                "totalTokens": usage["total_tokens"],
                "tokensPerSecond": _tokens_per_second(usage["completion_tokens"], elapsed_ms),
                "response": text or None,
                "benchmarkVersion": BENCHMARK_VERSION,
                "probeVersion": probe.version,
            }
            results.append(result_row)
            total_inserted += 1

            if not args.dry_run:
                bench_conn = db.connect(db_path)
                db.init_schema(bench_conn)
                db.mark_or_model_benchmarked(bench_conn, model_id, utc_now())
                bench_conn.commit()
                bench_conn.close()

            detail = f" validation={validation_error}" if validation_error else ""
            print(
                f"[{total_inserted}] {model_id} probe={probe.name} transport={int(ok)} "
                f"task={int(task_success)} quality={q_score:.1f} elapsed_ms={elapsed_ms} attempts={attempt_count}{detail}",
                flush=True,
            )

        timestamp = utc_now()
        run = compile_run(timestamp, probe.prompt, results)
        run.update(
            {
                "probe_name": probe.name,
                "benchmark_version": BENCHMARK_VERSION,
                "probe_version": probe.version,
                "temperature": args.temperature,
                "max_completion_tokens": args.max_completion_tokens,
            }
        )
        run_id = db.write_run(run, db_path=db_path, platform="openrouter")
        print(
            f"probe={probe.name} run_id={run_id} task_success="
            f"{run['summary']['successCount']}/{run['summary']['totalModels']}"
        )

    print(f"model_results_inserted={total_inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
