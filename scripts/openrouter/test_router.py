#!/usr/bin/env python3
"""Benchmark OpenRouter's openrouter/free router behavior with shared probes."""
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

REQUESTED_MODEL = "openrouter/free"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dry_response(probe_name: str) -> tuple[dict[str, Any], int]:
    if probe_name == "hermes_json_schema":
        content = json.dumps(
            {
                "verdict": "needs_more_evidence",
                "confidence": 0.8,
                "reasons": ["privileged setter"],
                "missing_evidence": ["unprivileged reachability"],
            }
        )
        message: dict[str, Any] = {"role": "assistant", "content": content}
    elif probe_name == "hermes_tool_probe":
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_model_probe_verdict",
                        "arguments": json.dumps(
                            {"verdict": "unclear", "confidence": 0.8, "reason": "missing evidence"}
                        ),
                    },
                }
            ],
        }
    else:
        message = {
            "role": "assistant",
            "content": (
                "Verdict: needs_more_evidence. Confidence: 0.82. Check whether accumulatedRewardDebt is reset, "
                "whether active position status gates claiming, and whether repeated claims are possible."
            ),
        }
    return {
        "model": "qwen/qwen3-coder:free",
        "choices": [{"finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 115, "completion_tokens": 48, "total_tokens": 163},
    }, 42


def tokens_per_second(completion_tokens: int | None, latency_ms: int | None) -> float | None:
    if not completion_tokens or not latency_ms or latency_ms <= 0:
        return None
    return round(completion_tokens / (latency_ms / 1000.0), 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark openrouter/free router behavior")
    parser.add_argument("--db", default=str(db.HISTORY_DB), help="SQLite DB path")
    parser.add_argument("--probe", action="append", help=f"Probe name from prompts/. Defaults to {DEFAULT_PROBE}.")
    parser.add_argument("--runs", type=int, default=3, help="Number of router calls per probe")
    parser.add_argument("--delay", type=float, default=float(os.getenv("OPENROUTER_STATS_DELAY", "3.5")))
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required unless --dry-run is used")

    conn = db.connect(Path(args.db))
    db.init_schema(conn)
    probes = load_probes(args.probe or [DEFAULT_PROBE])
    client = OpenRouterClient(timeout=args.timeout)

    total = 0
    provider_failure: str | None = None
    for probe in probes:
        run_id: int | None = None

        for _ in range(args.runs):
            if total > 0 and args.delay > 0 and not args.dry_run:
                time.sleep(args.delay)

            if args.dry_run:
                response_data, elapsed_ms = dry_response(probe.name)
                ok, status, err_type, err_msg, attempts = True, 200, None, None, 1
            else:
                tools, tool_choice = tools_for(probe)
                result = client.chat_completion(
                    model=REQUESTED_MODEL,
                    messages=messages_for(probe),
                    temperature=args.temperature,
                    max_completion_tokens=args.max_completion_tokens,
                    response_format=response_format_for(probe),
                    tools=tools,
                    tool_choice=tool_choice,
                )
                ok, status, response_data = result.ok, result.status, result.data
                err_type, err_msg = result.error_type, result.error_message
                elapsed_ms = result.total_elapsed_ms or result.latency_ms
                attempts = result.attempt_count

            error_text = None if ok else (f"HTTP {status}: {err_msg}" if err_msg else f"HTTP {status}: {err_type}")
            if not args.dry_run and error_text and breaker.is_provider_scoped_failure(error_text):
                provider_failure = error_text
                print(
                    f"Provider-scoped OpenRouter router failure detected; stopping without recording a bad router sample: {provider_failure}",
                    file=sys.stderr,
                    flush=True,
                )
                break

            text, resolved_model, _finish_reason, tool_call_valid = extract_text(response_data)
            usage = usage_from(response_data)
            validation_error = validate_probe_response(
                probe=probe,
                http_ok=ok,
                text=text,
                tool_call_valid=tool_call_valid,
            )
            task_success = bool(ok and validation_error is None)
            score = quality_score(probe=probe, http_ok=ok, text=text, tool_call_valid=tool_call_valid)

            if run_id is None:
                run_id = db.create_router_run(
                    conn,
                    timestamp=utc_now(),
                    probe_name=probe.name,
                    prompt=probe.prompt,
                    benchmark_version=BENCHMARK_VERSION,
                    probe_version=probe.version,
                    temperature=args.temperature,
                    max_completion_tokens=args.max_completion_tokens,
                )
                conn.commit()

            db.insert_router_result(
                conn,
                run_id,
                {
                    "requested_model": REQUESTED_MODEL,
                    "resolved_model": resolved_model,
                    "probe_name": probe.name,
                    "success": int(task_success),
                    "http_status": status,
                    "error_type": err_type,
                    "latency_ms": elapsed_ms,
                    "tokens_per_second": tokens_per_second(usage["completion_tokens"], elapsed_ms),
                    "score": score,
                    "created_at": utc_now(),
                    "validation_error": validation_error,
                    "attempt_count": attempts,
                    "total_elapsed_ms": elapsed_ms,
                    "benchmark_version": BENCHMARK_VERSION,
                    "probe_version": probe.version,
                },
            )
            conn.commit()
            total += 1
            print(
                f"[{total}] requested={REQUESTED_MODEL} resolved={resolved_model} probe={probe.name} "
                f"transport={int(ok)} task={int(task_success)} elapsed_ms={elapsed_ms} attempts={attempts} quality={score:.1f}"
            )

        if provider_failure:
            break

    conn.close()
    print(f"router_results_inserted={total}")
    if provider_failure:
        print(f"openrouter_router_provider_error={provider_failure}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
