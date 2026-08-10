#!/usr/bin/env python3
"""Benchmark OpenRouter's openrouter/free router behavior into the unified history.db."""
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
from openrouter_client import OpenRouterClient  # noqa: E402
from probes import (  # noqa: E402
    extract_text,
    is_valid_json_object,
    load_probes,
    messages_for,
    response_format_for,
    score_response,
    tools_for,
    usage_from,
)
from sample_data import SAMPLE_JSON, SAMPLE_TEXT  # noqa: E402

REQUESTED_MODEL = "openrouter/free"


def dry_response(probe_name: str) -> tuple[dict[str, Any], int]:
    content = SAMPLE_JSON if "json" in probe_name else SAMPLE_TEXT
    payload = {
        "model": "qwen/qwen3-coder:free",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 115, "completion_tokens": 48, "total_tokens": 163},
    }
    return payload, 42


def tokens_per_second(completion_tokens: int | None, latency_ms: int | None) -> float | None:
    if not completion_tokens or not latency_ms or latency_ms <= 0:
        return None
    return round(completion_tokens / (latency_ms / 1000.0), 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark openrouter/free router behavior")
    parser.add_argument("--db", default=str(db.HISTORY_DB), help="SQLite DB path")
    parser.add_argument("--probe", action="append", help="Probe name from prompts/. Can be repeated. Defaults to hermes_triage.")
    parser.add_argument("--runs", type=int, default=3, help="Number of router calls per probe")
    parser.add_argument("--delay", type=float, default=float(os.getenv("OPENROUTER_STATS_DELAY", "3.5")))
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required unless --dry-run is used")

    conn = db.connect(Path(args.db))
    db.init_schema(conn)
    probes = load_probes(args.probe or ["hermes_triage"])
    client = OpenRouterClient(timeout=args.timeout)

    total = 0
    for probe in probes:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        run_id = db.create_router_run(conn, timestamp=timestamp, probe_name=probe.name, prompt=probe.prompt)
        conn.commit()

        for _ in range(args.runs):
            if total > 0 and args.delay > 0 and not args.dry_run:
                time.sleep(args.delay)
            if args.dry_run:
                response_data, latency_ms = dry_response(probe.name)
                ok, status = True, 200
                err_type = None
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
                ok, status, response_data, err_type = result.ok, result.status, result.data, result.error_type
                latency_ms = result.latency_ms

            text, resolved_model, _finish_reason, tool_call_valid = extract_text(response_data)
            usage = usage_from(response_data)
            schema_valid = is_valid_json_object(text) if probe.expects_json and text else None
            score = score_response(success=ok, text=text, expects_json=probe.expects_json, schema_valid=schema_valid, tool_call_valid=tool_call_valid)

            db.insert_router_result(
                conn,
                run_id,
                {
                    "requested_model": REQUESTED_MODEL,
                    "resolved_model": resolved_model,
                    "probe_name": probe.name,
                    "success": int(ok),
                    "http_status": status,
                    "error_type": err_type,
                    "latency_ms": latency_ms,
                    "tokens_per_second": tokens_per_second(usage["completion_tokens"], latency_ms),
                    "score": score,
                    "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                },
            )
            conn.commit()
            total += 1
            print(f"[{total}] requested={REQUESTED_MODEL} resolved={resolved_model} probe={probe.name} success={int(ok)} latency_ms={latency_ms} score={score}")

    print(f"router_results_inserted={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
