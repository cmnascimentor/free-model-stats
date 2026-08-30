#!/usr/bin/env python3
"""CI merge step: ingest an OpenRouter scratch-db artifact into canonical history.db."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "common"))
import db  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest OpenRouter scratch artifact")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--db", default=str(db.HISTORY_DB))
    args = parser.parse_args()

    payload = json.loads(Path(args.in_path).read_text(encoding="utf-8"))
    conn = db.connect(Path(args.db))
    db.init_schema(conn)

    for model in payload.get("or_models", []):
        db.upsert_or_model(
            conn,
            {
                key: model.get(key)
                for key in (
                    "openrouter_id", "name", "context_length", "max_completion_tokens",
                    "input_modalities", "output_modalities", "supported_parameters",
                    "pricing_prompt", "pricing_completion", "expiration_date", "knowledge_cutoff",
                    "active", "last_seen_at",
                )
            },
        )
        conn.execute(
            """UPDATE or_models
               SET last_benchmarked_at = ?, benchmark_count = ?
               WHERE openrouter_id = ?""",
            (
                model.get("last_benchmarked_at"),
                model.get("benchmark_count", 0) or 0,
                model.get("openrouter_id"),
            ),
        )
    conn.commit()

    run_id_map: dict[int, int] = {}
    for run in payload.get("runs", []):
        cur = conn.execute(
            """INSERT INTO runs
               (timestamp, platform, run_kind, probe_name, prompt, success_count, total_models,
                fastest_model, fastest_time, benchmark_version, probe_version, temperature,
                max_completion_tokens)
               VALUES (?, 'openrouter', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.get("timestamp"),
                run.get("run_kind") or "benchmark",
                run.get("probe_name"),
                run.get("prompt"),
                run.get("success_count"),
                run.get("total_models"),
                run.get("fastest_model"),
                run.get("fastest_time"),
                run.get("benchmark_version"),
                run.get("probe_version"),
                run.get("temperature"),
                run.get("max_completion_tokens"),
            ),
        )
        run_id_map[int(run["id"])] = int(cur.lastrowid)
    conn.commit()

    inserted_results = 0
    for result in payload.get("model_results", []):
        new_run_id = run_id_map.get(int(result["run_id"]))
        if new_run_id is None:
            continue
        conn.execute(
            """INSERT INTO model_results
               (run_id, model, success, error, response_time, tokens_generated, total_tokens,
                response, validation_error, transport_success, format_success, task_success,
                quality_score, http_status, finish_reason, resolved_model, attempt_count,
                final_attempt_ms, total_elapsed_ms, ttft_ms, generation_ms, tokens_per_second,
                benchmark_version, probe_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                new_run_id,
                result.get("model"),
                result.get("success"),
                result.get("error"),
                result.get("response_time"),
                result.get("tokens_generated"),
                result.get("total_tokens"),
                db.cap_response(result.get("response")),
                result.get("validation_error"),
                result.get("transport_success"),
                result.get("format_success"),
                result.get("task_success"),
                result.get("quality_score"),
                result.get("http_status"),
                result.get("finish_reason"),
                result.get("resolved_model"),
                result.get("attempt_count"),
                result.get("final_attempt_ms"),
                result.get("total_elapsed_ms"),
                result.get("ttft_ms"),
                result.get("generation_ms"),
                result.get("tokens_per_second"),
                result.get("benchmark_version"),
                result.get("probe_version"),
            ),
        )
        inserted_results += 1
    conn.commit()

    inserted_router = 0
    for result in payload.get("router_results", []):
        new_run_id = run_id_map.get(int(result["run_id"]))
        if new_run_id is None:
            continue
        db.insert_router_result(
            conn,
            new_run_id,
            {
                "requested_model": result.get("requested_model"),
                "resolved_model": result.get("resolved_model"),
                "probe_name": result.get("probe_name"),
                "success": result.get("success"),
                "http_status": result.get("http_status"),
                "error_type": result.get("error_type"),
                "latency_ms": result.get("latency_ms"),
                "tokens_per_second": result.get("tokens_per_second"),
                "score": result.get("score"),
                "created_at": result.get("created_at"),
                "validation_error": result.get("validation_error"),
                "attempt_count": result.get("attempt_count"),
                "total_elapsed_ms": result.get("total_elapsed_ms"),
                "benchmark_version": result.get("benchmark_version"),
                "probe_version": result.get("probe_version"),
            },
        )
        inserted_router += 1
    conn.commit()

    conn.execute(
        f"DELETE FROM runs WHERE id NOT IN (SELECT id FROM runs ORDER BY timestamp DESC LIMIT {db.MAX_RUNS})"
    )
    conn.commit()
    conn.close()

    print(
        f"ingested runs={len(run_id_map)} model_results={inserted_results} "
        f"or_models={len(payload.get('or_models', []))} router_results={inserted_router}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
