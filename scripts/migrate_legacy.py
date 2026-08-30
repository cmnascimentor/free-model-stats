#!/usr/bin/env python3
"""One-off backfill: import both legacy projects' history.db files into the new
merged project's history.db, so the dashboard has real historical charts from
day one instead of starting empty.

Read-only on both legacy databases -- nothing in openrouter-free-stats/ or
NIMStats/ is ever modified.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"

import sys  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR / "common"))
import db  # noqa: E402


def read_only_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def probe_prompt_text(probe_name: str | None) -> str | None:
    if not probe_name:
        return None
    path = PROMPTS_DIR / f"{probe_name}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def migrate_nim(nim_db: Path, out_conn: sqlite3.Connection) -> tuple[int, int]:
    if not nim_db.exists():
        print(f"skip NIM migration: {nim_db} not found")
        return 0, 0
    src = read_only_connect(nim_db)
    runs = src.execute(
        "SELECT id, timestamp, prompt, success_count, total_models, fastest_model, fastest_time FROM runs ORDER BY id ASC"
    ).fetchall()
    run_count = 0
    result_count = 0
    for run in runs:
        cur = out_conn.execute(
            """INSERT INTO runs (timestamp, platform, run_kind, probe_name, prompt, success_count, total_models, fastest_model, fastest_time)
               VALUES (?, 'nim', 'benchmark', NULL, ?, ?, ?, ?, ?)""",
            (run["timestamp"], run["prompt"], run["success_count"], run["total_models"], run["fastest_model"], run["fastest_time"]),
        )
        new_run_id = cur.lastrowid
        run_count += 1
        results = src.execute(
            "SELECT model, success, error, response_time, tokens_generated, total_tokens, response FROM model_results WHERE run_id = ?",
            (run["id"],),
        ).fetchall()
        out_conn.executemany(
            """INSERT INTO model_results (run_id, model, success, error, response_time, tokens_generated, total_tokens, response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [(new_run_id, r["model"], r["success"], r["error"], r["response_time"], r["tokens_generated"], r["total_tokens"], db.cap_response(r["response"])) for r in results],
        )
        result_count += len(results)
    src.close()
    return run_count, result_count


def migrate_or_models(or_db: Path, out_conn: sqlite3.Connection) -> int:
    src = read_only_connect(or_db)
    rows = src.execute(
        """SELECT openrouter_id, name, context_length, max_completion_tokens, input_modalities,
                  output_modalities, supported_parameters, pricing_prompt, pricing_completion,
                  expiration_date, knowledge_cutoff, active, last_seen_at
           FROM models"""
    ).fetchall()
    for row in rows:
        db.upsert_or_model(out_conn, dict(row))
    src.close()
    return len(rows)


def migrate_openrouter_pinned(or_db: Path, out_conn: sqlite3.Connection) -> tuple[int, int]:
    src = read_only_connect(or_db)
    pinned_runs = src.execute("SELECT id, timestamp, prompt_set FROM runs WHERE kind = 'pinned_models'").fetchall()

    run_count = 0
    result_count = 0
    for run in pinned_runs:
        results = src.execute(
            """SELECT openrouter_id, probe_name, success, http_status, error_type, error_message,
                      latency_ms, completion_tokens, total_tokens, response_excerpt
               FROM model_results WHERE run_id = ?""",
            (run["id"],),
        ).fetchall()

        by_probe: dict[str, list[sqlite3.Row]] = {}
        for r in results:
            by_probe.setdefault(r["probe_name"], []).append(r)

        for probe_name, rows in by_probe.items():
            successful = [r for r in rows if r["success"]]
            if successful:
                fastest = min(successful, key=lambda r: r["latency_ms"] or float("inf"))
                fastest_model, fastest_time = fastest["openrouter_id"], fastest["latency_ms"] or 0
            else:
                fastest_model, fastest_time = "N/A", 0

            cur = out_conn.execute(
                """INSERT INTO runs (timestamp, platform, run_kind, probe_name, prompt, success_count, total_models, fastest_model, fastest_time)
                   VALUES (?, 'openrouter', 'benchmark', ?, ?, ?, ?, ?, ?)""",
                (run["timestamp"], probe_name, probe_prompt_text(probe_name), len(successful), len(rows), fastest_model, fastest_time),
            )
            new_run_id = cur.lastrowid
            run_count += 1

            for r in rows:
                error_text = None
                if not r["success"]:
                    error_text = f"HTTP {r['http_status']}: {r['error_message']}" if r["error_message"] else f"HTTP {r['http_status']}"
                out_conn.execute(
                    """INSERT INTO model_results (run_id, model, success, error, response_time, tokens_generated, total_tokens, response)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (new_run_id, r["openrouter_id"], r["success"], error_text, r["latency_ms"], r["completion_tokens"], r["total_tokens"], db.cap_response(r["response_excerpt"])),
                )
                result_count += 1

    src.close()
    return run_count, result_count


def migrate_openrouter_router(or_db: Path, out_conn: sqlite3.Connection) -> int:
    src = read_only_connect(or_db)
    router_runs = src.execute("SELECT id, timestamp, prompt_set FROM runs WHERE kind = 'router'").fetchall()

    inserted = 0
    for run in router_runs:
        new_run_id = db.create_router_run(out_conn, timestamp=run["timestamp"], probe_name=run["prompt_set"], prompt=probe_prompt_text(run["prompt_set"]))
        results = src.execute(
            """SELECT resolved_model, probe_name, success, http_status, error_type, latency_ms,
                      tokens_per_second, score, created_at
               FROM router_results WHERE run_id = ?""",
            (run["id"],),
        ).fetchall()
        for r in results:
            db.insert_router_result(
                out_conn,
                new_run_id,
                {
                    "requested_model": "openrouter/free",
                    "resolved_model": r["resolved_model"],
                    "probe_name": r["probe_name"],
                    "success": r["success"],
                    "http_status": r["http_status"],
                    "error_type": r["error_type"],
                    "latency_ms": r["latency_ms"],
                    "tokens_per_second": r["tokens_per_second"],
                    "score": r["score"],
                    "created_at": r["created_at"],
                },
            )
            inserted += 1
    src.close()
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill history.db from both legacy projects")
    parser.add_argument("--nim-db", default="/home/moi/Documents/NIMStats/history.db")
    parser.add_argument("--openrouter-db", default="/home/moi/Documents/openrouter-free-stats/history.db")
    parser.add_argument("--out", default=str(db.HISTORY_DB))
    args = parser.parse_args()

    out_conn = db.connect(Path(args.out))
    db.init_schema(out_conn)

    nim_runs, nim_results = migrate_nim(Path(args.nim_db), out_conn)
    out_conn.commit()
    print(f"NIM: migrated {nim_runs} runs, {nim_results} model_results")

    or_db = Path(args.openrouter_db)
    if or_db.exists():
        or_models = migrate_or_models(or_db, out_conn)
        out_conn.commit()
        print(f"OpenRouter: migrated {or_models} or_models")

        or_runs, or_results = migrate_openrouter_pinned(or_db, out_conn)
        out_conn.commit()
        print(f"OpenRouter: migrated {or_runs} runs, {or_results} model_results")

        or_router = migrate_openrouter_router(or_db, out_conn)
        out_conn.commit()
        print(f"OpenRouter: migrated {or_router} router_results")
    else:
        print(f"skip OpenRouter migration: {or_db} not found")

    out_conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
