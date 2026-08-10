#!/usr/bin/env python3
"""Dump a scratch history.db (used only inside the CI openrouter_benchmark job)
into a JSON artifact for the merge_and_update job to ingest into the real,
committed history.db. Keeps the OpenRouter job from writing to the committed
db directly, avoiding a git-push race with the parallel NIM jobs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "common"))
import db  # noqa: E402


def rows(conn, table: str) -> list[dict]:
    return [dict(row) for row in conn.execute(f"SELECT * FROM {table}").fetchall()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a scratch history.db to a JSON artifact")
    parser.add_argument("--db", required=True, help="Path to the scratch SQLite DB")
    parser.add_argument("--out", required=True, help="Path to write the JSON artifact")
    args = parser.parse_args()

    conn = db.connect(Path(args.db))
    db.init_schema(conn)

    payload = {
        "runs": rows(conn, "runs"),
        "model_results": rows(conn, "model_results"),
        "or_models": rows(conn, "or_models"),
        "router_results": rows(conn, "router_results"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"exported runs={len(payload['runs'])} model_results={len(payload['model_results'])} "
        f"or_models={len(payload['or_models'])} router_results={len(payload['router_results'])} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
