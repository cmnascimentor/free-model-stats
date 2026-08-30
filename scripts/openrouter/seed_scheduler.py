#!/usr/bin/env python3
"""Seed OpenRouter scheduling state from a previous history.db into scratch.db.

Only ``or_models`` state is copied; historical runs remain in the merge job's
canonical database, preventing duplicate artifact ingestion.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "common"))
import db  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.is_file():
        print(f"scheduler seed not found: {source_path}; starting fresh")
        return 0

    source = sqlite3.connect(str(source_path))
    source.row_factory = sqlite3.Row
    target = db.connect(Path(args.target))
    db.init_schema(target)
    try:
        source_cols = {row[1] for row in source.execute("PRAGMA table_info(or_models)").fetchall()}
        if "openrouter_id" not in source_cols:
            print("source database has no compatible or_models table; starting fresh")
            return 0

        rows = source.execute("SELECT * FROM or_models").fetchall()
        copied = 0
        for row in rows:
            record = dict(row)
            # First use the normal metadata upsert, then restore scheduler fields
            # when they exist in the source schema.
            db.upsert_or_model(
                target,
                {
                    "openrouter_id": record.get("openrouter_id"),
                    "name": record.get("name"),
                    "context_length": record.get("context_length"),
                    "max_completion_tokens": record.get("max_completion_tokens"),
                    "input_modalities": record.get("input_modalities"),
                    "output_modalities": record.get("output_modalities"),
                    "supported_parameters": record.get("supported_parameters"),
                    "pricing_prompt": record.get("pricing_prompt"),
                    "pricing_completion": record.get("pricing_completion"),
                    "expiration_date": record.get("expiration_date"),
                    "knowledge_cutoff": record.get("knowledge_cutoff"),
                    "active": record.get("active", 1),
                    "last_seen_at": record.get("last_seen_at"),
                },
            )
            target.execute(
                """UPDATE or_models
                   SET last_benchmarked_at = ?, benchmark_count = ?
                   WHERE openrouter_id = ?""",
                (
                    record.get("last_benchmarked_at"),
                    record.get("benchmark_count", 0) or 0,
                    record.get("openrouter_id"),
                ),
            )
            copied += 1
        target.commit()
        print(f"seeded_scheduler_models={copied}")
        return 0
    finally:
        source.close()
        target.close()


if __name__ == "__main__":
    raise SystemExit(main())
