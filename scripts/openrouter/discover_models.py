#!/usr/bin/env python3
"""Discover OpenRouter free models and store capability metadata in the unified history.db.

Adapted from openrouter-free-stats' scripts/discover_models.py: instead of the
original project's dedicated `models` table, this upserts a trimmed `or_models`
table that only feeds the merged dashboard's Capabilities tab.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "common"))

import db  # noqa: E402
from openrouter_client import OpenRouterClient  # noqa: E402
from sample_data import SAMPLE_MODELS  # noqa: E402


def _is_zero_price(value: object) -> bool:
    """True if a pricing field parses to exactly 0. Missing/unparseable values are treated as free."""
    if value is None:
        return True
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def is_free_model(model: dict) -> bool:
    model_id = str(model.get("id") or "")
    if not model_id.endswith(":free"):
        return False
    pricing = model.get("pricing") or {}
    return (
        _is_zero_price(pricing.get("prompt"))
        and _is_zero_price(pricing.get("completion"))
        and _is_zero_price(pricing.get("request"))
    )


def fetch_models(args: argparse.Namespace) -> list[dict]:
    if args.dry_run:
        return SAMPLE_MODELS
    client = OpenRouterClient(timeout=args.timeout)
    result = client.get_models()
    if not result.ok:
        raise SystemExit(f"OpenRouter model discovery failed: {result.error_type}: {result.error_message}")
    data = (result.data or {}).get("data") or []
    if not isinstance(data, list):
        raise SystemExit("OpenRouter /models response did not contain a data list")
    return data


def to_or_model_row(raw: dict, now: str) -> dict:
    architecture = raw.get("architecture") or {}
    pricing = raw.get("pricing") or {}
    top_provider = raw.get("top_provider") or {}
    return {
        "openrouter_id": raw.get("id"),
        "name": raw.get("name"),
        "context_length": raw.get("context_length"),
        "max_completion_tokens": top_provider.get("max_completion_tokens"),
        "input_modalities": db.as_json(architecture.get("input_modalities")),
        "output_modalities": db.as_json(architecture.get("output_modalities")),
        "supported_parameters": db.as_json(raw.get("supported_parameters")),
        "pricing_prompt": str(pricing.get("prompt")) if pricing.get("prompt") is not None else None,
        "pricing_completion": str(pricing.get("completion")) if pricing.get("completion") is not None else None,
        "expiration_date": raw.get("expiration_date"),
        "knowledge_cutoff": raw.get("knowledge_cutoff"),
        "active": 1,
        "last_seen_at": now,
    }


def db_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover OpenRouter free models into history.db")
    parser.add_argument("--db", default=str(db.HISTORY_DB), help="SQLite DB path")
    parser.add_argument("--dry-run", action="store_true", help="Use offline sample models instead of OpenRouter API")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of discovered free models stored")
    parser.add_argument("--mark-missing-inactive", action="store_true", help="Mark previously seen models inactive if absent now")
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required unless --dry-run is used")

    conn = db.connect(Path(args.db))
    db.init_schema(conn)

    raw_models = fetch_models(args)
    free_models = [model for model in raw_models if is_free_model(model)]
    free_models.sort(key=lambda item: str(item.get("id") or ""))
    if args.limit > 0:
        free_models = free_models[: args.limit]

    now = db_now()
    seen: list[str] = []
    for model in free_models:
        model_id = str(model.get("id"))
        db.upsert_or_model(conn, to_or_model_row(model, now))
        seen.append(model_id)
    conn.commit()

    inactive = 0
    if args.mark_missing_inactive and seen:
        inactive = db.mark_or_models_missing_inactive(conn, seen)
        conn.commit()

    print(f"discovered_free_models={len(free_models)} db={args.db} dry_run={int(args.dry_run)} marked_inactive={inactive}")
    for model_id in seen[:20]:
        print(f" - {model_id}")
    if len(seen) > 20:
        print(f" ... {len(seen) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
