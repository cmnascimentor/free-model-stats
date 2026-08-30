#!/usr/bin/env python3
"""Merge parallel NIM group results and append one versioned run to history.db."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "common"))
import db  # noqa: E402


def main() -> int:
    all_models: list[dict] = []
    metadata: dict = {}

    for group_file in ["results-group1.json", "results-group2.json"]:
        path = SCRIPT_DIR / group_file
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        all_models.extend(data.get("models", []))
        if not metadata:
            metadata = {
                key: data.get(key)
                for key in (
                    "timestamp",
                    "probe_name",
                    "prompt",
                    "benchmark_version",
                    "probe_version",
                    "temperature",
                    "max_completion_tokens",
                )
            }
        else:
            # Parallel groups must represent the same experiment.
            for key in ("probe_name", "benchmark_version", "probe_version", "temperature", "max_completion_tokens"):
                if metadata.get(key) != data.get(key):
                    raise SystemExit(f"NIM group metadata mismatch for {key}: {metadata.get(key)!r} != {data.get(key)!r}")

    if not all_models:
        print("No NIM results found!", file=sys.stderr)
        return 1

    successful = [m for m in all_models if m.get("taskSuccess", m.get("success"))]
    if successful:
        fastest = min(successful, key=lambda x: x.get("responseTime") or float("inf"))
        fastest_model = fastest.get("model", "N/A")
        fastest_time = fastest.get("responseTime", 0) or 0
    else:
        fastest_model, fastest_time = "N/A", 0

    merged_run = {
        **metadata,
        "models": all_models,
        "summary": {
            "successCount": len(successful),
            "totalModels": len(all_models),
            "fastestModel": fastest_model,
            "fastestTime": fastest_time,
        },
    }
    run_id = db.write_run(merged_run, platform="nim")
    print(
        f"Updated history.db with NIM probe={metadata.get('probe_name')} "
        f"({len(successful)}/{len(all_models)} task-valid, run_id={run_id})"
    )

    for group_file in ["results-group1.json", "results-group2.json"]:
        path = SCRIPT_DIR / group_file
        if path.exists():
            path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
