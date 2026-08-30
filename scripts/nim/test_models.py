#!/usr/bin/env python3
"""Benchmark NVIDIA NIM models. Ported from NIMStats, writes into the unified history.db."""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "common"))
import breaker  # noqa: E402
import db  # noqa: E402

API_BASE = os.getenv("API_BASE", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("NIM_API_KEY", "")
MODEL_GROUP = os.getenv("MODEL_GROUP", "all")
MODEL_LIMIT = int(os.getenv("NIM_MODEL_LIMIT", "0"))  # 0 or unset = benchmark every discovered chat model
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
DISCOVERY_TIMEOUT_SECONDS = int(os.getenv("DISCOVERY_TIMEOUT_SECONDS", "30"))
PROMPT = "Write a Python function that checks if a number is prime and returns True or False"

OUTPUT_FILE = SCRIPT_DIR / "results.json"
MODELS_ENDPOINT = f"{API_BASE}/models"

# Snapshot used only if live catalog discovery fails (network error, non-2xx,
# malformed response, or an empty/all-filtered model list) so benchmarking can
# still proceed with a reasonable set of models.
#
# Curated from a live GET /v1/models catalog pull (2026-08-10) filtered through
# is_chat_model_id, spanning multiple vendors/sizes. Verify against a fresh
# catalog pull before trusting an entry that isn't a recent addition here —
# NVIDIA rotates catalog IDs (version suffixes change, models are retired)
# faster than this snapshot gets refreshed. Notably, as of this snapshot the
# catalog has no `qwen/*` chat entries and no `meta/llama-4-*` entries at all,
# despite both having been present in a previous version of this list.
FALLBACK_MODELS = [
    "deepseek-ai/deepseek-v4-flash-0731",
    "google/gemma-3-12b-it",
    "google/gemma-4-31b-it",
    "ibm/granite-34b-code-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "meta/llama-3.3-70b-instruct",
    "minimaxai/minimax-m3",
    "mistralai/codestral-22b-instruct-v0.1",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mistral-nemotron",
    "moonshotai/kimi-k2.6",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "stepfun-ai/step-3.7-flash",
    "writer/palmyra-creative-122b",
    "z-ai/glm-5.2",
]

# Curated allowlist of model IDs confirmed to support /v1/chat/completions
# on NVIDIA's API catalog.  Each entry was verified by a live probe
# (max_tokens=1 POST) returning HTTP 200, not just listed in GET /v1/models
# (which includes many models that return 404 on chat/completions because
# they are base/completion-only, vision-only, embedding, or otherwise not
# general-purpose chat models).
#
# Last verified: 2026-08-10
CHAT_MODEL_ALLOWLIST = frozenset({
    "deepseek-ai/deepseek-v4-flash-0731",
    "google/gemma-3-12b-it",
    "google/gemma-4-31b-it",
    "ibm/granite-34b-code-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "meta/llama-3.3-70b-instruct",
    "minimaxai/minimax-m3",
    "mistralai/codestral-22b-instruct-v0.1",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mistral-nemotron",
    "moonshotai/kimi-k2.6",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "stepfun-ai/step-3.7-flash",
    "writer/palmyra-creative-122b",
    "z-ai/glm-5.2",
})


def fetch_catalog_model_ids() -> list[str]:
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    request = urllib.request.Request(MODELS_ENDPOINT, method="GET", headers=headers)
    with urllib.request.urlopen(request, timeout=DISCOVERY_TIMEOUT_SECONDS) as response:
        raw_body = response.read().decode("utf-8", errors="replace")
    data = json.loads(raw_body)
    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError("catalog response did not contain a 'data' list")
    return [
        str(entry["id"])
        for entry in entries
        if isinstance(entry, dict) and entry.get("id")
    ]


def is_chat_model_id(model_id: str) -> bool:
    """Check if a model ID is in the curated chat-model allowlist."""
    return model_id in CHAT_MODEL_ALLOWLIST


def discover_chat_models() -> list[str]:
    """Discover current chat-capable NIM catalog models, falling back to a pinned snapshot on failure."""
    try:
        catalog_ids = fetch_catalog_model_ids()
    except Exception as exc:  # noqa: BLE001 - any discovery failure should fall back, not crash the run.
        print(f"Warning: NIM catalog discovery failed ({exc}); using pinned fallback model list", file=sys.stderr)
        return list(FALLBACK_MODELS)

    chat_models = sorted({model_id for model_id in catalog_ids if is_chat_model_id(model_id)})
    if not chat_models:
        print("Warning: NIM catalog discovery returned no chat-capable models; using pinned fallback model list", file=sys.stderr)
        return list(FALLBACK_MODELS)

    if MODEL_LIMIT > 0:
        chat_models = chat_models[:MODEL_LIMIT]
    return chat_models


def selected_models(dry_run: bool = False) -> list[str]:
    models = list(FALLBACK_MODELS) if dry_run else discover_chat_models()
    if MODEL_GROUP == "group1":
        mid = (len(models) + 1) // 2
        return models[:mid]
    if MODEL_GROUP == "group2":
        mid = (len(models) + 1) // 2
        return models[mid:]
    return models


def failure_result(model: str, error: str) -> dict[str, Any]:
    return {
        "model": model,
        "success": False,
        "error": error,
        "responseTime": None,
        "tokensGenerated": None,
        "totalTokens": None,
        "response": None,
    }


def normalize_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def dry_run_result(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "success": True,
        "responseTime": 42,
        "tokensGenerated": 24,
        "totalTokens": 48,
        "response": "def is_prime(n):\n    ...  # dry-run sample response",
        "error": None,
    }


def call_model(model: str, prompt: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 500,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )

    started = time.perf_counter()
    raw_body = ""
    status_code = 0

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status_code = response.status
            raw_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status_code = getattr(exc, "code", 0) or 0
        raw_body = exc.read().decode("utf-8", errors="replace")
    except TimeoutError:
        return failure_result(model, f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s")
    except Exception as exc:
        return failure_result(model, f"Request failed: {exc}")

    response_time = int((time.perf_counter() - started) * 1000)

    if not raw_body.strip():
        return failure_result(model, "Empty response from API")

    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        return {
            "model": model,
            "success": False,
            "error": f"Invalid JSON response: {exc.msg} at line {exc.lineno} column {exc.colno}",
            "responseTime": response_time,
            "tokensGenerated": None,
            "totalTokens": None,
            "response": raw_body,
        }

    error_obj = data.get("error")
    error_message = ""
    if isinstance(error_obj, dict):
        error_message = str(error_obj.get("message") or "").strip()
    elif isinstance(error_obj, str):
        error_message = error_obj.strip()

    if status_code >= 400:
        if not error_message:
            error_message = f"HTTP {status_code} returned by API"
        else:
            error_message = f"HTTP {status_code}: {error_message}"
        return failure_result(model, error_message)

    if error_message:
        return failure_result(model, error_message)

    choices = data.get("choices")
    content = ""
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = normalize_content(message.get("content"))

    if not content.strip():
        return failure_result(model, "No content in response")

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    completion_tokens = to_int(usage.get("completion_tokens"))
    total_tokens = to_int(usage.get("total_tokens"))

    return {
        "model": model,
        "success": True,
        "responseTime": response_time,
        "tokensGenerated": completion_tokens,
        "totalTokens": total_tokens,
        "response": content,
        "error": None,
    }


def compile_output(timestamp: str, prompt: str, models: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in models if item.get("success")]
    success_count = len(successful)
    total_count = len(models)

    if successful:
        fastest = min(
            successful,
            key=lambda item: item.get("responseTime")
            if isinstance(item.get("responseTime"), int)
            else float("inf"),
        )
        fastest_model = fastest.get("model", "N/A")
        fastest_time = fastest.get("responseTime", 0) or 0
    else:
        fastest_model = "N/A"
        fastest_time = 0

    return {
        "timestamp": timestamp,
        "prompt": prompt,
        "models": models,
        "summary": {
            "successCount": success_count,
            "totalModels": total_count,
            "fastestModel": fastest_model,
            "fastestTime": fastest_time,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark NVIDIA NIM catalog models")
    parser.add_argument("--db", default=str(db.HISTORY_DB), help="SQLite DB path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip catalog discovery and network calls; use the pinned fallback list and deterministic sample results",
    )
    args = parser.parse_args()

    if not args.dry_run and not API_KEY:
        print("Error: NIM_API_KEY environment variable not set", file=sys.stderr)
        return 1

    models = selected_models(dry_run=args.dry_run)
    if not args.dry_run:
        runnable, skipped = breaker.tripped_models(Path(args.db), models, probe="nim_prime")
        for model in skipped:
            print(f"Skipping: {model} (circuit breaker: repeated failures recently)")
        models = runnable
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    group_label = f" (Group: {MODEL_GROUP})" if MODEL_GROUP else ""
    print(f"Starting NVIDIA NIM Model Benchmarks{group_label}...")
    print(f"Timestamp: {timestamp}")
    print(f"Testing {len(models)} models...")
    print()

    results: list[dict[str, Any]] = []
    for model in models:
        print(f"Testing: {model}")
        result = dry_run_result(model) if args.dry_run else call_model(model, PROMPT)
        if result.get("success"):
            breaker.record_success(Path(args.db), model)
            print(f"  ✓ Success ({result['responseTime']}ms, {result.get('tokensGenerated', 0)} tokens)")
        else:
            if not args.dry_run:
                breaker.record_failure(Path(args.db), model, result.get("error") or "", probe="nim_prime")
            print(f"  ✗ Failed: {result.get('error') or 'Unknown error'}")
        results.append(result)
        if not args.dry_run:
            time.sleep(0.5)

    print()
    print("Compiling results...")

    final_json = compile_output(timestamp, PROMPT, results)
    OUTPUT_FILE.write_text(json.dumps(final_json, indent=2), encoding="utf-8")

    success_count = final_json["summary"]["successCount"]
    total_count = final_json["summary"]["totalModels"]
    print(f"Results saved to {OUTPUT_FILE.name}")
    print(f"Summary: {success_count}/{total_count} successful")

    if MODEL_GROUP == "all":
        run_id = db.write_run(final_json, db_path=Path(args.db), platform="nim")
        print(f"History updated: {args.db} (run_id={run_id})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
