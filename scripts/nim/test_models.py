#!/usr/bin/env python3
"""Benchmark NVIDIA NIM models with the shared FreeModelStats probe suite."""
from __future__ import annotations

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
COMMON_DIR = SCRIPT_DIR.parent / "common"
sys.path.insert(0, str(COMMON_DIR))

import breaker  # noqa: E402
import db  # noqa: E402
from probe_suite import (  # noqa: E402
    BENCHMARK_VERSION,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_PROBE,
    DEFAULT_TEMPERATURE,
    Probe,
    extract_text,
    load_probes,
    messages_for,
    quality_score,
    usage_from,
    validate_probe_response,
)

API_BASE = os.getenv("API_BASE", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("NIM_API_KEY", "")
MODEL_GROUP = os.getenv("MODEL_GROUP", "all")
MODEL_LIMIT = int(os.getenv("NIM_MODEL_LIMIT", "0"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
DISCOVERY_TIMEOUT_SECONDS = int(os.getenv("DISCOVERY_TIMEOUT_SECONDS", "30"))
OUTPUT_FILE = SCRIPT_DIR / "results.json"
MODELS_ENDPOINT = f"{API_BASE}/models"

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

# NVIDIA's /models catalog includes entries that do not support chat/completions.
# This allowlist is intentionally a capability filter, not a static benchmark list:
# live discovery still determines which currently-listed entries are benchmarked.
CHAT_MODEL_ALLOWLIST = frozenset(FALLBACK_MODELS)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_catalog_model_ids() -> list[str]:
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    request = urllib.request.Request(MODELS_ENDPOINT, method="GET", headers=headers)
    with urllib.request.urlopen(request, timeout=DISCOVERY_TIMEOUT_SECONDS) as response:
        raw_body = response.read().decode("utf-8", errors="replace")
    data = json.loads(raw_body)
    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError("catalog response did not contain a 'data' list")
    return [str(entry["id"]) for entry in entries if isinstance(entry, dict) and entry.get("id")]


def is_chat_model_id(model_id: str) -> bool:
    return model_id in CHAT_MODEL_ALLOWLIST


def discover_chat_models() -> list[str]:
    try:
        catalog_ids = fetch_catalog_model_ids()
    except Exception as exc:  # noqa: BLE001 - discovery failure falls back safely.
        print(f"Warning: NIM catalog discovery failed ({exc}); using pinned fallback", file=sys.stderr)
        models = list(FALLBACK_MODELS)
    else:
        models = sorted({model_id for model_id in catalog_ids if is_chat_model_id(model_id)})
        if not models:
            print("Warning: no allowlisted chat models found; using pinned fallback", file=sys.stderr)
            models = list(FALLBACK_MODELS)

    if MODEL_LIMIT > 0:
        models = models[:MODEL_LIMIT]
    return models


def selected_models(dry_run: bool = False) -> list[str]:
    models = list(FALLBACK_MODELS) if dry_run else discover_chat_models()
    if MODEL_GROUP == "group1":
        mid = (len(models) + 1) // 2
        return models[:mid]
    if MODEL_GROUP == "group2":
        mid = (len(models) + 1) // 2
        return models[mid:]
    return models


def normalize_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def failure_result(model: str, error: str, *, response_time: int | None = None, status: int | None = None) -> dict[str, Any]:
    return {
        "model": model,
        "success": False,
        "transportSuccess": False,
        "formatSuccess": False,
        "taskSuccess": False,
        "qualityScore": 0.0,
        "error": error,
        "responseTime": response_time,
        "finalAttemptMs": response_time,
        "totalElapsedMs": response_time,
        "attemptCount": 1,
        "httpStatus": status,
        "tokensGenerated": None,
        "totalTokens": None,
        "tokensPerSecond": None,
        "response": None,
        "benchmarkVersion": BENCHMARK_VERSION,
    }


def dry_response(probe: Probe) -> dict[str, Any]:
    if probe.name == "hermes_json_schema":
        content = json.dumps(
            {
                "verdict": "needs_more_evidence",
                "confidence": 0.81,
                "reasons": ["setter is privileged", "downstream arithmetic may revert"],
                "missing_evidence": ["governance reachability", "other parameter bounds"],
            }
        )
    elif probe.name == "hermes_evidence_summary":
        content = (
            "Claim: a privileged fee update may deny swaps. Supporting evidence: onlyOwner controls the fee and downstream arithmetic can revert. "
            "Assumptions: no other bounds exist. Missing evidence: no ordinary-user setter path is confirmed. "
            "Next deterministic check: trace all fee setters and bounds."
        )
    elif probe.name == "hermes_code_reasoning":
        content = (
            "The denominator assumption is that totalSupply is non-zero. External token behavior matters because token.transfer can revert, "
            "return false, charge fees, or reenter. Burning before transfer is not automatically harmful because a revert is atomic, but a false-return token may matter. "
            "These assumptions need confirmation before calling this a vulnerability."
        )
    else:
        content = (
            "Verdict: needs_more_evidence. Confidence: 0.83. Check whether accumulatedRewardDebt is reset on close, "
            "whether claimReward is gated by active position status, and whether repeated claims are possible."
        )
    return {
        "model": "dry-run/nim",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 54, "total_tokens": 174},
    }


def call_model(model: str, probe: Probe, *, temperature: float, max_completion_tokens: int) -> tuple[dict[str, Any] | None, int, int, str | None]:
    payload = {
        "model": model,
        "messages": messages_for(probe),
        "temperature": temperature,
        "max_tokens": max_completion_tokens,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
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
    except Exception as exc:  # noqa: BLE001
        elapsed = int((time.perf_counter() - started) * 1000)
        return None, status_code, elapsed, f"Request failed: {exc}"

    elapsed = int((time.perf_counter() - started) * 1000)
    if not raw_body.strip():
        return None, status_code, elapsed, "Empty response from API"
    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        return None, status_code, elapsed, f"Invalid JSON response: {exc.msg}"

    error_obj = data.get("error") if isinstance(data, dict) else None
    error_message = ""
    if isinstance(error_obj, dict):
        error_message = str(error_obj.get("message") or error_obj).strip()
    elif error_obj:
        error_message = str(error_obj).strip()
    if status_code >= 400 or error_message:
        message = error_message or f"HTTP {status_code} returned by API"
        return data if isinstance(data, dict) else None, status_code, elapsed, f"HTTP {status_code}: {message}"
    return data if isinstance(data, dict) else None, status_code, elapsed, None


def tokens_per_second(tokens: int | None, elapsed_ms: int | None) -> float | None:
    if not tokens or not elapsed_ms or elapsed_ms <= 0:
        return None
    return round(tokens / (elapsed_ms / 1000.0), 3)


def evaluate_model(model: str, probe: Probe, *, dry_run: bool, temperature: float, max_completion_tokens: int) -> dict[str, Any]:
    if dry_run:
        data, status, elapsed, transport_error = dry_response(probe), 200, 42, None
    else:
        data, status, elapsed, transport_error = call_model(
            model, probe, temperature=temperature, max_completion_tokens=max_completion_tokens
        )

    if transport_error:
        return failure_result(model, transport_error, response_time=elapsed, status=status)

    text, resolved_model, finish_reason, tool_call_valid = extract_text(data)
    usage = usage_from(data)
    validation_error = validate_probe_response(
        probe=probe,
        http_ok=True,
        text=text,
        tool_call_valid=tool_call_valid,
    )
    task_success = validation_error is None
    if probe.name == "hermes_json_schema":
        try:
            format_success = isinstance(json.loads(text), dict)
        except (TypeError, json.JSONDecodeError):
            format_success = False
    else:
        format_success = bool(text.strip())
    q_score = quality_score(probe=probe, http_ok=True, text=text, tool_call_valid=tool_call_valid)

    return {
        "model": model,
        "success": task_success,
        "transportSuccess": True,
        "formatSuccess": format_success,
        "taskSuccess": task_success,
        "qualityScore": q_score,
        "error": None,
        "validationError": validation_error,
        "responseTime": elapsed,
        "finalAttemptMs": elapsed,
        "totalElapsedMs": elapsed,
        "attemptCount": 1,
        "httpStatus": status,
        "finishReason": finish_reason,
        "resolvedModel": resolved_model,
        "tokensGenerated": usage["completion_tokens"],
        "totalTokens": usage["total_tokens"],
        "tokensPerSecond": tokens_per_second(usage["completion_tokens"], elapsed),
        "response": text or None,
        "benchmarkVersion": BENCHMARK_VERSION,
        "probeVersion": probe.version,
    }


def compile_output(timestamp: str, probe: Probe, models: list[dict[str, Any]], *, temperature: float, max_completion_tokens: int) -> dict[str, Any]:
    successful = [item for item in models if item.get("taskSuccess")]
    if successful:
        fastest = min(successful, key=lambda item: item.get("responseTime") or float("inf"))
        fastest_model = fastest.get("model", "N/A")
        fastest_time = fastest.get("responseTime", 0) or 0
    else:
        fastest_model, fastest_time = "N/A", 0
    return {
        "timestamp": timestamp,
        "probe_name": probe.name,
        "prompt": probe.prompt,
        "benchmark_version": BENCHMARK_VERSION,
        "probe_version": probe.version,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "models": models,
        "summary": {
            "successCount": len(successful),
            "totalModels": len(models),
            "fastestModel": fastest_model,
            "fastestTime": fastest_time,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark NVIDIA NIM catalog models")
    parser.add_argument("--db", default=str(db.HISTORY_DB), help="SQLite DB path")
    parser.add_argument("--probe", default=DEFAULT_PROBE, help=f"Shared benchmark probe. Default: {DEFAULT_PROBE}")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not API_KEY:
        print("Error: NIM_API_KEY environment variable not set", file=sys.stderr)
        return 1

    probe = load_probes([args.probe])[0]
    if probe.uses_tools:
        print("Error: tool-use probe is provider-specific and is not enabled for NIM baseline runs", file=sys.stderr)
        return 2

    models = selected_models(dry_run=args.dry_run)
    db_path = Path(args.db)
    if not args.dry_run:
        runnable, skipped = breaker.tripped_models(db_path, models, probe=probe.name)
        for model in skipped:
            print(f"Skipping: {model} (circuit breaker: repeated transport failures on {probe.name})")
        models = runnable

    timestamp = utc_now()
    print(f"Starting NVIDIA NIM benchmark probe={probe.name} group={MODEL_GROUP} models={len(models)}")
    results: list[dict[str, Any]] = []
    provider_error: str | None = None

    for model in models:
        print(f"Testing: {model}", flush=True)
        result = evaluate_model(
            model,
            probe,
            dry_run=args.dry_run,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        )
        error_text = str(result.get("error") or "")
        if not args.dry_run and breaker.is_provider_scoped_failure(error_text):
            provider_error = error_text
            print(
                f"Provider-scoped NIM failure detected; stopping this group without tripping models: {provider_error}",
                file=sys.stderr,
                flush=True,
            )
            break

        if not args.dry_run:
            if result.get("transportSuccess"):
                breaker.record_success(db_path, model)
            else:
                breaker.record_failure(db_path, model, error_text, probe=probe.name)
        results.append(result)
        print(
            f"  transport={int(bool(result.get('transportSuccess')))} task={int(bool(result.get('taskSuccess')))} "
            f"quality={float(result.get('qualityScore') or 0):.1f} elapsed_ms={result.get('responseTime')} "
            f"validation={result.get('validationError') or '-'}",
            flush=True,
        )
        if not args.dry_run:
            time.sleep(0.5)

    final_json = compile_output(
        timestamp,
        probe,
        results,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )
    if provider_error:
        final_json["provider_error"] = provider_error
    OUTPUT_FILE.write_text(json.dumps(final_json, indent=2), encoding="utf-8")
    success_count = final_json["summary"]["successCount"]
    total_count = final_json["summary"]["totalModels"]
    print(f"Summary: {success_count}/{total_count} task-valid")

    if MODEL_GROUP == "all" and results:
        run_id = db.write_run(final_json, db_path=db_path, platform="nim")
        print(f"History updated: {args.db} (run_id={run_id})")
    elif MODEL_GROUP == "all" and provider_error:
        print("History not updated because NIM returned a provider-scoped failure.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
