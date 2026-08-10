#!/usr/bin/env python3
"""Probe loading and validation helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"


@dataclass(frozen=True)
class Probe:
    name: str
    prompt: str
    expects_json: bool = False
    uses_tools: bool = False


def load_probes(names: list[str] | None = None) -> list[Probe]:
    files = sorted(PROMPTS_DIR.glob("*.txt"))
    selected = []
    requested = set(names or [])
    for path in files:
        name = path.stem
        if requested and name not in requested:
            continue
        text = path.read_text(encoding="utf-8").strip()
        selected.append(
            Probe(
                name=name,
                prompt=text,
                expects_json=("json" in name.lower()),
                uses_tools=("tool" in name.lower()),
            )
        )
    if requested:
        found = {probe.name for probe in selected}
        missing = sorted(requested - found)
        if missing:
            raise SystemExit(f"Unknown probe(s): {', '.join(missing)}")
    if not selected:
        raise SystemExit(f"No probes found in {PROMPTS_DIR}")
    return selected


def messages_for(probe: Probe) -> list[dict[str, str]]:
    system = (
        "You are evaluating compact Hermes Web3 audit evidence. "
        "Be concise, deterministic, and avoid unsupported claims."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": probe.prompt},
    ]


def response_format_for(probe: Probe) -> dict[str, Any] | None:
    if not probe.expects_json:
        return None
    return {"type": "json_object"}


def tools_for(probe: Probe) -> tuple[list[dict[str, Any]] | None, str | dict[str, Any] | None]:
    if not probe.uses_tools:
        return None, None
    tool = {
        "type": "function",
        "function": {
            "name": "record_model_probe_verdict",
            "description": "Record a compact model probe verdict.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["pass", "fail", "unclear"]},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["verdict", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
    }
    return [tool], "auto"


def extract_text(data: dict[str, Any] | None) -> tuple[str, str | None, str | None, bool | None]:
    if not data:
        return "", None, None, None
    choices = data.get("choices") or []
    if not choices:
        return "", data.get("model"), None, None
    first = choices[0]
    message = first.get("message") or {}
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls")
    finish_reason = first.get("finish_reason")
    tool_call_valid = None
    if tool_calls is not None:
        tool_call_valid = bool(tool_calls and isinstance(tool_calls, list))
    return str(content), data.get("model"), finish_reason, tool_call_valid


def usage_from(data: dict[str, Any] | None) -> dict[str, int | None]:
    usage = (data or {}).get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def is_valid_json_object(text: str) -> bool:
    try:
        value = json.loads(text)
    except Exception:
        return False
    return isinstance(value, dict)


def score_response(*, success: bool, text: str, expects_json: bool, schema_valid: bool | None, tool_call_valid: bool | None) -> float:
    if not success:
        return 0.0
    score = 50.0
    if text.strip():
        score += 15.0
    if len(text.strip()) >= 80:
        score += 10.0
    if expects_json:
        score += 20.0 if schema_valid else -25.0
    if tool_call_valid is not None:
        score += 20.0 if tool_call_valid else -15.0
    if any(marker in text.lower() for marker in ["valid", "invalid", "needs", "confidence", "evidence", "assumption"]):
        score += 5.0
    return max(0.0, min(100.0, score))
