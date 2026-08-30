#!/usr/bin/env python3
"""Shared, versioned benchmark probes and deterministic validators.

The same core probe suite is used by every provider adapter so cross-provider
results are comparable. Provider-specific capability probes (for example tool
calling) may still be run, but should not be mixed into a cross-provider score
unless every compared provider used the same probe and configuration.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

BENCHMARK_VERSION = "2.0.0"
PROBE_SUITE_VERSION = "2.0.0"
DEFAULT_PROBE = "hermes_triage"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_COMPLETION_TOKENS = 512


@dataclass(frozen=True)
class Probe:
    name: str
    prompt: str
    expects_json: bool = False
    uses_tools: bool = False
    category: str = "reasoning"
    cross_provider: bool = True
    version: str = PROBE_SUITE_VERSION


def _probe_from_path(path: Path) -> Probe:
    name = path.stem
    return Probe(
        name=name,
        prompt=path.read_text(encoding="utf-8").strip(),
        expects_json=(name == "hermes_json_schema"),
        uses_tools=(name == "hermes_tool_probe"),
        category=(
            "structured_output" if name == "hermes_json_schema"
            else "tool_use" if name == "hermes_tool_probe"
            else "summarization" if name == "hermes_evidence_summary"
            else "code_reasoning" if name == "hermes_code_reasoning"
            else "triage"
        ),
        cross_provider=(name != "hermes_tool_probe"),
    )


def load_probes(names: list[str] | None = None) -> list[Probe]:
    files = sorted(PROMPTS_DIR.glob("*.txt"))
    requested = set(names or [])
    selected = [_probe_from_path(path) for path in files if not requested or path.stem in requested]
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
        "Be concise, deterministic, distinguish facts from assumptions, and do not invent missing code."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": probe.prompt},
    ]


def response_format_for(probe: Probe) -> dict[str, Any] | None:
    # Kept optional for providers that support OpenAI-compatible JSON mode.
    # Validation never trusts provider-side structured-output enforcement: the
    # response is always checked locally against the exact schema below.
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
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["verdict", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
    }
    return [tool], "auto"


def _valid_confidence(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def is_valid_tool_call(tool_call: Any) -> bool:
    if not isinstance(tool_call, dict):
        return False
    function = tool_call.get("function")
    if not isinstance(function, dict) or function.get("name") != "record_model_probe_verdict":
        return False
    args = function.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return False
    if not isinstance(args, dict) or set(args) != {"verdict", "confidence", "reason"}:
        return False
    return (
        args.get("verdict") in {"pass", "fail", "unclear"}
        and _valid_confidence(args.get("confidence"))
        and isinstance(args.get("reason"), str)
        and bool(args["reason"].strip())
    )


def extract_text(data: dict[str, Any] | None) -> tuple[str, str | None, str | None, bool | None]:
    if not data:
        return "", None, None, None
    choices = data.get("choices") or []
    if not choices:
        return "", data.get("model"), None, None
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    tool_calls = message.get("tool_calls")
    finish_reason = first.get("finish_reason")
    tool_call_valid = None
    if tool_calls is not None:
        tool_call_valid = isinstance(tool_calls, list) and any(is_valid_tool_call(call) for call in tool_calls)
    return str(content), data.get("model"), finish_reason, tool_call_valid


def usage_from(data: dict[str, Any] | None) -> dict[str, int | None]:
    usage = (data or {}).get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def is_valid_json_object(text: str) -> bool:
    return parse_json_object(text) is not None


def validate_exact_json_schema(text: str) -> bool:
    value = parse_json_object(text)
    if value is None:
        return False
    required = {"verdict", "confidence", "reasons", "missing_evidence"}
    if set(value) != required:
        return False
    if value.get("verdict") not in {"likely_valid", "likely_invalid", "needs_more_evidence"}:
        return False
    if not _valid_confidence(value.get("confidence")):
        return False
    for key in ("reasons", "missing_evidence"):
        items = value.get(key)
        if not isinstance(items, list) or not all(isinstance(item, str) and item.strip() for item in items):
            return False
    return True


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(needle in lower for needle in needles)


def _triage_valid(text: str) -> bool:
    lower = text.lower()
    verdict_ok = any(v in lower for v in ("likely_valid", "likely_invalid", "needs_more_evidence", "needs more evidence"))
    confidence_ok = bool(re.search(r"confidence\s*[:=\-]?\s*(?:0(?:\.\d+)?|1(?:\.0+)?)", lower))
    # The capsule explicitly names three unknowns. A useful answer must surface
    # at least two distinct deterministic checks rather than just restating a verdict.
    checks = sum(
        bool(_contains_any(lower, group))
        for group in (
            ("accumulatedrewarddebt", "reward debt", "reset"),
            ("active position", "onlyactiveposition", "position status", "gated"),
            ("repeat", "repeated", "multiple claim", "claim again"),
        )
    )
    return verdict_ok and confidence_ok and checks >= 2


def _summary_valid(text: str) -> bool:
    words = re.findall(r"\S+", text)
    if not words or len(words) > 180:
        return False
    lower = text.lower()
    sections = (
        ("claim",),
        ("supporting evidence", "evidence"),
        ("assumption", "assumptions"),
        ("missing evidence", "missing"),
        ("next deterministic check", "next check", "deterministic check"),
    )
    return all(_contains_any(lower, group) for group in sections)


def _code_reasoning_valid(text: str) -> bool:
    lower = text.lower()
    denominator = _contains_any(lower, ("totalsupply", "total supply", "denominator", "division by zero"))
    token = _contains_any(lower, ("token.transfer", "transfer return", "external token", "fee-on-transfer", "reentr"))
    burn = _contains_any(lower, ("burn", "before transfer", "revert", "atomic"))
    uncertainty = _contains_any(lower, ("assum", "depends", "need", "cannot", "not enough", "unknown"))
    return denominator and token and burn and uncertainty


def validate_probe_response(
    *,
    probe: Probe,
    http_ok: bool,
    text: str,
    tool_call_valid: bool | None,
) -> str | None:
    if not http_ok:
        return None
    if probe.uses_tools:
        return None if tool_call_valid is True else "invalid_tool_call"
    if not text.strip():
        return "empty_content"
    if probe.name == "hermes_json_schema":
        return None if validate_exact_json_schema(text) else "invalid_json_schema"
    if probe.name == "hermes_triage":
        return None if _triage_valid(text) else "triage_requirements_not_met"
    if probe.name == "hermes_evidence_summary":
        return None if _summary_valid(text) else "summary_requirements_not_met"
    if probe.name == "hermes_code_reasoning":
        return None if _code_reasoning_valid(text) else "reasoning_requirements_not_met"
    return None


def quality_score(
    *,
    probe: Probe,
    http_ok: bool,
    text: str,
    tool_call_valid: bool | None,
) -> float:
    """Deterministic 0-100 task-quality score.

    The score is intentionally conservative and rubric-based. It avoids an LLM
    judge so benchmark results remain reproducible and free of evaluator drift.
    """
    if not http_ok:
        return 0.0
    if probe.uses_tools:
        return 100.0 if tool_call_valid is True else 0.0
    if not text.strip():
        return 0.0
    if probe.name == "hermes_json_schema":
        value = parse_json_object(text)
        if value is None:
            return 0.0
        score = 20.0
        score += 35.0 if set(value) == {"verdict", "confidence", "reasons", "missing_evidence"} else 0.0
        score += 15.0 if value.get("verdict") in {"likely_valid", "likely_invalid", "needs_more_evidence"} else 0.0
        score += 15.0 if _valid_confidence(value.get("confidence")) else 0.0
        score += 15.0 if validate_exact_json_schema(text) else 0.0
        return min(100.0, score)
    if probe.name == "hermes_triage":
        lower = text.lower()
        score = 20.0
        score += 25.0 if any(v in lower for v in ("likely_valid", "likely_invalid", "needs_more_evidence", "needs more evidence")) else 0.0
        score += 20.0 if re.search(r"confidence\s*[:=\-]?\s*(?:0(?:\.\d+)?|1(?:\.0+)?)", lower) else 0.0
        checks = sum(
            bool(_contains_any(lower, group))
            for group in (
                ("accumulatedrewarddebt", "reward debt", "reset"),
                ("active position", "onlyactiveposition", "position status", "gated"),
                ("repeat", "repeated", "multiple claim", "claim again"),
            )
        )
        score += min(35.0, checks * 17.5)
        return min(100.0, score)
    if probe.name == "hermes_evidence_summary":
        score = 15.0 if len(re.findall(r"\S+", text)) <= 180 else 0.0
        lower = text.lower()
        for group in (
            ("claim",), ("evidence",), ("assumption",), ("missing",), ("next check", "deterministic check")
        ):
            score += 17.0 if _contains_any(lower, group) else 0.0
        return min(100.0, score)
    if probe.name == "hermes_code_reasoning":
        lower = text.lower()
        score = 10.0
        score += 25.0 if _contains_any(lower, ("totalsupply", "total supply", "denominator", "division by zero")) else 0.0
        score += 25.0 if _contains_any(lower, ("token.transfer", "transfer return", "external token", "fee-on-transfer", "reentr")) else 0.0
        score += 25.0 if _contains_any(lower, ("burn", "before transfer", "revert", "atomic")) else 0.0
        score += 15.0 if _contains_any(lower, ("assum", "depends", "need", "unknown")) else 0.0
        return min(100.0, score)
    return 100.0


def score_response(*, success: bool, text: str, expects_json: bool, schema_valid: bool | None, tool_call_valid: bool | None) -> float:
    """Backward-compatible router helper retained for older callers/tests."""
    if not success:
        return 0.0
    if expects_json:
        return 100.0 if schema_valid else 0.0
    if tool_call_valid is not None:
        return 100.0 if tool_call_valid else 0.0
    return 100.0 if text.strip() else 0.0
