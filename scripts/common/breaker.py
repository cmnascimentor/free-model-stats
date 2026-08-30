"""Shared failure circuit breaker for benchmark models.

Models that fail persistently (e.g. HTTP 404 after removal from a provider
catalog, or 403 "only available on agentic harnesses") waste quota and add
noise to every scheduled run. The breaker skips a (model, probe) pair after
`FAILURE_THRESHOLD` consecutive *transport* failures (HTTP/network errors —
validation-only failures where the model answered but misbehaved do NOT
count), and re-probes it after a cooldown so genuinely-restored models
rejoin the benchmark automatically. A success on ANY probe clears every
breaker entry for that model.

State keys are "model::probe". Legacy keys (bare "model", pre-splitting)
remain valid and apply to every probe of that model.

State lives in a small JSON file beside history.db so NIM and OpenRouter
halves (and the CI merge) stay consistent. Corrupt state is silently reset —
the breaker must never block a benchmark run.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

FAILURE_THRESHOLD = 3  # consecutive failures before tripping
COOLDOWN_DAYS = 7      # re-probe a tripped model this often
STATE_VERSION = 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def state_path(db_path: Path) -> Path:
    return Path(db_path).parent / "breaker_state.json"


def load_state(db_path: Path) -> dict:
    path = state_path(db_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == STATE_VERSION:
            return data
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return {"version": STATE_VERSION, "models": {}}


def save_state(db_path: Path, state: dict) -> None:
    path = state_path(db_path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _entry(state: dict, model: str) -> dict:
    return state.setdefault("models", {}).setdefault(model, {"consecutive_failures": 0})


def _key(model: str, probe: str) -> str:
    """State key for a (model, probe) pair; bare model for legacy/empty probe."""
    if probe:
        return f"{model}::{probe}"
    return model


def record_failure(db_path: Path, model: str, error: str = "", probe: str = "") -> None:
    """Record a transport failure for a (model, probe) pair.

    Empty `error` (validation-only failure — the model answered but misbehaved)
    is recorded as signal but never accumulates toward tripping the breaker.
    """
    if not error:
        return
    state = load_state(db_path)
    entry = _entry(state, _key(model, probe))
    entry["consecutive_failures"] = int(entry.get("consecutive_failures", 0)) + 1
    entry["last_error"] = str(error)[:300]
    entry["last_failure_at"] = _now().strftime("%Y-%m-%dT%H:%M:%SZ")
    if entry["consecutive_failures"] >= FAILURE_THRESHOLD:
        entry["tripped"] = True
        entry["trip_at"] = entry["last_failure_at"]
    save_state(db_path, state)


def record_success(db_path: Path, model: str) -> None:
    """Record a success — clears every breaker entry for the model (all probes):
    a successful response proves the model itself is alive."""
    state = load_state(db_path)
    models = state.get("models", {})
    doomed = [k for k in models if k == model or k.startswith(f"{model}::")]
    if doomed:
        for k in doomed:
            del models[k]
        save_state(db_path, state)


def is_tripped(db_path: Path, model: str, probe: str = "") -> bool:
    """True when a (model, probe) pair should be skipped (tripped + cooldown)."""
    state = load_state(db_path)
    entry = None
    for key in (_key(model, probe), model):  # pair entry first, then legacy model-level
        entry = state.get("models", {}).get(key)
        if entry:
            break
    if not entry or not entry.get("tripped"):
        return False
    trip_at = entry.get("trip_at")
    if not trip_at:
        return False
    try:
        tripped = datetime.strptime(trip_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return False  # unparseable timestamp — never silently skip a model
    return _now() < tripped + timedelta(days=COOLDOWN_DAYS)


def tripped_models(
    db_path: Path, candidates: list[str], probe: str = ""
) -> tuple[list[str], list[str]]:
    """Split candidates into (runnable, skipped) for a given probe.
    Skipped entries are logged by the caller with their last recorded error."""
    runnable: list[str] = []
    skipped: list[str] = []
    for model in candidates:
        if is_tripped(db_path, model, probe):
            skipped.append(model)
        else:
            runnable.append(model)
    return runnable, skipped