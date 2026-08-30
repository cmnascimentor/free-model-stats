#!/usr/bin/env python3
"""WebSocket benchmark runner server for the FreeModelStats dashboard.

Provides:
  - Static file serving for the dashboard (index.html, vendor/, etc.)
  - GET /api/probes       → list available probes with metadata
  - GET /api/models       → list known models grouped by platform
  - POST /api/run         → start a benchmark run, returns run_id
  - POST /api/stop/{id}   → terminate a running benchmark
  - WS   /ws/run/{run_id} → stream live events for a running benchmark

The runner executes the existing scripts (scripts/openrouter/test_models.py,
scripts/openrouter/test_router.py, scripts/nim/test_models.py) as subprocesses
and streams their stdout line-by-line to the browser over WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ─── Config ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
HISTORY_DB = REPO_ROOT / "history.db"
SCRIPTS_DIR = REPO_ROOT / "scripts"
PROMPTS_DIR = REPO_ROOT / "prompts"

HOST = os.getenv("RUNNER_HOST", "127.0.0.1")  # L2/H2: localhost-only by default
PORT = int(os.getenv("RUNNER_PORT", "8420"))
# Optional shared-secret auth. When set, mutating endpoints require the
# X-Runner-Token header (WebSocket: ?token= query param). Recommended whenever
# RUNNER_HOST is anything other than loopback.
RUNNER_TOKEN = os.getenv("RUNNER_TOKEN", "").strip()

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _env_file_value(key: str) -> str:
    """Look up KEY in the repo .env (same loose parser used for subprocess env)."""
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return ""
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, val = line.partition("=")
        if k.strip() == key:
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            return val
    return ""


def _check_token(request_token: str) -> bool:
    """Constant-time-ish comparison; empty RUNNER_TOKEN disables auth."""
    expected = RUNNER_TOKEN or _env_file_value("RUNNER_TOKEN")
    if not expected:
        return True
    import hmac

    return hmac.compare_digest(request_token, expected)

# ─── Probe / Model catalog ──────────────────────────────────────────────────

PROBES = [
    {
        "id": "nim_prime",
        "name": "NIM Prime Check",
        "platform": "nim",
        "difficulty": "trivial",
        "difficultyLabel": "Trivial",
        "type": ["text"],
        "description": "Simple coding task — write an is_prime function. Bare minimum: any model that responds passes.",
    },
    {
        "id": "hermes_triage",
        "name": "hermes_triage",
        "platform": "openrouter",
        "difficulty": "easy",
        "difficultyLabel": "Easy",
        "type": ["text"],
        "description": "Classify a Web3 finding. Validation only checks non-empty response.",
    },
    {
        "id": "hermes_evidence_summary",
        "name": "hermes_evidence_summary",
        "platform": "openrouter",
        "difficulty": "easy-moderate",
        "difficultyLabel": "Easy-Moderate",
        "type": ["text"],
        "description": "Summarize an audit capsule into 5 categories under 180 words.",
    },
    {
        "id": "hermes_code_reasoning",
        "name": "hermes_code_reasoning",
        "platform": "openrouter",
        "difficulty": "moderate",
        "difficultyLabel": "Moderate",
        "type": ["text"],
        "description": 'Analyze a Solidity snippet. Says "Do not invent missing code."',
    },
    {
        "id": "hermes_json_schema",
        "name": "hermes_json_schema",
        "platform": "openrouter",
        "difficulty": "moderate-hard",
        "difficultyLabel": "Moderate-Hard",
        "type": ["json"],
        "description": "Must return valid JSON with exact schema. Many free models fail this.",
    },
    {
        "id": "hermes_tool_probe",
        "name": "hermes_tool_probe",
        "platform": "openrouter",
        "difficulty": "hard",
        "difficultyLabel": "Hard",
        "type": ["tool-call"],
        "description": "Must produce a valid function call with exact name and schema. Hardest test.",
    },
]

VALID_PROBE_IDS = {p["id"] for p in PROBES}


def load_probe_prompts() -> dict[str, str]:
    """Load prompt text from files in prompts/ directory."""
    prompts: dict[str, str] = {}
    if PROMPTS_DIR.is_dir():
        for path in sorted(PROMPTS_DIR.glob("*.txt")):
            prompts[path.stem] = path.read_text(encoding="utf-8").strip()
    # NIM prompt is hardcoded in the script
    prompts["nim_prime"] = (
        "Write a Python function that checks if a number is prime and returns True or False"
    )
    return prompts


def get_known_models() -> dict[str, list[str]]:
    """Read model names from history.db grouped by platform."""
    models: dict[str, list[str]] = {"nim": [], "openrouter": []}
    try:
        import sqlite3

        conn = sqlite3.connect(str(HISTORY_DB))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT DISTINCT r.platform, mr.model "
            "FROM model_results mr JOIN runs r ON mr.run_id = r.id "
            "ORDER BY mr.model"
        ).fetchall()
        seen: set[str] = set()
        for row in rows:
            m = row["model"]
            if m in seen:
                continue
            seen.add(m)
            models.setdefault(row["platform"], []).append(m)
        conn.close()
    except Exception:
        pass
    return models


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="FreeModelStats Runner", docs_url=None, redoc_url=None)

# Allow the documented local flows: dashboard served from any localhost port
# (http.server :8000, another static server) can still reach the runner :8420.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["X-Runner-Token", "Content-Type"],
)

# Active runs: run_id → {process, ws_set, log, task}
active_runs: dict[str, dict[str, Any]] = {}

# ─── Static files ────────────────────────────────────────────────────────────

app.mount("/vendor", StaticFiles(directory=str(REPO_ROOT / "vendor")), name="vendor")


@app.get("/")
async def serve_dashboard():
    return FileResponse(str(REPO_ROOT / "index.html"), media_type="text/html")


@app.get("/history.db")
async def serve_db():
    return FileResponse(str(HISTORY_DB), media_type="application/octet-stream")


# ─── API endpoints ────────────────────────────────────────────────────────────


@app.get("/api/probes")
async def api_probes():
    prompts = load_probe_prompts()
    result = []
    for p in PROBES:
        entry = {**p, "prompt": prompts.get(p["id"], "")}
        result.append(entry)
    return result


@app.get("/api/models")
async def api_models():
    return get_known_models()


@app.post("/api/run")
async def start_run(request: Request, body: dict[str, Any]):
    """Start a benchmark run. Body must have 'platform' ('nim' or 'openrouter')
    and optionally 'probes' (list of probe ids) and 'models' (list of model ids
    for OpenRouter)."""
    if not _check_token(request.headers.get("X-Runner-Token", "")):
        raise HTTPException(403, "Invalid or missing X-Runner-Token")
    # Concurrency guard: benchmark scripts write to the shared history.db with
    # no cross-process locking strategy beyond SQLite defaults — serialize runs.
    if any(not run.get("complete") for run in active_runs.values()):
        raise HTTPException(409, "A run is already in progress")
    platform = body.get("platform")
    if platform not in ("nim", "openrouter"):
        raise HTTPException(400, "platform must be 'nim' or 'openrouter'")

    # M4: Validate probes against allowlist
    probes = body.get("probes", [])
    unknown = [p for p in probes if p not in VALID_PROBE_IDS]
    if unknown:
        raise HTTPException(400, f"Unknown probe(s): {', '.join(unknown)}")

    models = body.get("models", [])
    # Cap list sizes to prevent abuse
    if len(probes) > 20:
        raise HTTPException(400, "Too many probes (max 20)")
    if len(models) > 100:
        raise HTTPException(400, "Too many models (max 100)")

    dry_run = body.get("dry_run", False)
    run_id = uuid.uuid4().hex[:12]

    env = os.environ.copy()
    # Load .env if present
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            # L3: Strip surrounding quotes from values
            val = val.strip()
            # Inline comments: KEY=value # note (never inside quotes)
            if val and val[0] not in ('"', "'") and " #" in val:
                val = val.split(" #", 1)[0].strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            env[key.strip()] = val

    # M1: Force unbuffered Python output for live streaming
    env["PYTHONUNBUFFERED"] = "1"

    if platform == "nim":
        cmd = [sys.executable, "-u", str(SCRIPTS_DIR / "nim" / "test_models.py"), "--db", str(HISTORY_DB)]
        if dry_run:
            cmd.append("--dry-run")
    else:
        cmd = [sys.executable, "-u", str(SCRIPTS_DIR / "openrouter" / "test_models.py"), "--db", str(HISTORY_DB)]
        for probe in probes:
            cmd += ["--probe", probe]
        for model in models:
            cmd += ["--model", model]
        if dry_run:
            cmd.append("--dry-run")

    # H1: Buffer events per run so late-connecting WS clients get replayed history
    active_runs[run_id] = {
        "cmd": cmd,
        "platform": platform,
        "env": env,
        "ws_clients": set(),
        "log": [],        # buffered events for replay
        "complete": False, # whether the run has finished
        "task": None,      # L1: prevent GC of background task
        "process": None,   # M2: store subprocess handle for kill
    }

    # L1: Store task reference to prevent garbage collection
    task = asyncio.create_task(_run_process(run_id, cmd, env))
    active_runs[run_id]["task"] = task

    return {"run_id": run_id, "cmd": " ".join(cmd)}


# M2: Stop endpoint to kill a running subprocess
@app.post("/api/stop/{run_id}")
async def stop_run(run_id: str, request: Request):
    if not _check_token(request.headers.get("X-Runner-Token", "")):
        raise HTTPException(403, "Invalid or missing X-Runner-Token")
    run = active_runs.get(run_id)
    if not run:
        raise HTTPException(404, f"Unknown run {run_id}")
    proc = run.get("process")
    if proc and proc.returncode is None:
        proc.terminate()
        # Give it a moment, then kill if needed
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
        await _broadcast(run_id, {"type": "status", "message": "Stopped by user"})
    return {"stopped": run_id}


# ─── WebSocket ────────────────────────────────────────────────────────────────


@app.websocket("/ws/run/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str, token: str = ""):
    # Browsers cannot set custom headers on WebSocket — token rides the query string.
    if not _check_token(token):
        await websocket.close(code=4403)
        return
    await websocket.accept()

    run = active_runs.get(run_id)
    if not run:
        try:
            await websocket.send_json({"type": "error", "message": f"Unknown run {run_id}"})
        except Exception:
            pass
        await websocket.close()
        return

    # H1: Replay buffered events so late-connecting clients don't miss anything
    for event in run.get("log", []):
        try:
            await websocket.send_text(event if isinstance(event, str) else json.dumps(event, default=str))
        except Exception:
            break

    run["ws_clients"].add(websocket)
    try:
        # Keep the connection alive — client receives events
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Timeout is normal — just keep the connection open
                pass
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        run["ws_clients"].discard(websocket)


async def _broadcast(run_id: str, event: dict[str, Any]):
    """Send an event dict as JSON to all WebSocket clients for a run.
    Also buffers the event for replay to late-connecting clients (H1)."""
    run = active_runs.get(run_id)
    if not run:
        return
    payload = json.dumps(event, default=str)

    # H1: Buffer for replay
    run.setdefault("log", []).append(payload)
    if event.get("type") == "complete":
        run["complete"] = True

    # M3: Iterate a snapshot to avoid "Set changed size during iteration"
    dead: list[WebSocket] = []
    for ws in list(run.get("ws_clients", set())):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        run["ws_clients"].discard(ws)


async def _run_process(run_id: str, cmd: list[str], env: dict[str, str]):
    """Run a subprocess and stream stdout lines as WebSocket events."""
    await _broadcast(run_id, {"type": "start", "run_id": run_id, "cmd": " ".join(cmd)})

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(REPO_ROOT),
        )
        # M2: Store process handle so /api/stop can terminate it
        active_runs[run_id]["process"] = proc

        await _broadcast(run_id, {"type": "status", "message": "Process started"})

        # Stream output lines
        while proc.stdout:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                await _broadcast(run_id, {"type": "output", "line": text})

        returncode = await proc.wait()
        await _broadcast(
            run_id,
            {"type": "complete", "run_id": run_id, "returncode": returncode},
        )

    except Exception as exc:
        await _broadcast(run_id, {"type": "error", "message": str(exc)})
    finally:
        # Clean up after 30s so clients can still read the final events
        await asyncio.sleep(30)
        active_runs.pop(run_id, None)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if HOST not in LOOPBACK_HOSTS:
        print("=" * 62)
        print(f"⚠  WARNING: binding to {HOST} exposes this server to your network.")
        print("   Anyone reachable can start benchmarks (burning API quota),")
        print("   read history.db, or kill runs.")
        if not (RUNNER_TOKEN or _env_file_value("RUNNER_TOKEN")):
            print("⚠  RUNNER_TOKEN is NOT set — mutating endpoints are UNAUTHENTICATED.")
            print("   Set RUNNER_TOKEN=<secret> in .env or the environment.")
        print("=" * 62)
    print(f"FreeModelStats Runner starting on http://{HOST}:{PORT}")
    print(f"Serving dashboard from {REPO_ROOT}")
    print(f"History DB: {HISTORY_DB}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")