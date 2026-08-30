#!/usr/bin/env python3
"""Local FastAPI/WebSocket runner for FreeModelStats.

The server executes benchmark scripts as subprocesses, persists runner job/event
history in history.db, and streams output to connected browsers. When
RUNNER_TOKEN is configured and the dashboard is served by this runner, a small
runtime auth shim is injected into index.html so browser fetch/WebSocket calls
automatically carry the token without storing it in the repository.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent
HISTORY_DB = REPO_ROOT / "history.db"
SCRIPTS_DIR = REPO_ROOT / "scripts"
PROMPTS_DIR = REPO_ROOT / "prompts"
COMMON_DIR = SCRIPTS_DIR / "common"
sys.path.insert(0, str(COMMON_DIR))

import db  # noqa: E402
from probe_suite import DEFAULT_PROBE  # noqa: E402

HOST = os.getenv("RUNNER_HOST", "127.0.0.1")
PORT = int(os.getenv("RUNNER_PORT", "8420"))
RUNNER_TOKEN = os.getenv("RUNNER_TOKEN", "").strip()
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _env_file_value(key: str) -> str:
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, val = line.partition("=")
        if k.strip() != key:
            continue
        val = val.strip()
        if val and val[0] not in ('"', "'") and " #" in val:
            val = val.split(" #", 1)[0].strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        return val
    return ""


def _runner_token() -> str:
    return RUNNER_TOKEN or _env_file_value("RUNNER_TOKEN")


def _check_token(candidate: str) -> bool:
    expected = _runner_token()
    if not expected:
        return True
    return hmac.compare_digest(candidate, expected)


def _load_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip()
            if val and val[0] not in ('"', "'") and " #" in val:
                val = val.split(" #", 1)[0].strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            env[key.strip()] = val
    env["PYTHONUNBUFFERED"] = "1"
    return env


PROBES = [
    {
        "id": "hermes_triage",
        "name": "Hermes Triage",
        "platform": "all",
        "difficulty": "easy",
        "difficultyLabel": "Easy",
        "type": ["text"],
        "description": "Shared cross-provider Web3 triage baseline with deterministic verdict/check validation.",
    },
    {
        "id": "hermes_evidence_summary",
        "name": "Hermes Evidence Summary",
        "platform": "all",
        "difficulty": "easy-moderate",
        "difficultyLabel": "Easy-Moderate",
        "type": ["text"],
        "description": "Five-part evidence summary under 180 words.",
    },
    {
        "id": "hermes_code_reasoning",
        "name": "Hermes Code Reasoning",
        "platform": "all",
        "difficulty": "moderate",
        "difficultyLabel": "Moderate",
        "type": ["text"],
        "description": "Solidity state-assumption reasoning with deterministic concept coverage.",
    },
    {
        "id": "hermes_json_schema",
        "name": "Hermes Exact JSON Schema",
        "platform": "all",
        "difficulty": "moderate-hard",
        "difficultyLabel": "Moderate-Hard",
        "type": ["json"],
        "description": "Must return the exact expected object keys, enums and field types.",
    },
    {
        "id": "hermes_tool_probe",
        "name": "Hermes Tool Probe",
        "platform": "openrouter",
        "difficulty": "hard",
        "difficultyLabel": "Hard",
        "type": ["tool-call"],
        "description": "Provider-specific exact function-call structure test.",
    },
]
VALID_PROBE_IDS = {p["id"] for p in PROBES}
PROBE_META = {p["id"]: p for p in PROBES}


def load_probe_prompts() -> dict[str, str]:
    prompts: dict[str, str] = {}
    if PROMPTS_DIR.is_dir():
        for path in sorted(PROMPTS_DIR.glob("*.txt")):
            prompts[path.stem] = path.read_text(encoding="utf-8").strip()
    return prompts


def get_known_models() -> dict[str, list[str]]:
    models: dict[str, list[str]] = {"nim": [], "openrouter": []}
    try:
        conn = db.connect(HISTORY_DB)
        db.init_schema(conn)
        rows = conn.execute(
            "SELECT DISTINCT r.platform, mr.model FROM model_results mr "
            "JOIN runs r ON mr.run_id = r.id ORDER BY mr.model"
        ).fetchall()
        for row in rows:
            platform = row["platform"]
            model = row["model"]
            if model not in models.setdefault(platform, []):
                models[platform].append(model)
        conn.close()
    except Exception:
        pass
    return models


def _inject_runner_auth(html: str) -> str:
    token = _runner_token()
    if not token:
        return html
    # This exists only in the locally served page. It is never written to disk,
    # committed, or exposed by GitHub Pages. Any client able to load this local
    # runner page already has network access to the runner itself.
    token_js = json.dumps(token)
    shim = f"""
<script id="runner-auth-shim">
(() => {{
  const runnerToken = {token_js};
  const NativeFetch = window.fetch.bind(window);
  window.fetch = (input, init = {{}}) => {{
    const rawUrl = typeof input === 'string' ? input : (input && input.url) || '';
    let url;
    try {{ url = new URL(rawUrl, location.href); }} catch (_) {{ return NativeFetch(input, init); }}
    if (url.port === '{PORT}' && url.pathname.startsWith('/api/')) {{
      const headers = new Headers(init.headers || (input instanceof Request ? input.headers : undefined));
      headers.set('X-Runner-Token', runnerToken);
      init = {{...init, headers}};
    }}
    return NativeFetch(input, init);
  }};

  const NativeWebSocket = window.WebSocket;
  class AuthWebSocket extends NativeWebSocket {{
    constructor(url, protocols) {{
      const parsed = new URL(url, location.href);
      if (parsed.port === '{PORT}' && parsed.pathname.startsWith('/ws/run/')) {{
        parsed.searchParams.set('token', runnerToken);
      }}
      if (protocols === undefined) super(parsed.toString());
      else super(parsed.toString(), protocols);
    }}
  }}
  window.WebSocket = AuthWebSocket;
}})();
</script>
"""
    return html.replace("</head>", shim + "\n</head>", 1)


app = FastAPI(title="FreeModelStats Runner", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["X-Runner-Token", "Content-Type"],
)
app.mount("/vendor", StaticFiles(directory=str(REPO_ROOT / "vendor")), name="vendor")

# Active process state only. Durable status/events live in SQLite.
active_runs: dict[str, dict[str, Any]] = {}


@app.get("/")
async def serve_dashboard():
    html = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(_inject_runner_auth(html))


@app.get("/history.db")
async def serve_db():
    return FileResponse(str(HISTORY_DB), media_type="application/octet-stream")


@app.get("/api/probes")
async def api_probes():
    prompts = load_probe_prompts()
    return [{**probe, "prompt": prompts.get(probe["id"], "")} for probe in PROBES]


@app.get("/api/models")
async def api_models():
    return get_known_models()


@app.get("/api/run/{run_id}")
async def api_run_status(run_id: str):
    conn = db.connect(HISTORY_DB)
    db.init_schema(conn)
    row = conn.execute("SELECT * FROM runner_jobs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, f"Unknown run {run_id}")
    events = db.load_runner_events(conn, run_id, limit=1000)
    conn.close()
    return {"job": dict(row), "events": events}


def _validated_string_list(value: Any, *, name: str, max_items: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > max_items or not all(isinstance(item, str) and item.strip() for item in value):
        raise HTTPException(400, f"{name} must be a list of at most {max_items} non-empty strings")
    return [item.strip() for item in value]


@app.post("/api/run")
async def start_run(request: Request, body: dict[str, Any]):
    if not _check_token(request.headers.get("X-Runner-Token", "")):
        raise HTTPException(403, "Invalid or missing X-Runner-Token")
    if any(not run.get("complete") for run in active_runs.values()):
        raise HTTPException(409, "A run is already in progress")

    platform = body.get("platform")
    if platform not in ("nim", "openrouter"):
        raise HTTPException(400, "platform must be 'nim' or 'openrouter'")

    probes = _validated_string_list(body.get("probes"), name="probes", max_items=20)
    models = _validated_string_list(body.get("models"), name="models", max_items=100)
    if not probes:
        probes = [DEFAULT_PROBE]
    unknown = [probe for probe in probes if probe not in VALID_PROBE_IDS]
    if unknown:
        raise HTTPException(400, f"Unknown probe(s): {', '.join(unknown)}")
    incompatible = [probe for probe in probes if PROBE_META[probe]["platform"] not in ("all", platform)]
    if incompatible:
        raise HTTPException(400, f"Probe(s) not supported by {platform}: {', '.join(incompatible)}")
    if platform == "nim" and len(probes) != 1:
        raise HTTPException(400, "NIM live runs currently accept exactly one probe per run")
    if platform == "nim" and models:
        raise HTTPException(400, "NIM live runner model selection is not yet supported; leave models empty")

    dry_run = bool(body.get("dry_run", False))
    run_id = uuid.uuid4().hex[:12]
    env = _load_subprocess_env()

    if platform == "nim":
        cmd = [
            sys.executable,
            "-u",
            str(SCRIPTS_DIR / "nim" / "test_models.py"),
            "--db",
            str(HISTORY_DB),
            "--probe",
            probes[0],
        ]
    else:
        cmd = [
            sys.executable,
            "-u",
            str(SCRIPTS_DIR / "openrouter" / "test_models.py"),
            "--db",
            str(HISTORY_DB),
        ]
        for probe in probes:
            cmd += ["--probe", probe]
        for model in models:
            cmd += ["--model", model]
    if dry_run:
        cmd.append("--dry-run")

    conn = db.connect(HISTORY_DB)
    db.init_schema(conn)
    db.create_runner_job(conn, run_id=run_id, platform=platform, command=cmd, created_at=utc_now())
    # Keep durable local runner history bounded.
    conn.execute(
        """DELETE FROM runner_jobs WHERE run_id NOT IN
           (SELECT run_id FROM runner_jobs ORDER BY created_at DESC LIMIT 100)"""
    )
    conn.commit()
    conn.close()

    active_runs[run_id] = {
        "cmd": cmd,
        "platform": platform,
        "env": env,
        "ws_clients": set(),
        "complete": False,
        "task": None,
        "process": None,
    }
    task = asyncio.create_task(_run_process(run_id, cmd, env))
    active_runs[run_id]["task"] = task
    return {"run_id": run_id, "cmd": " ".join(cmd)}


@app.post("/api/stop/{run_id}")
async def stop_run(run_id: str, request: Request):
    if not _check_token(request.headers.get("X-Runner-Token", "")):
        raise HTTPException(403, "Invalid or missing X-Runner-Token")
    run = active_runs.get(run_id)
    if not run:
        raise HTTPException(404, f"Unknown active run {run_id}")
    proc = run.get("process")
    if proc and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    await _broadcast(run_id, {"type": "status", "message": "Stopped by user"})
    await _finish_job(run_id, status="stopped", returncode=proc.returncode if proc else None, message="Stopped by user")
    run["complete"] = True
    return {"stopped": run_id}


@app.websocket("/ws/run/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str, token: str = ""):
    if not _check_token(token):
        await websocket.close(code=4403)
        return

    conn = db.connect(HISTORY_DB)
    db.init_schema(conn)
    job = conn.execute("SELECT * FROM runner_jobs WHERE run_id = ?", (run_id,)).fetchone()
    events = db.load_runner_events(conn, run_id, limit=1000) if job else []
    conn.close()
    if not job:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": f"Unknown run {run_id}"})
        await websocket.close()
        return

    await websocket.accept()
    for event in events:
        await websocket.send_text(json.dumps(event, default=str))

    run = active_runs.get(run_id)
    if not run or run.get("complete"):
        await websocket.close()
        return

    run["ws_clients"].add(websocket)
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        run["ws_clients"].discard(websocket)


async def _persist_event(run_id: str, event: dict[str, Any]) -> None:
    conn = db.connect(HISTORY_DB)
    db.init_schema(conn)
    db.insert_runner_event(conn, run_id=run_id, created_at=utc_now(), event=event)
    conn.commit()
    conn.close()


async def _broadcast(run_id: str, event: dict[str, Any]) -> None:
    await _persist_event(run_id, event)
    run = active_runs.get(run_id)
    if not run:
        return
    if event.get("type") == "complete":
        run["complete"] = True
    payload = json.dumps(event, default=str)
    dead: list[WebSocket] = []
    for ws in list(run.get("ws_clients", set())):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        run["ws_clients"].discard(ws)


async def _finish_job(run_id: str, *, status: str, returncode: int | None, message: str | None = None) -> None:
    conn = db.connect(HISTORY_DB)
    db.init_schema(conn)
    db.update_runner_job(
        conn,
        run_id,
        status=status,
        finished_at=utc_now(),
        returncode=returncode,
        message=message,
    )
    conn.commit()
    conn.close()


async def _run_process(run_id: str, cmd: list[str], env: dict[str, str]):
    await _broadcast(run_id, {"type": "start", "run_id": run_id, "cmd": " ".join(cmd)})
    conn = db.connect(HISTORY_DB)
    db.init_schema(conn)
    db.update_runner_job(conn, run_id, status="running", started_at=utc_now())
    conn.commit()
    conn.close()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(REPO_ROOT),
        )
        active_runs[run_id]["process"] = proc
        await _broadcast(run_id, {"type": "status", "message": "Process started"})

        while proc.stdout:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                await _broadcast(run_id, {"type": "output", "line": text})

        returncode = await proc.wait()
        status = "completed" if returncode == 0 else "failed"
        await _broadcast(run_id, {"type": "complete", "run_id": run_id, "returncode": returncode})
        await _finish_job(run_id, status=status, returncode=returncode)
    except Exception as exc:  # noqa: BLE001
        run = active_runs.get(run_id)
        if run:
            run["complete"] = True
        await _broadcast(run_id, {"type": "error", "message": str(exc)})
        await _finish_job(run_id, status="failed", returncode=None, message=str(exc))
    finally:
        await asyncio.sleep(30)
        active_runs.pop(run_id, None)


if __name__ == "__main__":
    conn = db.connect(HISTORY_DB)
    db.init_schema(conn)
    conn.commit()
    conn.close()

    if HOST not in LOOPBACK_HOSTS:
        print("=" * 66)
        print(f"WARNING: binding to {HOST} exposes the runner to your network.")
        print("Anyone reachable may read history.db; authenticated users can start/stop runs.")
        if not _runner_token():
            print("WARNING: RUNNER_TOKEN is not set; mutating endpoints are unauthenticated.")
        print("=" * 66)
    print(f"FreeModelStats Runner starting on http://{HOST}:{PORT}")
    print(f"History DB: {HISTORY_DB}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
