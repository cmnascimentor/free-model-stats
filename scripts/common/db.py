"""Unified SQLite schema and write helpers for the merged NIM + OpenRouter dashboard.

Keeps the `runs` / `model_results` shape NIMStats' index.html already knows how to
read (one run = one prompt tested against N models), tagged with a `platform`
column so both benchmark suites can share the same history.db and dashboard.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_DB = REPO_ROOT / "history.db"
MAX_RUNS = 2000
MAX_RESPONSE_CHARS = 2000


def as_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def from_json(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def connect(db_path: Path = HISTORY_DB) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")  # wait up to 30s on lock contention
    return conn


def cap_response(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= MAX_RESPONSE_CHARS:
        return text
    return text[:MAX_RESPONSE_CHARS] + "\n[truncated]"


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        -- DELETE (not WAL): the shipped history.db is uploaded as a single-file
        -- CI artifact and served statically, so it must never depend on a
        -- side-car -wal/-shm file that wouldn't be shipped with it.
        PRAGMA journal_mode=DELETE;

        CREATE TABLE IF NOT EXISTS runs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            platform      TEXT    NOT NULL DEFAULT 'nim',
            run_kind      TEXT    NOT NULL DEFAULT 'benchmark',
            probe_name    TEXT,
            prompt        TEXT,
            success_count INTEGER,
            total_models  INTEGER,
            fastest_model TEXT,
            fastest_time  INTEGER
        );

        CREATE TABLE IF NOT EXISTS model_results (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id           INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            model            TEXT    NOT NULL,
            success          INTEGER NOT NULL DEFAULT 0,
            error            TEXT,
            response_time    INTEGER,
            tokens_generated INTEGER,
            total_tokens     INTEGER,
            response         TEXT,
            validation_error TEXT
        );

        CREATE TABLE IF NOT EXISTS or_models (
            openrouter_id         TEXT PRIMARY KEY,
            name                  TEXT,
            context_length        INTEGER,
            max_completion_tokens INTEGER,
            input_modalities      TEXT,
            output_modalities     TEXT,
            supported_parameters  TEXT,
            pricing_prompt        TEXT,
            pricing_completion    TEXT,
            expiration_date       TEXT,
            knowledge_cutoff      TEXT,
            active                INTEGER NOT NULL DEFAULT 1,
            last_seen_at          TEXT
        );

        CREATE TABLE IF NOT EXISTS router_results (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id           INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            requested_model  TEXT NOT NULL,
            resolved_model   TEXT,
            probe_name       TEXT,
            success          INTEGER NOT NULL,
            http_status      INTEGER,
            error_type       TEXT,
            latency_ms       INTEGER,
            tokens_per_second REAL,
            score            REAL,
            created_at       TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_mr_run       ON model_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_mr_model     ON model_results(model);
        CREATE INDEX IF NOT EXISTS idx_runs_ts      ON runs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_runs_platform ON runs(platform);
        CREATE INDEX IF NOT EXISTS idx_router_run    ON router_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_router_resolved ON router_results(resolved_model);
        """
    )
    ensure_column(conn, "runs", "run_kind", "TEXT NOT NULL DEFAULT 'benchmark'")
    ensure_column(conn, "model_results", "validation_error", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_kind ON runs(run_kind)")
    conn.execute(
        """UPDATE runs
           SET run_kind = 'router'
           WHERE id IN (SELECT DISTINCT run_id FROM router_results)
             AND COALESCE(run_kind, 'benchmark') != 'router'"""
    )


def write_run(run: dict[str, Any], db_path: Path = HISTORY_DB, platform: str = "nim") -> int:
    """Insert a benchmark run (NIM-shaped: one prompt vs N models) and prune old runs."""
    summary = run.get("summary", {})
    conn = connect(db_path)
    try:
        init_schema(conn)
        cur = conn.execute(
            """INSERT INTO runs (timestamp, platform, run_kind, probe_name, prompt, success_count, total_models, fastest_model, fastest_time)
               VALUES (?, ?, 'benchmark', ?, ?, ?, ?, ?, ?)""",
            (
                run.get("timestamp"),
                platform,
                run.get("probe_name"),
                run.get("prompt"),
                summary.get("successCount"),
                summary.get("totalModels"),
                summary.get("fastestModel"),
                summary.get("fastestTime"),
            ),
        )
        run_id = cur.lastrowid
        conn.executemany(
            """INSERT INTO model_results
               (run_id, model, success, error, response_time, tokens_generated, total_tokens, response, validation_error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    run_id,
                    m.get("model"),
                    1 if m.get("success") else 0,
                    m.get("error"),
                    m.get("responseTime"),
                    m.get("tokensGenerated"),
                    m.get("totalTokens"),
                    cap_response(m.get("response")),
                    m.get("validationError") or m.get("validation_error"),
                )
                for m in run.get("models", [])
            ],
        )
        conn.execute(
            f"DELETE FROM runs WHERE id NOT IN "
            f"(SELECT id FROM runs ORDER BY timestamp DESC LIMIT {MAX_RUNS})"
        )
        conn.commit()
        return int(run_id)
    finally:
        conn.close()


def upsert_or_model(conn: sqlite3.Connection, model: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO or_models (
          openrouter_id, name, context_length, max_completion_tokens,
          input_modalities, output_modalities, supported_parameters,
          pricing_prompt, pricing_completion, expiration_date, knowledge_cutoff,
          active, last_seen_at
        ) VALUES (
          :openrouter_id, :name, :context_length, :max_completion_tokens,
          :input_modalities, :output_modalities, :supported_parameters,
          :pricing_prompt, :pricing_completion, :expiration_date, :knowledge_cutoff,
          :active, :last_seen_at
        )
        ON CONFLICT(openrouter_id) DO UPDATE SET
          name=excluded.name,
          context_length=excluded.context_length,
          max_completion_tokens=excluded.max_completion_tokens,
          input_modalities=excluded.input_modalities,
          output_modalities=excluded.output_modalities,
          supported_parameters=excluded.supported_parameters,
          pricing_prompt=excluded.pricing_prompt,
          pricing_completion=excluded.pricing_completion,
          expiration_date=excluded.expiration_date,
          knowledge_cutoff=excluded.knowledge_cutoff,
          active=excluded.active,
          last_seen_at=excluded.last_seen_at
        """,
        model,
    )


def mark_or_models_missing_inactive(conn: sqlite3.Connection, seen_ids: list[str]) -> int:
    if not seen_ids:
        return 0
    placeholders = ",".join("?" for _ in seen_ids)
    cur = conn.execute(
        f"UPDATE or_models SET active = 0 WHERE openrouter_id NOT IN ({placeholders})",
        seen_ids,
    )
    return int(cur.rowcount)


def insert_router_result(conn: sqlite3.Connection, run_id: int, result: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO router_results
           (run_id, requested_model, resolved_model, probe_name, success, http_status,
            error_type, latency_ms, tokens_per_second, score, created_at)
           VALUES (:run_id, :requested_model, :resolved_model, :probe_name, :success, :http_status,
                   :error_type, :latency_ms, :tokens_per_second, :score, :created_at)""",
        {**result, "run_id": run_id},
    )


def create_router_run(conn: sqlite3.Connection, *, timestamp: str, probe_name: str, prompt: str | None = None) -> int:
    cur = conn.execute(
        """INSERT INTO runs (timestamp, platform, run_kind, probe_name, prompt) VALUES (?, 'openrouter', 'router', ?, ?)""",
        (timestamp, probe_name, prompt),
    )
    return int(cur.lastrowid)
