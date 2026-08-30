"""Unified SQLite schema and write helpers for FreeModelStats.

Schema v2 preserves the legacy ``runs`` / ``model_results`` shape consumed by
index.html while recording enough evidence to distinguish transport health,
format compliance and task correctness. Existing databases are migrated in
place with additive columns.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_DB = REPO_ROOT / "history.db"
MAX_RUNS = 2000
MAX_RESPONSE_CHARS = 4000


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
    conn.execute("PRAGMA busy_timeout=30000")
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
        -- DELETE rather than WAL because history.db is shipped as one static file.
        PRAGMA journal_mode=DELETE;

        CREATE TABLE IF NOT EXISTS runs (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp             TEXT    NOT NULL,
            platform              TEXT    NOT NULL DEFAULT 'nim',
            run_kind              TEXT    NOT NULL DEFAULT 'benchmark',
            probe_name            TEXT,
            prompt                TEXT,
            success_count         INTEGER,
            total_models          INTEGER,
            fastest_model         TEXT,
            fastest_time          INTEGER,
            benchmark_version     TEXT,
            probe_version         TEXT,
            temperature           REAL,
            max_completion_tokens INTEGER
        );

        CREATE TABLE IF NOT EXISTS model_results (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id             INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            model              TEXT    NOT NULL,
            success            INTEGER NOT NULL DEFAULT 0,
            error              TEXT,
            response_time      INTEGER,
            tokens_generated   INTEGER,
            total_tokens       INTEGER,
            response           TEXT,
            validation_error   TEXT,
            transport_success  INTEGER,
            format_success     INTEGER,
            task_success       INTEGER,
            quality_score      REAL,
            http_status        INTEGER,
            finish_reason      TEXT,
            resolved_model     TEXT,
            attempt_count      INTEGER,
            final_attempt_ms   INTEGER,
            total_elapsed_ms   INTEGER,
            ttft_ms            INTEGER,
            generation_ms      INTEGER,
            tokens_per_second  REAL,
            benchmark_version  TEXT,
            probe_version      TEXT
        );

        CREATE TABLE IF NOT EXISTS or_models (
            openrouter_id          TEXT PRIMARY KEY,
            name                   TEXT,
            context_length         INTEGER,
            max_completion_tokens  INTEGER,
            input_modalities       TEXT,
            output_modalities      TEXT,
            supported_parameters   TEXT,
            pricing_prompt         TEXT,
            pricing_completion     TEXT,
            expiration_date        TEXT,
            knowledge_cutoff       TEXT,
            active                 INTEGER NOT NULL DEFAULT 1,
            last_seen_at           TEXT,
            last_benchmarked_at    TEXT,
            benchmark_count        INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS router_results (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id            INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            requested_model   TEXT NOT NULL,
            resolved_model    TEXT,
            probe_name        TEXT,
            success           INTEGER NOT NULL,
            http_status       INTEGER,
            error_type        TEXT,
            latency_ms        INTEGER,
            tokens_per_second REAL,
            score             REAL,
            created_at        TEXT NOT NULL,
            validation_error  TEXT,
            attempt_count     INTEGER,
            total_elapsed_ms  INTEGER,
            benchmark_version TEXT,
            probe_version     TEXT
        );

        CREATE TABLE IF NOT EXISTS runner_jobs (
            run_id       TEXT PRIMARY KEY,
            platform     TEXT NOT NULL,
            command_json TEXT NOT NULL,
            status       TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            started_at   TEXT,
            finished_at  TEXT,
            returncode   INTEGER,
            message      TEXT
        );

        CREATE TABLE IF NOT EXISTS runner_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id     TEXT NOT NULL REFERENCES runner_jobs(run_id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            event_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_mr_run              ON model_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_mr_model            ON model_results(model);
        CREATE INDEX IF NOT EXISTS idx_runs_ts             ON runs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_runs_platform       ON runs(platform);
        CREATE INDEX IF NOT EXISTS idx_runs_probe          ON runs(probe_name);
        CREATE INDEX IF NOT EXISTS idx_router_run          ON router_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_router_resolved     ON router_results(resolved_model);
        CREATE INDEX IF NOT EXISTS idx_or_last_benchmarked ON or_models(last_benchmarked_at);
        CREATE INDEX IF NOT EXISTS idx_runner_events_run   ON runner_events(run_id, id);
        """
    )

    # Additive migration from pre-v2 databases.
    for column, definition in (
        ("run_kind", "TEXT NOT NULL DEFAULT 'benchmark'"),
        ("benchmark_version", "TEXT"),
        ("probe_version", "TEXT"),
        ("temperature", "REAL"),
        ("max_completion_tokens", "INTEGER"),
    ):
        ensure_column(conn, "runs", column, definition)

    for column, definition in (
        ("validation_error", "TEXT"),
        ("transport_success", "INTEGER"),
        ("format_success", "INTEGER"),
        ("task_success", "INTEGER"),
        ("quality_score", "REAL"),
        ("http_status", "INTEGER"),
        ("finish_reason", "TEXT"),
        ("resolved_model", "TEXT"),
        ("attempt_count", "INTEGER"),
        ("final_attempt_ms", "INTEGER"),
        ("total_elapsed_ms", "INTEGER"),
        ("ttft_ms", "INTEGER"),
        ("generation_ms", "INTEGER"),
        ("tokens_per_second", "REAL"),
        ("benchmark_version", "TEXT"),
        ("probe_version", "TEXT"),
    ):
        ensure_column(conn, "model_results", column, definition)

    for column, definition in (
        ("last_benchmarked_at", "TEXT"),
        ("benchmark_count", "INTEGER NOT NULL DEFAULT 0"),
    ):
        ensure_column(conn, "or_models", column, definition)

    for column, definition in (
        ("validation_error", "TEXT"),
        ("attempt_count", "INTEGER"),
        ("total_elapsed_ms", "INTEGER"),
        ("benchmark_version", "TEXT"),
        ("probe_version", "TEXT"),
    ):
        ensure_column(conn, "router_results", column, definition)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_kind ON runs(run_kind)")
    conn.execute(
        """UPDATE runs
           SET run_kind = 'router'
           WHERE id IN (SELECT DISTINCT run_id FROM router_results)
             AND COALESCE(run_kind, 'benchmark') != 'router'"""
    )
    # Backfill legacy semantics: historical success meant transport/non-empty success.
    conn.execute(
        """UPDATE model_results
           SET transport_success = COALESCE(transport_success, success),
               format_success = COALESCE(format_success, success),
               task_success = COALESCE(task_success, success),
               quality_score = COALESCE(quality_score, CASE WHEN success = 1 THEN 100.0 ELSE 0.0 END),
               total_elapsed_ms = COALESCE(total_elapsed_ms, response_time)
           WHERE transport_success IS NULL OR task_success IS NULL OR quality_score IS NULL OR total_elapsed_ms IS NULL"""
    )


def _bool_int(value: Any, default: bool = False) -> int:
    if value is None:
        return int(default)
    return 1 if bool(value) else 0


def write_run(run: dict[str, Any], db_path: Path = HISTORY_DB, platform: str = "nim") -> int:
    """Insert one benchmark run and prune old benchmark/router run rows."""
    summary = run.get("summary", {})
    conn = connect(db_path)
    try:
        init_schema(conn)
        cur = conn.execute(
            """INSERT INTO runs
               (timestamp, platform, run_kind, probe_name, prompt, success_count, total_models,
                fastest_model, fastest_time, benchmark_version, probe_version, temperature,
                max_completion_tokens)
               VALUES (?, ?, 'benchmark', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.get("timestamp"),
                platform,
                run.get("probe_name"),
                run.get("prompt"),
                summary.get("successCount"),
                summary.get("totalModels"),
                summary.get("fastestModel"),
                summary.get("fastestTime"),
                run.get("benchmark_version"),
                run.get("probe_version"),
                run.get("temperature"),
                run.get("max_completion_tokens"),
            ),
        )
        run_id = int(cur.lastrowid)
        rows = []
        for model in run.get("models", []):
            task_success = model.get("taskSuccess")
            if task_success is None:
                task_success = model.get("success", False)
            transport_success = model.get("transportSuccess")
            if transport_success is None:
                transport_success = bool(model.get("success")) or not bool(model.get("error"))
            format_success = model.get("formatSuccess")
            if format_success is None:
                format_success = task_success
            quality = model.get("qualityScore")
            if quality is None:
                quality = 100.0 if task_success else 0.0
            rows.append(
                (
                    run_id,
                    model.get("model"),
                    _bool_int(task_success),  # legacy dashboard field now means task-valid success
                    model.get("error"),
                    model.get("responseTime"),
                    model.get("tokensGenerated"),
                    model.get("totalTokens"),
                    cap_response(model.get("response")),
                    model.get("validationError") or model.get("validation_error"),
                    _bool_int(transport_success),
                    _bool_int(format_success),
                    _bool_int(task_success),
                    float(quality),
                    model.get("httpStatus"),
                    model.get("finishReason"),
                    model.get("resolvedModel"),
                    model.get("attemptCount"),
                    model.get("finalAttemptMs"),
                    model.get("totalElapsedMs") or model.get("responseTime"),
                    model.get("ttftMs"),
                    model.get("generationMs"),
                    model.get("tokensPerSecond"),
                    model.get("benchmarkVersion") or run.get("benchmark_version"),
                    model.get("probeVersion") or run.get("probe_version"),
                )
            )
        conn.executemany(
            """INSERT INTO model_results
               (run_id, model, success, error, response_time, tokens_generated, total_tokens,
                response, validation_error, transport_success, format_success, task_success,
                quality_score, http_status, finish_reason, resolved_model, attempt_count,
                final_attempt_ms, total_elapsed_ms, ttft_ms, generation_ms, tokens_per_second,
                benchmark_version, probe_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.execute(
            f"DELETE FROM runs WHERE id NOT IN "
            f"(SELECT id FROM runs ORDER BY timestamp DESC LIMIT {MAX_RUNS})"
        )
        conn.commit()
        return run_id
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


def mark_or_model_benchmarked(conn: sqlite3.Connection, model_id: str, timestamp: str) -> None:
    conn.execute(
        """UPDATE or_models
           SET last_benchmarked_at = ?, benchmark_count = COALESCE(benchmark_count, 0) + 1
           WHERE openrouter_id = ?""",
        (timestamp, model_id),
    )


def insert_router_result(conn: sqlite3.Connection, run_id: int, result: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO router_results
           (run_id, requested_model, resolved_model, probe_name, success, http_status,
            error_type, latency_ms, tokens_per_second, score, created_at, validation_error,
            attempt_count, total_elapsed_ms, benchmark_version, probe_version)
           VALUES (:run_id, :requested_model, :resolved_model, :probe_name, :success, :http_status,
                   :error_type, :latency_ms, :tokens_per_second, :score, :created_at, :validation_error,
                   :attempt_count, :total_elapsed_ms, :benchmark_version, :probe_version)""",
        {**result, "run_id": run_id},
    )


def create_router_run(
    conn: sqlite3.Connection,
    *,
    timestamp: str,
    probe_name: str,
    prompt: str | None = None,
    benchmark_version: str | None = None,
    probe_version: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO runs
           (timestamp, platform, run_kind, probe_name, prompt, benchmark_version, probe_version,
            temperature, max_completion_tokens)
           VALUES (?, 'openrouter', 'router', ?, ?, ?, ?, ?, ?)""",
        (timestamp, probe_name, prompt, benchmark_version, probe_version, temperature, max_completion_tokens),
    )
    return int(cur.lastrowid)


def create_runner_job(
    conn: sqlite3.Connection, *, run_id: str, platform: str, command: list[str], created_at: str
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO runner_jobs
           (run_id, platform, command_json, status, created_at)
           VALUES (?, ?, ?, 'queued', ?)""",
        (run_id, platform, as_json(command) or "[]", created_at),
    )


def update_runner_job(conn: sqlite3.Connection, run_id: str, **fields: Any) -> None:
    allowed = {"status", "started_at", "finished_at", "returncode", "message"}
    updates = {key: value for key, value in fields.items() if key in allowed}
    if not updates:
        return
    sql = ", ".join(f"{key} = ?" for key in updates)
    conn.execute(f"UPDATE runner_jobs SET {sql} WHERE run_id = ?", [*updates.values(), run_id])


def insert_runner_event(conn: sqlite3.Connection, *, run_id: str, created_at: str, event: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO runner_events (run_id, created_at, event_json) VALUES (?, ?, ?)",
        (run_id, created_at, as_json(event) or "{}"),
    )


def load_runner_events(conn: sqlite3.Connection, run_id: str, limit: int = 1000) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT event_json FROM runner_events WHERE run_id = ? ORDER BY id ASC LIMIT ?",
        (run_id, limit),
    ).fetchall()
    return [from_json(row["event_json"], {}) for row in rows]
