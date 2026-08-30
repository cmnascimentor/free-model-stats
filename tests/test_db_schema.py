import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import load_module  # noqa: E402

db = load_module("db", "scripts/common/db.py")


class DbSchemaTests(unittest.TestCase):
    def test_run_kind_is_set_for_benchmark_and_router_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "history.db"
            conn = db.connect(db_path)
            db.init_schema(conn)
            conn.close()
            benchmark_id = db.write_run(
                {
                    "timestamp": "2026-08-10T00:00:00Z",
                    "prompt": "p",
                    "models": [{"model": "vendor/model", "success": True, "response": "ok"}],
                    "summary": {"successCount": 1, "totalModels": 1},
                },
                db_path=db_path,
                platform="openrouter",
            )
            conn = db.connect(db_path)
            db.init_schema(conn)
            router_id = db.create_router_run(conn, timestamp="2026-08-10T00:01:00Z", probe_name="probe")
            conn.commit()

            rows = {
                row["id"]: row["run_kind"]
                for row in conn.execute("SELECT id, run_kind FROM runs").fetchall()
            }
            self.assertEqual(rows[benchmark_id], "benchmark")
            self.assertEqual(rows[router_id], "router")
            conn.close()

    def test_foreign_key_cascade_prunes_model_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "history.db"
            run_id = db.write_run(
                {
                    "timestamp": "2026-08-10T00:00:00Z",
                    "prompt": "p",
                    "models": [{"model": "vendor/model", "success": True, "response": "ok"}],
                    "summary": {"successCount": 1, "totalModels": 1},
                },
                db_path=db_path,
                platform="nim",
            )
            conn = db.connect(db_path)
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()
            count = conn.execute("SELECT COUNT(*) FROM model_results").fetchone()[0]
            conn.close()
            self.assertEqual(count, 0)

    def test_response_is_capped_on_insert(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "history.db"
            oversized = "x" * (db.MAX_RESPONSE_CHARS + 1000)
            db.write_run(
                {
                    "timestamp": "2026-08-10T00:00:00Z",
                    "prompt": "p",
                    "models": [{"model": "vendor/model", "success": True, "response": oversized}],
                    "summary": {"successCount": 1, "totalModels": 1},
                },
                db_path=db_path,
                platform="nim",
            )
            conn = sqlite3.connect(db_path)
            response = conn.execute("SELECT response FROM model_results").fetchone()[0]
            conn.close()
            self.assertLessEqual(len(response), db.MAX_RESPONSE_CHARS + len("\n[truncated]"))
            self.assertTrue(response.endswith("[truncated]"))

    def test_real_pre_v2_schema_migrates_and_backfills_success(self):
        """Regression test for the schema that existed immediately before v2."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "history.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                PRAGMA foreign_keys=ON;
                CREATE TABLE runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    platform TEXT NOT NULL DEFAULT 'nim',
                    run_kind TEXT NOT NULL DEFAULT 'benchmark',
                    probe_name TEXT,
                    prompt TEXT,
                    success_count INTEGER,
                    total_models INTEGER,
                    fastest_model TEXT,
                    fastest_time INTEGER
                );
                CREATE TABLE model_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    success INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    response_time INTEGER,
                    tokens_generated INTEGER,
                    total_tokens INTEGER,
                    response TEXT,
                    validation_error TEXT
                );
                CREATE TABLE or_models (
                    openrouter_id TEXT PRIMARY KEY,
                    name TEXT,
                    context_length INTEGER,
                    max_completion_tokens INTEGER,
                    input_modalities TEXT,
                    output_modalities TEXT,
                    supported_parameters TEXT,
                    pricing_prompt TEXT,
                    pricing_completion TEXT,
                    expiration_date TEXT,
                    knowledge_cutoff TEXT,
                    active INTEGER NOT NULL DEFAULT 1,
                    last_seen_at TEXT
                );
                CREATE TABLE router_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    requested_model TEXT NOT NULL,
                    resolved_model TEXT,
                    probe_name TEXT,
                    success INTEGER NOT NULL,
                    http_status INTEGER,
                    error_type TEXT,
                    latency_ms INTEGER,
                    tokens_per_second REAL,
                    score REAL,
                    created_at TEXT NOT NULL
                );
                INSERT INTO runs(id,timestamp,platform,probe_name)
                VALUES(1,'2026-08-01T00:00:00Z','nim','legacy_probe');
                INSERT INTO model_results(run_id,model,success,response_time)
                VALUES(1,'vendor/model',1,42);
                INSERT INTO or_models(openrouter_id,name,active,last_seen_at)
                VALUES('vendor/free:free','Legacy Free',1,'2026-08-01T00:00:00Z');
                """
            )
            conn.commit()
            conn.close()

            conn = db.connect(db_path)
            db.init_schema(conn)
            row = conn.execute(
                "SELECT transport_success, format_success, task_success, quality_score, total_elapsed_ms "
                "FROM model_results"
            ).fetchone()
            or_row = conn.execute(
                "SELECT last_benchmarked_at, benchmark_count FROM or_models WHERE openrouter_id='vendor/free:free'"
            ).fetchone()
            indexes = {r[1] for r in conn.execute("PRAGMA index_list(or_models)").fetchall()}
            conn.close()

            self.assertEqual(tuple(row), (1, 1, 1, 100.0, 42))
            self.assertEqual(tuple(or_row), (None, 0))
            self.assertIn("idx_or_last_benchmarked", indexes)


if __name__ == "__main__":
    unittest.main()
