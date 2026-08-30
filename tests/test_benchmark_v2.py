import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import load_module  # noqa: E402

db = load_module("db_v2", "scripts/common/db.py")
or_test_models = load_module("or_test_models_v2", "scripts/openrouter/test_models.py")


class BenchmarkV2SchemaTests(unittest.TestCase):
    def test_write_run_separates_transport_and_task_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.db"
            run_id = db.write_run(
                {
                    "timestamp": "2026-08-30T12:00:00Z",
                    "probe_name": "hermes_triage",
                    "prompt": "p",
                    "benchmark_version": "2.0.0",
                    "probe_version": "2.0.0",
                    "temperature": 0.0,
                    "max_completion_tokens": 512,
                    "models": [
                        {
                            "model": "vendor/model",
                            "success": False,
                            "transportSuccess": True,
                            "formatSuccess": True,
                            "taskSuccess": False,
                            "qualityScore": 55,
                            "validationError": "triage_requirements_not_met",
                            "responseTime": 123,
                            "attemptCount": 2,
                            "finalAttemptMs": 30,
                            "totalElapsedMs": 123,
                        }
                    ],
                    "summary": {"successCount": 0, "totalModels": 1},
                },
                db_path=path,
                platform="openrouter",
            )
            conn = sqlite3.connect(path)
            row = conn.execute(
                "SELECT success, transport_success, format_success, task_success, quality_score, "
                "attempt_count, final_attempt_ms, total_elapsed_ms FROM model_results WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            conn.close()
            self.assertEqual(row[:4], (0, 1, 1, 0))
            self.assertEqual(row[4], 55.0)
            self.assertEqual(row[5:], (2, 30, 123))

    def test_runner_events_are_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.db"
            conn = db.connect(path)
            db.init_schema(conn)
            db.create_runner_job(
                conn,
                run_id="abc123",
                platform="openrouter",
                command=["python", "bench.py"],
                created_at="2026-08-30T12:00:00+00:00",
            )
            db.insert_runner_event(
                conn,
                run_id="abc123",
                created_at="2026-08-30T12:00:01+00:00",
                event={"type": "output", "line": "hello"},
            )
            conn.commit()
            events = db.load_runner_events(conn, "abc123")
            conn.close()
            self.assertEqual(events, [{"line": "hello", "type": "output"}])


class OpenRouterRotationTests(unittest.TestCase):
    def test_new_then_oldest_models_are_selected_before_alphabetical_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.db"
            conn = db.connect(path)
            db.init_schema(conn)
            for model_id, last, count in [
                ("a/alphabetically-first:free", "2026-08-30T12:00:00Z", 20),
                ("z/new-model:free", None, 0),
                ("m/old-model:free", "2026-08-20T12:00:00Z", 3),
            ]:
                db.upsert_or_model(
                    conn,
                    {
                        "openrouter_id": model_id,
                        "name": model_id,
                        "context_length": 1,
                        "max_completion_tokens": 1,
                        "input_modalities": None,
                        "output_modalities": None,
                        "supported_parameters": None,
                        "pricing_prompt": "0",
                        "pricing_completion": "0",
                        "expiration_date": None,
                        "knowledge_cutoff": None,
                        "active": 1,
                        "last_seen_at": "2026-08-30T12:00:00Z",
                    },
                )
                conn.execute(
                    "UPDATE or_models SET last_benchmarked_at=?, benchmark_count=? WHERE openrouter_id=?",
                    (last, count, model_id),
                )
            conn.commit()
            selected = or_test_models.get_active_models(conn, None, 2)
            conn.close()
            self.assertEqual(selected, ["z/new-model:free", "m/old-model:free"])


if __name__ == "__main__":
    unittest.main()
