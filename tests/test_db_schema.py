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
            db.write_run(
                {
                    "timestamp": "2026-08-10T00:00:00Z",
                    "prompt": "p",
                    "models": [{"model": "vendor/model", "success": True, "response": "x" * 3000}],
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


if __name__ == "__main__":
    unittest.main()
