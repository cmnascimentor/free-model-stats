"""Regression test: empty NIM artifacts must not fail the canonical merge."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tests._load import load_module

merge_results = load_module("nim_merge_results_under_test", "scripts/nim/merge_results.py")


class NimEmptyMergeTests(unittest.TestCase):
    def test_empty_provider_results_are_a_successful_noop(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            original_script_dir = merge_results.SCRIPT_DIR
            merge_results.SCRIPT_DIR = tmp_path
            try:
                payload = {
                    "timestamp": "2026-09-03T00:00:00Z",
                    "probe_name": "hermes_triage",
                    "prompt": "probe",
                    "benchmark_version": "2.0.0",
                    "probe_version": "2.0.0",
                    "temperature": 0.0,
                    "max_completion_tokens": 512,
                    "models": [],
                    "provider_error": "HTTP 401: HTTP 401 returned by API",
                    "summary": {
                        "successCount": 0,
                        "totalModels": 0,
                        "fastestModel": "N/A",
                        "fastestTime": 0,
                    },
                }
                for filename in merge_results.GROUP_FILES:
                    (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")

                with mock.patch.object(merge_results.db, "write_run") as write_run:
                    self.assertEqual(merge_results.main(), 0)
                    write_run.assert_not_called()

                for filename in merge_results.GROUP_FILES:
                    self.assertFalse((tmp_path / filename).exists())
            finally:
                merge_results.SCRIPT_DIR = original_script_dir


if __name__ == "__main__":
    unittest.main()
