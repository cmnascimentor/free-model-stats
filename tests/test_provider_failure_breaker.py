"""Regression tests for provider-wide failures poisoning model breaker state."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests._load import load_module

breaker = load_module("provider_breaker_under_test", "scripts/common/breaker.py")


class ProviderFailureBreakerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.db = Path(self.tmp.name) / "history.db"
        self.db.touch()

    def tearDown(self):
        self.tmp.cleanup()

    def test_provider_scoped_statuses_never_trip_models(self):
        errors = (
            "HTTP 401: Missing Authentication header",
            "HTTP 402: account has insufficient credits",
            "HTTP 407: proxy authentication required",
        )
        for idx, error in enumerate(errors):
            model = f"m/provider-{idx}"
            for _ in range(breaker.FAILURE_THRESHOLD + 2):
                breaker.record_failure(self.db, model, error, probe="p")
            self.assertEqual(breaker.tripped_models(self.db, [model], probe="p"), ([model], []))

        self.assertEqual(breaker.load_state(self.db)["models"], {})

    def test_auth_scoped_403_never_trips_model(self):
        error = "HTTP 403: Invalid API key credentials for this account"
        for _ in range(breaker.FAILURE_THRESHOLD + 1):
            breaker.record_failure(self.db, "m/auth", error, probe="p")
        self.assertEqual(breaker.tripped_models(self.db, ["m/auth"], probe="p"), (["m/auth"], []))

    def test_model_scoped_403_still_trips(self):
        error = "HTTP 403: model is only available on agentic harnesses"
        for _ in range(breaker.FAILURE_THRESHOLD):
            breaker.record_failure(self.db, "m/model-only", error, probe="p")
        self.assertEqual(
            breaker.tripped_models(self.db, ["m/model-only"], probe="p"),
            ([], ["m/model-only"]),
        )

    def test_429_remains_model_scoped_for_existing_breaker_semantics(self):
        error = "HTTP 429: upstream model rate limit exceeded"
        for _ in range(breaker.FAILURE_THRESHOLD):
            breaker.record_failure(self.db, "m/rate-limited", error, probe="p")
        self.assertEqual(
            breaker.tripped_models(self.db, ["m/rate-limited"], probe="p"),
            ([], ["m/rate-limited"]),
        )

    def test_old_poisoned_state_version_is_ignored(self):
        state_path = breaker.state_path(self.db)
        state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "models": {
                        "m/poisoned::p": {
                            "consecutive_failures": 3,
                            "tripped": True,
                            "trip_at": "2099-01-01T00:00:00Z",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(
            breaker.tripped_models(self.db, ["m/poisoned"], probe="p"),
            (["m/poisoned"], []),
        )
        self.assertEqual(breaker.load_state(self.db)["version"], breaker.STATE_VERSION)
        self.assertEqual(breaker.load_state(self.db)["models"], {})


if __name__ == "__main__":
    unittest.main()
