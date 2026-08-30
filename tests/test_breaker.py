"""Offline tests for the shared circuit breaker (scripts/common/breaker.py)."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests._load import load_module

breaker = load_module("breaker_under_test", "scripts/common/breaker.py")


class BreakerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.db = Path(self.tmp.name) / "history.db"
        self.db.touch()

    def tearDown(self):
        self.tmp.cleanup()

    # ── per-pair scoping ──────────────────────────────────────────────

    def test_starts_clean(self):
        runnable, skipped = breaker.tripped_models(self.db, ["m/a", "m/b"])
        self.assertEqual(runnable, ["m/a", "m/b"])
        self.assertEqual(skipped, [])

    def test_trip_is_probe_scoped(self):
        """3 transport failures on probe A trip only probe A; probe B stays runnable."""
        for _ in range(breaker.FAILURE_THRESHOLD):
            breaker.record_failure(self.db, "m/dead", "HTTP 404 returned by API", probe="probe_a")
        runnable_a, skipped_a = breaker.tripped_models(self.db, ["m/dead", "m/live"], probe="probe_a")
        self.assertEqual(skipped_a, ["m/dead"])
        self.assertEqual(runnable_a, ["m/live"])
        runnable_b, skipped_b = breaker.tripped_models(self.db, ["m/dead", "m/live"], probe="probe_b")
        self.assertEqual(runnable_b, ["m/dead", "m/live"])
        self.assertEqual(skipped_b, [])

    def test_legacy_model_level_key_applies_to_every_probe(self):
        """Pre-splitting state (bare model key, no '::') trips every probe."""
        state = {
            "version": breaker.STATE_VERSION,
            "models": {"m/old": {"consecutive_failures": 9, "tripped": True, "trip_at": "2000-01-01T00:00:00Z"}},
        }
        breaker.save_state(self.db, state)
        # trip_at far in the past means cooldown expired → runnable again
        runnable, skipped = breaker.tripped_models(self.db, ["m/old"], probe="any")
        self.assertEqual(runnable, ["m/old"])
        # fresh legacy trip is respected for any probe
        breaker.record_failure(  # will use bare key only when probe="" (legacy path)
            breaker.state_path(self.db).parent / "history.db", "m/old2", "HTTP 404", probe=""
        )
        for _ in range(breaker.FAILURE_THRESHOLD - 1):
            breaker.record_failure(self.db, "m/old2", "HTTP 404", probe="")
        runnable, skipped = breaker.tripped_models(self.db, ["m/old2"], probe="whatever")
        self.assertEqual(skipped, ["m/old2"])

    # ── thresholds ─────────────────────────────────────────────────────

    def test_trips_after_threshold(self):
        for _ in range(breaker.FAILURE_THRESHOLD):
            breaker.record_failure(self.db, "m/dead", "HTTP 404", probe="p")
        runnable, skipped = breaker.tripped_models(self.db, ["m/dead", "m/live"], probe="p")
        self.assertEqual(runnable, ["m/live"])
        self.assertEqual(skipped, ["m/dead"])

    def test_below_threshold_does_not_trip(self):
        for _ in range(breaker.FAILURE_THRESHOLD - 1):
            breaker.record_failure(self.db, "m/flaky", "HTTP 500", probe="p")
        runnable, skipped = breaker.tripped_models(self.db, ["m/flaky"], probe="p")
        self.assertEqual(runnable, ["m/flaky"])
        self.assertEqual(skipped, [])

    def test_success_resets_failure_count(self):
        breaker.record_failure(self.db, "m/flaky", "HTTP 500", probe="p")
        breaker.record_failure(self.db, "m/flaky", "HTTP 500", probe="p")
        breaker.record_success(self.db, "m/flaky")
        breaker.record_failure(self.db, "m/flaky", "HTTP 500", probe="p")
        breaker.record_failure(self.db, "m/flaky", "HTTP 500", probe="p")
        runnable, skipped = breaker.tripped_models(self.db, ["m/flaky"], probe="p")
        self.assertEqual(runnable, ["m/flaky"])  # only 2 consecutive

    # ── transport vs validation semantics ─────────────────────────────

    def test_validation_only_failures_never_trip(self):
        """No error_text (model answered but misbehaved) → ignored entirely, never trips."""
        for _ in range(breaker.FAILURE_THRESHOLD + 2):
            breaker.record_failure(self.db, "m/smart-but-wrong", "", probe="p")
        runnable, skipped = breaker.tripped_models(self.db, ["m/smart-but-wrong"], probe="p")
        self.assertEqual(runnable, ["m/smart-but-wrong"])
        self.assertEqual(skipped, [])
        # no state was written at all for signal-only failures
        state = breaker.load_state(self.db)
        self.assertEqual(state["models"], {})

    # ── success clearing semantics ─────────────────────────────────────

    def test_success_on_any_probe_clears_all_probes(self):
        for _ in range(breaker.FAILURE_THRESHOLD):
            breaker.record_failure(self.db, "m/recovered", "HTTP 404", probe="probe_a")
        self.assertEqual(breaker.tripped_models(self.db, ["m/recovered"], probe="probe_a")[1], ["m/recovered"])
        breaker.record_success(self.db, "m/recovered")  # success on any probe
        for probe in ("probe_a", "probe_b"):
            runnable, skipped = breaker.tripped_models(self.db, ["m/recovered"], probe=probe)
            self.assertEqual(runnable, ["m/recovered"])
            self.assertEqual(skipped, [])

    def test_success_clears_tripped_after_cooldown_expiry(self):
        for _ in range(breaker.FAILURE_THRESHOLD):
            breaker.record_failure(self.db, "m/recovered", "HTTP 404", probe="p")
        state = breaker.load_state(self.db)
        state["models"]["m/recovered::p"]["trip_at"] = "2000-01-01T00:00:00Z"  # cooldown long past
        breaker.save_state(self.db, state)
        self.assertFalse(breaker.is_tripped(self.db, "m/recovered", probe="p"))
        breaker.record_success(self.db, "m/recovered")
        self.assertEqual(breaker.tripped_models(self.db, ["m/recovered"], probe="p"), (["m/recovered"], []))

    # ── robustness ─────────────────────────────────────────────────────

    def test_corrupt_state_is_reset(self):
        breaker.state_path(self.db).write_text("{not json", encoding="utf-8")
        runnable, skipped = breaker.tripped_models(self.db, ["m/any"], probe="p")
        self.assertEqual(runnable, ["m/any"])
        breaker.record_failure(self.db, "m/any", "HTTP 404", probe="p")
        state = breaker.load_state(self.db)
        self.assertEqual(state["models"]["m/any::p"]["consecutive_failures"], 1)

    def test_unparseable_trip_timestamp_never_skips(self):
        breaker.record_failure(self.db, "m/x", "HTTP 404", probe="p")
        state = breaker.load_state(self.db)
        state["models"]["m/x::p"]["trip_at"] = "not-a-date"
        breaker.save_state(self.db, state)
        runnable, skipped = breaker.tripped_models(self.db, ["m/x"], probe="p")
        self.assertEqual(runnable, ["m/x"])

    def test_state_file_shape(self):
        breaker.record_failure(self.db, "m/x", "boom", probe="my_probe")
        data = json.loads(breaker.state_path(self.db).read_text())
        self.assertEqual(data["version"], breaker.STATE_VERSION)
        entry = data["models"]["m/x::my_probe"]
        self.assertEqual(entry["consecutive_failures"], 1)
        self.assertEqual(entry["last_error"], "boom")
        self.assertIn("last_failure_at", entry)
        self.assertNotIn("tripped", entry)


if __name__ == "__main__":
    unittest.main()