import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import load_module  # noqa: E402

client_mod = load_module("openrouter_client", "scripts/openrouter/openrouter_client.py")
ApiResult = client_mod.ApiResult
OpenRouterClient = client_mod.OpenRouterClient
MAX_RETRIES = client_mod.MAX_RETRIES


def _result(status, ok=None):
    if ok is None:
        ok = status == 200
    return ApiResult(
        ok,
        status,
        {} if ok else None,
        None if ok else f"http_{status}",
        None if ok else "err",
        1,
    )


class RetryLogicTests(unittest.TestCase):
    def setUp(self):
        self._orig_sleep = client_mod.time.sleep
        self._orig_do_request = client_mod.OpenRouterClient._do_request
        client_mod.time.sleep = lambda *_a, **_k: None
        self.attempts = 0

    def tearDown(self):
        client_mod.time.sleep = self._orig_sleep
        client_mod.OpenRouterClient._do_request = self._orig_do_request

    def _install(self, scripted):
        def fake(_self, _req):
            result = scripted[min(self.attempts, len(scripted) - 1)]
            self.attempts += 1
            return result
        client_mod.OpenRouterClient._do_request = fake

    def test_502_retries_max_then_returns_last_with_attempt_count(self):
        self._install([_result(502)])
        client = OpenRouterClient(api_key="x")
        result = client.request("GET", "/models")
        self.assertEqual(result.status, 502)
        self.assertEqual(self.attempts, MAX_RETRIES + 1)
        self.assertEqual(result.attempt_count, MAX_RETRIES + 1)
        self.assertIsNotNone(result.total_elapsed_ms)

    def test_transient_failure_then_success_reports_two_attempts(self):
        self._install([_result(503), _result(200)])
        client = OpenRouterClient(api_key="x")
        result = client.request("GET", "/models")
        self.assertTrue(result.ok)
        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(self.attempts, 2)

    def test_200_returns_immediately(self):
        self._install([_result(200)])
        client = OpenRouterClient(api_key="x")
        result = client.request("GET", "/models")
        self.assertTrue(result.ok)
        self.assertEqual(self.attempts, 1)
        self.assertEqual(result.attempt_count, 1)

    def test_429_does_not_retry(self):
        self._install([_result(429)])
        client = OpenRouterClient(api_key="x")
        result = client.request("GET", "/models")
        self.assertEqual(result.status, 429)
        self.assertEqual(self.attempts, 1)
        self.assertEqual(result.attempt_count, 1)


if __name__ == "__main__":
    unittest.main()
