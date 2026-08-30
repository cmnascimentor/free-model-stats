import importlib.util
import unittest
from pathlib import Path


try:
    spec = importlib.util.spec_from_file_location("runner_server_test", Path(__file__).resolve().parents[1] / "runner_server.py")
    runner = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(runner)
    RUNNER_AVAILABLE = True
except ModuleNotFoundError:
    runner = None
    RUNNER_AVAILABLE = False


@unittest.skipUnless(RUNNER_AVAILABLE, "optional FastAPI runner dependencies are not installed")
class RunnerAuthTests(unittest.TestCase):
    def setUp(self):
        self.original_token = runner.RUNNER_TOKEN
        runner.RUNNER_TOKEN = "test-secret-value"

    def tearDown(self):
        runner.RUNNER_TOKEN = self.original_token

    def test_session_value_is_derived_not_equal_to_secret(self):
        session = runner._session_value()
        self.assertTrue(session)
        self.assertNotEqual(session, runner.RUNNER_TOKEN)
        self.assertTrue(runner._check_session(session))
        self.assertFalse(runner._check_session("wrong"))

    def test_dashboard_html_never_contains_runner_token(self):
        html = runner._dashboard_html()
        self.assertNotIn(runner.RUNNER_TOKEN, html)
        self.assertIn('vendor/benchmark-v2.js', html)

    def test_direct_token_check_uses_configured_secret(self):
        self.assertTrue(runner._check_token("test-secret-value"))
        self.assertFalse(runner._check_token("test-secret-valuE"))


if __name__ == "__main__":
    unittest.main()
