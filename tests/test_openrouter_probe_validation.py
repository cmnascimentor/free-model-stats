import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import load_module  # noqa: E402

probes = load_module("openrouter_probes_validation", "scripts/openrouter/probes.py")


class ProbeValidationTests(unittest.TestCase):
    def test_empty_http_success_fails_validation(self):
        probe = probes.Probe(name="plain", prompt="p")
        self.assertEqual(
            probes.validate_probe_response(probe=probe, http_ok=True, text="", tool_call_valid=None),
            "empty_content",
        )

    def test_json_probe_requires_json_object(self):
        probe = probes.Probe(name="json", prompt="p", expects_json=True)
        self.assertEqual(
            probes.validate_probe_response(probe=probe, http_ok=True, text="not json", tool_call_valid=None),
            "invalid_json",
        )
        self.assertIsNone(
            probes.validate_probe_response(probe=probe, http_ok=True, text='{"ok": true}', tool_call_valid=None)
        )

    def test_tool_probe_requires_valid_tool_call(self):
        probe = probes.Probe(name="tool", prompt="p", uses_tools=True)
        invalid = {"function": {"name": "wrong", "arguments": "{}"}}
        valid = {
            "function": {
                "name": "record_model_probe_verdict",
                "arguments": '{"verdict":"pass","confidence":0.8,"reason":"clear evidence"}',
            }
        }
        self.assertFalse(probes.is_valid_tool_call(invalid))
        self.assertTrue(probes.is_valid_tool_call(valid))
        self.assertEqual(
            probes.validate_probe_response(probe=probe, http_ok=True, text="", tool_call_valid=False),
            "invalid_tool_call",
        )
        self.assertIsNone(
            probes.validate_probe_response(probe=probe, http_ok=True, text="", tool_call_valid=True)
        )


if __name__ == "__main__":
    unittest.main()
