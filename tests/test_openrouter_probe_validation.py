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

    def test_json_probe_requires_exact_schema(self):
        probe = probes.Probe(name="hermes_json_schema", prompt="p", expects_json=True)
        invalid_cases = [
            "not json",
            '{"ok": true}',
            '{"verdict":"likely_valid","confidence":2,"reasons":[],"missing_evidence":[]}',
            '{"verdict":"likely_valid","confidence":0.8,"reasons":[],"missing_evidence":[],"extra":1}',
        ]
        for text in invalid_cases:
            self.assertEqual(
                probes.validate_probe_response(probe=probe, http_ok=True, text=text, tool_call_valid=None),
                "invalid_json_schema",
                text,
            )
        valid = (
            '{"verdict":"needs_more_evidence","confidence":0.8,'
            '"reasons":["privileged setter"],"missing_evidence":["reachability"]}'
        )
        self.assertIsNone(
            probes.validate_probe_response(probe=probe, http_ok=True, text=valid, tool_call_valid=None)
        )
        self.assertEqual(probes.quality_score(probe=probe, http_ok=True, text=valid, tool_call_valid=None), 100.0)

    def test_triage_requires_verdict_confidence_and_two_checks(self):
        probe = probes.Probe(name="hermes_triage", prompt="p")
        weak = "Verdict: needs_more_evidence. Confidence: 0.8. More review is needed."
        self.assertEqual(
            probes.validate_probe_response(probe=probe, http_ok=True, text=weak, tool_call_valid=None),
            "triage_requirements_not_met",
        )
        strong = (
            "Verdict: needs_more_evidence. Confidence: 0.8. Check whether reward debt is reset and whether "
            "claimReward is gated by active position status. Also test repeated claims."
        )
        self.assertIsNone(
            probes.validate_probe_response(probe=probe, http_ok=True, text=strong, tool_call_valid=None)
        )
        self.assertGreaterEqual(probes.quality_score(probe=probe, http_ok=True, text=strong, tool_call_valid=None), 90)

    def test_summary_requires_all_sections_and_word_limit(self):
        probe = probes.Probe(name="hermes_evidence_summary", prompt="p")
        good = (
            "Claim: privileged misconfiguration may block swaps. Evidence: onlyOwner controls the fee. "
            "Assumptions: no hidden bound exists. Missing evidence: ordinary-user reachability is unconfirmed. "
            "Next deterministic check: trace every setter and bound."
        )
        self.assertIsNone(
            probes.validate_probe_response(probe=probe, http_ok=True, text=good, tool_call_valid=None)
        )
        too_long = "Claim Evidence Assumptions Missing Next deterministic check " + "x " * 181
        self.assertEqual(
            probes.validate_probe_response(probe=probe, http_ok=True, text=too_long, tool_call_valid=None),
            "summary_requirements_not_met",
        )

    def test_code_reasoning_requires_all_requested_concepts(self):
        probe = probes.Probe(name="hermes_code_reasoning", prompt="p")
        good = (
            "We need to confirm totalSupply denominator behavior. External token.transfer behavior may revert, return false, "
            "or reenter. Burning before transfer is not automatically harmful because a revert is atomic. These assumptions need confirmation."
        )
        self.assertIsNone(
            probes.validate_probe_response(probe=probe, http_ok=True, text=good, tool_call_valid=None)
        )

    def test_tool_probe_requires_exact_valid_tool_call(self):
        probe = probes.Probe(name="hermes_tool_probe", prompt="p", uses_tools=True)
        invalid = {"function": {"name": "wrong", "arguments": "{}"}}
        invalid_extra = {
            "function": {
                "name": "record_model_probe_verdict",
                "arguments": '{"verdict":"pass","confidence":0.8,"reason":"ok","extra":1}',
            }
        }
        valid = {
            "function": {
                "name": "record_model_probe_verdict",
                "arguments": '{"verdict":"pass","confidence":0.8,"reason":"clear evidence"}',
            }
        }
        self.assertFalse(probes.is_valid_tool_call(invalid))
        self.assertFalse(probes.is_valid_tool_call(invalid_extra))
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
