import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import load_module  # noqa: E402

discover = load_module("openrouter_discover_models", "scripts/openrouter/discover_models.py")


class IsFreeModelTests(unittest.TestCase):
    def test_free_suffixed_zero_priced_model_is_free(self):
        model = {
            "id": "meta-llama/llama-3.3-70b-instruct:free",
            "pricing": {"prompt": "0", "completion": "0", "request": "0"},
        }
        self.assertTrue(discover.is_free_model(model))

    def test_missing_request_price_key_defaults_to_free(self):
        # Matches OpenRouter's live /models response shape, which omits "request" for
        # most free models rather than including an explicit "0".
        model = {
            "id": "google/gemma-4-31b-it:free",
            "pricing": {"prompt": "0", "completion": "0"},
        }
        self.assertTrue(discover.is_free_model(model))

    def test_model_without_free_suffix_is_excluded(self):
        model = {
            "id": "meta-llama/llama-3.3-70b-instruct",
            "pricing": {"prompt": "0", "completion": "0", "request": "0"},
        }
        self.assertFalse(discover.is_free_model(model))

    def test_free_suffixed_but_nonzero_priced_model_is_excluded(self):
        model = {
            "id": "some-vendor/some-model:free",
            "pricing": {"prompt": "0.000002", "completion": "0", "request": "0"},
        }
        self.assertFalse(discover.is_free_model(model))


if __name__ == "__main__":
    unittest.main()
