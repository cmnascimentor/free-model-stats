import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import load_module  # noqa: E402

nim = load_module("nim_test_models", "scripts/nim/test_models.py")


class IsChatModelIdTests(unittest.TestCase):
    def test_known_chat_models_pass(self):
        for model_id in [
            "meta/llama-3.3-70b-instruct",
            "google/gemma-3-12b-it",
            "openai/gpt-oss-120b",
            "z-ai/glm-5.2",
        ]:
            self.assertTrue(nim.is_chat_model_id(model_id), model_id)

    def test_non_allowlisted_models_are_excluded(self):
        for model_id in [
            "nvidia/nv-embedqa-mistral-7b-v2",
            "meta/llama-guard-4-12b",
            "nvidia/nemotron-3.5-content-safety",
            "nvidia/nemotron-4-340b-reward",
            "google/deplot",
            "google/diffusiongemma-26b-a4b-it",
            "01-ai/yi-large",            # in catalog but 404 on chat/completions
            "databricks/dbrx-instruct",  # in catalog but 404
            "meta/llama2-70b",           # base model, not chat
            "bigcode/starcoder2-15b",    # code completion, not chat
            "adept/fuyu-8b",             # vision-only
        ]:
            self.assertFalse(nim.is_chat_model_id(model_id), model_id)


class DiscoverChatModelsTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, nim, "MODEL_LIMIT", nim.MODEL_LIMIT)

    def test_default_limit_is_unlimited(self):
        # NIM_MODEL_LIMIT is an opt-in cap: unset or 0 must benchmark every
        # discovered chat model, not just an alphabetically-first slice.
        self.assertEqual(nim.MODEL_LIMIT, 0)

    def test_zero_limit_returns_full_discovered_list(self):
        nim.fetch_catalog_model_ids = lambda: list(nim.CHAT_MODEL_ALLOWLIST) + ["nvidia/nv-embed-v1"]
        nim.MODEL_LIMIT = 0
        result = nim.discover_chat_models()
        self.assertEqual(result, sorted(nim.CHAT_MODEL_ALLOWLIST))

    def test_filters_to_allowlist_only(self):
        nim.fetch_catalog_model_ids = lambda: [
            "meta/llama-3.3-70b-instruct",
            "nvidia/nv-embed-v1",
            "openai/gpt-oss-120b",
            "meta/llama-guard-4-12b",
            "01-ai/yi-large",  # in catalog but not in allowlist
        ]
        nim.MODEL_LIMIT = 0
        result = nim.discover_chat_models()
        self.assertEqual(result, ["meta/llama-3.3-70b-instruct", "openai/gpt-oss-120b"])

    def test_respects_model_limit(self):
        nim.fetch_catalog_model_ids = lambda: list(nim.CHAT_MODEL_ALLOWLIST)
        nim.MODEL_LIMIT = 2
        result = nim.discover_chat_models()
        self.assertEqual(len(result), 2)
        self.assertEqual(result, sorted(nim.CHAT_MODEL_ALLOWLIST)[:2])

    def test_falls_back_when_fetch_raises(self):
        def boom():
            raise RuntimeError("network unreachable")

        nim.fetch_catalog_model_ids = boom
        result = nim.discover_chat_models()
        self.assertEqual(result, nim.FALLBACK_MODELS)

    def test_falls_back_when_all_models_filtered_out(self):
        nim.fetch_catalog_model_ids = lambda: ["nvidia/nv-embed-v1", "meta/llama-guard-4-12b"]
        result = nim.discover_chat_models()
        self.assertEqual(result, nim.FALLBACK_MODELS)


class SelectedModelsGroupSplitTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, nim, "MODEL_GROUP", nim.MODEL_GROUP)
        self.fixed_models = list(nim.CHAT_MODEL_ALLOWLIST)[:7]
        nim.discover_chat_models = lambda: list(self.fixed_models)

    def test_group1_and_group2_partition_all_models(self):
        nim.MODEL_GROUP = "group1"
        group1 = nim.selected_models()
        nim.MODEL_GROUP = "group2"
        group2 = nim.selected_models()

        self.assertEqual(set(group1) | set(group2), set(self.fixed_models))
        self.assertEqual(set(group1) & set(group2), set())
        self.assertEqual(sorted(group1 + group2), sorted(self.fixed_models))

    def test_all_group_returns_full_list(self):
        nim.MODEL_GROUP = "all"
        self.assertEqual(nim.selected_models(), self.fixed_models)

    def test_dry_run_uses_fallback_without_network(self):
        nim.MODEL_GROUP = "all"
        self.assertEqual(nim.selected_models(dry_run=True), nim.FALLBACK_MODELS)


if __name__ == "__main__":
    unittest.main()