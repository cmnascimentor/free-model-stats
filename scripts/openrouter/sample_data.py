#!/usr/bin/env python3
"""Offline fixtures used by dry-run and tests.

Sanitized (no keys/tokens - this is public catalog metadata) snapshot taken
from a live GET /v1/models pull against openrouter.ai on 2026-08-10. Includes
one zero-priced ":free" model and one non-free model with the same base name
so is_free_model()'s pricing check (not just the id suffix) has a fixture to
exercise.
"""
SAMPLE_MODELS = [
    {
        "id": "google/gemma-4-31b-it:free",
        "name": "Google: Gemma 4 31B (free)",
        "canonical_slug": "google/gemma-4-31b-it-20260402",
        "context_length": 262144,
        "architecture": {
            "input_modalities": ["image", "text", "video"],
            "output_modalities": ["text"],
            "modality": "text+image+video->text",
            "tokenizer": "Gemma",
        },
        "pricing": {"prompt": "0", "completion": "0"},
        "supported_parameters": [
            "include_reasoning",
            "max_tokens",
            "reasoning",
            "response_format",
            "seed",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        ],
        "top_provider": {"context_length": 262144, "max_completion_tokens": 32768, "is_moderated": False},
        "description": "Offline sample model for dry-run smoke tests.",
        "created": 1775148486,
        "expiration_date": None,
        "knowledge_cutoff": None,
    },
    {
        "id": "nvidia/nemotron-3-super-120b-a12b:free",
        "name": "NVIDIA: Nemotron 3 Super (free)",
        "canonical_slug": "nvidia/nemotron-3-super-120b-a12b-20230311",
        "context_length": 262144,
        "architecture": {
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "modality": "text->text",
            "tokenizer": "Other",
        },
        "pricing": {"prompt": "0", "completion": "0"},
        "supported_parameters": [
            "include_reasoning",
            "max_tokens",
            "reasoning",
            "reasoning_effort",
            "response_format",
            "seed",
            "structured_outputs",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        ],
        "top_provider": {"context_length": 262144, "max_completion_tokens": 262144, "is_moderated": False},
        "description": "Offline sample coder model for dry-run smoke tests.",
        "created": 1773245239,
        "expiration_date": None,
        "knowledge_cutoff": None,
    },
    {
        "id": "meta-llama/llama-3.3-70b-instruct",
        "name": "Meta: Llama 3.3 70B Instruct",
        "canonical_slug": "meta-llama/llama-3.3-70b-instruct",
        "context_length": 131072,
        "architecture": {
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "modality": "text->text",
            "tokenizer": "Llama",
        },
        "pricing": {"prompt": "0.00000012", "completion": "0.0000003"},
        "supported_parameters": ["temperature", "top_p", "max_tokens", "response_format"],
        "top_provider": {"context_length": 131072, "max_completion_tokens": 4096, "is_moderated": True},
        "description": "Offline sample non-free model (no ':free' suffix, non-zero pricing) for dry-run smoke tests.",
        "created": 1735689600,
        "expiration_date": None,
        "knowledge_cutoff": None,
    },
]

SAMPLE_TEXT = (
    "Verdict: needs more evidence. The capsule suggests a possible authorization or state-transition issue, "
    "but the exploitability claim depends on caller constraints and whether the affected path is reachable. "
    "Confidence: 0.62. Missing checks: role gates, invariant impact, and reproducible transaction trace."
)

SAMPLE_JSON = '{"verdict":"needs_more_evidence","confidence":0.62,"reasons":["reachability not proven","impact not quantified"],"missing_evidence":["role gate check","PoC trace"]}'
