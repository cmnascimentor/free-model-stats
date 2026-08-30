#!/usr/bin/env python3
"""Compatibility shim for the shared benchmark probe suite.

New code should import ``scripts/common/probe_suite.py`` directly. This module
remains so older scripts/tests that import ``scripts/openrouter/probes.py`` keep
working while NIM and OpenRouter use the same validators and benchmark version.
"""
from __future__ import annotations

import sys
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parent.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from probe_suite import *  # noqa: F401,F403,E402
