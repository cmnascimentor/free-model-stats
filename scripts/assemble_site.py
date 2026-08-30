#!/usr/bin/env python3
"""Assemble the static dashboard used by GitHub Pages.

The legacy dashboard intentionally remains one buildless ``index.html``. Benchmark
methodology v2 lives in a separate, auditable adapter; this assembler injects the
adapter into the published HTML without requiring an npm/frontend build step.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ADAPTER_SRC = "vendor/benchmark-v2.js"
ADAPTER_TAG = f'<script src="{ADAPTER_SRC}"></script>'


def assemble_html(source: str) -> str:
    """Return dashboard HTML with exactly one benchmark-v2 adapter tag."""
    if source.count(ADAPTER_TAG) > 1:
        raise ValueError("benchmark-v2 adapter is referenced more than once")
    if ADAPTER_TAG in source:
        return source
    marker = "</body>"
    if marker not in source:
        raise ValueError("index.html has no </body> marker")
    return source.replace(marker, f"{ADAPTER_TAG}\n{marker}", 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble FreeModelStats static dashboard")
    parser.add_argument("--source", default="index.html")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    html = assemble_html(source_path.read_text(encoding="utf-8"))
    if html.count(ADAPTER_TAG) != 1:
        raise SystemExit("assembled dashboard must contain exactly one benchmark-v2 adapter tag")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"assembled {output_path} with {ADAPTER_SRC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
