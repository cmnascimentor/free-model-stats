#!/usr/bin/env python3
"""Minimal OpenRouter HTTP client using only Python standard library."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_TITLE = "free-model-stats"
MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.getenv("OPENROUTER_RETRY_BASE_DELAY", "5"))
# 429 is deliberately excluded: on the free tier, retrying a rate-limit burns
# more quota. Only transient gateway/server failures are retried.
RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})


@dataclass
class ApiResult:
    ok: bool
    status: int | None
    data: dict[str, Any] | None
    error_type: str | None
    error_message: str | None
    latency_ms: int  # final HTTP attempt only
    retry_after: float | None = None
    attempt_count: int = 1
    total_elapsed_ms: int | None = None  # includes retries and retry backoff


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, timeout: int = 90) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", DEFAULT_TITLE),
            "X-OpenRouter-Metadata": "enabled",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        referer = os.getenv("OPENROUTER_SITE_URL")
        if referer:
            headers["HTTP-Referer"] = referer
        return headers

    def _do_request(self, req: urllib.request.Request) -> ApiResult:
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                latency_ms = int((time.perf_counter() - start) * 1000)
                try:
                    data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    return ApiResult(False, response.status, None, "invalid_json", raw[:500], latency_ms)
                return ApiResult(True, response.status, data, None, None, latency_ms)
        except urllib.error.HTTPError as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            raw = exc.read().decode("utf-8", errors="replace")
            err_type = f"http_{exc.code}"
            err_msg = raw[:1000]
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    error = parsed.get("error") or parsed
                    err_msg = str(error.get("message") or error)[:1000] if isinstance(error, dict) else str(error)[:1000]
            except json.JSONDecodeError:
                pass
            retry_after = _parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
            return ApiResult(False, exc.code, None, err_type, err_msg, latency_ms, retry_after)
        except Exception as exc:  # noqa: BLE001 - benchmark client records operational failures.
            latency_ms = int((time.perf_counter() - start) * 1000)
            return ApiResult(False, None, None, exc.__class__.__name__, str(exc)[:1000], latency_ms)

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> ApiResult:
        url = f"{OPENROUTER_BASE_URL}{path}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        operation_started = time.perf_counter()
        result = ApiResult(False, None, None, "max_retries_exceeded", "Max retries exceeded", 0)

        for attempt in range(MAX_RETRIES + 1):
            req = urllib.request.Request(url=url, data=body, method=method, headers=self._headers())
            result = self._do_request(req)
            result.attempt_count = attempt + 1

            if result.status in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                if result.retry_after is not None:
                    delay = max(delay, result.retry_after)
                print(
                    f"  HTTP {result.status} from OpenRouter, retrying in {delay:.0f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES + 1})",
                    flush=True,
                )
                time.sleep(delay)
                continue

            result.total_elapsed_ms = int((time.perf_counter() - operation_started) * 1000)
            return result

        result.total_elapsed_ms = int((time.perf_counter() - operation_started) * 1000)
        return result

    def get_models(self) -> ApiResult:
        return self.request("GET", "/models")

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ApiResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        return self.request("POST", "/chat/completions", payload)
