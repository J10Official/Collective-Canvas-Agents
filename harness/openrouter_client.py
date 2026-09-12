"""Small dependency-free OpenRouter client for multimodal canvas agents."""

from __future__ import annotations

import base64
import json
import os
import random
from pathlib import Path
from time import sleep
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-vision-exp"
DEFAULT_PROVIDER = "fireworks"
RATE_LIMIT_BACKOFF_SECONDS = (10, 10, 20, 20, 60, 60, 120)


def load_env(path: Path) -> None:
    """Load simple KEY=VALUE entries without printing or overwriting existing values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def image_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_vision_request(
    prompt: str,
    image_path: Path,
    response_format: dict[str, Any],
    *,
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    reasoning: str = "disabled",
    reasoning_max_tokens: int | None = None,
    reference_image_path: Path | None = None,
) -> dict[str, Any]:
    model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    provider = provider or os.environ.get("OPENROUTER_PROVIDER", DEFAULT_PROVIDER)
    uses_effort_only_reasoning = model == "openai/gpt-5.6-luna"
    if reasoning == "disabled":
        reasoning_config = {"enabled": False, "exclude": True}
    elif reasoning_max_tokens is not None and not uses_effort_only_reasoning:
        reasoning_config = {"max_tokens": reasoning_max_tokens, "exclude": True}
    else:
        reasoning_config = {"effort": reasoning, "exclude": True}
    provider_config: dict[str, Any] = {"order": [provider], "allow_fallbacks": False, "require_parameters": True}
    effective_response_format = response_format
    if provider in {"relace/fp4", "streamlake/fp8"}:
        # These exact endpoints are classified as fallback endpoints by
        # OpenRouter. `only` still prevents routing to any other endpoint.
        provider_config.update({"only": [provider], "allow_fallbacks": True, "quantizations": [provider.rsplit("/", 1)[1]]})
    if provider in {"relace/fp4", "streamlake/fp8"}:
        # These endpoints support JSON object but not OpenRouter's strict schema mode.
        effective_response_format = {"type": "json_object"}
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
    ]
    if reference_image_path is not None:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(reference_image_path)}})
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "provider": provider_config,
        "response_format": effective_response_format,
        "reasoning": reasoning_config,
        "max_tokens": max_tokens,
        "stream": False,
    }
    # OpenAI's Luna endpoints expose reasoning effort but do not accept a
    # temperature parameter. Keeping require_parameters=True prevents silent
    # parameter dropping, so omit temperature explicitly for this model.
    if model != "openai/gpt-5.6-luna":
        payload["temperature"] = temperature
    return payload


def request_preview(payload: dict[str, Any], image_path: Path, reference_image_path: Path | None = None) -> dict[str, Any]:
    """Return a persistable copy without embedding the large base64 image."""
    preview = json.loads(json.dumps(payload))
    image_paths = [image_path] + ([reference_image_path] if reference_image_path is not None else [])
    image_items = [item for item in preview["messages"][0]["content"] if item.get("type") == "image_url"]
    for item, path in zip(image_items, image_paths, strict=True):
        item["image_url"]["url"] = f"<base64 PNG: {path.name}, {path.stat().st_size} bytes>"
    return preview


def chat_completion(
    payload: dict[str, Any],
    api_key: str,
    timeout: int = 180,
    max_rate_limit_retries: int = len(RATE_LIMIT_BACKOFF_SECONDS),
    max_transient_retries: int = 2,
    max_invalid_output_retries: int = 3,
    validate_response: Callable[[dict[str, Any]], Any] | None = None,
    on_retry: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is empty; add it to harness/.env")
    rate_limit_retries = 0
    transient_retries = 0
    invalid_output_retries = 0
    attempts = 0
    retry_events: list[dict[str, Any]] = []

    def retry_delay(retry_number: int) -> float:
        return min(2.0**retry_number, 15.0) + random.uniform(0.15, 0.65)

    def report_retry(
        kind: str,
        retry_number: int,
        max_retries: int,
        delay_seconds: float,
        message: str,
        *,
        invalid_response: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "kind": kind,
            "retry": retry_number,
            "max_retries": max_retries,
            "delay_seconds": delay_seconds,
            "message": message,
        }
        retry_events.append(dict(event))
        callback_event = dict(event)
        if invalid_response is not None:
            callback_event["invalid_response"] = invalid_response
        if on_retry is not None:
            on_retry(callback_event)

    def record_client_attempts(result: dict[str, Any]) -> None:
        if attempts <= 1:
            return
        metadata: dict[str, Any] = {"attempts": attempts}
        if rate_limit_retries:
            metadata["rate_limit_retries"] = rate_limit_retries
        if transient_retries:
            metadata["transient_retries"] = transient_retries
        if invalid_output_retries:
            metadata["invalid_output_retries"] = invalid_output_retries
        metadata["retry_events"] = retry_events
        result["_client"] = metadata

    while True:
        attempts += 1
        request = Request(
            OPENROUTER_URL,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "Collective Canvas Pilot",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            if validate_response is not None:
                try:
                    validate_response(result)
                except (RuntimeError, ValueError) as exc:
                    if invalid_output_retries >= max_invalid_output_retries:
                        raise RuntimeError(
                            f"model output remained invalid after {invalid_output_retries + 1} attempts: {exc}"
                        ) from exc
                    invalid_output_retries += 1
                    message = (
                        "Model output was invalid; retrying immediately "
                        f"({invalid_output_retries}/{max_invalid_output_retries}): {exc}"
                    )
                    print(message)
                    report_retry(
                        "invalid_output",
                        invalid_output_retries,
                        max_invalid_output_retries,
                        0.0,
                        str(exc),
                        invalid_response=result,
                    )
                    continue
            record_client_attempts(result)
            return result
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and rate_limit_retries < max_rate_limit_retries:
                delay = float(RATE_LIMIT_BACKOFF_SECONDS[min(rate_limit_retries, len(RATE_LIMIT_BACKOFF_SECONDS) - 1)])
                rate_limit_retries += 1
                print(f"OpenRouter rate-limited a request; retrying in {delay:.1f}s ({rate_limit_retries}/{max_rate_limit_retries}).")
                report_retry("rate_limit", rate_limit_retries, max_rate_limit_retries, delay, f"HTTP 429: {detail}")
                sleep(delay)
                continue
            if exc.code in {408, 425, 500, 502, 503, 504} and transient_retries < max_transient_retries:
                transient_retries += 1
                delay = retry_delay(transient_retries - 1)
                print(f"OpenRouter returned transient HTTP {exc.code}; retrying in {delay:.1f}s ({transient_retries}/{max_transient_retries}).")
                report_retry("transient_http", transient_retries, max_transient_retries, delay, f"HTTP {exc.code}: {detail}")
                sleep(delay)
                continue
            suffix = f" after {attempts} attempts" if attempts > 1 else ""
            raise RuntimeError(f"OpenRouter rejected the request ({exc.code}){suffix}: {detail}") from exc
        except URLError as exc:
            if transient_retries < max_transient_retries:
                transient_retries += 1
                delay = retry_delay(transient_retries - 1)
                print(f"OpenRouter connection failed ({exc.reason}); retrying in {delay:.1f}s ({transient_retries}/{max_transient_retries}).")
                report_retry("connection", transient_retries, max_transient_retries, delay, str(exc.reason))
                sleep(delay)
                continue
            raise RuntimeError(f"Could not reach OpenRouter after {attempts} attempts: {exc.reason}") from exc
        except TimeoutError as exc:
            if transient_retries < max_transient_retries:
                transient_retries += 1
                delay = retry_delay(transient_retries - 1)
                print(f"OpenRouter read timed out; retrying in {delay:.1f}s ({transient_retries}/{max_transient_retries}).")
                report_retry("timeout", transient_retries, max_transient_retries, delay, str(exc) or "read timed out")
                sleep(delay)
                continue
            raise RuntimeError(f"OpenRouter read timed out after {attempts} attempts") from exc


def response_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenRouter response did not contain assistant content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if content is None:
        choice = response.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason", "unknown") if isinstance(choice, dict) else "unknown"
        usage = response.get("usage", {})
        details = usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}
        reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if finish_reason == "length" and reasoning_tokens:
            raise RuntimeError(f"model exhausted the output limit using {reasoning_tokens} reasoning tokens and returned no answer")
        raise RuntimeError(f"model returned no assistant content (finish_reason={finish_reason})")
    raise RuntimeError("OpenRouter assistant content had an unexpected type")
