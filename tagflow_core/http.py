"""HTTP reliability helpers for local and cloud AI endpoints."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

import requests

TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class ApiRequestError(RuntimeError):
    """A normalized AI API request or response failure."""


@dataclass(frozen=True)
class HttpPolicy:
    """Timeout and retry policy for an AI inference request."""

    connect_timeout: float = 10.0
    read_timeout: float = 300.0
    retries: int = 2
    backoff_seconds: float = 0.75
    max_error_body_chars: int = 1000

    def __post_init__(self) -> None:
        if self.connect_timeout <= 0 or self.read_timeout <= 0:
            raise ValueError("HTTPタイムアウトは0より大きい必要があります。")
        if self.retries < 0:
            raise ValueError("HTTP再試行回数は0以上である必要があります。")
        if self.backoff_seconds < 0:
            raise ValueError("HTTPバックオフ秒数は0以上である必要があります。")
        if self.max_error_body_chars < 0:
            raise ValueError("HTTPエラー本文の上限文字数は0以上である必要があります。")


class _Response(Protocol):
    status_code: int
    text: str
    headers: Mapping[str, str]

    def json(self) -> Any: ...


class _Session(Protocol):
    def post(self, url: str, **kwargs: Any) -> _Response: ...


def _retry_delay(response: _Response | None, attempt: int, policy: HttpPolicy) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), 30.0)
            except ValueError:
                pass
    return policy.backoff_seconds * (2**attempt)


def _trim_error_body(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "…"


def post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    policy: HttpPolicy | None = None,
    session: _Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] | None = None,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """POST JSON with bounded connection retries and normalized diagnostics.

    Read timeouts are deliberately not retried: a provider may already have
    completed an expensive generation, and blindly repeating it can double the
    workload or cost. Connection failures and explicitly transient HTTP
    statuses are retried with exponential backoff.
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError("API URLが空です。")
    if not isinstance(payload, Mapping):
        raise TypeError("API payloadはマッピングである必要があります。")

    request_policy = policy or HttpPolicy()
    client: _Session = session or requests
    attempts = request_policy.retries + 1

    for attempt in range(attempts):
        if should_stop and should_stop():
            raise ApiRequestError("停止要求によりAPI通信を中断しました。")

        response: _Response | None = None
        try:
            request_kwargs: dict[str, Any] = {
                "json": dict(payload),
                "timeout": (request_policy.connect_timeout, request_policy.read_timeout),
                "allow_redirects": False,
            }
            if headers:
                request_kwargs["headers"] = dict(headers)
            response = client.post(url.strip(), **request_kwargs)
        except requests.exceptions.ReadTimeout as exc:
            raise ApiRequestError(
                f"AI APIの応答待ちが{request_policy.read_timeout:g}秒でタイムアウトしました。"
            ) from exc
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as exc:
            if attempt < attempts - 1:
                sleep(_retry_delay(None, attempt, request_policy))
                continue
            raise ApiRequestError(f"AI APIへ接続できません: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise ApiRequestError(f"AI API通信に失敗しました: {exc}") from exc

        if 200 <= response.status_code < 300:
            try:
                data = response.json()
            except ValueError as exc:
                raise ApiRequestError("AI APIレスポンスが有効なJSONではありません。") from exc
            if not isinstance(data, dict):
                raise ApiRequestError("AI APIレスポンスのルートがJSONオブジェクトではありません。")
            return data

        if response.status_code in TRANSIENT_STATUS_CODES and attempt < attempts - 1:
            sleep(_retry_delay(response, attempt, request_policy))
            continue

        body = _trim_error_body(response.text, request_policy.max_error_body_chars)
        suffix = f" - {body}" if body else ""
        raise ApiRequestError(f"AI APIエラー: HTTP {response.status_code}{suffix}")

    raise AssertionError("HTTP retry loop terminated unexpectedly")
