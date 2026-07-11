from __future__ import annotations

import unittest

import requests

from tagflow_core.http import ApiRequestError, HttpPolicy, post_json


class FakeResponse:
    def __init__(self, status_code: int, data=None, text: str = "", headers=None):
        self.status_code = status_code
        self._data = data
        self.text = text
        self.headers = headers or {}

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class PostJsonTests(unittest.TestCase):
    def test_rejects_negative_error_body_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "上限文字数"):
            HttpPolicy(max_error_body_chars=-1)

    def test_retries_connection_error_then_returns_json(self) -> None:
        session = FakeSession(
            [
                requests.exceptions.ConnectionError("offline"),
                FakeResponse(200, {"response": "ok"}),
            ]
        )
        sleeps = []
        result = post_json(
            "http://localhost/api",
            {"model": "x"},
            policy=HttpPolicy(retries=1, backoff_seconds=0.25),
            session=session,
            sleep=sleeps.append,
        )
        self.assertEqual(result, {"response": "ok"})
        self.assertEqual(sleeps, [0.25])
        self.assertEqual(session.calls[0][1]["timeout"], (10.0, 300.0))
        self.assertIs(session.calls[0][1]["allow_redirects"], False)

    def test_passes_explicit_headers_without_mutating_them(self) -> None:
        session = FakeSession([FakeResponse(200, {"response": "ok"})])
        headers = {"Authorization": "Bearer secret", "X-Test": "1"}

        post_json(
            "https://example.test/v1/responses",
            {"model": "x"},
            session=session,
            headers=headers,
        )

        self.assertEqual(session.calls[0][1]["headers"], headers)
        self.assertEqual(headers, {"Authorization": "Bearer secret", "X-Test": "1"})

    def test_honors_retry_after_for_transient_status(self) -> None:
        session = FakeSession(
            [
                FakeResponse(429, {}, "busy", {"Retry-After": "2"}),
                FakeResponse(200, {"response": "ok"}),
            ]
        )
        sleeps = []
        post_json(
            "http://localhost/api",
            {},
            policy=HttpPolicy(retries=1),
            session=session,
            sleep=sleeps.append,
        )
        self.assertEqual(sleeps, [2.0])

    def test_does_not_retry_read_timeout(self) -> None:
        session = FakeSession([requests.exceptions.ReadTimeout("slow")])
        with self.assertRaisesRegex(ApiRequestError, "応答待ち"):
            post_json(
                "http://localhost/api",
                {},
                policy=HttpPolicy(retries=3, read_timeout=1),
                session=session,
                sleep=lambda _: self.fail("read timeout must not be retried"),
            )
        self.assertEqual(len(session.calls), 1)

    def test_rejects_non_object_json(self) -> None:
        session = FakeSession([FakeResponse(200, ["not", "object"])])
        with self.assertRaisesRegex(ApiRequestError, "JSONオブジェクト"):
            post_json("http://localhost/api", {}, session=session)

    def test_trims_error_body(self) -> None:
        session = FakeSession([FakeResponse(400, {}, "x" * 50)])
        with self.assertRaises(ApiRequestError) as context:
            post_json(
                "http://localhost/api",
                {},
                policy=HttpPolicy(retries=0, max_error_body_chars=10),
                session=session,
            )
        self.assertIn("xxxxxxxxxx…", str(context.exception))


if __name__ == "__main__":
    unittest.main()
