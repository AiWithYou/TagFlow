from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from tagflow_core.http import HttpPolicy
from tagflow_core.providers import (
    API_PROVIDER_CODEX_CLI,
    API_PROVIDER_OPENAI,
    ProviderConfigurationError,
    ProviderExecutionError,
    _wrap_codex_command_for_platform,
    build_ai_request_payload,
    build_codex_command,
    codex_subscription_environment,
    execute_ai_request,
    extract_ai_response_text,
    provider_default_api_url,
    provider_default_model,
    provider_request_headers,
    run_codex_cli,
    validate_provider_location,
)


class FakeResponse:
    status_code = 200
    text = ""
    headers = {}

    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


class FakeSession:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.data)


class ProviderPayloadTests(unittest.TestCase):
    def test_openai_responses_payload_uses_base64_image_input(self) -> None:
        payload = build_ai_request_payload(
            API_PROVIDER_OPENAI,
            "gpt-5.6",
            "describe",
            images=["YWJj"],
            image_mime_types=["image/png"],
            temperature=0.2,
            top_p=0.8,
        )

        self.assertEqual(payload["model"], "gpt-5.6")
        content = payload["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "describe"})
        self.assertEqual(
            content[1],
            {"type": "input_image", "image_url": "data:image/png;base64,YWJj"},
        )
        self.assertIs(payload["store"], False)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)

    def test_codex_payload_uses_resolved_local_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(b"image")
            payload = build_ai_request_payload(
                API_PROVIDER_CODEX_CLI,
                "custom-codex-model",
                "describe",
                image_paths=[image],
            )

        self.assertEqual(payload["model"], "custom-codex-model")
        self.assertEqual(payload["prompt"], "describe")
        self.assertEqual(payload["image_paths"], [str(image.resolve())])

    def test_openai_raw_response_text_is_extracted(self) -> None:
        result = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": " first "},
                        {"type": "output_text", "text": "second"},
                    ],
                }
            ]
        }
        self.assertEqual(extract_ai_response_text(API_PROVIDER_OPENAI, result), "first\nsecond")

    def test_online_defaults_are_explicit(self) -> None:
        self.assertEqual(provider_default_api_url(API_PROVIDER_OPENAI), "https://api.openai.com/v1/responses")
        self.assertEqual(provider_default_model(API_PROVIDER_OPENAI), "gpt-5.6")
        self.assertEqual(provider_default_model(API_PROVIDER_CODEX_CLI), "")


class OpenAIExecutionTests(unittest.TestCase):
    def test_openai_location_accepts_only_the_official_responses_endpoint(self) -> None:
        self.assertEqual(
            validate_provider_location(
                API_PROVIDER_OPENAI,
                "https://api.openai.com/v1/responses/",
            ),
            "https://api.openai.com/v1/responses",
        )

        rejected_locations = (
            "http://api.openai.com/v1/responses",
            "https://evil.example/v1/responses",
            "https://api.openai.com/v1/responses?x=1",
            "https://api.openai.com:abc/v1/responses",
            "https://user:password@api.openai.com/v1/responses",
        )
        for location in rejected_locations:
            with self.subTest(location=location):
                with self.assertRaisesRegex(ProviderConfigurationError, "APIキー保護"):
                    validate_provider_location(API_PROVIDER_OPENAI, location)

    def test_api_key_is_required_and_never_part_of_payload(self) -> None:
        with self.assertRaisesRegex(ProviderConfigurationError, "OPENAI_API_KEY"):
            provider_request_headers(API_PROVIDER_OPENAI, environ={})

        payload = build_ai_request_payload(API_PROVIDER_OPENAI, "gpt-5.6", "hello")
        self.assertNotIn("api_key", payload)
        self.assertNotIn("Authorization", payload)

    def test_execute_openai_adds_environment_headers(self) -> None:
        session = FakeSession({"output_text": "answer"})
        payload = build_ai_request_payload(API_PROVIDER_OPENAI, "gpt-5.6", "hello")

        result = execute_ai_request(
            API_PROVIDER_OPENAI,
            "https://api.openai.com/v1/responses",
            payload,
            session=session,
            environ={
                "OPENAI_API_KEY": "test-key",
                "OPENAI_ORG_ID": "org-test",
                "OPENAI_PROJECT_ID": "proj-test",
            },
            policy=HttpPolicy(retries=0),
        )

        self.assertEqual(result, {"output_text": "answer"})
        request_headers = session.calls[0][1]["headers"]
        self.assertEqual(request_headers["Authorization"], "Bearer test-key")
        self.assertEqual(request_headers["OpenAI-Organization"], "org-test")
        self.assertEqual(request_headers["OpenAI-Project"], "proj-test")
        self.assertNotIn("test-key", repr(session.calls[0][1]["json"]))

    def test_rejected_openai_location_never_reaches_http_session(self) -> None:
        session = FakeSession({"output_text": "answer"})
        payload = build_ai_request_payload(API_PROVIDER_OPENAI, "gpt-5.6", "hello")

        with self.assertRaisesRegex(ProviderConfigurationError, "APIキー保護"):
            execute_ai_request(
                API_PROVIDER_OPENAI,
                "https://example.test/v1/responses",
                payload,
                session=session,
                environ={"OPENAI_API_KEY": "test-key"},
            )

        self.assertEqual(session.calls, [])


class CodexExecutionTests(unittest.TestCase):
    def test_subscription_environment_removes_one_shot_api_credentials(self) -> None:
        cleaned = codex_subscription_environment(
            {
                "PATH": "bin",
                "CODEX_HOME": "state",
                "CODEX_API_KEY": "codex-key",
                "codex_access_token": "access-token",
                "OPENAI_API_KEY": "openai-key",
                "OPENAI_ORG_ID": "org",
                "OPENAI_PROJECT_ID": "project",
            }
        )

        self.assertEqual(cleaned, {"PATH": "bin", "CODEX_HOME": "state"})

    def test_build_codex_command_is_noninteractive_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            executable.write_text("", encoding="utf-8")
            image = root / "image.png"
            image.write_bytes(b"image")
            output = root / "answer.txt"

            args = build_codex_command(
                str(executable),
                model="custom-codex-model",
                image_paths=[image],
                output_path=output,
            )

        self.assertEqual(args[1], "exec")
        self.assertIn("--ephemeral", args)
        self.assertIn("--skip-git-repo-check", args)
        self.assertIn("--ignore-user-config", args)
        self.assertIn("--ignore-rules", args)
        self.assertEqual(args[args.index("--sandbox") + 1], "read-only")
        self.assertEqual(args[args.index("--ask-for-approval") + 1], "never")
        config_values = [
            args[index + 1]
            for index, value in enumerate(args)
            if value == "--config"
        ]
        self.assertIn('web_search="disabled"', config_values)
        self.assertIn("features.shell_tool=false", config_values)
        self.assertIn("features.apps=false", config_values)
        self.assertIn("project_doc_max_bytes=0", config_values)
        self.assertEqual(args[args.index("--model") + 1], "custom-codex-model")
        self.assertEqual(args[args.index("--image") + 1], str(image.resolve()))
        self.assertEqual(args[-1], "-")

    def test_build_codex_command_omits_model_when_blank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "codex"
            executable.write_text("", encoding="utf-8")
            output = Path(directory) / "answer.txt"

            args = build_codex_command(
                str(executable),
                model="",
                output_path=output,
            )

        self.assertNotIn("--model", args)

    def test_windows_codex_cmd_wrapper_uses_cmd_exe_with_safe_quoting(self) -> None:
        wrapped = _wrap_codex_command_for_platform(
            r"C:\Program Files\npm\codex.cmd",
            ["exec", "--model", "model with spaces"],
            platform_name="nt",
            command_processor=r"C:\Windows\System32\cmd.exe",
        )

        self.assertEqual(
            wrapped[:4],
            [r"C:\Windows\System32\cmd.exe", "/d", "/s", "/c"],
        )
        self.assertIn('"C:\\Program Files\\npm\\codex.cmd"', wrapped[4])
        self.assertIn('"model with spaces"', wrapped[4])

    def test_posix_codex_executable_is_invoked_directly(self) -> None:
        wrapped = _wrap_codex_command_for_platform(
            "/opt/codex/bin/codex",
            ["exec", "-"],
            platform_name="posix",
        )
        self.assertEqual(wrapped, ["/opt/codex/bin/codex", "exec", "-"])

    def test_run_codex_reads_only_the_final_message_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "codex"
            executable.write_text("", encoding="utf-8")
            captured = {}

            def runner(args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                output_path = Path(args[args.index("--output-last-message") + 1])
                output_path.write_text(" final answer \n", encoding="utf-8")
                return subprocess.CompletedProcess(args, 0, stdout="progress", stderr="")

            result = run_codex_cli(
                str(executable),
                model="custom-codex-model",
                prompt="tag this image",
                environ={
                    "PATH": "bin",
                    "CODEX_HOME": "state",
                    "CODEX_API_KEY": "must-not-leak",
                },
                runner=runner,
            )

        self.assertEqual(result, "final answer")
        self.assertIn("TagFlow desktop application", captured["kwargs"]["input"])
        self.assertTrue(captured["kwargs"]["capture_output"])
        self.assertFalse(captured["kwargs"]["check"])
        self.assertEqual(
            captured["kwargs"]["env"],
            {"PATH": "bin", "CODEX_HOME": "state"},
        )

    def test_codex_nonzero_exit_has_bounded_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "codex"
            executable.write_text("", encoding="utf-8")

            def runner(args, **kwargs):
                return subprocess.CompletedProcess(args, 7, stdout="", stderr="x" * 2000)

            with self.assertRaises(ProviderExecutionError) as context:
                run_codex_cli(
                    str(executable),
                    model="custom-codex-model",
                    prompt="hello",
                    runner=runner,
                )

        self.assertIn("終了コード7", str(context.exception))
        self.assertLess(len(str(context.exception)), 1700)


if __name__ == "__main__":
    unittest.main()
