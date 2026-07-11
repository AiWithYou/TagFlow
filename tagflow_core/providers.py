"""AI provider definitions and execution helpers for TagFlow.

The GUI treats every backend as a provider with a location field.  HTTP
providers use that field as an endpoint URL; the ChatGPT subscription backend
uses it as the Codex CLI executable name/path.  Secrets are intentionally read
from environment variables at request time and are never part of TagFlow's
saved configuration.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .http import HttpPolicy, post_json

DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_CODEX_COMMAND = "codex"

API_PROVIDER_OLLAMA = "ollama"
API_PROVIDER_LM_STUDIO = "lm_studio"
API_PROVIDER_OPENAI = "openai"
API_PROVIDER_CODEX_CLI = "codex_cli"
DEFAULT_API_PROVIDER = API_PROVIDER_OLLAMA

API_PROVIDER_CHOICES = (
    (API_PROVIDER_OLLAMA, "Ollama（ローカル）"),
    (API_PROVIDER_LM_STUDIO, "LM Studio（ローカル）"),
    (API_PROVIDER_OPENAI, "OpenAI API（従量課金）"),
    (API_PROVIDER_CODEX_CLI, "ChatGPTサブスク（Codex CLI）"),
)
API_PROVIDER_LABELS = dict(API_PROVIDER_CHOICES)
DEFAULT_API_URLS = {
    API_PROVIDER_OLLAMA: DEFAULT_OLLAMA_API_URL,
    API_PROVIDER_LM_STUDIO: DEFAULT_LM_STUDIO_API_URL,
    API_PROVIDER_OPENAI: DEFAULT_OPENAI_API_URL,
    API_PROVIDER_CODEX_CLI: DEFAULT_CODEX_COMMAND,
}
DEFAULT_PROVIDER_MODELS = {
    API_PROVIDER_OLLAMA: "gemma4:12b",
    API_PROVIDER_LM_STUDIO: "gemma4:12b",
    API_PROVIDER_OPENAI: "gpt-5.6",
    API_PROVIDER_CODEX_CLI: "",
}
HTTP_PROVIDERS = frozenset(
    {API_PROVIDER_OLLAMA, API_PROVIDER_LM_STUDIO, API_PROVIDER_OPENAI}
)
CLOUD_PROVIDERS = frozenset({API_PROVIDER_OPENAI, API_PROVIDER_CODEX_CLI})

# The subscription backend must use the Codex CLI's persisted login selected
# by ``codex login``.  One-shot automation credentials take precedence for
# ``codex exec`` and could silently turn a subscription request into metered
# API usage, so they are deliberately removed from the child process.
CODEX_SUBSCRIPTION_BLOCKED_ENV = frozenset(
    {
        "CODEX_API_KEY",
        "CODEX_ACCESS_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
    }
)

CODEX_INFERENCE_PREAMBLE = (
    "You are an inference backend embedded in the TagFlow desktop application. "
    "Answer the supplied request directly. Do not inspect unrelated files, run shell commands, "
    "edit files, or use tools. Return only the requested answer, without progress commentary."
)


class ProviderConfigurationError(ValueError):
    """Raised when a selected provider is missing required local configuration."""


class ProviderExecutionError(RuntimeError):
    """Raised when a non-HTTP provider process fails."""


def normalize_api_provider(api_provider: str) -> str:
    if not isinstance(api_provider, str) or api_provider not in API_PROVIDER_LABELS:
        valid = ", ".join(API_PROVIDER_LABELS.keys())
        raise ValueError(f"未対応のAI接続先です: {api_provider!r}。有効値: {valid}")
    return api_provider


def api_provider_display_name(api_provider: str) -> str:
    return API_PROVIDER_LABELS[normalize_api_provider(api_provider)]


def provider_default_api_url(api_provider: str) -> str:
    return DEFAULT_API_URLS[normalize_api_provider(api_provider)]


def provider_default_model(api_provider: str) -> str:
    return DEFAULT_PROVIDER_MODELS[normalize_api_provider(api_provider)]


def provider_is_http(api_provider: str) -> bool:
    return normalize_api_provider(api_provider) in HTTP_PROVIDERS


def provider_uses_local_image_paths(api_provider: str) -> bool:
    return normalize_api_provider(api_provider) == API_PROVIDER_CODEX_CLI


def provider_requires_model(api_provider: str) -> bool:
    # Codex can use the model selected in its own config when the field is blank.
    return normalize_api_provider(api_provider) != API_PROVIDER_CODEX_CLI


def provider_is_cloud(api_provider: str) -> bool:
    return normalize_api_provider(api_provider) in CLOUD_PROVIDERS


def validate_provider_location(api_provider: str, location: str) -> str:
    """Validate and normalize the selected endpoint or executable location.

    OpenAI credentials must never be forwarded to arbitrary hosts.  The
    built-in OpenAI provider therefore accepts only the official Responses API
    endpoint.  Custom OpenAI-compatible local endpoints remain available via
    the LM Studio provider, which does not attach an OpenAI API key.
    """

    provider = normalize_api_provider(api_provider)
    normalized = location.strip() if isinstance(location, str) else ""
    if not normalized:
        raise ProviderConfigurationError(f"{provider_location_label(provider)}が空です。")

    if provider == API_PROVIDER_OPENAI:
        parsed = urlparse(normalized)
        path = parsed.path.rstrip("/")
        try:
            port = parsed.port
        except ValueError:
            port = -1
        if (
            parsed.scheme.lower() != "https"
            or (parsed.hostname or "").lower() != "api.openai.com"
            or parsed.username is not None
            or parsed.password is not None
            or port not in {None, 443}
            or path != "/v1/responses"
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise ProviderConfigurationError(
                "OpenAI APIキー保護のため、接続先は "
                f"{DEFAULT_OPENAI_API_URL} のみ使用できます。"
            )
        return DEFAULT_OPENAI_API_URL

    return normalized


def provider_location_label(api_provider: str) -> str:
    if normalize_api_provider(api_provider) == API_PROVIDER_CODEX_CLI:
        return "Codexコマンド"
    return "API URL"


def provider_location_placeholder(api_provider: str) -> str:
    provider = normalize_api_provider(api_provider)
    if provider == API_PROVIDER_CODEX_CLI:
        return "codex または codex.exe / codex.cmd のフルパス"
    return provider_default_api_url(provider)


def provider_location_tooltip(api_provider: str) -> str:
    provider = normalize_api_provider(api_provider)
    if provider == API_PROVIDER_OPENAI:
        return "OpenAI Responses APIのURL。認証には環境変数 OPENAI_API_KEY を使います。"
    if provider == API_PROVIDER_CODEX_CLI:
        return "ChatGPTでサインイン済みのCodex CLIコマンド。事前に codex を起動してログインしてください。"
    if provider == API_PROVIDER_LM_STUDIO:
        return "LM StudioのOpenAI互換 Chat Completions API URL。"
    return "Ollama generate API URL。"


def provider_notice(api_provider: str) -> str:
    provider = normalize_api_provider(api_provider)
    if provider == API_PROVIDER_OPENAI:
        return "クラウド送信: 画像・入力テキストをOpenAI APIへ送信します。OPENAI_API_KEYが必要です。"
    if provider == API_PROVIDER_CODEX_CLI:
        return "クラウド送信: ChatGPTで認証したCodex CLIを使います。サブスク上限が適用され、大量処理は低速です。"
    return "ローカル処理: TagFlowは指定したローカルAPIへ送信します。"


def _normalized_model(api_provider: str, model: str) -> str:
    normalized = model.strip() if isinstance(model, str) else ""
    if not normalized and provider_requires_model(api_provider):
        raise ValueError("モデル名が空です。")
    return normalized


def _normalized_prompt(prompt: str) -> str:
    normalized = prompt.strip() if isinstance(prompt, str) else ""
    if not normalized:
        raise ValueError("プロンプトが空です。")
    return normalized


def _validated_base64_images(
    images: Sequence[str] | None,
    image_mime_types: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    image_list = list(images or [])
    mime_list = list(image_mime_types or [])
    if len(image_list) != len(mime_list):
        raise ValueError("画像データとMIMEタイプの件数が一致しません。")
    for index, (image, mime_type) in enumerate(zip(image_list, mime_list), start=1):
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"{index}件目の画像データが空です。")
        if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
            raise ValueError(f"{index}件目の画像MIMEタイプが不正です: {mime_type!r}")
    return image_list, mime_list


def build_ai_request_payload(
    api_provider: str,
    model: str,
    prompt: str,
    images: Sequence[str] | None = None,
    image_mime_types: Sequence[str] | None = None,
    image_paths: Sequence[str | os.PathLike[str]] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict[str, Any]:
    """Build a provider-specific request without adding credentials."""

    provider = normalize_api_provider(api_provider)
    model_name = _normalized_model(provider, model)
    prompt_text = _normalized_prompt(prompt)

    if provider == API_PROVIDER_CODEX_CLI:
        if images:
            raise ValueError("Codex CLIにはBase64画像ではなくローカル画像パスを渡してください。")
        normalized_paths = [str(Path(path).expanduser().resolve()) for path in (image_paths or [])]
        return {
            "model": model_name,
            "prompt": prompt_text,
            "image_paths": normalized_paths,
        }

    image_list, mime_list = _validated_base64_images(images, image_mime_types)
    if image_paths:
        raise ValueError("HTTP APIにはローカル画像パスを直接渡せません。")

    if provider == API_PROVIDER_OLLAMA:
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt_text,
            "stream": False,
        }
        if image_list:
            payload["images"] = image_list
        options: dict[str, float] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if top_p is not None:
            options["top_p"] = float(top_p)
        if options:
            payload["options"] = options
        return payload

    if provider == API_PROVIDER_LM_STUDIO:
        if image_list:
            content: str | list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
            for image, mime_type in zip(image_list, mime_list):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image}"},
                    }
                )
        else:
            content = prompt_text
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        return payload

    if provider == API_PROVIDER_OPENAI:
        content = [{"type": "input_text", "text": prompt_text}]
        for image, mime_type in zip(image_list, mime_list):
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{image}",
                }
            )
        # GPT-5 family model support for sampling controls can differ.  TagFlow
        # deliberately omits temperature/top_p here and relies on the prompt.
        return {
            "model": model_name,
            "input": [{"role": "user", "content": content}],
            # Avoid retaining each one-shot tagging response for later API retrieval.
            "store": False,
        }

    raise AssertionError(f"Provider normalization missed: {provider}")


def _extract_text_parts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return parts


def extract_ai_response_text(api_provider: str, result: Mapping[str, Any]) -> str:
    """Extract the generated text from a raw provider response."""

    provider = normalize_api_provider(api_provider)
    provider_name = api_provider_display_name(provider)
    if not isinstance(result, Mapping):
        raise ValueError(f"{provider_name}のレスポンスがJSONオブジェクトではありません。")

    if provider in {API_PROVIDER_OLLAMA, API_PROVIDER_CODEX_CLI}:
        response = result.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()
        raise ValueError(f"{provider_name}のレスポンスに有効なresponseがありません。")

    if provider == API_PROVIDER_LM_STUDIO:
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise ValueError(f"{provider_name}のレスポンスにchoicesがありません。")
        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise ValueError(f"{provider_name}のレスポンスにmessageがありません。")
        text = "\n".join(_extract_text_parts(message.get("content"))).strip()
        if text:
            return text
        raise ValueError(f"{provider_name}のレスポンス本文が空です。")

    if provider == API_PROVIDER_OPENAI:
        convenience_text = result.get("output_text")
        if isinstance(convenience_text, str) and convenience_text.strip():
            return convenience_text.strip()

        text_parts: list[str] = []
        output = result.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, Mapping):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, Mapping):
                        continue
                    if part.get("type") not in {"output_text", "text"}:
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text.strip())
        if text_parts:
            return "\n".join(text_parts)
        raise ValueError("OpenAI APIのレスポンスに出力テキストがありません。")

    raise AssertionError(f"Provider normalization missed: {provider}")


def provider_request_headers(
    api_provider: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return request headers, reading secrets only from the process environment."""

    provider = normalize_api_provider(api_provider)
    if provider != API_PROVIDER_OPENAI:
        return {}

    environment = os.environ if environ is None else environ
    api_key = environment.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ProviderConfigurationError(
            "環境変数 OPENAI_API_KEY が設定されていません。TagFlowはAPIキーを設定ファイルへ保存しません。"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    organization = environment.get("OPENAI_ORG_ID", "").strip()
    project = environment.get("OPENAI_PROJECT_ID", "").strip()
    if organization:
        headers["OpenAI-Organization"] = organization
    if project:
        headers["OpenAI-Project"] = project
    return headers


def _resolve_codex_executable(command: str) -> str:
    raw = (command or DEFAULT_CODEX_COMMAND).strip().strip('"')
    if not raw:
        raw = DEFAULT_CODEX_COMMAND
    expanded = os.path.expandvars(os.path.expanduser(raw))
    path = Path(expanded)
    if path.exists() and path.is_file():
        return str(path.resolve())
    resolved = shutil.which(expanded)
    if resolved:
        return resolved
    raise ProviderConfigurationError(
        f"Codex CLIが見つかりません: {raw!r}。Codexをインストールし、最初に codex を起動してChatGPTでログインしてください。"
    )


def _wrap_codex_command_for_platform(
    executable: str,
    arguments: Sequence[str],
    *,
    platform_name: str | None = None,
    command_processor: str | None = None,
) -> list[str]:
    """Return a directly executable command on POSIX and native Windows.

    npm installs expose Codex as ``codex.cmd`` on Windows.  ``subprocess`` cannot
    execute a batch wrapper directly with ``shell=False``, so only that wrapper
    is delegated to ``cmd.exe``.  The executable and every argument are quoted
    with Python's Windows command-line encoder before they reach ``cmd /c``.
    """

    selected_platform = os.name if platform_name is None else platform_name
    suffix = Path(executable).suffix.lower()
    argument_list = [str(value) for value in arguments]
    if selected_platform == "nt" and suffix in {".cmd", ".bat"}:
        shell = (
            command_processor
            or os.environ.get("COMSPEC")
            or shutil.which("cmd.exe")
            or "cmd.exe"
        )
        command_line = subprocess.list2cmdline([executable, *argument_list])
        return [shell, "/d", "/s", "/c", command_line]
    return [executable, *argument_list]


def codex_subscription_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy the process environment without one-shot API auth overrides."""

    source = os.environ if environ is None else environ
    blocked = {name.casefold() for name in CODEX_SUBSCRIPTION_BLOCKED_ENV}
    return {
        str(key): str(value)
        for key, value in source.items()
        if str(key).casefold() not in blocked
    }


def build_codex_command(
    command: str,
    *,
    model: str = "",
    image_paths: Sequence[str | os.PathLike[str]] | None = None,
    output_path: str | os.PathLike[str],
) -> list[str]:
    """Build the non-interactive, read-only Codex invocation."""

    executable = _resolve_codex_executable(command)
    arguments = [
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--ask-for-approval",
        "never",
        "--color",
        "never",
        # TagFlow needs only direct model inference.  Remove unrelated tools
        # even when a future Codex default would otherwise expose them.
        "--config",
        'web_search="disabled"',
        "--config",
        "features.shell_tool=false",
        "--config",
        "features.apps=false",
        "--config",
        "project_doc_max_bytes=0",
        "--output-last-message",
        str(Path(output_path)),
    ]
    model_name = model.strip() if isinstance(model, str) else ""
    if model_name:
        arguments.extend(["--model", model_name])
    for raw_path in image_paths or []:
        image_path = Path(raw_path).expanduser().resolve()
        if not image_path.is_file():
            raise ProviderConfigurationError(f"Codexへ渡す画像が見つかりません: {image_path}")
        arguments.extend(["--image", str(image_path)])
    arguments.append("-")
    return _wrap_codex_command_for_platform(executable, arguments)


def run_codex_cli(
    command: str,
    *,
    model: str,
    prompt: str,
    image_paths: Sequence[str | os.PathLike[str]] | None = None,
    timeout_seconds: float = 600.0,
    should_stop: Callable[[], bool] | None = None,
    environ: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Run Codex once and return only its final assistant message."""

    prompt_text = _normalized_prompt(prompt)
    if should_stop and should_stop():
        raise ProviderExecutionError("停止要求によりCodex CLIの実行を中断しました。")
    if timeout_seconds <= 0:
        raise ValueError("Codex CLIタイムアウトは0より大きい必要があります。")

    with tempfile.TemporaryDirectory(prefix="tagflow-codex-") as temp_dir:
        output_path = Path(temp_dir) / "last-message.txt"
        args = build_codex_command(
            command,
            model=model,
            image_paths=image_paths,
            output_path=output_path,
        )
        run_kwargs: dict[str, Any] = {
            "input": f"{CODEX_INFERENCE_PREAMBLE}\n\n{prompt_text}\n",
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "capture_output": True,
            "timeout": float(timeout_seconds),
            "cwd": temp_dir,
            "check": False,
            "env": codex_subscription_environment(environ),
        }
        if os.name == "nt":
            run_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            completed = runner(args, **run_kwargs)
        except subprocess.TimeoutExpired as exc:
            raise ProviderExecutionError(
                f"Codex CLIの応答待ちが{timeout_seconds:g}秒でタイムアウトしました。"
            ) from exc
        except OSError as exc:
            raise ProviderExecutionError(f"Codex CLIを起動できません: {exc}") from exc

        if should_stop and should_stop():
            raise ProviderExecutionError("停止要求によりCodex CLIの結果を破棄しました。")
        if completed.returncode != 0:
            diagnostic = " ".join((completed.stderr or completed.stdout or "").split())
            if len(diagnostic) > 1500:
                diagnostic = diagnostic[:1500] + "…"
            suffix = f" - {diagnostic}" if diagnostic else ""
            raise ProviderExecutionError(
                f"Codex CLIが終了コード{completed.returncode}で失敗しました{suffix}"
            )

        output = ""
        if output_path.exists():
            output = output_path.read_text(encoding="utf-8", errors="replace").strip()
        if not output:
            output = (completed.stdout or "").strip()
        if not output:
            raise ProviderExecutionError("Codex CLIの最終応答が空です。")
        return output


def execute_ai_request(
    api_provider: str,
    location: str,
    payload: Mapping[str, Any],
    *,
    policy: HttpPolicy | None = None,
    should_stop: Callable[[], bool] | None = None,
    environ: Mapping[str, str] | None = None,
    session: Any = None,
    sleep: Callable[[float], None] | None = None,
    codex_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Execute an already-built request through the selected backend."""

    provider = normalize_api_provider(api_provider)
    validated_location = validate_provider_location(provider, location)
    if not isinstance(payload, Mapping):
        raise TypeError("AI request payloadはマッピングである必要があります。")

    if provider == API_PROVIDER_CODEX_CLI:
        timeout = policy.read_timeout if policy is not None else 600.0
        response = run_codex_cli(
            validated_location,
            model=str(payload.get("model", "")),
            prompt=str(payload.get("prompt", "")),
            image_paths=payload.get("image_paths") or [],
            timeout_seconds=timeout,
            should_stop=should_stop,
            environ=environ,
            runner=codex_runner,
        )
        return {"response": response}

    kwargs: dict[str, Any] = {
        "policy": policy,
        "should_stop": should_stop,
        "headers": provider_request_headers(provider, environ=environ),
    }
    if session is not None:
        kwargs["session"] = session
    if sleep is not None:
        kwargs["sleep"] = sleep
    return post_json(validated_location, payload, **kwargs)
