"""Apply testable reliability services to the existing TagFlow PySide6 GUI.

The large legacy GUI remains in :mod:`TagFlow`.  ``tagflow_entry.py`` imports
that module without starting its event loop, then calls :func:`install` to
replace only I/O-heavy methods.  This keeps UI behavior stable and makes the
new logic testable without importing PySide6 in the core test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .files import atomic_write_json, atomic_write_text, transfer_image_family
from .http import ApiRequestError, HttpPolicy, post_json

DEFAULT_IMAGE_POLICY = HttpPolicy(
    connect_timeout=10.0,
    read_timeout=600.0,
    retries=2,
    backoff_seconds=1.0,
)
DEFAULT_CHAT_POLICY = HttpPolicy(
    connect_timeout=10.0,
    read_timeout=300.0,
    retries=2,
    backoff_seconds=0.75,
)
_INSTALL_MARKER = "_tagflow_core_installed"


def _config_path(app: Any, filepath: str | Path | None) -> Path:
    return Path(filepath).expanduser() if filepath is not None else Path(app.APP_DIR) / "app_config.json"


def _install_config_io(app: Any) -> None:
    def load_app_config(filepath: str | Path | None = None) -> dict[str, Any]:
        config_file = _config_path(app, filepath)
        if not config_file.exists():
            app.logger.info("設定ファイルが見つかりません。空の設定を使用します。")
            return {}
        try:
            config_data = app.json.loads(config_file.read_text(encoding="utf-8-sig"))
            if not isinstance(config_data, dict):
                raise ValueError("設定ファイルのルートはJSONオブジェクトである必要があります。")
            app.logger.info("設定ファイル %s から読み込みました。", config_file)
            return config_data
        except (app.json.JSONDecodeError, OSError, UnicodeError, ValueError) as exc:
            app.logger.warning("設定ファイル読み込みエラー: %s", exc)
            return {}

    def save_app_config(config: dict[str, Any], filepath: str | Path | None = None) -> None:
        if not isinstance(config, dict):
            app.logger.error("設定保存エラー: 保存する設定は辞書である必要があります。")
            return
        config_file = _config_path(app, filepath)
        try:
            atomic_write_json(config_file, config)
            app.logger.info("設定を %s に保存しました。", config_file)
        except Exception as exc:  # Preserve the GUI's existing non-fatal save behavior.
            app.logger.error("設定保存エラー: %s", exc)

    app.load_app_config = load_app_config
    app.save_app_config = save_app_config


def _install_image_analysis(app: Any) -> None:
    def analyze_image(self: Any, image_path: str | Path, should_stop=None) -> str:
        try:
            base64_image, mime_type = self.encode_image_data(image_path)
            payload = app.build_ai_request_payload(
                api_provider=self.api_provider,
                model=self.model,
                prompt=self.get_prompt(),
                images=[base64_image],
                image_mime_types=[mime_type],
            )
            result = post_json(
                self.api_url,
                payload,
                policy=DEFAULT_IMAGE_POLICY,
                should_stop=should_stop,
            )
            response_text = app.extract_ai_response_text(self.api_provider, result)
            return response_text if not self.clean_custom_response else self.clean_response_text(response_text)
        except ApiRequestError as exc:
            app.logger.error("画像分析APIエラー: %s", exc)
            raise RuntimeError(str(exc)) from exc
        except Exception as exc:
            app.logger.error("画像分析エラー: %s", exc)
            raise

    def analysis_worker_run(self: Any) -> None:
        total = len(self.image_paths)
        if total == 0:
            self.analysis_complete.emit(0, 0)
            return

        processed = 0
        errors = 0
        for index, image_path in enumerate(self.image_paths, start=1):
            if self.stop_requested:
                break
            try:
                app.logger.info("処理中: %s", Path(image_path).name)
                result = self.analyzer.analyze_image(
                    image_path,
                    should_stop=lambda: bool(self.stop_requested),
                )
                if self.stop_requested:
                    break
                atomic_write_text(Path(image_path).with_suffix(".txt"), result)
                self.image_analyzed.emit(str(image_path), result)
                processed += 1
            except Exception as exc:
                app.logger.error("分析エラー: %s", exc)
                errors += 1
            self.progress_updated.emit(index / total * 100)

        self.analysis_complete.emit(processed, errors)

    app.ImageAnalyzer.analyze_image = analyze_image
    app.AnalysisWorker.run = analysis_worker_run


def _install_file_transfer(app: Any) -> None:
    def file_operation_worker_run(self: Any) -> None:
        target_path = Path(self.target_dir)
        try:
            target_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.operation_error.emit(f"出力先フォルダを作成できません: {exc}")
            return

        total = len(self.file_paths)
        if total == 0:
            self.operation_complete.emit(0)
            return

        processed = 0
        failures: list[str] = []
        for index, file_path in enumerate(self.file_paths, start=1):
            if self.stop_requested:
                break
            try:
                result = transfer_image_family(file_path, target_path, self.operation)
                processed += 1
                app.logger.info(
                    "%s完了: %s -> %s (%d files)",
                    "コピー" if self.operation == "copy" else "移動",
                    result.source_image,
                    result.destination_image,
                    len(result.transferred_files),
                )
            except Exception as exc:
                message = f"{Path(file_path).name}: {exc}"
                failures.append(message)
                app.logger.error("ファイル操作エラー: %s", message)
            self.progress_updated.emit(index / total * 100)

        if failures:
            summary = "\n".join(failures[:10])
            if len(failures) > 10:
                summary += f"\n...ほか {len(failures) - 10} 件"
            self.operation_error.emit(f"{len(failures)} 件の処理に失敗しました。\n{summary}")
        self.operation_complete.emit(processed)

    app.FileOperationWorker.run = file_operation_worker_run


def _install_chat(app: Any) -> None:
    def chat_worker_run(self: Any) -> None:
        try:
            result = post_json(
                self.api_url,
                self.payload,
                policy=DEFAULT_CHAT_POLICY,
                should_stop=self.isInterruptionRequested,
            )
            self.result_ready.emit(app.extract_ai_response_text(self.api_provider, result))
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    app.ChatWorker.run = chat_worker_run


def _install_text_transform(app: Any) -> None:
    def service_init(self: Any, api_provider: str, api_url: str, model: str, timeout: float = 120) -> None:
        self.api_provider = app.normalize_api_provider(api_provider)
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.http_policy = HttpPolicy(
            connect_timeout=10.0,
            read_timeout=float(timeout),
            retries=2,
            backoff_seconds=0.75,
        )

    def transform_text(self: Any, text: str, preset: dict[str, Any], should_stop=None) -> str:
        provider_name = app.api_provider_display_name(self.api_provider)
        payload = app.build_ai_request_payload(
            api_provider=self.api_provider,
            model=self.model,
            prompt=self.build_prompt(preset, text, self.model),
            temperature=preset["temperature"],
            top_p=0.9,
        )
        try:
            result = post_json(
                self.api_url,
                payload,
                policy=self.http_policy,
                should_stop=should_stop,
            )
        except ApiRequestError as exc:
            raise RuntimeError(f"{provider_name} API通信エラー: {exc}") from exc
        response_text = app.extract_ai_response_text(self.api_provider, result)
        return self.clean_output(response_text, preset["mode"])

    def write_transform_result(
        self: Any,
        image_path: Path,
        source_path: Path,
        output_path: Path,
        preset: dict[str, Any],
        input_text: str,
        output_text: str,
        overwrite: bool,
    ) -> None:
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"出力先が既に存在するためスキップします: {output_path.name}")

        had_previous = output_path.exists()
        previous_text = output_path.read_text(encoding="utf-8") if had_previous else None
        atomic_write_text(output_path, output_text)
        try:
            self.update_metadata(
                image_path=Path(image_path),
                base_caption_path=Path(source_path),
                output_path=output_path,
                preset=preset,
                input_text=input_text,
                output_text=output_text,
            )
        except Exception:
            if had_previous and previous_text is not None:
                atomic_write_text(output_path, previous_text)
            else:
                output_path.unlink(missing_ok=True)
            raise

    def update_metadata(
        self: Any,
        image_path: Path,
        base_caption_path: Path,
        output_path: Path,
        preset: dict[str, Any],
        input_text: str,
        output_text: str,
    ) -> None:
        metadata_path = image_path.with_name(f"{image_path.stem}.tagflow.json")
        metadata = self._load_metadata(metadata_path, image_path, base_caption_path)
        metadata["schema_version"] = 1
        metadata["image_file"] = image_path.name
        metadata["base_caption_file"] = base_caption_path.name
        metadata["transforms"].append(
            {
                "mode": preset["mode"],
                "source_file": base_caption_path.name,
                "output_file": output_path.name,
                "api_provider": self.api_provider,
                "model": self.model,
                "api_url": self.api_url,
                "created_at": app.datetime.now().astimezone().isoformat(timespec="seconds"),
                "input_sha256": self._sha256_text(input_text),
                "output_sha256": self._sha256_text(output_text),
            }
        )
        atomic_write_json(metadata_path, metadata)

    def transform_worker_run(self: Any) -> None:
        total = len(self.image_paths)
        if total == 0:
            self.transform_complete.emit(0, 0)
            return

        service = app.TextTransformService(self.api_provider, self.api_url, self.model)
        success = 0
        failed = 0
        for index, image_path in enumerate(self.image_paths, start=1):
            if self.stop_requested:
                self.log_message.emit("停止要求を受け取ったため、残りの処理を中断しました。")
                break

            source_path = image_path.with_suffix(".txt")
            output_path = service.get_output_path(image_path, self.preset)
            try:
                if not source_path.exists():
                    raise FileNotFoundError(f"入力txtが見つかりません: {source_path.name}")
                if output_path.exists() and not self.overwrite:
                    raise FileExistsError(f"出力先が既に存在するためスキップします: {output_path.name}")
                input_text = source_path.read_text(encoding="utf-8-sig")
                output_text = service.transform_text(
                    input_text,
                    self.preset,
                    should_stop=lambda: bool(self.stop_requested),
                )
                if self.stop_requested:
                    self.log_message.emit("停止要求を受け取ったため、現在の結果は保存せずに中断しました。")
                    break
                service.write_transform_result(
                    image_path=image_path,
                    source_path=source_path,
                    output_path=output_path,
                    preset=self.preset,
                    input_text=input_text,
                    output_text=output_text,
                    overwrite=self.overwrite,
                )
                success += 1
                self.log_message.emit(f"{image_path.name}: {output_path.name} を保存しました。")
                self.item_completed.emit(str(image_path), str(output_path))
            except Exception as exc:
                failed += 1
                self.log_message.emit(f"{image_path.name}: {exc}")
                self.item_failed.emit(str(image_path), str(exc))
            self.progress_updated.emit(index / total * 100)

        self.transform_complete.emit(success, failed)

    app.TextTransformService.__init__ = service_init
    app.TextTransformService.transform_text = transform_text
    app.TextTransformService.write_transform_result = write_transform_result
    app.TextTransformService.update_metadata = update_metadata
    app.TextTransformWorker.run = transform_worker_run


def install(app: Any) -> None:
    """Install the hardened core exactly once into a loaded TagFlow GUI module."""

    if getattr(app, _INSTALL_MARKER, False):
        return

    required = (
        "APP_DIR",
        "logger",
        "json",
        "datetime",
        "build_ai_request_payload",
        "extract_ai_response_text",
        "normalize_api_provider",
        "api_provider_display_name",
        "ImageAnalyzer",
        "AnalysisWorker",
        "FileOperationWorker",
        "ChatWorker",
        "TextTransformService",
        "TextTransformWorker",
    )
    missing = [name for name in required if not hasattr(app, name)]
    if missing:
        raise RuntimeError("TagFlow GUIとの統合に必要な要素がありません: " + ", ".join(missing))

    _install_config_io(app)
    _install_image_analysis(app)
    _install_file_transfer(app)
    _install_chat(app)
    _install_text_transform(app)
    setattr(app, _INSTALL_MARKER, True)
