from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import ModuleType
from unittest import mock

from tagflow_core.integration import install


class SignalRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def emit(self, *args: object) -> None:
        self.calls.append(args)


class ImageAnalyzer:
    def encode_image_data(self, image_path):
        return "encoded", "image/png"

    def get_prompt(self):
        return "describe"

    def clean_response_text(self, text):
        return f"clean:{text}"


class AnalysisWorker:
    pass


class FileOperationWorker:
    pass


class ChatWorker:
    pass


class TextTransformWorker:
    pass


class TextTransformService:
    def build_prompt(self, preset, text, model):
        return preset["prompt"].replace("{TEXT}", text)

    def clean_output(self, text, mode):
        return f"{mode}:{text}"

    def _load_metadata(self, metadata_path, image_path, base_caption_path):
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            "schema_version": 1,
            "image_file": image_path.name,
            "base_caption_file": base_caption_path.name,
            "transforms": [],
        }

    def _sha256_text(self, text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IntegrationTests(unittest.TestCase):
    def make_app(self, root: Path) -> ModuleType:
        app = ModuleType("fake_tagflow")
        app.APP_DIR = root
        app.logger = mock.Mock()
        app.json = json
        app.datetime = datetime
        app.ImageAnalyzer = type("ImageAnalyzerForApp", (ImageAnalyzer,), {})
        app.AnalysisWorker = type("AnalysisWorkerForApp", (AnalysisWorker,), {})
        app.FileOperationWorker = type("FileOperationWorkerForApp", (FileOperationWorker,), {})
        app.ChatWorker = type("ChatWorkerForApp", (ChatWorker,), {})
        app.TextTransformService = type("TextTransformServiceForApp", (TextTransformService,), {})
        app.TextTransformWorker = type("TextTransformWorkerForApp", (TextTransformWorker,), {})
        app.normalize_api_provider = lambda provider: provider
        app.api_provider_display_name = lambda provider: provider
        app.provider_uses_local_image_paths = lambda provider: provider == "codex_cli"
        app.build_ai_request_payload = lambda **kwargs: kwargs
        app.extract_ai_response_text = lambda provider, result: result["response"]
        app.execute_ai_request = mock.Mock(return_value={"response": "default"})
        return app

    def test_install_is_idempotent_and_config_is_app_relative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = self.make_app(root)
            install(app)
            first_method = app.ImageAnalyzer.analyze_image
            install(app)
            self.assertIs(app.ImageAnalyzer.analyze_image, first_method)

            app.save_app_config({"theme": "dark"})
            self.assertEqual(
                json.loads((root / "app_config.json").read_text(encoding="utf-8")),
                {"theme": "dark"},
            )
            self.assertEqual(app.load_app_config(), {"theme": "dark"})

    def test_invalid_config_root_falls_back_to_empty_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app_config.json").write_text("[]", encoding="utf-8")
            app = self.make_app(root)
            install(app)
            self.assertEqual(app.load_app_config(), {})
            app.logger.warning.assert_called()

    def test_image_analyzer_uses_reliable_http_and_cleaning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(Path(directory))
            install(app)
            analyzer = app.ImageAnalyzer()
            analyzer.api_provider = "ollama"
            analyzer.model = "vision"
            analyzer.api_url = "http://localhost/api"
            analyzer.clean_custom_response = True

            app.execute_ai_request.return_value = {"response": "raw"}
            result = analyzer.analyze_image("image.png")

            self.assertEqual(result, "clean:raw")
            app.execute_ai_request.assert_called_once()
            payload = app.execute_ai_request.call_args.args[2]
            self.assertEqual(payload["images"], ["encoded"])
            self.assertEqual(payload["image_mime_types"], ["image/png"])

    def test_codex_image_analyzer_passes_local_path_without_base64_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.write_bytes(b"image")
            app = self.make_app(root)
            install(app)
            analyzer = app.ImageAnalyzer()
            analyzer.api_provider = "codex_cli"
            analyzer.model = "custom-codex-model"
            analyzer.api_url = "codex"
            analyzer.clean_custom_response = False
            analyzer.encode_image_data = mock.Mock(side_effect=AssertionError("must not encode"))
            app.execute_ai_request.return_value = {"response": "caption"}

            result = analyzer.analyze_image(image)

            self.assertEqual(result, "caption")
            analyzer.encode_image_data.assert_not_called()
            payload = app.execute_ai_request.call_args.args[2]
            self.assertEqual(payload["image_paths"], [str(image.resolve())])

    def test_analysis_worker_writes_caption_and_emits_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.write_bytes(b"not decoded in this test")
            app = self.make_app(root)
            install(app)

            worker = app.AnalysisWorker()
            worker.image_paths = [str(image)]
            worker.stop_requested = False
            worker.analyzer = mock.Mock()
            worker.analyzer.analyze_image.return_value = "caption"
            worker.progress_updated = SignalRecorder()
            worker.analysis_complete = SignalRecorder()
            worker.image_analyzed = SignalRecorder()

            worker.run()

            self.assertEqual((root / "image.txt").read_text(encoding="utf-8"), "caption")
            self.assertEqual(worker.progress_updated.calls, [(100.0,)])
            self.assertEqual(worker.analysis_complete.calls, [(1, 0)])
            self.assertEqual(worker.image_analyzed.calls, [(str(image), "caption")])

    def test_file_operation_worker_keeps_transformed_sidecars_together(self) -> None:
        with tempfile.TemporaryDirectory() as source_directory, tempfile.TemporaryDirectory() as target_directory:
            source = Path(source_directory)
            target = Path(target_directory)
            image = source / "sample.png"
            image.write_bytes(b"image")
            (source / "sample.txt").write_text("base", encoding="utf-8")
            (source / "sample.en.txt").write_text("translated", encoding="utf-8")
            (source / "sample.tagflow.json").write_text("{}", encoding="utf-8")

            app = self.make_app(source)
            install(app)
            worker = app.FileOperationWorker()
            worker.target_dir = str(target)
            worker.file_paths = [str(image)]
            worker.operation = "copy"
            worker.stop_requested = False
            worker.progress_updated = SignalRecorder()
            worker.operation_complete = SignalRecorder()
            worker.operation_error = SignalRecorder()

            worker.run()

            self.assertEqual(
                {path.name for path in target.iterdir()},
                {"sample.png", "sample.txt", "sample.en.txt", "sample.tagflow.json"},
            )
            self.assertEqual(worker.operation_complete.calls, [(1,)])
            self.assertEqual(worker.operation_error.calls, [])

    def test_chat_worker_emits_normalized_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(Path(directory))
            install(app)
            worker = app.ChatWorker()
            worker.api_provider = "ollama"
            worker.api_url = "http://localhost/api"
            worker.payload = {"model": "chat"}
            worker.result_ready = SignalRecorder()
            worker.error_occurred = SignalRecorder()
            worker.isInterruptionRequested = lambda: False

            app.execute_ai_request.return_value = {"response": "hello"}
            worker.run()

            self.assertEqual(worker.result_ready.calls, [("hello",)])
            self.assertEqual(worker.error_occurred.calls, [])
            app.execute_ai_request.assert_called_once()

    def test_text_transform_uses_selected_provider_executor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(Path(directory))
            install(app)
            service = app.TextTransformService("openai", "https://api.openai.com/v1/responses", "gpt-5.6")
            app.execute_ai_request.return_value = {"response": "translated"}
            preset = {
                "prompt": "Translate: {TEXT}",
                "temperature": 0.1,
                "mode": "translate_ja_to_en",
            }

            result = service.transform_text("入力", preset)

            self.assertEqual(result, "translate_ja_to_en:translated")
            call = app.execute_ai_request.call_args
            self.assertEqual(call.args[0], "openai")
            self.assertEqual(call.args[1], "https://api.openai.com/v1/responses")
            self.assertEqual(call.args[2]["prompt"], "Translate: 入力")

    def test_transform_write_restores_previous_output_when_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = self.make_app(root)
            install(app)
            service = app.TextTransformService("ollama", "http://localhost/api", "model")
            output = root / "image.en.txt"
            output.write_text("previous", encoding="utf-8")
            service.update_metadata = mock.Mock(side_effect=OSError("disk full"))

            with self.assertRaisesRegex(OSError, "disk full"):
                service.write_transform_result(
                    image_path=root / "image.png",
                    source_path=root / "image.txt",
                    output_path=output,
                    preset={"mode": "translate_ja_to_en"},
                    input_text="入力",
                    output_text="new",
                    overwrite=True,
                )

            self.assertEqual(output.read_text(encoding="utf-8"), "previous")


if __name__ == "__main__":
    unittest.main()
