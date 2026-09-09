"""The CLI must work when Windows redirects output using a legacy code page."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class AuditCliEncodingTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        script = Path(__file__).resolve().parents[1] / "tools" / "audit_dataset.py"
        environment = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
        return subprocess.run(
            [sys.executable, str(script), *arguments],
            env=environment, capture_output=True, timeout=20, check=False,
        )

    def test_help_is_utf8_even_with_legacy_pipe_encoding(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
        self.assertIn("監査対象フォルダ", result.stdout.decode("utf-8"))
        self.assertEqual(result.stderr, b"")

    def test_json_preserves_non_ascii_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "日本語画像"
            root.mkdir()
            (root / "photo.png").write_bytes(b"image")
            (root / "photo.txt").write_text("説明文", encoding="utf-8")
            result = self.run_cli(str(root), "--skip-image-verify", "--skip-duplicates", "--json-stdout")
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
            report = json.loads(result.stdout.decode("utf-8"))
            self.assertEqual(report["root"], str(root.resolve()))
            self.assertEqual(report["issue_count"], 0)
            self.assertEqual(result.stderr, b"")

    def test_errors_are_utf8_and_keep_normalized_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "存在しないフォルダ"
            result = self.run_cli(str(missing))
            self.assertEqual(result.returncode, 2)
            self.assertIn("監査エラー", result.stderr.decode("utf-8"))
            self.assertNotIn("Traceback", result.stderr.decode("utf-8"))
            self.assertEqual(result.stdout, b"")


if __name__ == "__main__":
    unittest.main()
