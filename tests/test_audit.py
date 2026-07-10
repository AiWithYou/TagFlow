from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tagflow_core.audit import audit_dataset


class DatasetAuditTests(unittest.TestCase):
    def _image(self, path: Path, value: int = 0) -> None:
        Image.new("RGB", (4, 4), (value, value, value)).save(path)

    def test_reports_pairing_validity_orphans_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._image(root / "good.png", 1)
            (root / "good.txt").write_text("caption", encoding="utf-8")
            (root / "good.en.txt").write_text("caption", encoding="utf-8")

            self._image(root / "missing.png", 2)

            self._image(root / "empty.png", 3)
            (root / "empty.txt").write_text("  \n", encoding="utf-8")

            self._image(root / "duplicate.png", 1)
            (root / "duplicate.txt").write_text("duplicate", encoding="utf-8")

            (root / "orphan.txt").write_text("orphan", encoding="utf-8")
            (root / "broken.png").write_bytes(b"not an image")

            report = audit_dataset(root)
            codes = [issue.code for issue in report.issues]
            self.assertIn("missing_caption", codes)
            self.assertIn("empty_caption", codes)
            self.assertIn("orphan_sidecar", codes)
            self.assertIn("invalid_image", codes)
            self.assertIn("duplicate_image", codes)
            self.assertEqual(report.image_count, 5)

    def test_reports_same_stem_images_as_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._image(root / "same.png", 1)
            self._image(root / "same.jpg", 2)
            (root / "same.txt").write_text("shared", encoding="utf-8")

            report = audit_dataset(root, verify_images=False, find_duplicates=False)
            ambiguous = [issue for issue in report.issues if issue.code == "ambiguous_caption_owner"]
            self.assertEqual(len(ambiguous), 2)

    def test_json_report_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._image(root / "image.png")
            (root / "image.txt").write_text("caption", encoding="utf-8")
            output = root / "report.json"

            report = audit_dataset(root, verify_images=False, find_duplicates=False)
            report.write_json(output)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(data["issue_count"], 0)


if __name__ == "__main__":
    unittest.main()
