from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tagflow_core.files import (
    TransferError,
    atomic_write_json,
    atomic_write_text,
    discover_image_family,
    transfer_image_family,
)


class AtomicWriteTests(unittest.TestCase):
    def test_atomic_write_text_replaces_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "caption.txt"
            path.write_text("old", encoding="utf-8")
            atomic_write_text(path, "新しいcaption")
            self.assertEqual(path.read_text(encoding="utf-8"), "新しいcaption")
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_atomic_write_json_uses_utf8_and_newline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.json"
            atomic_write_json(path, {"日本語": [1, 2]})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "日本語": [\n    1,\n    2\n  ]\n}\n')


class ImageFamilyTransferTests(unittest.TestCase):
    def _make_family(self, directory: Path, stem: str = "sample") -> Path:
        image = directory / f"{stem}.png"
        image.write_bytes(b"image")
        (directory / f"{stem}.txt").write_text("base", encoding="utf-8")
        (directory / f"{stem}.en.txt").write_text("english", encoding="utf-8")
        (directory / f"{stem}.custom.txt").write_text("future preset", encoding="utf-8")
        (directory / f"{stem}.tagflow.json").write_text("{}", encoding="utf-8")
        return image

    def test_discovers_base_transform_and_metadata_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = self._make_family(root)
            names = [path.name for path in discover_image_family(image)]
            self.assertEqual(
                names,
                [
                    "sample.png",
                    "sample.custom.txt",
                    "sample.en.txt",
                    "sample.tagflow.json",
                    "sample.txt",
                ],
            )

    def test_discovers_sidecars_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "Sample.PNG"
            image.write_bytes(b"image")
            (root / "sample.TXT").write_text("base", encoding="utf-8")
            (root / "SAMPLE.EN.TXT").write_text("english", encoding="utf-8")

            names = {path.name for path in discover_image_family(image)}
            self.assertEqual(names, {"Sample.PNG", "sample.TXT", "SAMPLE.EN.TXT"})

    def test_rejects_ambiguous_same_stem_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "sample.png"
            first.write_bytes(b"png")
            (root / "sample.jpg").write_bytes(b"jpg")
            (root / "sample.txt").write_text("caption", encoding="utf-8")

            with self.assertRaisesRegex(TransferError, "同一stem"):
                discover_image_family(first)

    def test_copy_keeps_all_sidecars_and_uses_collision_free_stem(self) -> None:
        with tempfile.TemporaryDirectory() as source_directory, tempfile.TemporaryDirectory() as target_directory:
            source = Path(source_directory)
            target = Path(target_directory)
            image = self._make_family(source)
            # Even a sidecar-only collision must prevent mixing two families.
            (target / "sample.prompt.txt").write_text("existing", encoding="utf-8")

            result = transfer_image_family(image, target, "copy")

            self.assertEqual(result.destination_image.name, "sample_1.png")
            self.assertEqual(
                {path.name for path in result.transferred_files},
                {
                    "sample_1.png",
                    "sample_1.txt",
                    "sample_1.en.txt",
                    "sample_1.custom.txt",
                    "sample_1.tagflow.json",
                },
            )
            self.assertTrue(image.exists())
            self.assertEqual((target / "sample.prompt.txt").read_text(encoding="utf-8"), "existing")

    def test_move_rolls_back_when_a_later_member_fails(self) -> None:
        with tempfile.TemporaryDirectory() as source_directory, tempfile.TemporaryDirectory() as target_directory:
            source = Path(source_directory)
            target = Path(target_directory)
            image = self._make_family(source)
            real_move = __import__("shutil").move
            call_count = 0

            def flaky_move(src: str, dst: str):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("simulated failure")
                return real_move(src, dst)

            with mock.patch("tagflow_core.files.shutil.move", side_effect=flaky_move):
                with self.assertRaises(TransferError):
                    transfer_image_family(image, target, "move")

            self.assertTrue(image.exists())
            self.assertTrue((source / "sample.txt").exists())
            self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
