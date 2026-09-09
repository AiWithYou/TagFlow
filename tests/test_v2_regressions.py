"""Fault injection and compatibility coverage for the v2 data path."""
from __future__ import annotations

import errno
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from tagflow_core import __version__
from tagflow_core import audit, files
from tagflow_core.sidecars import resolve_sidecar


class TransferRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.target = self.root / "target"
        self.source.mkdir()
        self.target.mkdir()
        self.image = self.source / "photo.png"
        self.image.write_bytes(b"image bytes")
        (self.source / "photo.txt").write_bytes(b"original caption\r\n")
        (self.source / "photo.en.txt").write_bytes(b"translated")
        self.metadata = self.source / "photo.tagflow.json"
        self.metadata.write_text(json.dumps({
            "image_file": "photo.png", "base_caption_file": "photo.txt",
            "transforms": [{"source_file": "photo.txt", "output_file": "photo.en.txt",
                            "model": "photo.png", "input_sha256": "unchanged"}],
        }), encoding="utf-8")
        self.before = {p.name: p.read_bytes() for p in self.source.iterdir()}

    def assert_originals(self):
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, self.before)

    def assert_no_temporary_files(self):
        self.assertFalse(list(self.target.glob(".tagflow-*")))

    def test_dotted_image_does_not_lose_its_sidecars(self):
        (self.source / "photo.en.png").write_bytes(b"different image")
        (self.source / "photo.en.lora.txt").write_text("separate", encoding="utf-8")
        family = {p.name for p in files.discover_image_family(self.image)}
        self.assertNotIn("photo.en.txt", family)
        self.assertNotIn("photo.en.lora.txt", family)
        result = files.transfer_image_family(self.image, self.target, "move")
        self.assertEqual(len(result.transferred_files), 3)
        self.assertTrue((self.source / "photo.en.txt").exists())
        self.assertTrue((self.source / "photo.en.lora.txt").exists())

    def test_metadata_references_follow_collision_rename(self):
        (self.target / "photo.png").write_bytes(b"existing image")
        result = files.transfer_image_family(self.image, self.target)
        data = json.loads((self.target / "photo_1.tagflow.json").read_text(encoding="utf-8"))
        self.assertEqual(result.destination_image.name, "photo_1.png")
        self.assertEqual(data["image_file"], "photo_1.png")
        self.assertEqual(data["base_caption_file"], "photo_1.txt")
        self.assertEqual(data["transforms"][0]["source_file"], "photo_1.txt")
        self.assertEqual(data["transforms"][0]["output_file"], "photo_1.en.txt")
        self.assertEqual(data["transforms"][0]["model"], "photo.png")
        self.assert_originals()
        self.assert_no_temporary_files()

    def test_invalid_metadata_on_rename_preserves_sources(self):
        self.metadata.write_bytes(b"{invalid")
        (self.target / "photo.png").write_bytes(b"existing")
        with self.assertRaises(files.TransferError):
            files.transfer_image_family(self.image, self.target, "move")
        self.assertTrue(self.image.exists())
        self.assertEqual(self.metadata.read_bytes(), b"{invalid")
        self.assertEqual([p.name for p in self.target.iterdir()], ["photo.png"])

    def test_unchanged_metadata_is_copied_byte_for_byte(self):
        files.transfer_image_family(self.image, self.target)
        self.assertEqual((self.target / self.metadata.name).read_bytes(), self.metadata.read_bytes())
        self.assert_no_temporary_files()

    def test_partial_staging_copy_is_cleaned(self):
        def fail_mid_copy(source, destination, *args, **kwargs):
            Path(destination).write_bytes(b"incomplete")
            raise OSError("disk full")
        with mock.patch.object(files.shutil, "copyfile", side_effect=fail_mid_copy):
            with self.assertRaisesRegex(files.TransferError, "disk full"):
                files.transfer_image_family(self.image, self.target, "move")
        self.assert_originals()
        self.assertEqual(list(self.target.iterdir()), [])

    def test_publish_collision_never_overwrites_or_deletes_other_writer(self):
        real_publish = files._publish
        calls = 0
        def race(staged, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                destination.write_bytes(b"other writer")
            return real_publish(staged, destination)
        with mock.patch.object(files, "_publish", side_effect=race):
            with self.assertRaises(files.TransferError):
                files.transfer_image_family(self.image, self.target, "move")
        self.assert_originals()
        leftovers = list(self.target.iterdir())
        self.assertEqual(len(leftovers), 1)
        self.assertEqual(leftovers[0].read_bytes(), b"other writer")

    def test_all_outputs_exist_before_any_source_is_deleted(self):
        real_unlink = Path.unlink
        deleted = []
        def check(path, *args, **kwargs):
            if path.parent == self.source:
                for name, content in self.before.items():
                    self.assertEqual((self.target / name).read_bytes(), content)
                deleted.append(path)
            return real_unlink(path, *args, **kwargs)
        with mock.patch.object(Path, "unlink", new=check):
            result = files.transfer_image_family(self.image, self.target, "move")
        self.assertEqual(len(deleted), 4)
        self.assertEqual(len(result.transferred_files), 4)
        self.assertEqual(list(self.source.iterdir()), [])
        self.assert_no_temporary_files()

    def test_failed_rollback_keeps_last_copies_and_recovery_manifest(self):
        real_unlink = Path.unlink
        calls = 0
        def conflict(path, *args, **kwargs):
            nonlocal calls
            if path.parent == self.source:
                calls += 1
                if calls == 2:
                    self.image.write_bytes(b"newly recreated source")
                    raise OSError("cannot delete second source")
            return real_unlink(path, *args, **kwargs)
        with mock.patch.object(Path, "unlink", new=conflict):
            with self.assertRaisesRegex(files.TransferError, "復旧用原本"):
                files.transfer_image_family(self.image, self.target, "move")
        self.assertEqual(self.image.read_bytes(), b"newly recreated source")
        for name, content in self.before.items():
            self.assertEqual((self.target / name).read_bytes(), content)
        stages = list(self.target.glob(".tagflow-transfer-*"))
        self.assertEqual(len(stages), 1)
        manifest = json.loads((stages[0] / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["recovery_files"]:
            self.assertTrue(Path(entry["backup"]).exists())
        self.assertFalse(list(self.target.glob(".tagflow-lock-*")))

    def test_rollback_after_rename_restores_original_metadata_bytes(self):
        (self.target / "photo.png").write_bytes(b"existing")
        real_unlink = Path.unlink
        def fail_last(path, *args, **kwargs):
            if path == self.source / "photo.txt":
                raise OSError("cannot delete last source")
            return real_unlink(path, *args, **kwargs)
        with mock.patch.object(Path, "unlink", new=fail_last):
            with self.assertRaises(files.TransferError):
                files.transfer_image_family(self.image, self.target, "move")
        self.assert_originals()
        self.assertEqual([p.name for p in self.target.iterdir()], ["photo.png"])

    def test_changed_source_aborts_without_deleting_it(self):
        real_publish = files._publish
        def edit(staged, destination):
            real_publish(staged, destination)
            if destination.suffix == ".png":
                self.image.write_bytes(b"edited while copying")
        with mock.patch.object(files, "_publish", side_effect=edit):
            with self.assertRaisesRegex(files.TransferError, "変更"):
                files.transfer_image_family(self.image, self.target, "move")
        self.assertEqual(self.image.read_bytes(), b"edited while copying")
        self.assertEqual(list(self.target.iterdir()), [])
        self.assertTrue(self.metadata.exists())

    def test_concurrent_copies_get_distinct_complete_families(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: files.transfer_image_family(self.image, self.target), range(12)))
        self.assertEqual(len({r.destination_image.name.casefold() for r in results}), 12)
        self.assertEqual(len(list(self.target.iterdir())), 48)
        for result in results:
            data = json.loads(result.destination_image.with_suffix(".tagflow.json").read_text(encoding="utf-8"))
            self.assertEqual(data["image_file"], result.destination_image.name)
        self.assert_originals()
        self.assert_no_temporary_files()

    def test_move_into_same_directory_is_noop(self):
        result = files.transfer_image_family(self.image, self.source, "move")
        self.assertEqual(result.destination_image, self.image)
        self.assert_originals()

    def test_copy_into_same_directory_creates_new_family(self):
        result = files.transfer_image_family(self.image, self.source)
        self.assertEqual(result.destination_image.name, "photo_1.png")
        self.assertEqual(len(list(self.source.iterdir())), 8)
        self.assertEqual(self.image.read_bytes(), self.before[self.image.name])

    def test_unicode_casefold_does_not_corrupt_suffix(self):
        image = self.source / "Straße.png"
        image.write_bytes(b"image")
        (self.source / "STRASSE.EN.TXT").write_bytes(b"caption")
        result = files.transfer_image_family(image, self.target)
        self.assertIn("Straße.EN.TXT", {p.name for p in result.transferred_files})

    def test_readonly_copy_and_collision_metadata_cleanup(self):
        self.image.chmod(stat.S_IREAD)
        self.metadata.chmod(stat.S_IREAD)
        (self.target / "photo.png").write_bytes(b"existing")
        result = files.transfer_image_family(self.image, self.target)
        self.assertEqual(result.destination_image.read_bytes(), self.before["photo.png"])
        self.assert_no_temporary_files()
        self.assert_originals()

    def test_symlink_image_is_rejected(self):
        link = self.source / "linked.png"
        try:
            link.symlink_to(self.image)
        except OSError as exc:
            self.skipTest(f"symlinks not available: {exc}")
        with self.assertRaises(files.TransferError):
            files.transfer_image_family(link, self.target)
        self.assertTrue(self.image.exists())
        self.assertEqual(list(self.target.iterdir()), [])

    def test_symlink_sidecar_is_rejected(self):
        link = self.source / "photo.extra.txt"
        try:
            link.symlink_to(self.source / "photo.txt")
        except OSError as exc:
            self.skipTest(f"symlinks not available: {exc}")
        with self.assertRaises(files.TransferError):
            files.transfer_image_family(self.image, self.target, "move")
        self.assertTrue(self.image.exists())
        self.assertEqual(list(self.target.iterdir()), [])

    @unittest.skipIf(os.name == "nt", "Windows uses no-clobber rename instead")
    def test_filesystem_without_hardlinks_uses_exclusive_copy(self):
        with mock.patch.object(files.os, "link", side_effect=OSError(errno.ENOTSUP, "unsupported")):
            files.transfer_image_family(self.image, self.target)
        for name, content in self.before.items():
            self.assertEqual((self.target / name).read_bytes(), content)
        self.assert_no_temporary_files()

    def test_exclusive_copy_cleans_partial_output(self):
        output = self.target / "partial.png"
        def fail(incoming, outgoing, **kwargs):
            outgoing.write(b"partial")
            raise OSError("disk full")
        with mock.patch.object(files.shutil, "copyfileobj", side_effect=fail):
            with self.assertRaises(OSError):
                files._copy_exclusive(self.image, output)
        self.assertFalse(output.exists())
        self.assert_originals()

    def test_exclusive_copy_preserves_existing_output(self):
        output = self.target / "existing.png"
        output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            files._copy_exclusive(self.image, output)
        self.assertEqual(output.read_bytes(), b"keep")

    def test_atomic_replace_failure_keeps_original_text(self):
        caption = self.source / "photo.txt"
        with mock.patch.object(files.os, "replace", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                files.atomic_write_text(caption, "new")
        self.assert_originals()
        self.assertFalse(list(self.source.glob("*.tmp")))


class AuditRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def image(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        path.with_suffix(".txt").write_text("caption", encoding="utf-8")
        return path

    def run_audit(self, **kwargs):
        return audit.audit_dataset(self.root, verify_images=False, **kwargs)

    def test_singleton_sizes_are_not_hashed(self):
        for i in range(20):
            self.image(f"img{i}.png", b"x" * (i + 1))
        with mock.patch.object(audit, "_sha256_file", side_effect=AssertionError("unnecessary hash")):
            report = self.run_audit()
        self.assertEqual(report.hashed_image_count, 0)
        self.assertEqual(report.issue_count, 0)

    def test_only_same_size_candidates_are_hashed(self):
        self.image("a.png", b"duplicate")
        self.image("b.png", b"duplicate")
        self.image("c.png", b"unique size")
        with mock.patch.object(audit, "_sha256_file", wraps=audit._sha256_file) as hashed:
            report = self.run_audit()
        self.assertEqual(hashed.call_count, 2)
        self.assertEqual(report.hashed_image_count, 2)
        self.assertEqual(report.issue_counts, {"duplicate_image": 1})

    def test_same_size_different_content_is_not_a_duplicate(self):
        self.image("a.png", b"abcd")
        self.image("b.png", b"efgh")
        self.assertEqual(self.run_audit().issue_count, 0)

    def test_uppercase_base_caption_matches_on_every_platform(self):
        (self.root / "Photo.PNG").write_bytes(b"img")
        (self.root / "PHOTO.TXT").write_text("caption", encoding="utf-8")
        report = self.run_audit(find_duplicates=False)
        self.assertEqual(report.caption_count, 1)
        self.assertEqual(report.issue_count, 0)

    def test_dotted_owners_match_transfer_rules(self):
        self.image("foo.png", b"first")
        self.image("foo.en.png", b"second")
        (self.root / "foo.en.lora.txt").write_text("caption", encoding="utf-8")
        report = self.run_audit(find_duplicates=False)
        self.assertEqual(report.issue_count, 0)
        self.assertEqual(report.caption_count, 2)

    def test_malformed_metadata_is_reported(self):
        self.image("img.png", b"image")
        (self.root / "img.tagflow.json").write_text("{broken", encoding="utf-8")
        self.assertEqual(self.run_audit().issue_counts, {"invalid_metadata": 1})

    def test_invalid_metadata_shape_is_reported(self):
        self.image("img.png", b"image")
        path = self.root / "img.tagflow.json"
        for value in ([], {"transforms": {}}, {"transforms": [7]}):
            with self.subTest(value=value):
                path.write_text(json.dumps(value), encoding="utf-8")
                self.assertEqual(self.run_audit().issue_counts, {"invalid_metadata": 1})

    def test_cancellation_does_not_return_partial_clean_report(self):
        with self.assertRaises(audit.AuditCancelled):
            self.run_audit(should_stop=lambda: True)

    def test_hashing_checks_cancellation_between_chunks(self):
        image = self.image("big.png", b"x" * 32)
        calls = 0
        def stop():
            nonlocal calls
            calls += 1
            return calls >= 3
        with self.assertRaises(audit.AuditCancelled):
            audit._sha256_file(image, chunk_size=4, should_stop=stop)
        self.assertEqual(calls, 3)

    def test_scan_errors_are_not_silently_clean(self):
        def denied(root, onerror, **kwargs):
            onerror(PermissionError(errno.EACCES, "denied", str(root / "private")))
            return iter([])
        with mock.patch.object(audit.os, "walk", side_effect=denied):
            report = self.run_audit()
        self.assertEqual(report.issue_counts, {"unreadable_directory": 1})

    def test_hash_read_error_is_reported(self):
        self.image("a.png", b"same")
        self.image("b.png", b"size")
        with mock.patch.object(audit, "_sha256_file", side_effect=OSError("unreadable")):
            report = self.run_audit()
        self.assertEqual(report.issue_counts, {"unreadable_image": 2})

    def test_nonrecursive_audit_stays_in_requested_directory(self):
        self.image("root.png", b"one")
        sub = self.root / "sub"
        sub.mkdir()
        (sub / "sub.png").write_bytes(b"two")
        self.assertEqual(self.run_audit(recursive=False).image_count, 1)
        self.assertEqual(self.run_audit(recursive=True).image_count, 2)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX-only special files")
    def test_named_pipes_are_not_opened_as_images(self):
        os.mkfifo(self.root / "pipe.png")
        self.assertEqual(audit.audit_dataset(self.root).image_count, 0)


class OwnershipAndVersionTests(unittest.TestCase):
    def test_lookup_cost_depends_on_filename_not_dataset_size(self):
        class CountingSet(set):
            calls = 0
            def __contains__(self, value):
                self.calls += 1
                return super().__contains__(value)
        stems = CountingSet(f"image{i}" for i in range(10000))
        stems.add("target")
        self.assertEqual(resolve_sidecar("target.future.custom.txt", stems), ("target", ".future.custom.txt"))
        self.assertLessEqual(stems.calls, 3)

    def test_metadata_does_not_use_a_shorter_stem_owner(self):
        self.assertIsNone(resolve_sidecar("foo.bar.tagflow.json", {"foo"}))
        self.assertEqual(resolve_sidecar("foo.bar.tagflow.json", {"foo.bar"}), ("foo.bar", ".tagflow.json"))

    def test_longest_unicode_owner_keeps_original_suffix(self):
        self.assertEqual(resolve_sidecar("STRASSE.EN.TXT", {"strasse", "strasse.en"}),
                         ("strasse.en", ".TXT"))

    def test_version_command_runs_without_importing_gui(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root / "tagflow_entry.py"), "--version"],
                                capture_output=True, text=True, timeout=20, check=True)
        self.assertEqual(result.stdout.strip(), f"TagFlow {__version__}")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
