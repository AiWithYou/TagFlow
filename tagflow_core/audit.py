"""Dataset integrity audit with indexed ownership and size-filtered hashing."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from .files import SUPPORTED_IMAGE_SUFFIXES, atomic_write_json
from .sidecars import resolve_sidecar


class AuditCancelled(RuntimeError):
    """An audit was cancelled; no incomplete report is returned as a clean one."""


@dataclass(frozen=True)
class AuditIssue:
    code: str
    path: str
    message: str
    related_paths: tuple[str, ...] = ()


@dataclass
class DatasetAuditReport:
    root: str
    recursive: bool
    image_count: int = 0
    caption_count: int = 0
    sidecar_count: int = 0
    issues: list[AuditIssue] = field(default_factory=list)
    hashed_image_count: int = 0

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def issue_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for issue in self.issues:
            counts[issue.code] += 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "root": self.root,
            "recursive": self.recursive,
            "image_count": self.image_count,
            "caption_count": self.caption_count,
            "sidecar_count": self.sidecar_count,
            "hashed_image_count": self.hashed_image_count,
            "issue_count": self.issue_count,
            "issue_counts": self.issue_counts,
            "issues": [asdict(issue) for issue in self.issues],
        }

    def write_json(self, path: str | os.PathLike[str]) -> None:
        atomic_write_json(path, self.to_dict())


def _check_cancel(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise AuditCancelled("停止要求によりデータセット監査を中断しました。")


def _iter_files(root: Path, recursive: bool, onerror=None, should_stop=None) -> Iterable[Path]:
    # os.walk exposes scan errors (unlike rglob versions which suppress them).
    for directory, directories, filenames in os.walk(root, onerror=onerror, followlinks=False):
        _check_cancel(should_stop)
        if not recursive:
            directories.clear()
        for name in filenames:
            _check_cancel(should_stop)
            path = Path(directory) / name
            if path.is_file():  # Do not open FIFOs/devices as images.
                yield path


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024, *, should_stop=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            _check_cancel(should_stop)
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        return str(exc)
    return None


def _owner_stem_for_sidecar(sidecar: Path, image_stems: set[str]) -> str | None:
    """Compatibility helper; the audit itself reuses its per-directory index."""
    originals = {stem.casefold(): stem for stem in sorted(image_stems)}
    identity = resolve_sidecar(sidecar.name, originals.keys())
    return originals[identity[0]] if identity else None


def audit_dataset(
    root: str | os.PathLike[str],
    *,
    recursive: bool = True,
    verify_images: bool = True,
    find_duplicates: bool = True,
    should_stop: Callable[[], bool] | None = None,
) -> DatasetAuditReport:
    """Audit images/captions without modifying them.

    Equal contents necessarily have equal byte sizes, so only size groups of
    two or more images are hashed. Cancellation is checked between files and
    hash chunks; Pillow's individual image verification is not interruptible.
    """
    _check_cancel(should_stop)
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"監査対象フォルダが見つかりません: {root_path}")
    report = DatasetAuditReport(root=str(root_path), recursive=recursive)

    def scan_error(exc: OSError) -> None:
        report.issues.append(AuditIssue("unreadable_directory", str(exc.filename or root_path),
                                       f"フォルダを走査できません: {exc}"))

    files = sorted(_iter_files(root_path, recursive, scan_error, should_stop),
                   key=lambda p: (str(p).casefold(), str(p)))
    images = [p for p in files if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    texts = [p for p in files if p.name.casefold().endswith(".txt")]
    metadata = [p for p in files if p.name.casefold().endswith(".tagflow.json")]
    report.image_count = len(images)
    report.sidecar_count = len(texts) + len(metadata)

    by_stem: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    stems_by_dir: dict[Path, set[str]] = defaultdict(set)
    captions: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for image in images:
        by_stem[(image.parent, image.stem.casefold())].append(image)
        stems_by_dir[image.parent].add(image.stem.casefold())
    for text in texts:
        captions[(text.parent, text.name.casefold())].append(text)
    for group in by_stem.values():
        if len(group) > 1:
            for image in group:
                report.issues.append(AuditIssue(
                    "ambiguous_caption_owner", str(image),
                    "同じフォルダに同一stemの画像が複数あり、captionを共有してしまいます。",
                    tuple(str(p) for p in group if p != image),
                ))
    for group in captions.values():
        if len(group) > 1:
            report.issues.append(AuditIssue(
                "ambiguous_caption_case", str(group[0]),
                "大文字小文字だけが異なるサイドカーが重複しています。",
                tuple(str(p) for p in group[1:]),
            ))

    found_captions: set[Path] = set()
    for image in images:
        _check_cancel(should_stop)
        matches = captions.get((image.parent, f"{image.stem}.txt".casefold()), [])
        if not matches:
            report.issues.append(AuditIssue("missing_caption", str(image),
                                           f"ベースcaptionがありません: {image.stem}.txt"))
        else:
            caption = matches[0]
            found_captions.add(caption)
            try:
                content = caption.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError) as exc:
                report.issues.append(AuditIssue("unreadable_caption", str(caption),
                                               f"captionをUTF-8として読めません: {exc}", (str(image),)))
            else:
                if not content.strip():
                    report.issues.append(AuditIssue("empty_caption", str(caption),
                                                   "captionが空です。", (str(image),)))
        if verify_images:
            error = _verify_image(image)
            if error:
                report.issues.append(AuditIssue("invalid_image", str(image),
                                               f"Pillowで画像を検証できません: {error}"))
    report.caption_count = len(found_captions)

    for sidecar in (*texts, *metadata):
        _check_cancel(should_stop)
        if resolve_sidecar(sidecar.name, stems_by_dir[sidecar.parent]) is None:
            report.issues.append(AuditIssue("orphan_sidecar", str(sidecar),
                                           "対応する画像が見つからないサイドカーです。"))
    for sidecar in metadata:
        _check_cancel(should_stop)
        try:
            value = json.loads(sidecar.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict) or not isinstance(value.get("transforms", []), list):
                raise ValueError("メタデータはオブジェクト、transformsは配列である必要があります。")
            if any(not isinstance(item, dict) for item in value.get("transforms", [])):
                raise ValueError("変換履歴の各要素はオブジェクトである必要があります。")
        except (OSError, UnicodeError, ValueError) as exc:
            report.issues.append(AuditIssue("invalid_metadata", str(sidecar),
                                           f"TagFlowメタデータを読めません: {exc}"))

    if find_duplicates:
        by_size: dict[int, list[Path]] = defaultdict(list)
        for image in images:
            _check_cancel(should_stop)
            try:
                by_size[image.stat().st_size].append(image)
            except OSError as exc:
                report.issues.append(AuditIssue("unreadable_image", str(image),
                                               f"画像のサイズを取得できません: {exc}"))
        hashes: dict[str, list[Path]] = defaultdict(list)
        for group in by_size.values():
            if len(group) < 2:
                continue
            for image in group:
                _check_cancel(should_stop)
                try:
                    hashes[_sha256_file(image, should_stop=should_stop)].append(image)
                    report.hashed_image_count += 1
                except OSError as exc:
                    report.issues.append(AuditIssue("unreadable_image", str(image),
                                                   f"画像をハッシュ計算用に読めません: {exc}"))
        for digest, group in hashes.items():
            for duplicate in group[1:]:
                report.issues.append(AuditIssue("duplicate_image", str(duplicate),
                                               f"画像内容が重複しています (sha256={digest})。",
                                               (str(group[0]),)))
    _check_cancel(should_stop)
    report.issues.sort(key=lambda issue: (issue.code, issue.path.casefold(), issue.path))
    return report
