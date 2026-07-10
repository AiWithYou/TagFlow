"""Dataset integrity audit for TagFlow image/caption collections."""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image

from .files import SUPPORTED_IMAGE_SUFFIXES, atomic_write_json


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
            "issue_count": self.issue_count,
            "issue_counts": self.issue_counts,
            "issues": [asdict(issue) for issue in self.issues],
        }

    def write_json(self, path: str | os.PathLike[str]) -> None:
        atomic_write_json(path, self.to_dict())


def _iter_files(root: Path, recursive: bool) -> Iterable[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    return (path for path in iterator if path.is_file())


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # Pillow exposes format-specific exception types.
        return str(exc)
    return None


def _owner_stem_for_sidecar(sidecar: Path, image_stems: set[str]) -> str | None:
    name = sidecar.name.casefold()
    # Longest first prevents an image named ``foo.en`` from being assigned to ``foo``.
    for stem in sorted(image_stems, key=len, reverse=True):
        folded_stem = stem.casefold()
        if name == f"{folded_stem}.txt" or name == f"{folded_stem}.tagflow.json":
            return stem
        if name.startswith(f"{folded_stem}.") and name.endswith(".txt"):
            return stem
    return None


def audit_dataset(
    root: str | os.PathLike[str],
    *,
    recursive: bool = True,
    verify_images: bool = True,
    find_duplicates: bool = True,
) -> DatasetAuditReport:
    """Audit image/caption pair integrity and optional image validity/hashes."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"監査対象フォルダが見つかりません: {root_path}")

    files = sorted(_iter_files(root_path, recursive), key=lambda path: str(path).casefold())
    images = [path for path in files if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    text_sidecars = [path for path in files if path.name.lower().endswith(".txt")]
    metadata_sidecars = [path for path in files if path.name.lower().endswith(".tagflow.json")]

    report = DatasetAuditReport(
        root=str(root_path),
        recursive=recursive,
        image_count=len(images),
        caption_count=len(
            {
                image.with_suffix(".txt")
                for image in images
                if image.with_suffix(".txt").exists()
            }
        ),
        sidecar_count=len(text_sidecars) + len(metadata_sidecars),
    )

    by_directory_stem: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for image in images:
        by_directory_stem[(image.parent, image.stem.casefold())].append(image)

    for same_stem_images in by_directory_stem.values():
        if len(same_stem_images) > 1:
            paths = tuple(str(path) for path in same_stem_images)
            for path in same_stem_images:
                report.issues.append(
                    AuditIssue(
                        code="ambiguous_caption_owner",
                        path=str(path),
                        message="同じフォルダに同一stemの画像が複数あり、captionを共有してしまいます。",
                        related_paths=tuple(other for other in paths if other != str(path)),
                    )
                )

    for image in images:
        caption = image.with_suffix(".txt")
        if not caption.exists():
            report.issues.append(
                AuditIssue(
                    code="missing_caption",
                    path=str(image),
                    message=f"ベースcaptionがありません: {caption.name}",
                )
            )
        else:
            try:
                content = caption.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError) as exc:
                report.issues.append(
                    AuditIssue(
                        code="unreadable_caption",
                        path=str(caption),
                        message=f"captionをUTF-8として読めません: {exc}",
                        related_paths=(str(image),),
                    )
                )
            else:
                if not content.strip():
                    report.issues.append(
                        AuditIssue(
                            code="empty_caption",
                            path=str(caption),
                            message="captionが空です。",
                            related_paths=(str(image),),
                        )
                    )

        if verify_images:
            error = _verify_image(image)
            if error:
                report.issues.append(
                    AuditIssue(
                        code="invalid_image",
                        path=str(image),
                        message=f"Pillowで画像を検証できません: {error}",
                    )
                )

    image_stems_by_dir: dict[Path, set[str]] = defaultdict(set)
    for image in images:
        image_stems_by_dir[image.parent].add(image.stem)

    for sidecar in (*text_sidecars, *metadata_sidecars):
        owner = _owner_stem_for_sidecar(sidecar, image_stems_by_dir[sidecar.parent])
        if owner is None:
            report.issues.append(
                AuditIssue(
                    code="orphan_sidecar",
                    path=str(sidecar),
                    message="対応する画像が見つからないサイドカーです。",
                )
            )

    if find_duplicates:
        hash_groups: dict[str, list[Path]] = defaultdict(list)
        for image in images:
            try:
                hash_groups[_sha256_file(image)].append(image)
            except OSError as exc:
                report.issues.append(
                    AuditIssue(
                        code="unreadable_image",
                        path=str(image),
                        message=f"画像をハッシュ計算用に読めません: {exc}",
                    )
                )
        for digest, duplicate_paths in hash_groups.items():
            if len(duplicate_paths) < 2:
                continue
            canonical = duplicate_paths[0]
            for duplicate in duplicate_paths[1:]:
                report.issues.append(
                    AuditIssue(
                        code="duplicate_image",
                        path=str(duplicate),
                        message=f"画像内容が重複しています (sha256={digest})。",
                        related_paths=(str(canonical),),
                    )
                )

    report.issues.sort(key=lambda issue: (issue.code, issue.path.casefold()))
    return report
