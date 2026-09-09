"""Transactional image-family transfers and atomic text writes.

Transfers stage every member before publishing, never overwrite destination
files, and delete move sources only after the entire family was copied. This
is exception-safe, not a multi-file filesystem transaction: do not edit either
folder during a transfer. A power loss can leave duplicates/staging files.
"""
from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from .sidecars import resolve_sidecar

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".avif"}
)


class TransferError(RuntimeError):
    """A failed transfer; recovery locations are included when cleanup fails."""


@dataclass(frozen=True)
class TransferResult:
    source_image: Path
    destination_image: Path
    transferred_files: tuple[Path, ...]


def atomic_write_text(path: str | os.PathLike[str], text: str, *, encoding: str = "utf-8") -> None:
    """Write through an adjacent temporary file and an atomic replacement."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | os.PathLike[str], value: Any, *, encoding: str = "utf-8") -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding=encoding)


def discover_image_family(image_path: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Return the image and only sidecars owned by its longest matching stem."""
    image = Path(image_path)
    if image.is_symlink():
        raise TransferError(f"シンボリックリンクの画像は転送できません: {image}")
    if not image.is_file():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image}")
    if image.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"未対応の画像形式です: {image.suffix or '(拡張子なし)'}")
    entries = tuple(image.parent.iterdir())
    images = [p for p in entries if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
              and (p.is_file() or p.is_symlink())]
    stem = image.stem.casefold()
    ambiguous = [p for p in images if p != image and p.stem.casefold() == stem]
    if ambiguous:
        names = ", ".join(p.name for p in sorted(ambiguous))
        raise TransferError(f"同一stemの画像が複数あり、サイドカーの所有者を安全に判定できません: {image.name}, {names}")
    stems = {p.stem.casefold() for p in images}
    sidecars = []
    for candidate in entries:
        identity = resolve_sidecar(candidate.name, stems)
        if identity is None or identity[0] != stem:
            continue
        if candidate.is_symlink():
            raise TransferError(f"シンボリックリンクのサイドカーは転送できません: {candidate}")
        if candidate.is_file():
            sidecars.append(candidate)
    sidecars.sort(key=lambda p: (p.name.casefold(), p.name))
    names = [p.name.casefold() for p in sidecars]
    if len(set(names)) != len(names):
        raise TransferError("大文字小文字だけが異なるサイドカーが重複しています。")
    return image, *sidecars


def _destination_has_stem(target_dir: Path, stem: str) -> bool:
    prefix = f"{stem}.".casefold()
    return any(p.name.casefold().startswith(prefix) for p in target_dir.iterdir())


def _occupied_stems(target_dir: Path) -> set[str]:
    occupied: set[str] = set()
    for child in target_dir.iterdir():
        name = child.name.casefold()
        occupied.update(name[:i] for i, char in enumerate(name) if char == ".")
    return occupied


@contextmanager
def _reserve_destination(source_image: Path, target_dir: Path) -> Iterator[Path]:
    """Serialize cooperating transfers by case-insensitive destination stem.

    The hash avoids invalid/overlong Windows lock names. Stale locks are not
    stolen: after a crash a subsequent transfer simply chooses another suffix.
    """
    occupied = _occupied_stems(target_dir)
    counter = 0
    while True:
        stem = source_image.stem if counter == 0 else f"{source_image.stem}_{counter}"
        counter += 1
        if stem.casefold() in occupied:
            continue
        token = hashlib.sha256(stem.casefold().encode("utf-8")).hexdigest()
        lock = target_dir / f".tagflow-lock-{token}"
        try:
            lock.mkdir()
        except FileExistsError:
            continue
        try:
            if _destination_has_stem(target_dir, stem):
                continue
            yield target_dir / f"{stem}{source_image.suffix}"
            return
        finally:
            try:
                lock.rmdir()
            except OSError as exc:
                logger.warning("転送予約ファイルが残っています: %s: %s", lock, exc)


def _map_family_destination(source_member: Path, source_image: Path, destination_image: Path) -> Path:
    if source_member == source_image:
        return destination_image
    identity = resolve_sidecar(source_member.name, {source_image.stem.casefold()})
    if identity is None:
        raise TransferError(f"サイドカー名を対応付けできません: {source_member}")
    return destination_image.with_name(destination_image.stem + identity[1])


def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise TransferError(f"通常ファイルではありません: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _copy_exclusive(source: Path, destination: Path) -> None:
    """Portable no-clobber copy, including cleanup of a partially written file."""
    created = False
    try:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            created = True
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        shutil.copystat(source, destination)
    except Exception:
        if created:
            _unlink_output(destination)
        raise


def _publish(staged: Path, destination: Path) -> None:
    """Publish without overwrite; Windows rename also supports exFAT.

    Windows rename fails if the destination exists; POSIX rename overwrites,
    so POSIX uses link instead. Both paths keep publication on one filesystem.
    """
    if os.name == "nt":
        os.rename(staged, destination)
        return
    try:
        os.link(staged, destination)
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EACCES, errno.EXDEV, errno.ENOSYS,
                             errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
        _copy_exclusive(staged, destination)


def _unlink_output(path: Path) -> None:
    """Remove only an output created by this transfer, including read-only copies."""
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        if os.name != "nt":
            raise
        path.chmod(stat.S_IWRITE)
        path.unlink()


def _cleanup_stage(stage: Path) -> None:
    def onerror(function, path, error_info):
        if os.name != "nt" or not isinstance(error_info[1], PermissionError):
            raise error_info[1]
        os.chmod(path, stat.S_IWRITE)
        function(path)
    try:
        shutil.rmtree(stage, onerror=onerror)
    except OSError as exc:
        logger.warning("転送の一時ファイルが残っています: %s: %s", stage, exc)


def _rewrite_metadata(staged: Path, names: dict[str, str]) -> None:
    """Update known filename fields, never arbitrary prompt/model/history text."""
    metadata = json.loads(staged.read_text(encoding="utf-8-sig"))
    if not isinstance(metadata, dict):
        raise ValueError("TagFlowメタデータのルートはJSONオブジェクトである必要があります。")
    def remap(record: dict[str, Any], key: str) -> None:
        value = record.get(key)
        if isinstance(value, str) and value.casefold() in names:
            record[key] = names[value.casefold()]
    for key in ("image_file", "base_caption_file"):
        remap(metadata, key)
    transforms = metadata.get("transforms", [])
    if not isinstance(transforms, list):
        raise ValueError("TagFlowメタデータのtransformsは配列である必要があります。")
    for transform in transforms:
        if not isinstance(transform, dict):
            raise ValueError("TagFlowの変換履歴はJSONオブジェクトである必要があります。")
        for key in ("source_file", "output_file"):
            remap(transform, key)
    atomic_write_json(staged, metadata)


def transfer_image_family(
    image_path: str | os.PathLike[str],
    target_dir: str | os.PathLike[str],
    operation: Literal["copy", "move"] = "copy",
) -> TransferResult:
    """Transfer a complete family with staged copies and no-clobber publication.

    On source-delete failure, restore removed originals from staging. If that
    rollback itself fails, retain *all* destinations and the original staging
    copies, and report their location. Never discard the last surviving copy.
    """
    if operation not in {"copy", "move"}:
        raise ValueError(f"未対応のファイル操作です: {operation!r}")
    source_image = Path(image_path)
    family = discover_image_family(source_image)
    destination_dir = Path(target_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    if operation == "move" and source_image.parent.resolve() == destination_dir.resolve():
        return TransferResult(source_image, source_image, family)

    with _reserve_destination(source_image, destination_dir) as destination_image:
        mapping = [(p, _map_family_destination(p, source_image, destination_image)) for p in family]
        stage = Path(tempfile.mkdtemp(prefix=".tagflow-transfer-", dir=destination_dir))
        published: list[Path] = []
        removed: list[tuple[Path, Path]] = []
        keep_stage = False
        originals: dict[Path, Path] = {}
        manifest = {"schema_version": 1, "operation": operation,
                    "files": [{"source": str(src.resolve()), "destination": str(dst.resolve()),
                               "staged_original": str(stage / str(i))}
                              for i, (src, dst) in enumerate(mapping)]}
        try:
            atomic_write_json(stage / "manifest.json", manifest)
            snapshots = {p: _fingerprint(p) for p in family}
            for index, (source, _) in enumerate(mapping):
                backup = stage / str(index)
                shutil.copyfile(source, backup)
                # Flush before copying read-only attributes (also works on Windows).
                with backup.open("r+b") as stream:
                    os.fsync(stream.fileno())
                shutil.copystat(source, backup)
                if _fingerprint(source) != snapshots[source]:
                    raise TransferError(f"転送中に入力ファイルが変更されました: {source}")
                originals[source] = backup
            names = {source.name.casefold(): dest.name for source, dest in mapping}
            for source, destination in mapping:
                staged = originals[source]
                if (source_image.stem != destination_image.stem
                        and source.name.casefold().endswith(".tagflow.json")):
                    staged = stage / "renamed-metadata.json"
                    shutil.copyfile(originals[source], staged)
                    _rewrite_metadata(staged, names)
                    shutil.copymode(originals[source], staged)
                _publish(staged, destination)
                published.append(destination)
                if staged == originals[source] and not staged.exists():
                    # Windows rename consumed the staging path; the destination
                    # is now the rollback copy. Renamed metadata retains its original.
                    originals[source] = destination
            if operation == "move":
                # Check the whole family before deleting even its first member.
                for source in family:
                    if _fingerprint(source) != snapshots[source]:
                        raise TransferError(f"転送中に入力ファイルが変更されました: {source}")
                for source in family:
                    if _fingerprint(source) != snapshots[source]:
                        raise TransferError(f"転送中に入力ファイルが変更されました: {source}")
                    source.unlink()
                    removed.append((source, originals[source]))
        except Exception as exc:
            recovery_errors: list[str] = []
            for source, backup in reversed(removed):
                try:
                    _copy_exclusive(backup, source)
                except Exception as rollback_exc:
                    recovery_errors.append(f"{source}: {rollback_exc}")
            if not recovery_errors:
                for destination in reversed(published):
                    try:
                        _unlink_output(destination)
                    except OSError as cleanup_exc:
                        recovery_errors.append(f"{destination}: {cleanup_exc}")
            keep_stage = bool(recovery_errors)
            detail = f"画像ファミリーの{operation}に失敗しました: {source_image}: {exc}"
            if recovery_errors:
                manifest["recovery_files"] = [{"source": str(src.resolve()), "backup": str(backup.resolve())}
                                              for src, backup in originals.items()]
                try:
                    atomic_write_json(stage / "manifest.json", manifest)
                except OSError as manifest_exc:
                    recovery_errors.append(f"復旧情報の保存失敗: {manifest_exc}")
                detail += f" / 復旧用原本: {stage} / 復旧失敗: " + "; ".join(recovery_errors)
            raise TransferError(detail) from exc
        finally:
            if not keep_stage:
                _cleanup_stage(stage)
        return TransferResult(source_image, destination_image, tuple(published))
