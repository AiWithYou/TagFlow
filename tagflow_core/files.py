"""Safe file and sidecar operations used by TagFlow.

The GUI treats an image and every caption/metadata sidecar sharing its stem as
one logical unit.  This module keeps those files together and avoids partial or
overwriting operations.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".avif"}
)


class TransferError(RuntimeError):
    """Raised when an image-family copy/move cannot be completed safely."""


@dataclass(frozen=True)
class TransferResult:
    """Result of copying or moving an image and its sidecars."""

    source_image: Path
    destination_image: Path
    transferred_files: tuple[Path, ...]


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write text using an adjacent temporary file and an atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
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


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    encoding: str = "utf-8",
) -> None:
    """Serialize JSON and replace the destination atomically."""

    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, payload, encoding=encoding)


def _is_sidecar_for(image_path: Path, candidate: Path) -> bool:
    if not candidate.is_file():
        return False
    image_stem = image_path.stem.casefold()
    name = candidate.name.casefold()
    if name == f"{image_stem}.tagflow.json":
        return True
    if name == f"{image_stem}.txt":
        return True
    return name.startswith(f"{image_stem}.") and name.endswith(".txt")


def discover_image_family(image_path: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Return the image followed by all same-stem text/TagFlow sidecars.

    Besides built-in suffixes such as ``.en.txt`` and ``.lora.txt``, unknown
    ``<stem>.*.txt`` sidecars are included so future transform presets remain
    compatible without changing this function.
    """

    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image}")
    if image.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"未対応の画像形式です: {image.suffix or '(拡張子なし)'}")

    ambiguous = sorted(
        (
            path
            for path in image.parent.iterdir()
            if path != image
            and path.is_file()
            and path.stem.casefold() == image.stem.casefold()
            and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if ambiguous:
        names = ", ".join(path.name for path in ambiguous)
        raise TransferError(
            "同一stemの画像が複数あり、サイドカーの所有者を安全に判定できません: "
            f"{image.name}, {names}"
        )

    sidecars = [
        candidate
        for candidate in image.parent.iterdir()
        if candidate != image and _is_sidecar_for(image, candidate)
    ]
    sidecars.sort(key=lambda path: path.name.casefold())
    return (image, *sidecars)


def _destination_has_stem(target_dir: Path, stem: str) -> bool:
    """Check a stem collision case-insensitively (matching Windows behavior)."""

    prefix = f"{stem}.".casefold()
    try:
        children = target_dir.iterdir()
    except FileNotFoundError:
        return False
    return any(child.name.casefold().startswith(prefix) for child in children)


def _choose_destination_image(source_image: Path, target_dir: Path) -> Path:
    base_stem = source_image.stem
    suffix = source_image.suffix
    counter = 0
    while True:
        candidate_stem = base_stem if counter == 0 else f"{base_stem}_{counter}"
        if not _destination_has_stem(target_dir, candidate_stem):
            return target_dir / f"{candidate_stem}{suffix}"
        counter += 1


def _map_family_destination(
    source_member: Path,
    source_image: Path,
    destination_image: Path,
) -> Path:
    if source_member == source_image:
        return destination_image
    trailing_name = source_member.name[len(source_image.stem) :]
    return destination_image.with_name(f"{destination_image.stem}{trailing_name}")


def transfer_image_family(
    image_path: str | os.PathLike[str],
    target_dir: str | os.PathLike[str],
    operation: Literal["copy", "move"] = "copy",
) -> TransferResult:
    """Copy or move an image and all sidecars without overwriting any family.

    A destination stem is selected only when no existing file starts with that
    stem.  Copies are removed on failure.  Moves are rolled back in reverse
    order when possible, preventing the image and captions from being split.
    """

    if operation not in {"copy", "move"}:
        raise ValueError(f"未対応のファイル操作です: {operation!r}")

    source_image = Path(image_path)
    family = discover_image_family(source_image)
    destination_dir = Path(target_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_image = _choose_destination_image(source_image, destination_dir)
    mapping = tuple(
        (member, _map_family_destination(member, source_image, destination_image))
        for member in family
    )

    completed: list[tuple[Path, Path]] = []
    try:
        for source, destination in mapping:
            if destination.exists():
                # Defensive check against a race after destination selection.
                raise FileExistsError(f"出力先が既に存在します: {destination}")
            if operation == "copy":
                shutil.copy2(source, destination)
            else:
                shutil.move(str(source), str(destination))
            completed.append((source, destination))
    except Exception as exc:
        cleanup_errors: list[str] = []
        if operation == "copy":
            for _, destination in reversed(completed):
                try:
                    destination.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    cleanup_errors.append(f"{destination}: {cleanup_exc}")
        else:
            for source, destination in reversed(completed):
                try:
                    if destination.exists() and not source.exists():
                        shutil.move(str(destination), str(source))
                except Exception as rollback_exc:  # noqa: BLE001 - preserve all rollback diagnostics.
                    cleanup_errors.append(f"{destination} -> {source}: {rollback_exc}")

        detail = f"画像ファミリーの{operation}に失敗しました: {source_image}: {exc}"
        if cleanup_errors:
            detail += " / 復旧失敗: " + "; ".join(cleanup_errors)
        raise TransferError(detail) from exc

    return TransferResult(
        source_image=source_image,
        destination_image=destination_image,
        transferred_files=tuple(destination for _, destination in completed),
    )
