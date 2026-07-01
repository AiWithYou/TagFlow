#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = ROOT / "presets"

VALID_DETAIL_LEVELS = {"brief", "standard", "detailed"}
VALID_TRANSFORM_MODES = {
    "translate_ja_to_en",
    "translate_en_to_ja",
    "danbooru_tags",
    "natural_prompt_en",
    "natural_prompt_ja",
    "lora_caption_en",
}


def load_json_array(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{path} が見つかりません。") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} のJSON形式が不正です: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} は空でないJSON配列である必要があります。")
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path} の {index} 件目はJSONオブジェクトである必要があります。")
    return data


def require_text(item: dict, field: str, path: Path, index: int) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} の {index} 件目に必須文字列フィールド {field!r} がありません。")
    return value


def require_bool(item: dict, field: str, path: Path, index: int) -> bool:
    value = item.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{path} の {index} 件目に bool フィールド {field!r} が必要です。")
    return value


def require_unique(value: str, seen: set[str], path: Path, field: str, index: int) -> None:
    if value in seen:
        raise ValueError(f"{path} の {field!r} が重複しています: {value!r} ({index} 件目)")
    seen.add(value)


def validate_model_presets() -> None:
    path = PRESETS_DIR / "model_presets.json"
    data = load_json_array(path)
    seen_models: set[str] = set()
    for index, item in enumerate(data, start=1):
        require_text(item, "label", path, index)
        model = require_text(item, "model", path, index)
        require_unique(model, seen_models, path, "model", index)


def validate_prompt_presets() -> None:
    path = PRESETS_DIR / "prompt_presets.json"
    data = load_json_array(path)
    seen_ids: set[str] = set()
    for index, item in enumerate(data, start=1):
        preset_id = require_text(item, "id", path, index)
        require_unique(preset_id, seen_ids, path, "id", index)
        require_text(item, "label", path, index)
        require_text(item, "prompt", path, index)
        detail_level = require_text(item, "detail_level", path, index)
        if detail_level not in VALID_DETAIL_LEVELS:
            raise ValueError(f"{path} の {index} 件目の detail_level が不正です: {detail_level!r}")
        require_bool(item, "clean_response", path, index)
        require_bool(item, "use_japanese", path, index)


def validate_transform_presets() -> None:
    path = PRESETS_DIR / "transform_presets.json"
    data = load_json_array(path)
    seen_ids: set[str] = set()
    for index, item in enumerate(data, start=1):
        preset_id = require_text(item, "id", path, index)
        require_unique(preset_id, seen_ids, path, "id", index)
        require_text(item, "label", path, index)
        mode = require_text(item, "mode", path, index)
        if mode not in VALID_TRANSFORM_MODES:
            raise ValueError(f"{path} の {index} 件目の mode が不正です: {mode!r}")
        output_suffix = require_text(item, "output_suffix", path, index)
        if not output_suffix.startswith(".") or output_suffix == ".txt":
            raise ValueError(f"{path} の {index} 件目の output_suffix が不正です: {output_suffix!r}")
        require_text(item, "recommended_model", path, index)
        prompt = require_text(item, "prompt", path, index)
        if "{TEXT}" not in prompt:
            raise ValueError(f"{path} の {index} 件目の prompt に {{TEXT}} が必要です。")
        temperature = item.get("temperature")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError(f"{path} の {index} 件目に number フィールド 'temperature' が必要です。")


def main() -> int:
    validators = (
        validate_model_presets,
        validate_prompt_presets,
        validate_transform_presets,
    )
    errors: list[str] = []
    for validator in validators:
        try:
            validator()
        except Exception as exc:  # noqa: BLE001 - CLI validator should aggregate all failures.
            errors.append(str(exc))

    if errors:
        print("Preset validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Preset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
