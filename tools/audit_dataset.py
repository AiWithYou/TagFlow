#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit a TagFlow image/caption dataset from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tagflow_core.audit import DatasetAuditReport, audit_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="画像・caption・変換サイドカーの整合性を監査します。",
    )
    parser.add_argument("folder", type=Path, help="監査対象フォルダ")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="サブフォルダを再帰走査しない",
    )
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        help="Pillowによる画像デコード検証を省略する",
    )
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="SHA-256による画像重複検出を省略する",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        help="詳細レポートをJSONファイルへ保存する",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="人間向け表示の代わりにJSONを標準出力する",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="問題が1件以上あれば終了コード1を返す（CI向け）",
    )
    return parser


def print_human_report(report: DatasetAuditReport) -> None:
    print(f"対象: {report.root}")
    print(f"画像: {report.image_count} / サイドカー: {report.sidecar_count} / 問題: {report.issue_count}")
    if not report.issues:
        print("問題は見つかりませんでした。")
        return

    print("\n問題内訳:")
    for code, count in report.issue_counts.items():
        print(f"- {code}: {count}")

    print("\n詳細:")
    for issue in report.issues:
        print(f"[{issue.code}] {issue.path}")
        print(f"  {issue.message}")
        if issue.related_paths:
            print("  関連: " + ", ".join(issue.related_paths))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_dataset(
            args.folder,
            recursive=not args.no_recursive,
            verify_images=not args.skip_image_verify,
            find_duplicates=not args.skip_duplicates,
        )
        if args.json_path:
            report.write_json(args.json_path)
        if args.json_stdout:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print_human_report(report)
    except Exception as exc:  # noqa: BLE001 - CLI should return one normalized failure.
        print(f"監査エラー: {exc}", file=sys.stderr)
        return 2

    return 1 if args.fail_on_issues and report.issue_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
