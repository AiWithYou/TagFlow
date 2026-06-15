# TagFlow 実装前確認

このリポジトリで Codex に実装を依頼する場合は、先に次の順番で確認してください。

1. この `00_README_FIRST.md` を読む。
2. `codex_prompt_ja.md` の内容をそのまま Codex に貼る。
3. `docs/implementation_spec_ja.md` を仕様として参照する。
4. `presets/prompt_presets.json` と `presets/model_presets.json` を実装候補データとして使う。
5. 実装後に `docs/acceptance_tests.md` の項目を確認する。

## 実装方針

- 既存の `TagFlow.py` の構造を大きく変えず、画像タグ付けタブと詳細設定ダイアログに小さく追加する。
- モデル候補とプロンプト候補は `presets/` 配下の JSON を正とする。
- JSON が存在しない、空、または形式不正の場合は明示的なエラーにする。
- 新しい production dependency は追加しない。
