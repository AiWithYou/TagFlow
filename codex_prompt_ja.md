# Codex 貼り付け用プロンプト

TagFlow にモデル候補とプロンプト候補のプリセット機能を実装してください。

最初に `00_README_FIRST.md` を読み、`docs/implementation_spec_ja.md` を仕様として参照してください。実装候補データとして `presets/prompt_presets.json` と `presets/model_presets.json` を使ってください。実装後は `docs/acceptance_tests.md` の項目を確認し、可能な範囲で検証コマンドを実行してください。

要件:

- `TagFlow.py` の既存構造を尊重し、画像タグ付けタブと詳細設定ダイアログを中心に最小変更で実装する。
- Ollama モデル候補は `presets/model_presets.json` から読み込み、画像タグ付けタブのモデルコンボボックスに反映する。
- プロンプト候補は `presets/prompt_presets.json` から読み込み、詳細設定ダイアログで選べるようにする。
- プロンプト候補を選択したら、プロンプト本文、前置き削除設定、詳細度、日本語出力設定が反映されるようにする。
- プリセット JSON が欠落、空、または不正な形式の場合は、黙って既定値へ戻さず明示的に失敗させる。
- 依存ライブラリは追加しない。
- 最後に commit と push まで行う。
