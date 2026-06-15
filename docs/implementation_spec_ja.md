# プリセット機能 実装仕様

## 目的

TagFlow の画像タグ付け機能で、Ollama モデル名とカスタムプロンプトを毎回手入力しなくても、管理された JSON 候補から選べるようにする。

## 対象ファイル

- `TagFlow.py`
- `presets/model_presets.json`
- `presets/prompt_presets.json`

## モデル候補

`presets/model_presets.json` は空でない JSON 配列とする。各要素は次のフィールドを持つ。

- `label`: UI 補足表示用の名称
- `model`: Ollama API に渡すモデル名
- `description`: 任意の説明

`TagFlow.py` はこのファイルを読み込み、画像タグ付けタブの `Ollamaモデル` コンボボックスに `model` を候補として追加する。コンボボックスは既存どおり編集可能にし、ユーザーが任意のモデル名を入力できる状態は維持する。

## プロンプト候補

`presets/prompt_presets.json` は空でない JSON 配列とする。各要素は次のフィールドを持つ。

- `id`: 一意な識別子
- `label`: UI に表示する候補名
- `prompt`: カスタムプロンプト本文
- `detail_level`: `brief`, `standard`, `detailed` のいずれか
- `clean_response`: 前置きや余計な表現を削除するかどうか
- `use_japanese`: 日本語出力チェックへ反映するかどうか
- `description`: 任意の説明

詳細設定ダイアログに `プロンプト候補` のコンボボックスを追加する。候補を選択した場合、`prompt`, `detail_level`, `clean_response` をフォームに反映し、ダイアログ確定後に `use_japanese` を画像タグ付けタブのチェックボックスへ反映する。

## エラー処理

- プリセットファイルが存在しない場合は明示的な例外を出す。
- JSON として読めない場合は明示的な例外を出す。
- 空配列、配列以外、必須フィールド不足、不正な `detail_level`、重複 `id` は明示的な例外にする。
- ハードコードされた候補へ黙って戻さない。

## 非対象

- Ollama API の通信仕様変更
- 画像分析処理の並列化変更
- ファイル移動、分析結果編集、AI チャットの大幅変更
- 新規 production dependency の追加
