# TagFlow テキスト変換機能 実装仕様

## 1. 目的

TagFlowを「画像キャプション生成ツール」から「画像プロンプト資産管理ツール」へ拡張する。

既存の画像同名 `.txt` を入力として、翻訳、Danbooruタグ化、自然文プロンプト化、LoRAキャプション化を行う。

## 2. 現在リポジトリとの差分

現在の更新では、モデル候補と画像分析プロンプト候補のJSONプリセット化は入っている。
一方、テキスト変換専用のUI、処理ワーカー、保存形式、変換履歴管理は未実装。

今回追加するのは以下。

- `TextTransformTab`
- `TextTransformService`
- `TextTransformWorker`
- `presets/transform_presets.json`
- `.tagflow.json` 変換履歴
- README追記

## 3. 変換モード定義

| mode | 表示名 | 入力 | 出力 | suffix |
| --- | --- | --- | --- | --- |
| `translate_ja_to_en` | 日本語→英語 | `.txt` | 英語翻訳 | `.en.txt` |
| `translate_en_to_ja` | 英語→日本語 | `.txt` | 日本語翻訳 | `.ja.txt` |
| `danbooru_tags` | Danbooruタグ化 | `.txt` | 英語タグ列 | `.danbooru.txt` |
| `natural_prompt_en` | 英語自然文プロンプト | `.txt` | 英語プロンプト | `.prompt.txt` |
| `natural_prompt_ja` | 日本語自然文プロンプト | `.txt` | 日本語プロンプト | `.prompt_ja.txt` |
| `lora_caption_en` | LoRAキャプション | `.txt` | 英語LoRA向けキャプション | `.lora.txt` |

## 4. ファイル探索

対象画像拡張子は既存の対応形式に合わせる。

```python
SUPPORTED_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.avif'}
```

画像 `path/to/image.jpg` に対して、入力テキストは `path/to/image.txt` を基本とする。

## 5. TextTransformService

責務:

- transform presetの読み込み
- 入力テキストからプロンプトを構築
- Ollama APIへPOST
- レスポンス抽出
- 出力クリーニング
- メタデータ生成

主なメソッド:

```python
class TextTransformService:
    def __init__(self, api_url: str, model: str, timeout: int = 120): ...
    def build_prompt(self, preset: dict, text: str, model: str) -> str: ...
    def transform_text(self, text: str, preset: dict) -> str: ...
    def clean_output(self, text: str, mode: str) -> str: ...
```

## 6. TextTransformWorker

責務:

- 複数ファイルを順番に処理
- `stop_requested` を監視
- 進捗、ログ、エラー、個別結果をSignalでUIへ返す

Signal:

```python
progress_updated = Signal(float)
log_message = Signal(str)
item_completed = Signal(str, str)
item_failed = Signal(str, str)
transform_complete = Signal(int, int)
```

## 7. TextTransformTab

UI部品:

- フォルダ入力欄 + 参照ボタン
- サブフォルダを含めるチェック
- 対象拡張子表示
- 変換モードコンボ
- モデルコンボ
- API URL入力欄
- 上書き許可チェック
- 入力テキストプレビュー
- 出力テキストプレビュー
- 選択項目をプレビュー変換
- 一括変換
- 停止
- プログレスバー
- ログ表示

## 8. 保存ポリシー

デフォルトでは既存 `image.txt` を上書きしない。

変換結果は次のサイドカーファイルへ保存する。

```text
image.en.txt
image.ja.txt
image.danbooru.txt
image.prompt.txt
image.prompt_ja.txt
image.lora.txt
```

出力先が既に存在し、上書き不可の場合:

- スキップする
- ログに出す

## 9. `.tagflow.json`

1画像につき1つ作る。
既存ファイルが存在する場合はJSONを読み、`transforms` に追記する。
JSONが壊れている場合は `.tagflow.bak.json` に退避して作り直す。

必須項目:

- `schema_version`
- `image_file`
- `base_caption_file`
- `transforms`
- 各transformの `mode`, `source_file`, `output_file`, `model`, `api_url`, `created_at`, `input_sha256`, `output_sha256`

## 10. クリーニング

Danbooruタグ化:

- 改行をカンマへ
- セミコロンをカンマへ
- 日本語読点をカンマへ
- 前後空白削除
- スペースを `_` へ
- 小文字化
- 重複削除

翻訳/自然文プロンプト:

- 前置きの削除
- Markdownコードフェンス削除
- 先頭末尾の引用符削除
- 空行の過剰削除

## 11. API URL

既存の `http://localhost:11434/api/generate` を初期値にする。
将来的にOpenAI互換APIへ移行しやすいよう、`TextTransformService` にAPI呼び出しを閉じ込める。

## 12. README追記ポイント

READMEに以下を追記済み。

- Text Transformタブの用途
- 変換モード一覧
- 出力サイドカーファイル
- 推奨モデル
- 既存 `.txt` を上書きしない安全設計
- TranslateGemma向けの使い方
