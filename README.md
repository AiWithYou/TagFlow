# TagFlow: AI画像タグ付け・ファイル整理ツール

TagFlow は、ローカルで動作する Ollama の画像対応モデルを使って、画像の説明文や検索用タグを生成し、生成した `.txt` をもとに画像を検索、コピー、移動、一括編集できる Windows 向けデスクトップアプリです。

画像そのものをクラウドへ送らず、ローカルの Ollama API に送信して分析するため、個人写真、制作素材、学習データ、業務用画像を手元で整理したい場面に向いています。

## できること

- 画像をドラッグ＆ドロップして、Ollama で説明文やタグを生成する。
- 生成結果を画像と同じフォルダの `.txt` ファイルとして保存する。
- `.txt` 内のタグや説明文を検索して、該当画像をまとめてコピーまたは移動する。
- 画像に対応する `.txt` を個別編集、またはフォルダ単位で一括編集する。
- Ollama とテキストチャットして、タグ付け方針や分類ルールを相談する。
- 削除パターンを GUI で管理し、LLM が出しがちな前置き文を自動整形する。
- モデル候補とプロンプト候補を JSON プリセットで管理する。

## 想定ユースケース

| 用途 | 使い方 | 効果 |
| --- | --- | --- |
| SNS 投稿用写真の整理 | 写真に説明タグを付け、人物、場所、雰囲気で検索 | 手作業の目視選別を短縮 |
| AI 学習データ整備 | 画像ごとにタグ `.txt` を生成し、一括置換で表記ゆれを直す | キャプション作成とクリーニングを効率化 |
| 商品画像管理 | 商品カテゴリ、色、形、状態をタグ化し、検索でフォルダ分け | EC 登録前の素材整理を高速化 |
| 旅行写真アルバム | 場所、イベント、食事、景色などで自動分類 | 後から探しやすい写真棚卸し |
| 社内画像の安全管理 | 顔、書類、ホワイトボードなどを検出し、対象画像を移動 | 公開前チェックの補助 |
| 素材ライブラリ運用 | 背景、ポーズ、構図、色味などをタグ化 | 制作素材を再利用しやすくする |

## 全体の流れ

1. Ollama を起動する。
2. TagFlow を起動する。
3. `画像タグ付け` タブへ画像またはフォルダを追加する。
4. モデル、URL、日本語出力、詳細設定、プロンプト候補を選ぶ。
5. `分析実行` で画像ごとに `.txt` を生成する。
6. `ファイル移動` タブで `.txt` の内容を検索し、画像と `.txt` をまとめてコピーまたは移動する。
7. `分析結果編集` タブでタグの不要語削除、接頭辞、接尾辞、置換を行う。

## 画面と機能

### 画像タグ付けタブ

画像分析の中心になるタブです。

主な操作:

- 画像ファイルまたはフォルダをリストへ追加する。
- 複数画像をまとめて分析する。
- 選択画像だけを分析する。
- 分析中に処理を停止する。
- 画像プレビューと分析結果を確認する。
- 画像と同名の `.txt` に分析結果を保存する。

対応画像形式:

- `.jpg`
- `.jpeg`
- `.png`
- `.gif`
- `.bmp`
- `.webp`
- `.heic`
- `.avif`

HEIC/AVIF を扱う場合は `pillow_heif` が必要です。

![画像タグ付けタブ](https://github.com/user-attachments/assets/bb7b15fd-5c89-4e33-823b-517d3b4f97ff)

### モデル候補

ツールバーの `Ollamaモデル` は編集可能なコンボボックスです。候補は `presets/model_presets.json` から読み込まれます。

初期候補:

| ラベル | モデル名 | 用途 |
| --- | --- | --- |
| Gemma 3 27B Vision | `gemma3:27b` | 高精度寄りの画像説明 |
| Gemma 3 4B Vision | `gemma3:4b` | 軽量、高速確認 |
| LLaVA Latest | `llava:latest` | 汎用的な画像説明 |
| LLaVA 13B | `llava:13b` | 詳細な説明の試行 |

候補にないモデルも、コンボボックスへ直接入力できます。入力したモデル名は Ollama API の `model` としてそのまま送られます。

### 詳細設定とプロンプト候補

`詳細設定` ボタンを押すと、カスタムプロンプト、クリーニング、詳細度を設定できます。

追加された `プロンプト候補` では、`presets/prompt_presets.json` に定義された候補を選択できます。候補を選ぶと次の項目が反映されます。

- カスタムプロンプト本文
- 前置きや余計な表現を削除するかどうか
- 詳細度
- 日本語出力チェックのオン/オフ

初期候補:

| ID | 表示名 | 内容 |
| --- | --- | --- |
| `ja_caption_standard` | 日本語: 標準説明 | 2から3文の日本語説明 |
| `ja_caption_detailed` | 日本語: 詳細説明 | 4から5文の詳しい日本語説明 |
| `ja_tag_keywords` | 日本語: タグ抽出 | 日本語タグをカンマ区切りで出力 |
| `en_caption_standard` | English: Standard caption | 英語の標準説明 |
| `en_tag_keywords` | English: Tag keywords | 英語タグをカンマ区切りで出力 |

手入力したい場合は `手動入力` のままカスタムプロンプト欄を編集します。空欄の場合は、詳細度と日本語出力設定に応じた既存の標準プロンプトが使われます。

### 分析結果のクリーニング

LLM はしばしば次のような前置きを含めます。

- `この画像は...`
- `The image shows...`
- `Here is a description...`

TagFlow は `clean_patterns` に定義された正規表現でこれらを削除し、検索しやすいテキストへ整えます。削除パターンは `設定` メニューの `削除パターン設定` から編集できます。

### ファイル移動タブ

生成済みの `.txt` を検索し、対応する画像をまとめてコピーまたは移動できます。

主な操作:

- ソースフォルダを選ぶ。
- 宛先フォルダを選ぶ。
- 検索キーワードをカンマ区切りで入力する。
- AND 検索または OR 検索を選ぶ。
- 正規表現検索を使う。
- 検索結果から対象ファイルを選ぶ。
- 対応する `.txt` と画像をまとめてコピーまたは移動する。

画像ファイルと `.txt` は同じベース名で対応します。

例:

```text
photo_001.jpg
photo_001.txt
```

宛先に同名ファイルがある場合は、自動で連番が付与されます。

例:

```text
photo_001.jpg
photo_001_1.jpg
photo_001_2.jpg
```

![ファイル移動タブ](https://github.com/user-attachments/assets/c3a82446-cded-44f1-94dd-20b4e51754d3)

### 分析結果編集タブ

画像に対応する `.txt` を確認、編集、一括加工できます。

できること:

- フォルダ内の画像一覧を表示する。
- 選択画像のプレビューを表示する。
- 対応する `.txt` を個別編集する。
- 文字列を削除する。
- 先頭に文字列を追加する。
- 末尾に文字列を追加する。
- 文字列を置換する。
- 一括処理のログを確認する。

タグ表記をそろえるときに便利です。

例:

```text
girl, outdoors, sunset
```

を

```text
1girl, outdoors, sunset, warm lighting
```

のように整理できます。

![分析結果編集タブ](https://github.com/user-attachments/assets/fda5ecf7-0cd8-447b-878a-7cd83b1353fb)

### AIチャットタブ

画像タグ付けタブで指定している Ollama URL とモデルを使い、テキストチャットできます。

使い道:

- タグ付け方針を相談する。
- 検索キーワードの設計を考える。
- 画像分類ルールを作る。
- 生成されたタグの改善案を聞く。
- 英語タグと日本語タグの使い分けを相談する。

送信方法:

- `送信` ボタン
- `Shift+Enter`

![AIチャットタブ](https://github.com/user-attachments/assets/2f5d15ab-509e-4fdd-a5f5-cd5e68bded6a)

### 設定メニュー

`設定` メニューから削除パターンを編集できます。

削除パターン設定では、次の2段階を扱います。

- 初期クリーニング
- 追加クリーニング

パターンテストでサンプル文字列に対する効果を確認できます。

![削除パターン設定](https://github.com/user-attachments/assets/7ed7d31e-1c2f-489d-8483-263bf9872c2b)

### AIサーバ管理

`AIサーバ` メニューから `ollama serve` を起動、停止できます。

Windows では別コンソールで Ollama を起動します。macOS/Linux ではバックグラウンドプロセスとして起動します。

![AIサーバ管理](https://github.com/user-attachments/assets/0d8abe93-1d86-4c45-9b7a-6e0ae870ce4d)

## インストール

### 1. Python を用意する

Python 3.8 以上を使用します。

確認:

```powershell
python --version
```

### 2. 仮想環境を作る

Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. 依存ライブラリを入れる

```powershell
pip install -r requirements.txt
```

依存ライブラリ:

| ライブラリ | 用途 |
| --- | --- |
| `PySide6` | デスクトップ GUI |
| `Pillow` | 画像読み込み、変換、プレビュー |
| `requests` | Ollama API 通信 |
| `pillow_heif` | HEIC/HEIF 対応 |

### 4. Ollama を用意する

Ollama をインストールし、使いたい画像対応モデルを取得します。

例:

```powershell
ollama pull gemma3:4b
ollama pull gemma3:27b
ollama pull llava:latest
```

起動:

```powershell
ollama serve
```

TagFlow のメニューから起動することもできます。

## 起動方法

Windows では次のどれかで起動できます。

```powershell
.\start.ps1
```

```powershell
.\start.bat
```

```powershell
python TagFlow.py
```

macOS/Linux:

```bash
python TagFlow.py
```

## 画像タグ付けの詳しい手順

1. Ollama が起動していることを確認する。
2. TagFlow を起動する。
3. `画像タグ付け` タブを開く。
4. `Ollamaモデル` を選ぶ。
5. `Ollama URL` を確認する。
6. 日本語で出したい場合は `日本語で出力` をオンにする。
7. `詳細設定` を開く。
8. 必要なら `プロンプト候補` を選ぶ。
9. 必要ならカスタムプロンプトを直接編集する。
10. `OK` で詳細設定を閉じる。
11. `フォルダ追加`、`ファイル追加`、またはドラッグ＆ドロップで画像を追加する。
12. 一部だけ分析したい場合は画像を選択する。
13. `分析実行` を押す。
14. 確認ダイアログが出た場合は、選択画像だけか全画像かを選ぶ。
15. 処理完了後、画像と同じ場所に `.txt` が作られていることを確認する。

## 検索とファイル整理の詳しい手順

1. `ファイル移動` タブを開く。
2. `ソースフォルダ` を選ぶ。
3. `宛先フォルダ` を選ぶ。
4. 検索キーワードを入力する。
5. AND/OR を選ぶ。
6. 必要なら正規表現をオンにする。
7. `検索` を押す。
8. 結果一覧から対象画像を選ぶ。
9. `コピー` または `移動` を押す。

検索キーワード例:

```text
夕焼け, 海, 人物
```

OR 検索では、どれか1つが含まれれば一致します。AND 検索では、すべて含まれる必要があります。

## プリセットファイル

TagFlow ではモデル候補とプロンプト候補を `presets/` 配下の JSON で管理します。

```text
presets/
  model_presets.json
  prompt_presets.json
```

### model_presets.json

空でない JSON 配列です。

必須フィールド:

- `label`
- `model`

任意フィールド:

- `description`

例:

```json
{
  "label": "Gemma 3 4B Vision",
  "model": "gemma3:4b",
  "description": "軽量寄りの画像説明候補。高速確認や低VRAM環境向け。"
}
```

### prompt_presets.json

空でない JSON 配列です。

必須フィールド:

- `id`
- `label`
- `prompt`
- `detail_level`
- `clean_response`
- `use_japanese`

任意フィールド:

- `description`

`detail_level` は次のいずれかです。

- `brief`
- `standard`
- `detailed`

例:

```json
{
  "id": "ja_tag_keywords",
  "label": "日本語: タグ抽出",
  "prompt": "画像検索に使いやすいタグを日本語でカンマ区切りで出力してください。",
  "detail_level": "brief",
  "clean_response": true,
  "use_japanese": true,
  "description": "ファイル移動タブのキーワード検索に使いやすいタグ生成候補。"
}
```

### プリセットのエラー

プリセットはアプリ起動時に読み込まれます。次の場合は明示的なエラーになります。

- ファイルが存在しない。
- JSON として壊れている。
- 配列ではない。
- 空配列である。
- 必須フィールドがない。
- 必須フィールドが空文字列である。
- `prompt_presets.json` の `id` が重複している。
- `detail_level` が `brief`, `standard`, `detailed` 以外である。
- `clean_response` または `use_japanese` が真偽値ではない。

ハードコードされた候補へ黙って戻る処理はありません。プリセットファイルを修正してから起動してください。

## app_config.json

`app_config.json` はアプリ設定や削除パターンを保存するための JSON です。

現在の主な用途:

- 削除パターンの保存
- モデル名、API URL、日本語設定、カスタムプロンプトなどを読み込むための設定

削除パターンは次の構造です。

```json
{
  "clean_patterns": {
    "initial": [
      "^Here.*?:",
      "^説明[:：]\\s*"
    ],
    "additional": [
      "^また、?",
      "^そして、?"
    ]
  }
}
```

`initial` は前置き削除などの初期クリーニング、`additional` は接続詞や箇条書き記号などの追加クリーニングに使われます。

## ファイル構成

```text
TagFlow/
  00_README_FIRST.md              # Codex 作業前の確認手順
  app_config.json                 # アプリ設定、削除パターン
  codex_prompt_ja.md              # Codex に貼る日本語プロンプト
  LICENSE                         # ライセンス
  README.md                       # このファイル
  requirements.txt                # Python 依存ライブラリ
  start.bat                       # Windows 起動バッチ
  start.ps1                       # Windows PowerShell 起動スクリプト
  TagFlow.py                      # メインアプリ
  docs/
    acceptance_tests.md           # 受け入れ確認
    implementation_spec_ja.md     # プリセット機能の実装仕様
  presets/
    model_presets.json            # Ollama モデル候補
    prompt_presets.json           # プロンプト候補
```

## 開発メモ

### 構文チェック

```powershell
python -m py_compile TagFlow.py
```

### JSON チェック

```powershell
python -m json.tool presets/model_presets.json
python -m json.tool presets/prompt_presets.json
```

### 受け入れ確認

`docs/acceptance_tests.md` を確認してください。プリセット機能では、JSON の存在と妥当性、GUI への候補反映、候補選択時の設定反映、fail-fast 動作を確認します。

## トラブルシュート

### アプリ起動時にプリセットエラーが出る

`presets/model_presets.json` または `presets/prompt_presets.json` を確認してください。

確認ポイント:

- ファイルが存在するか。
- JSON として正しいか。
- 配列になっているか。
- 空配列ではないか。
- 必須フィールドが入っているか。
- `detail_level` が正しい値か。
- `clean_response` と `use_japanese` が `true` または `false` か。

### Ollama API エラーが出る

確認ポイント:

- Ollama が起動しているか。
- `Ollama URL` が正しいか。
- モデルを `ollama pull` 済みか。
- 選択したモデルが画像入力に対応しているか。

確認例:

```powershell
ollama list
```

### 画像が読み込めない

確認ポイント:

- 拡張子が対応形式か。
- ファイルが壊れていないか。
- HEIC/AVIF の場合は `pillow_heif` が入っているか。

### `.txt` が見つからない

画像タグ付けを実行すると、画像と同じ場所に同名の `.txt` が作られます。

例:

```text
sample.png
sample.txt
```

検索や移動では、この対応関係を使います。

### 検索結果が出ない

確認ポイント:

- `.txt` が存在するか。
- 検索キーワードの表記が `.txt` の中身と一致しているか。
- AND 検索で条件を厳しくしすぎていないか。
- 正規表現が不正ではないか。

## ライセンス

このソフトウェアは商用利用可能です。使用しているライブラリは、それぞれのライセンスに従ってください。
