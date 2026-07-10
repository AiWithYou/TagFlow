# 信頼性改善とデータセット監査

## 起動方法

`start.bat` または `start.ps1` を使用してください。これらは `tagflow_entry.py` を起動し、既存の `TagFlow.py` GUIへ信頼性改善を適用してから画面を開きます。

```powershell
.\start.ps1
```

仮想環境を既に有効化している場合は、次でも起動できます。

```powershell
python tagflow_entry.py
```

`TagFlow.py` は既存GUI実装として残しています。直接 `python TagFlow.py` を実行した場合、以下の信頼性改善は適用されません。

## 適用される改善

### 画像ファミリー単位のコピー・移動

画像と同じstemを持つ次のファイルを一体として扱います。

- ベースcaption: `image.txt`
- 変換サイドカー: `image.en.txt`、`image.danbooru.txt`、`image.prompt.txt` など
- 将来追加される `image.*.txt`
- 変換履歴: `image.tagflow.json`

出力先に画像またはいずれかのサイドカーが存在する場合、ファミリー全体へ同じ連番を付けます。コピー途中で失敗した場合は作成済みファイルを削除し、移動途中で失敗した場合は可能な限り元へロールバックします。

### 原子的な保存

caption、変換結果、設定、`.tagflow.json` は隣接一時ファイルへ書き込み、flushとfsyncの後に `os.replace()` で置換します。処理中断時に既存ファイルが空になるリスクを抑えます。

設定ファイルの既定位置は、起動時のカレントディレクトリではなく、アプリ本体と同じ場所の `app_config.json` です。

### AI API通信

OllamaおよびLM Studio向け通信に以下を追加しています。

- 接続タイムアウト: 10秒
- 応答タイムアウト: 画像600秒、チャット300秒、テキスト変換は既存設定値
- 接続失敗と一時的HTTPエラーの限定再試行
- `Retry-After` 対応
- 長大なエラー本文の切り詰め
- 不正JSONレスポンスの明示的なエラー化

読み取りタイムアウトは再試行しません。推論がサーバ側で完了している可能性があり、同じ重い処理を重複実行するのを避けるためです。

## データセット監査

`tools/audit_dataset.py` は画像学習データを再帰走査し、次を検出します。

- ベースcaption不足
- 空caption
- UTF-8として読めないcaption
- 対応画像のない孤立サイドカー
- Pillowで検証できない画像
- 同じフォルダ内の同一stem画像
- SHA-256が一致する重複画像

基本実行:

```powershell
python tools/audit_dataset.py D:\dataset
```

JSONレポートを保存し、問題があれば終了コード1を返す場合:

```powershell
python tools/audit_dataset.py D:\dataset `
  --json D:\reports\tagflow-audit.json `
  --fail-on-issues
```

主なオプション:

| オプション | 内容 |
| --- | --- |
| `--no-recursive` | 監査対象フォルダ直下だけを確認 |
| `--skip-image-verify` | Pillowによる画像検証を省略 |
| `--skip-duplicates` | SHA-256重複検出を省略 |
| `--json PATH` | JSONレポートを保存 |
| `--json-stdout` | JSONを標準出力 |
| `--fail-on-issues` | 問題検出時に終了コード1 |

## 開発時の確認

```powershell
python tools/validate_presets.py
python -m py_compile TagFlow.py tagflow_entry.py tagflow_core/*.py tools/*.py
python -m unittest discover -s tests -v
```

GUI受け入れ確認では、画像タグ付け、チャット、テキスト変換、コピー、移動、停止操作をOllamaとLM Studioの双方で確認してください。特に、変換サイドカーと `.tagflow.json` が画像と同じ連番で移動することを確認します。
