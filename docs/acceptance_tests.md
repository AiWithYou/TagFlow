# 受け入れ確認

## 静的確認

- [ ] `00_README_FIRST.md` が存在し、作業順が書かれている。
- [ ] `codex_prompt_ja.md` が存在し、そのまま貼れる実装依頼になっている。
- [ ] `docs/implementation_spec_ja.md` が存在し、モデル候補とプロンプト候補の仕様が書かれている。
- [ ] `presets/model_presets.json` が存在し、空でない JSON 配列である。
- [ ] `presets/prompt_presets.json` が存在し、空でない JSON 配列である。
- [ ] `TagFlow.py` が `presets/model_presets.json` からモデル候補を読み込む。
- [ ] `TagFlow.py` が `presets/prompt_presets.json` からプロンプト候補を読み込む。
- [ ] ハードコードされたモデル候補へ黙って戻る処理がない。
- [ ] プリセット JSON の欠落、不正形式、必須フィールド不足が明示的なエラーになる。
- [ ] 新しい production dependency が追加されていない。

## 動作確認

- [ ] 画像タグ付けタブの `Ollamaモデル` コンボボックスに `model_presets.json` の候補が表示される。
- [ ] 詳細設定ダイアログに `プロンプト候補` コンボボックスが表示される。
- [ ] プロンプト候補を選択すると、カスタムプロンプト本文が反映される。
- [ ] プロンプト候補を選択すると、前置き削除チェックが反映される。
- [ ] プロンプト候補を選択すると、詳細度ラジオボタンが反映される。
- [ ] 日本語プリセットを OK すると、画像タグ付けタブの `日本語で出力` がオンになる。
- [ ] 英語プリセットを OK すると、画像タグ付けタブの `日本語で出力` がオフになる。

## 推奨コマンド

```powershell
python -m json.tool presets/model_presets.json
python -m json.tool presets/prompt_presets.json
python -m py_compile TagFlow.py
```
