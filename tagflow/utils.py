import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ----------------- デフォルト削除パターン -----------------
default_initial_patterns = [
    r"^Here.*?:",
    r"^説明[:：]\s*",
    r"^This image shows",
    r"^The image shows",
    r"^In this image,",
    r"^The image depicts",
    r"^This image depicts",
    r"^The photo shows",
    r"^The picture shows",
    r"^I will describe",
    r"^I'll describe",
    r"^Let me describe",
    r"^I can see",
    r"^画像には",
    r"^この画像には",
    r"^この画像は",
    r"^この絵には",
    r"^この絵は",
    r"^この写真には",
    r"^この写真は",
    r"^写真には",
    r"^画像は",
    r"^この画像を.*?で説明します。?\n?",
    r"^以下.*?で説明します。?\n?",
    r"^、\s*",
]

default_additional_patterns = [
    r"^また、?",
    r"^そして、?",
    r"^なお、?",
    r"^さらに、?",
    r"^加えて、?",
    r"^特に、?",
    r"^具体的には、?",
    r"^Additionally,\s*",
    r"^Moreover,\s*",
    r"^Furthermore,\s*",
    r"^Also,\s*",
    r"^And\s*",
    r"^Specifically,\s*",
    r"^There\s+(?:is|are)\s+",
    r"^We\s+can\s+see\s+",
    r"^You\s+can\s+see\s+",
    r"^It\s+appears\s+",
    r"これは",
    r"それは",
    r"以下は",
    r"次のような",
    r"(?:以下の)?(?:画像に適用できる)?(?:Danbooru|だんぼーる|ダンボール|ダンボール)?タグ(?:です|となります)。?\n?",
    r"タグ(?:一覧|リスト)：\n?",
    r"^[*＊・]",
    r"^、",
    r"主な(?:特徴|要素)(?:：|は)(?:以下の)?(?:通り|とおり)(?:です)?。?\n?",
]

# ----------------- ユーティリティ関数 -----------------

def load_app_config(filepath=None):
    """JSON設定ファイルを読み込み、辞書を返す"""
    config_file = Path(filepath) if filepath else Path("app_config.json")
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                logger.info(f"設定ファイル {config_file} から読み込みました。")
                return config_data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"設定ファイル読み込みエラー: {e}")
    else:
        logger.info("設定ファイルが見つかりません。空の設定を使用します。")
    return {}

def save_app_config(config, filepath=None):
    """設定をJSONファイルに保存する"""
    config_file = Path(filepath) if filepath else Path("app_config.json")
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            logger.info(f"設定を {config_file} に保存しました。")
    except Exception as e:
        logger.error(f"設定保存エラー: {e}")

def apply_clean_patterns(text, patterns):
    """与えられたテキストに対して削除パターンを適用し整形する"""
    text = text.strip()
    for pattern in patterns.get("initial", []):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL).strip()
    for pattern in patterns.get("additional", []):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL).strip()
    text = re.sub(r'[・\-•*＊\n] ?', ', ', text, flags=re.MULTILINE | re.DOTALL)
    parts = [part.strip() for part in text.split(',') if part.strip() and not part.startswith('、')]
    return ', '.join(parts)

def fetch_latest_models(url="https://ollama.ai/library"):
    """Ollamaライブラリページからモデル一覧を取得"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        models = re.findall(r'x-test-model-title title="([^"]+)"', response.text)
        return sorted(set(models))
    except Exception as e:
        logger.warning(f"モデル一覧取得エラー: {e}")
        return []
