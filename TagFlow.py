#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import base64
import requests
import subprocess
import re
import io
import logging
import sys
import json
import shutil
import platform
import hashlib
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageQt
from threading import Thread

# PySide6インポート
from PySide6.QtGui import (
    QAction, QActionGroup, QPixmap, QIcon, QImage, QColor, QPalette,
    QFont, QTextCursor
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox, QProgressBar,
    QFileDialog, QSplitter, QListWidget, QListWidgetItem, QTextEdit,
    QGroupBox, QRadioButton, QStatusBar, QToolBar, QDialog, QMessageBox,
    QGridLayout
)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QEvent

# HEIC画像対応（pillow_heifがインストールされている場合）
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# ----------------- ロギング設定 -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
MODEL_PRESETS_FILE = APP_DIR / "presets" / "model_presets.json"
PROMPT_PRESETS_FILE = APP_DIR / "presets" / "prompt_presets.json"
TRANSFORM_PRESETS_FILE = APP_DIR / "presets" / "transform_presets.json"
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
API_PROVIDER_OLLAMA = "ollama"
API_PROVIDER_LM_STUDIO = "lm_studio"
DEFAULT_API_PROVIDER = API_PROVIDER_OLLAMA
API_PROVIDER_CHOICES = (
    (API_PROVIDER_OLLAMA, "Ollama"),
    (API_PROVIDER_LM_STUDIO, "LM Studio"),
)
API_PROVIDER_LABELS = dict(API_PROVIDER_CHOICES)
DEFAULT_API_URLS = {
    API_PROVIDER_OLLAMA: DEFAULT_OLLAMA_API_URL,
    API_PROVIDER_LM_STUDIO: DEFAULT_LM_STUDIO_API_URL,
}
DEFAULT_THEME = "light"
VALID_THEMES = {"light", "dark"}
SUPPORTED_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.avif'}
VALID_DETAIL_LEVELS = {"brief", "standard", "detailed"}
VALID_TRANSFORM_MODES = {
    "translate_ja_to_en",
    "translate_en_to_ja",
    "danbooru_tags",
    "natural_prompt_en",
    "natural_prompt_ja",
    "lora_caption_en",
}

# ----------------- デフォルト削除パターン -----------------
default_initial_patterns = [
    r"^Here.*?:",            # "Here..." で始まる前置きを削除
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
    r"^、\s*"
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
    r"(?:以下の)?(?:画像に適用できる)?(?:Danbooru|だんぼーる|ダンボール|ダンボーる)?タグ(?:です|となります)。?\n?",
    r"タグ(?:一覧|リスト)：\n?",
    r"^[*＊・]",
    r"^、",
    r"主な(?:特徴|要素)(?:：|は)(?:以下の)?(?:通り|とおり)(?:です)?。?\n?"
]

# ----------------- ユーティリティ関数 -----------------
def load_app_config(filepath=None):
    """
    JSON設定ファイルを読み込み、辞書を返す
    """
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
    """
    設定をJSONファイルに保存する
    """
    config_file = Path(filepath) if filepath else Path("app_config.json")
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            logger.info(f"設定を {config_file} に保存しました。")
    except Exception as e:
        logger.error(f"設定保存エラー: {e}")

def load_json_list(filepath, required_text_fields):
    """
    JSON配列のプリセットファイルを読み込み、必須文字列フィールドを検証する
    """
    preset_file = Path(filepath)
    try:
        with open(preset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"プリセットファイルが見つかりません: {preset_file}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"プリセットファイルのJSON形式が不正です: {preset_file}: {e}") from e
    except OSError as e:
        raise OSError(f"プリセットファイルを読み込めません: {preset_file}: {e}") from e

    if not isinstance(data, list) or not data:
        raise ValueError(f"プリセットファイルは空でないJSON配列である必要があります: {preset_file}")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{preset_file} の {index + 1} 件目はオブジェクトである必要があります。")
        for field in required_text_fields:
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{preset_file} の {index + 1} 件目に必須文字列フィールド {field} がありません。")

    return data

def load_model_presets():
    """
    AIモデル候補を読み込む
    """
    return load_json_list(MODEL_PRESETS_FILE, ("label", "model"))

def load_prompt_presets():
    """
    プロンプト候補を読み込み、詳細設定で使う項目を検証する
    """
    presets = load_json_list(PROMPT_PRESETS_FILE, ("id", "label", "prompt", "detail_level"))
    seen_ids = set()
    for index, preset in enumerate(presets):
        preset_id = preset["id"]
        if preset_id in seen_ids:
            raise ValueError(f"{PROMPT_PRESETS_FILE} の id が重複しています: {preset_id}")
        seen_ids.add(preset_id)

        if preset["detail_level"] not in VALID_DETAIL_LEVELS:
            raise ValueError(
                f"{PROMPT_PRESETS_FILE} の {index + 1} 件目の detail_level が不正です: "
                f"{preset['detail_level']}"
            )
        if not isinstance(preset.get("clean_response"), bool):
            raise ValueError(f"{PROMPT_PRESETS_FILE} の {index + 1} 件目に clean_response(bool) が必要です。")
        if not isinstance(preset.get("use_japanese"), bool):
            raise ValueError(f"{PROMPT_PRESETS_FILE} の {index + 1} 件目に use_japanese(bool) が必要です。")

    return presets

def load_transform_presets():
    """
    テキスト変換候補を読み込み、変換に必要な項目を検証する
    """
    presets = load_json_list(
        TRANSFORM_PRESETS_FILE,
        ("id", "label", "mode", "output_suffix", "recommended_model", "prompt")
    )
    seen_ids = set()
    for index, preset in enumerate(presets):
        preset_id = preset["id"]
        if preset_id in seen_ids:
            raise ValueError(f"{TRANSFORM_PRESETS_FILE} の id が重複しています: {preset_id}")
        seen_ids.add(preset_id)

        mode = preset["mode"]
        if mode not in VALID_TRANSFORM_MODES:
            raise ValueError(f"{TRANSFORM_PRESETS_FILE} の {index + 1} 件目の mode が不正です: {mode}")

        output_suffix = preset["output_suffix"]
        if not output_suffix.startswith(".") or output_suffix == ".txt":
            raise ValueError(
                f"{TRANSFORM_PRESETS_FILE} の {index + 1} 件目の output_suffix が不正です: "
                f"{output_suffix}"
            )

        if "{TEXT}" not in preset["prompt"]:
            raise ValueError(f"{TRANSFORM_PRESETS_FILE} の {index + 1} 件目の prompt に {{TEXT}} が必要です。")

        temperature = preset.get("temperature")
        if not isinstance(temperature, (int, float)):
            raise ValueError(f"{TRANSFORM_PRESETS_FILE} の {index + 1} 件目に temperature(number) が必要です。")

    return presets

def strip_model_channel_markers(text):
    """
    一部のチャット系モデルが返す channel 制御トークンを取り除く。
    """
    text = re.sub(
        r"^\s*<\|channel\>\s*(?:thought|analysis|final)?\s*(?:\r?\n)?\s*<channel\|>\s*",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"<\|channel\>\s*(?:thought|analysis|final)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<channel\|>\s*", "", text, flags=re.IGNORECASE)
    return text.strip()

def normalize_api_provider(api_provider):
    if not isinstance(api_provider, str) or api_provider not in API_PROVIDER_LABELS:
        valid = ", ".join(API_PROVIDER_LABELS.keys())
        raise ValueError(f"未対応のAI接続先です: {api_provider!r}。有効値: {valid}")
    return api_provider

def api_provider_display_name(api_provider):
    return API_PROVIDER_LABELS[normalize_api_provider(api_provider)]

def provider_default_api_url(api_provider):
    api_provider = normalize_api_provider(api_provider)
    return DEFAULT_API_URLS[api_provider]

def normalize_theme(theme):
    if not isinstance(theme, str) or theme not in VALID_THEMES:
        valid = ", ".join(sorted(VALID_THEMES))
        raise ValueError(f"未対応のテーマです: {theme!r}。有効値: {valid}")
    return theme

def build_ai_request_payload(
    api_provider,
    model,
    prompt,
    images=None,
    image_mime_types=None,
    temperature=None,
    top_p=None,
):
    """
    選択されたAI接続先に合わせてリクエストpayloadを構築する。
    """
    api_provider = normalize_api_provider(api_provider)
    model = model.strip() if isinstance(model, str) else ""
    prompt = prompt.strip() if isinstance(prompt, str) else ""
    if not model:
        raise ValueError("モデル名が空です。")
    if not prompt:
        raise ValueError("プロンプトが空です。")

    images = images or []
    if api_provider == API_PROVIDER_OLLAMA:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if images:
            payload["images"] = images
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if options:
            payload["options"] = options
        return payload

    if api_provider == API_PROVIDER_LM_STUDIO:
        if images:
            if not image_mime_types or len(image_mime_types) != len(images):
                raise ValueError("LM Studioの画像入力には画像ごとのMIMEタイプが必要です。")
            content = [{"type": "text", "text": prompt}]
            for base64_image, mime_type in zip(images, image_mime_types):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                })
        else:
            content = prompt

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": content}
            ],
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        return payload

    raise ValueError(f"未対応のAI接続先です: {api_provider!r}")

def extract_ai_response_text(api_provider, result):
    """
    選択されたAI接続先のレスポンスから本文だけを取り出す。
    """
    api_provider = normalize_api_provider(api_provider)
    provider_name = api_provider_display_name(api_provider)
    if not isinstance(result, dict):
        raise RuntimeError(f"{provider_name} APIレスポンスのルートがJSONオブジェクトではありません。")

    if api_provider == API_PROVIDER_OLLAMA:
        response_text = result.get("response")
        if isinstance(response_text, str) and response_text.strip():
            return response_text
        raise RuntimeError("Ollama APIレスポンスに response テキストがありません。")

    if api_provider == API_PROVIDER_LM_STUDIO:
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LM Studio APIレスポンスに choices がありません。")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("LM Studio APIレスポンスの choices[0] がオブジェクトではありません。")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("LM Studio APIレスポンスに message がありません。")
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        raise RuntimeError("LM Studio APIレスポンスに message.content テキストがありません。")

    raise ValueError(f"未対応のAI接続先です: {api_provider!r}")

def apply_clean_patterns(text, patterns):
    """
    与えられたテキストに対して、初期および追加パターンを適用してクリーニングを行う
    """
    text = strip_model_channel_markers(text)
    # 初期パターン適用
    for pattern in patterns.get("initial", []):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL).strip()
    # 追加パターン適用
    for pattern in patterns.get("additional", []):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL).strip()

    # 特殊文字削除処理: 改行やいくつかの記号をまとめてカンマに変換
    text = re.sub(r'[・\-•*＊\n] ?', ', ', text, flags=re.MULTILINE | re.DOTALL)

    # 不要な空白や行頭にある読点を調整
    parts = [part.strip() for part in text.split(',') if part.strip() and not part.startswith('、')]
    return ', '.join(parts)

# ----------------- GUIウィジェット -----------------
class DropListWidget(QListWidget):
    """
    ドラッグ＆ドロップで画像ファイルを追加できるリストウィジェット
    """
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setIconSize(QSize(100, 100))
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setSpacing(10)
        self.setDragDropMode(QListWidget.InternalMove)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            file_paths = [url.toLocalFile() for url in event.mimeData().urls()]
            self.files_dropped.emit(file_paths)
        else:
            super().dropEvent(event)

class ImagePreviewWidget(QWidget):
    """
    画像のプレビュー表示ウィジェット
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.pixmap = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("background-color: #202124; color: #f1f3f4; border-radius: 5px;")
        layout.addWidget(self.image_label)

    def set_image(self, image_path):
        """
        指定されたパスの画像をラベルに表示する
        HEICなどの特殊フォーマットはPILで開いてから変換
        """
        self.image_path = image_path
        if not image_path or not Path(image_path).exists():
            self.image_label.setText("画像なし")
            self.pixmap = None
            return
        try:
            # まずPILで画像を開く
            with Image.open(image_path) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGBA')
                # PILイメージをQImageに変換
                qimg = ImageQt.ImageQt(img)
                self.pixmap = QPixmap.fromImage(qimg)
            if self.pixmap.isNull():
                self.image_label.setText("画像読み込み失敗")
                return
            self.update_pixmap()
        except Exception as e:
            logger.error(f"画像プレビューエラー: {str(e)}")
            self.image_label.setText(f"エラー: {str(e)}")

    def update_pixmap(self):
        """
        ウィジェットのサイズに合わせて画像をリサイズして表示する
        """
        if self.pixmap and not self.pixmap.isNull():
            scaled_pixmap = self.pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """
        ウィジェットのリサイズイベント時に呼び出され、画像サイズを更新
        """
        self.update_pixmap()
        super().resizeEvent(event)

class TagDisplayWidget(QWidget):
    """
    分析結果（タグ）を表示するためのテキスト表示ウィジェット
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setMinimumHeight(100)
        self.text_display.setObjectName("tagDisplay")
        layout.addWidget(self.text_display)

    def set_tags(self, tags_text):
        """
        表示用テキストをセットする
        """
        self.text_display.setText(tags_text)

class ImageListItem(QListWidgetItem):
    """
    画像ファイルの一覧に表示する各アイテム
    """
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = Path(image_path)
        self.setText(self.image_path.name)
        self.setToolTip(str(self.image_path))
        try:
            with Image.open(image_path) as img:
                # RGBモードに変換して処理
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGBA')
                img.thumbnail((100, 100))
                qimg = ImageQt.ImageQt(img)
                self.setIcon(QIcon(QPixmap.fromImage(qimg)))
                self.setSizeHint(QSize(120, 120))
        except Exception as e:
            logger.error(f"サムネイル生成エラー: {str(e)}")
            self.setIcon(QIcon())

# ----------------- 画像分析関連 -----------------
class ImageAnalyzer:
    """
    画像分析のためのクラス
    """
    def __init__(
        self,
        model,
        use_japanese=False,
        detail_level="standard",
        custom_prompt=None,
        clean_custom_response=True,
        api_provider=API_PROVIDER_OLLAMA,
        api_url=DEFAULT_OLLAMA_API_URL,
        clean_patterns=None
    ):
        """
        :param model: 使用するモデル名
        :param use_japanese: Trueの場合、日本語で説明させる
        :param detail_level: 'brief', 'standard', 'detailed' の3段階
        :param custom_prompt: カスタムプロンプト文字列
        :param clean_custom_response: Trueの場合、余計な前置きを自動的に削除
        :param api_provider: AI接続先（ollama / lm_studio）
        :param api_url: APIエンドポイントのURL
        :param clean_patterns: 削除・置換パターン辞書
        """
        self.model = model
        self.use_japanese = use_japanese
        self.detail_level = detail_level
        self.custom_prompt = custom_prompt
        self.clean_custom_response = clean_custom_response
        self.api_provider = normalize_api_provider(api_provider)
        self.api_url = api_url
        # HEIC対応のため、拡張子を追加
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.avif'}
        self.clean_patterns = clean_patterns or {
            "initial": default_initial_patterns,
            "additional": default_additional_patterns
        }

    def encode_image_data(self, image_path):
        """
        Pillowで画像を開き、Base64文字列とMIMEタイプを返す
        """
        try:
            with Image.open(image_path) as img:
                source_format = img.format or "PNG"
                # RGBモードに変換して処理
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGBA')
                img_buffer = io.BytesIO()
                save_format = source_format
                if save_format.upper() in ["HEIF", "WEBP"]:
                    save_format = "PNG"
                if save_format.upper() in ["JPEG", "JPG"] and img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(img_buffer, format=save_format)
                mime_format = "jpeg" if save_format.upper() in ["JPEG", "JPG"] else save_format.lower()
                mime_type = f"image/{mime_format}"
                return base64.b64encode(img_buffer.getvalue()).decode('utf-8'), mime_type
        except Exception as e:
            logger.error(f"画像エンコードエラー: {str(e)}")
            raise

    def encode_image(self, image_path):
        """
        Pillowで画像を開き、Base64にエンコードして返す
        """
        base64_image, _ = self.encode_image_data(image_path)
        return base64_image

    def get_prompt(self):
        """
        カスタムプロンプトがあればそれを使用し、
        なければ detail_level と use_japanese に応じたデフォルトを返す
        """
        if self.custom_prompt:
            prompt = self.custom_prompt.strip()
            if self.clean_custom_response:
                if self.use_japanese:
                    prompt += " 主要な要素や行動に焦点を当て、余計な前置きは不要です。"
                else:
                    prompt += " Focus on key elements and actions, and omit unnecessary introductory phrases."
            return prompt

        # デフォルトのプロンプト
        if self.use_japanese:
            if self.detail_level == "brief":
                return "この画像を1文で簡潔に説明してください。余計な前置きは不要です。"
            elif self.detail_level == "standard":
                return "この画像を2〜3文で説明してください。主要な要素や行動に焦点を当て、余計な前置きは不要です。"
            else:
                return "この画像を4〜5文で詳しく説明してください。視覚的な要素、行動、雰囲気などを含めて説明し、余計な前置きは不要です。"
        else:
            if self.detail_level == "brief":
                return "Describe this image in a single concise sentence, without any introductory phrases."
            elif self.detail_level == "standard":
                return "Describe this image in 2-3 sentences, focusing on key elements and actions. No introductory phrases."
            else:
                return "Describe this image in 4-5 sentences, including visual elements, actions, and atmosphere. No introductory phrases."

    def clean_response_text(self, response):
        """
        得られたレスポンステキストを設定されたパターンに基づいてクリーンアップする
        """
        return apply_clean_patterns(response, self.clean_patterns)

    def analyze_image(self, image_path):
        """
        画像をAPIに送信して分析し、テキスト（タグ）を返す
        """
        try:
            base64_image, mime_type = self.encode_image_data(image_path)
            payload = build_ai_request_payload(
                api_provider=self.api_provider,
                model=self.model,
                prompt=self.get_prompt(),
                images=[base64_image],
                image_mime_types=[mime_type],
            )
            response = requests.post(self.api_url, json=payload)
            if response.status_code != 200:
                logger.error(f"APIエラー: status_code={response.status_code}, text={response.text}")
                raise Exception(f"APIエラー: {response.status_code} - {response.text}")

            result = response.json()
            response_text = extract_ai_response_text(self.api_provider, result)
            if not self.clean_custom_response:
                return response_text
            else:
                return self.clean_response_text(response_text)
        except requests.exceptions.RequestException as e:
            logger.error(f"API通信エラー: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"画像分析エラー: {str(e)}")
            raise

class AnalysisWorker(QThread):
    """
    画像分析処理を別スレッドで実行するためのワーカークラス
    """
    progress_updated = Signal(float)
    analysis_complete = Signal(int, int)
    analysis_error = Signal(str)
    image_analyzed = Signal(str, str)

    def __init__(self, analyzer, image_paths):
        super().__init__()
        self.analyzer = analyzer
        self.image_paths = image_paths
        self.stop_requested = False

    def run(self):
        """
        指定された画像リストを順番に分析し、結果を.txtファイルに書き出す
        """
        total = len(self.image_paths)
        if total == 0:
            return
        processed = 0
        errors = 0
        for image_path in self.image_paths:
            if self.stop_requested:
                break
            try:
                logger.info(f"処理中: {Path(image_path).name}")
                result = self.analyzer.analyze_image(image_path)
                text_path = Path(image_path).with_suffix('.txt')
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(result)
                self.image_analyzed.emit(image_path, result)
                processed += 1
            except Exception as e:
                logger.error(f"分析エラー: {str(e)}")
                errors += 1
            progress = (processed + errors) / total * 100
            self.progress_updated.emit(progress)
        self.analysis_complete.emit(processed, errors)

    def stop(self):
        """
        分析処理を停止するフラグを立てる
        """
        self.stop_requested = True

class FileSearchWorker(QThread):
    """
    テキストファイル内のタグ検索を別スレッドで実行するためのワーカークラス
    """
    progress_updated = Signal(float)
    search_complete = Signal(list)
    search_error = Signal(str)

    def __init__(self, source_dir, search_terms, search_mode="OR", use_regex=False):
        super().__init__()
        self.source_dir = source_dir
        self.search_terms = search_terms
        self.search_mode = search_mode
        self.use_regex = use_regex
        self.stop_requested = False

    def run(self):
        """
        指定フォルダ直下の画像ファイルと同名の.txtファイルを検索し、
        検索キーワードに合致するファイルパスを収集する
        """
        try:
            source_path = Path(self.source_dir)
            if not source_path.exists():
                self.search_error.emit(f"ソースフォルダが存在しません: {self.source_dir}")
                return

            supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.avif'}
            image_files = []
            for ext in supported_formats:
                image_files.extend(source_path.glob(f"*{ext}"))
                image_files.extend(source_path.glob(f"*{ext.upper()}"))

            total_files = len(image_files)
            if total_files == 0:
                self.search_error.emit("画像ファイルが見つかりません")
                return

            matched_files = set()
            processed = 0
            for image_path in image_files:
                if self.stop_requested:
                    break
                text_path = image_path.with_suffix('.txt')
                if text_path.exists():
                    try:
                        with open(text_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if self._match_content(content):
                            matched_files.add(str(image_path))
                    except Exception as e:
                        logger.error(f"ファイル読み込みエラー: {str(e)}")
                processed += 1
                self.progress_updated.emit(processed / total_files * 100)

            self.search_complete.emit(list(matched_files))
        except Exception as e:
            self.search_error.emit(f"検索エラー: {str(e)}")

    def _match_content(self, content):
        """
        検索キーワードのOR/ANDマッチを行う（正規表現オプション対応）
        """
        if not self.search_terms:
            return False

        matches = []
        for term in self.search_terms:
            if not term:
                continue
            if self.use_regex:
                try:
                    pattern = re.compile(term, re.IGNORECASE)
                    match = bool(pattern.search(content))
                except re.error:
                    match = term.lower() in content.lower()
            else:
                match = term.lower() in content.lower()
            matches.append(match)

        if self.search_mode == "AND":
            return all(matches)
        else:
            return any(matches)

    def stop(self):
        """
        検索処理を停止するフラグを立てる
        """
        self.stop_requested = True

class FileOperationWorker(QThread):
    """
    ファイルのコピー・移動処理を別スレッドで実行するためのワーカークラス
    """
    progress_updated = Signal(float)
    operation_complete = Signal(int)
    operation_error = Signal(str)

    def __init__(self, file_paths, target_dir, operation="copy"):
        super().__init__()
        self.file_paths = file_paths
        self.target_dir = target_dir
        self.operation = operation
        self.stop_requested = False

    def run(self):
        """
        選択されたファイルをコピーまたは移動し、対応する.txtファイルも同様にコピー/移動する
        """
        try:
            target_path = Path(self.target_dir)
            if not target_path.exists():
                target_path.mkdir(parents=True, exist_ok=True)

            total_files = len(self.file_paths)
            if total_files == 0:
                return

            processed = 0
            for file_path in self.file_paths:
                if self.stop_requested:
                    break
                src_path = Path(file_path)
                if not src_path.exists():
                    continue
                dst_path = target_path / src_path.name

                # 同名ファイルがある場合は連番を付ける
                if dst_path.exists():
                    base_name = dst_path.stem
                    extension = dst_path.suffix
                    counter = 1
                    while dst_path.exists():
                        new_name = f"{base_name}_{counter}{extension}"
                        dst_path = target_path / new_name
                        counter += 1

                try:
                    # 画像コピー/移動
                    if self.operation == "copy":
                        shutil.copy2(src_path, dst_path)
                        # テキストファイルも同様に
                        text_path = src_path.with_suffix('.txt')
                        if text_path.exists():
                            text_target = dst_path.with_suffix('.txt')
                            shutil.copy2(text_path, text_target)
                    else:
                        shutil.move(src_path, dst_path)
                        text_path = src_path.with_suffix('.txt')
                        if text_path.exists():
                            text_target = dst_path.with_suffix('.txt')
                            shutil.move(text_path, text_target)
                    processed += 1
                except Exception as e:
                    logger.error(f"ファイル操作エラー: {str(e)}")

                self.progress_updated.emit(processed / total_files * 100)
            self.operation_complete.emit(processed)
        except Exception as e:
            self.operation_error.emit(f"ファイル操作エラー: {str(e)}")

    def stop(self):
        """
        ファイル操作処理を停止するフラグを立てる
        """
        self.stop_requested = True

class ChatWorker(QThread):
    """
    AIチャットAPIへの通信を非同期で実行するワーカー
    """
    result_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, api_provider, api_url, payload):
        super().__init__()
        self.api_provider = normalize_api_provider(api_provider)
        self.api_url = api_url
        self.payload = payload

    def run(self):
        """
        APIにPOSTリクエストを送り、結果を受け取る
        """
        try:
            response = requests.post(self.api_url, json=self.payload)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code} Error: {response.text}")
            result = response.json()
            self.result_ready.emit(extract_ai_response_text(self.api_provider, result))
        except Exception as e:
            self.error_occurred.emit(str(e))

# ----------------- テキスト変換関連 -----------------
class TextTransformService:
    """
    既存txtを選択されたAI接続先に渡して翻訳/タグ化/プロンプト化するサービス。
    """
    def __init__(self, api_provider, api_url, model, timeout=120):
        self.api_provider = normalize_api_provider(api_provider)
        self.api_url = api_url
        self.model = model
        self.timeout = timeout

    def build_prompt(self, preset, text, model):
        """
        変換プリセットからAI接続先へ渡すプロンプトを構築する。
        """
        if not text.strip():
            raise ValueError("変換元テキストが空です。")
        return preset["prompt"].replace("{TEXT}", text.strip())

    def transform_text(self, text, preset):
        """
        AI APIにテキスト変換を依頼し、整形済みの結果を返す。
        """
        provider_name = api_provider_display_name(self.api_provider)
        payload = build_ai_request_payload(
            api_provider=self.api_provider,
            model=self.model,
            prompt=self.build_prompt(preset, text, self.model),
            temperature=preset["temperature"],
            top_p=0.9,
        )
        try:
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"{provider_name} API通信エラー: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(f"{provider_name} APIエラー: HTTP {response.status_code} - {response.text}")

        try:
            result = response.json()
        except ValueError as e:
            raise RuntimeError(f"{provider_name} APIレスポンスのJSON形式が不正です: {e}") from e

        response_text = extract_ai_response_text(self.api_provider, result)

        return self.clean_output(response_text, preset["mode"])

    def clean_output(self, text, mode):
        """
        変換モードに応じて出力を検索・保存しやすい形へ整える。
        """
        text = strip_model_channel_markers(text)
        text = self._strip_markdown_fence(text)
        if mode == "danbooru_tags":
            return self._clean_danbooru_tags(text)

        text = re.sub(
            r"^\s*(?:translation|translated text|output|result|prompt|caption|翻訳|結果|出力)\s*[:：]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.strip().strip('"').strip("'").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _strip_markdown_fence(self, text):
        text = text.strip()
        text = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        return text.strip()

    def _clean_danbooru_tags(self, text):
        text = re.sub(
            r"^\s*(?:danbooru[- ]style tags|danbooru tags|tags|tag list|タグ|タグ一覧)\s*[:：]\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        )
        text = text.replace("\n", ",")
        text = text.replace(";", ",")
        text = text.replace("、", ",")
        parts = []
        seen = set()
        for raw_tag in text.split(","):
            tag = re.sub(r"^\s*(?:[-*・•]+|\d+[.)）])\s*", "", raw_tag.strip())
            tag = tag.lower()
            tag = re.sub(r"\s+", "_", tag)
            tag = re.sub(r"_+", "_", tag)
            tag = tag.strip(" _.,:;[](){}\"'#")
            if tag and tag not in seen:
                parts.append(tag)
                seen.add(tag)
        return ", ".join(parts)

    def get_output_path(self, image_path, preset):
        image_path = Path(image_path)
        return image_path.with_name(f"{image_path.stem}{preset['output_suffix']}")

    def write_transform_result(self, image_path, source_path, output_path, preset, input_text, output_text, overwrite):
        """
        サイドカーファイルと .tagflow.json の変換履歴を書き込む。
        """
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"出力先が既に存在するためスキップします: {output_path.name}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        self.update_metadata(
            image_path=Path(image_path),
            base_caption_path=Path(source_path),
            output_path=output_path,
            preset=preset,
            input_text=input_text,
            output_text=output_text,
        )

    def update_metadata(self, image_path, base_caption_path, output_path, preset, input_text, output_text):
        metadata_path = image_path.with_name(f"{image_path.stem}.tagflow.json")
        metadata = self._load_metadata(metadata_path, image_path, base_caption_path)
        metadata["schema_version"] = 1
        metadata["image_file"] = image_path.name
        metadata["base_caption_file"] = base_caption_path.name
        metadata["transforms"].append({
            "mode": preset["mode"],
            "source_file": base_caption_path.name,
            "output_file": output_path.name,
            "api_provider": self.api_provider,
            "model": self.model,
            "api_url": self.api_url,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "input_sha256": self._sha256_text(input_text),
            "output_sha256": self._sha256_text(output_text),
        })
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _load_metadata(self, metadata_path, image_path, base_caption_path):
        default_metadata = {
            "schema_version": 1,
            "image_file": image_path.name,
            "base_caption_file": base_caption_path.name,
            "transforms": [],
        }
        if not metadata_path.exists():
            return default_metadata

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                raise ValueError("メタデータのルートはオブジェクトである必要があります。")
            transforms = metadata.get("transforms")
            if transforms is None:
                metadata["transforms"] = []
            elif not isinstance(transforms, list):
                raise ValueError("transforms は配列である必要があります。")
            return metadata
        except (json.JSONDecodeError, OSError, ValueError) as e:
            backup_path = self._backup_invalid_metadata(metadata_path)
            logger.warning(f"壊れたメタデータを退避しました: {backup_path} ({e})")
            return default_metadata

    def _backup_invalid_metadata(self, metadata_path):
        backup_path = metadata_path.with_name(f"{metadata_path.stem}.bak.json")
        if backup_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_path = metadata_path.with_name(f"{metadata_path.stem}.bak.{timestamp}.json")
        metadata_path.replace(backup_path)
        return backup_path

    def _sha256_text(self, text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

class TextTransformPreviewWorker(QThread):
    """
    選択ファイル1件のプレビュー変換をバックグラウンドで実行する。
    """
    result_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, text, preset, api_provider, api_url, model):
        super().__init__()
        self.text = text
        self.preset = preset
        self.api_provider = normalize_api_provider(api_provider)
        self.api_url = api_url
        self.model = model

    def run(self):
        try:
            service = TextTransformService(self.api_provider, self.api_url, self.model)
            self.result_ready.emit(service.transform_text(self.text, self.preset))
        except Exception as e:
            self.error_occurred.emit(str(e))

class TextTransformWorker(QThread):
    """
    バッチ変換をバックグラウンドで実行するワーカー。
    """
    progress_updated = Signal(float)
    log_message = Signal(str)
    item_completed = Signal(str, str)
    item_failed = Signal(str, str)
    transform_complete = Signal(int, int)

    def __init__(self, image_paths, preset, api_provider, api_url, model, overwrite=False):
        super().__init__()
        self.image_paths = [Path(path) for path in image_paths]
        self.preset = preset
        self.api_provider = normalize_api_provider(api_provider)
        self.api_url = api_url
        self.model = model
        self.overwrite = overwrite
        self.stop_requested = False

    def run(self):
        total = len(self.image_paths)
        if total == 0:
            self.transform_complete.emit(0, 0)
            return

        service = TextTransformService(self.api_provider, self.api_url, self.model)
        success = 0
        failed = 0
        for index, image_path in enumerate(self.image_paths, start=1):
            if self.stop_requested:
                self.log_message.emit("停止要求を受け取ったため、残りの処理を中断しました。")
                break

            source_path = image_path.with_suffix(".txt")
            output_path = service.get_output_path(image_path, self.preset)
            try:
                if not source_path.exists():
                    raise FileNotFoundError(f"入力txtが見つかりません: {source_path.name}")
                if output_path.exists() and not self.overwrite:
                    raise FileExistsError(f"出力先が既に存在するためスキップします: {output_path.name}")

                with open(source_path, "r", encoding="utf-8") as f:
                    input_text = f.read()
                output_text = service.transform_text(input_text, self.preset)
                service.write_transform_result(
                    image_path=image_path,
                    source_path=source_path,
                    output_path=output_path,
                    preset=self.preset,
                    input_text=input_text,
                    output_text=output_text,
                    overwrite=self.overwrite,
                )
                success += 1
                self.log_message.emit(f"{image_path.name}: {output_path.name} を保存しました。")
                self.item_completed.emit(str(image_path), str(output_path))
            except Exception as e:
                failed += 1
                self.log_message.emit(f"{image_path.name}: {e}")
                self.item_failed.emit(str(image_path), str(e))
            self.progress_updated.emit(index / total * 100)

        self.transform_complete.emit(success, failed)

    def stop(self):
        self.stop_requested = True

# ----------------- 削除パターンテストダイアログ -----------------
class PatternTestDialog(QDialog):
    """
    ユーザーが入力した削除パターンをサンプルテキストに対してテストできるダイアログ
    """
    def __init__(self, patterns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("削除パターンテスト")
        self.resize(600, 400)
        self.patterns = patterns

        layout = QVBoxLayout(self)

        sample_label = QLabel("サンプルテキスト:")
        layout.addWidget(sample_label)

        self.sample_edit = QTextEdit()
        self.sample_edit.setPlainText("ここにサンプルテキストを入力してください。")
        layout.addWidget(self.sample_edit)

        test_button = QPushButton("テスト実行")
        test_button.clicked.connect(self.run_test)
        layout.addWidget(test_button)

        result_label = QLabel("テスト結果:")
        layout.addWidget(result_label)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        layout.addWidget(self.result_display)

        close_button = QPushButton("閉じる")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

    def run_test(self):
        """
        サンプルテキストに現在のパターンを適用した結果を表示
        """
        sample_text = self.sample_edit.toPlainText()
        result = apply_clean_patterns(sample_text, self.patterns)
        self.result_display.setPlainText(result)

# ----------------- 削除パターン設定ダイアログ -----------------
class DeletionPatternDialog(QDialog):
    """
    削除パターンの設定を行うダイアログ
    """
    def __init__(self, parent=None, current_patterns=None):
        super().__init__(parent)
        self.setWindowTitle("削除パターン設定")
        self.setMinimumWidth(600)
        self.current_patterns = current_patterns or {
            "initial": default_initial_patterns,
            "additional": default_additional_patterns
        }
        layout = QVBoxLayout(self)

        init_group = QGroupBox("初期クリーニングパターン（1行1パターン）")
        init_layout = QVBoxLayout()
        self.initial_patterns_edit = QTextEdit()
        self.initial_patterns_edit.setPlainText("\n".join(self.current_patterns.get("initial", default_initial_patterns)))
        init_layout.addWidget(self.initial_patterns_edit)
        init_group.setLayout(init_layout)
        layout.addWidget(init_group)

        add_group = QGroupBox("追加クリーニングパターン（1行1パターン）")
        add_layout = QVBoxLayout()
        self.additional_patterns_edit = QTextEdit()
        self.additional_patterns_edit.setPlainText("\n".join(self.current_patterns.get("additional", default_additional_patterns)))
        add_layout.addWidget(self.additional_patterns_edit)
        add_group.setLayout(add_layout)
        layout.addWidget(add_group)

        # 正規表現の情報ボタン
        self.regex_info_button = QPushButton("正規表現について")
        self.regex_info_button.clicked.connect(self.show_regex_info)
        layout.addWidget(self.regex_info_button)

        # パターンテストボタン
        self.test_pattern_button = QPushButton("パターンテスト")
        self.test_pattern_button.clicked.connect(self.open_pattern_test)
        layout.addWidget(self.test_pattern_button)

        # 設定ファイルのロード／セーブボタン
        file_button_layout = QHBoxLayout()
        self.load_button = QPushButton("設定ファイルを読み込み")
        self.save_button = QPushButton("設定ファイルの保存先選択")
        self.load_button.clicked.connect(self.load_patterns)
        self.save_button.clicked.connect(self.save_patterns)
        file_button_layout.addWidget(self.load_button)
        file_button_layout.addWidget(self.save_button)
        layout.addLayout(file_button_layout)

        close_layout = QHBoxLayout()
        self.close_button = QPushButton("閉じる")
        self.close_button.clicked.connect(self.accept)
        close_layout.addStretch()
        close_layout.addWidget(self.close_button)
        layout.addLayout(close_layout)

    def show_regex_info(self):
        """
        正規表現の簡易的なガイドを表示
        """
        info = (
            "【正規表現の基本例】\n"
            "  - '^abc' : 文字列の先頭が 'abc' である場合にマッチ\n"
            "  - 'xyz$' : 文字列の末尾が 'xyz' である場合にマッチ\n"
            "  - '[0-9]+' : 1文字以上の数字にマッチ\n"
            "  - '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}' : メールアドレスの簡易マッチ\n"
            "  - 'ab|cd' : 'ab'または'cd'にマッチ\n"
            "  - '\\s+' : 1文字以上の空白文字にマッチ\n"
            "  - '.*' : 任意の文字が0文字以上連続（貪欲マッチ）\n"
            "【特別な記号例】\n"
            "  - '.' : 任意の1文字（改行を除く; re.DOTALLで改行含む）\n"
            "  - '*' : 直前の要素を0回以上繰り返し\n"
            "  - '+' : 直前の要素を1回以上繰り返し\n"
            "  - '?' : 直前の要素を0回または1回\n"
            "  - '(?m)' : マルチラインモード（^と$が行頭・行末にマッチ）\n"
            "  - '(?s)' : DOTALLモード（'.'が改行にもマッチ）\n"
        )
        QMessageBox.information(self, "正規表現について", info)

    def open_pattern_test(self):
        """
        パターンテストダイアログを表示する
        """
        current_patterns = {
            "initial": [line.strip() for line in self.initial_patterns_edit.toPlainText().splitlines() if line.strip()],
            "additional": [line.strip() for line in self.additional_patterns_edit.toPlainText().splitlines() if line.strip()]
        }
        dialog = PatternTestDialog(current_patterns, self)
        dialog.exec()

    def load_patterns(self):
        """
        JSONファイルから削除パターンをロードする
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "設定ファイルを選択", "", "JSON Files (*.json)")
        if file_path:
            config = load_app_config(file_path)
            patterns = config.get("clean_patterns", {})
            init = patterns.get("initial", default_initial_patterns)
            add = patterns.get("additional", default_additional_patterns)
            self.initial_patterns_edit.setPlainText("\n".join(init))
            self.additional_patterns_edit.setPlainText("\n".join(add))
            QMessageBox.information(self, "ロード完了", "削除パターンをロードしました。")

    def save_patterns(self):
        """
        現在の削除パターンをJSONファイルに保存する
        """
        file_path, _ = QFileDialog.getSaveFileName(self, "設定ファイルの保存先を選択", "", "JSON Files (*.json)")
        if file_path:
            init = [line.strip() for line in self.initial_patterns_edit.toPlainText().splitlines() if line.strip()]
            add = [line.strip() for line in self.additional_patterns_edit.toPlainText().splitlines() if line.strip()]
            config = load_app_config(file_path)
            config["clean_patterns"] = {"initial": init, "additional": add}
            save_app_config(config, file_path)
            QMessageBox.information(self, "保存完了", "削除パターンを保存しました。")

    def get_patterns(self):
        """
        現在の削除パターンを辞書形式で取得
        """
        init = [line.strip() for line in self.initial_patterns_edit.toPlainText().splitlines() if line.strip()]
        add = [line.strip() for line in self.additional_patterns_edit.toPlainText().splitlines() if line.strip()]
        return {"initial": init, "additional": add}

# ----------------- 詳細設定ダイアログ -----------------
class SettingsDialog(QDialog):
    """
    画像タグ付けの詳細設定を行うダイアログ
    """
    def __init__(self, parent=None, prompt_presets=None):
        super().__init__(parent)
        self.prompt_presets = prompt_presets or []
        self.setWindowTitle("詳細設定")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)

        # カスタムプロンプト設定
        prompt_group = QGroupBox("カスタムプロンプト")
        prompt_layout = QVBoxLayout()

        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("プロンプト候補:"))
        self.prompt_preset_combo = QComboBox()
        self.prompt_preset_combo.addItem("手動入力", None)
        for preset in self.prompt_presets:
            self.prompt_preset_combo.addItem(preset["label"], preset)
        self.prompt_preset_combo.currentIndexChanged.connect(self.apply_prompt_preset)
        preset_layout.addWidget(self.prompt_preset_combo, 1)
        prompt_layout.addLayout(preset_layout)

        self.custom_prompt = QTextEdit()
        self.custom_prompt.setPlaceholderText("カスタムプロンプトを入力（空白ならデフォルト使用）")
        self.custom_prompt.setMinimumHeight(100)
        self.clean_response = QCheckBox("前置きや余計な表現を削除")
        self.clean_response.setChecked(True)
        prompt_layout.addWidget(self.custom_prompt)
        prompt_layout.addWidget(self.clean_response)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        # 詳細度設定
        detail_group = QGroupBox("説明の詳細度（カスタムプロンプト使用時は無効）")
        detail_layout = QHBoxLayout()
        self.detail_level = "standard"
        self.brief_radio = QRadioButton("簡潔（1文）")
        self.standard_radio = QRadioButton("標準（2-3文）")
        self.detailed_radio = QRadioButton("詳細（4-5文）")
        self.standard_radio.setChecked(True)
        self.brief_radio.toggled.connect(self._update_detail_level)
        self.standard_radio.toggled.connect(self._update_detail_level)
        self.detailed_radio.toggled.connect(self._update_detail_level)
        detail_layout.addWidget(self.brief_radio)
        detail_layout.addWidget(self.standard_radio)
        detail_layout.addWidget(self.detailed_radio)
        detail_group.setLayout(detail_layout)
        layout.addWidget(detail_group)

        file_button_layout = QHBoxLayout()
        self.load_config_button = QPushButton("設定ファイルを読み込み")
        self.save_config_button = QPushButton("設定ファイルの保存先選択")
        self.load_config_button.clicked.connect(self.load_config)
        self.save_config_button.clicked.connect(self.save_config)
        file_button_layout.addWidget(self.load_config_button)
        file_button_layout.addWidget(self.save_config_button)
        layout.addLayout(file_button_layout)

        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("キャンセル")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _update_detail_level(self):
        """
        ラジオボタンの選択状態を detail_level に反映
        """
        if self.brief_radio.isChecked():
            self.detail_level = "brief"
        elif self.standard_radio.isChecked():
            self.detail_level = "standard"
        else:
            self.detail_level = "detailed"

    def _set_detail_level(self, detail_level):
        """
        detail_level を検証してラジオボタンへ反映する
        """
        if detail_level not in VALID_DETAIL_LEVELS:
            raise ValueError(f"未知の detail_level です: {detail_level}")
        self.detail_level = detail_level
        if detail_level == "brief":
            self.brief_radio.setChecked(True)
        elif detail_level == "standard":
            self.standard_radio.setChecked(True)
        else:
            self.detailed_radio.setChecked(True)

    def apply_prompt_preset(self, index):
        """
        選択したプロンプト候補をフォームへ反映する
        """
        preset = self.prompt_preset_combo.itemData(index)
        if preset is None:
            return
        self.custom_prompt.setPlainText(preset["prompt"])
        self.clean_response.setChecked(preset["clean_response"])
        self._set_detail_level(preset["detail_level"])

    def _matching_prompt_preset_id(self, prompt):
        """
        現在の本文に一致するプリセットIDを返す
        """
        for preset in self.prompt_presets:
            if prompt == preset["prompt"].strip():
                return preset["id"]
        return ""

    def _select_prompt_preset(self, preset_id):
        """
        指定IDのプロンプト候補を選択状態にする
        """
        if not preset_id:
            self.prompt_preset_combo.setCurrentIndex(0)
            return
        for index in range(1, self.prompt_preset_combo.count()):
            preset = self.prompt_preset_combo.itemData(index)
            if preset and preset["id"] == preset_id:
                self.prompt_preset_combo.setCurrentIndex(index)
                return
        raise ValueError(f"未知のプロンプト候補IDです: {preset_id}")

    def _sync_prompt_preset_combo(self):
        """
        現在の本文と一致する候補をコンボボックスへ反映する
        """
        preset_id = self._matching_prompt_preset_id(self.custom_prompt.toPlainText().strip())
        self._select_prompt_preset(preset_id)

    def load_config(self):
        """
        設定ファイル（JSON）から読み込んでフォームに適用
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "設定ファイルを選択", "", "JSON Files (*.json)")
        if file_path:
            config = load_app_config(file_path)
            if "custom_prompt" in config:
                self.custom_prompt.setPlainText(config["custom_prompt"])
            if "clean_response" in config:
                self.clean_response.setChecked(config["clean_response"])
            if "detail_level" in config:
                self._set_detail_level(config["detail_level"])
            prompt_preset_id = config.get("prompt_preset_id", "")
            if prompt_preset_id:
                self._select_prompt_preset(prompt_preset_id)
            else:
                self._sync_prompt_preset_combo()
            QMessageBox.information(self, "ロード完了", "設定ファイルをロードしました。")

    def save_config(self):
        """
        現在の設定をJSONファイルとして保存
        """
        file_path, _ = QFileDialog.getSaveFileName(self, "設定ファイルの保存先を選択", "", "JSON Files (*.json)")
        if file_path:
            config = self.get_settings()
            save_app_config(config, file_path)
            QMessageBox.information(self, "保存完了", "設定を保存しました。")

    def get_settings(self):
        """
        現在のフォーム入力から設定を辞書形式で取得
        """
        prompt = self.custom_prompt.toPlainText().strip()
        prompt_preset_id = self._matching_prompt_preset_id(prompt)
        settings = {
            "custom_prompt": prompt,
            "clean_response": self.clean_response.isChecked(),
            "detail_level": self.detail_level,
            "prompt_preset_id": prompt_preset_id
        }
        for preset in self.prompt_presets:
            if preset["id"] == prompt_preset_id:
                settings["use_japanese"] = preset["use_japanese"]
                break
        if not prompt_preset_id:
            del settings["prompt_preset_id"]
        return settings

    def set_settings(self, settings):
        """
        外部から与えられた設定をフォームに反映
        """
        if "custom_prompt" in settings:
            self.custom_prompt.setPlainText(settings["custom_prompt"])
        if "clean_response" in settings:
            self.clean_response.setChecked(settings["clean_response"])
        if "detail_level" in settings:
            self._set_detail_level(settings["detail_level"])
        prompt_preset_id = settings.get("prompt_preset_id", "")
        if prompt_preset_id:
            self._select_prompt_preset(prompt_preset_id)
        else:
            self._sync_prompt_preset_combo()

# ----------------- AIチャットタブ -----------------
class ChatTab(QWidget):
    """
    AIチャット用タブ
    """
    def __init__(self, parent=None, get_api_provider_func=None, get_api_url_func=None, get_model_func=None):
        super().__init__(parent)
        self.get_api_provider_func = get_api_provider_func
        self.get_api_url_func = get_api_url_func
        self.get_model_func = get_model_func
        self.waiting_message_displayed = False

        layout = QVBoxLayout(self)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        # Shift+Enter で送信できるようにイベントをフック
        self.input_edit.installEventFilter(self)
        self.send_button = QPushButton("送信")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)

        self.setLayout(layout)

    def eventFilter(self, source, event):
        """
        Shift+Enter でメッセージ送信
        """
        if source == self.input_edit and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return and (event.modifiers() & Qt.ShiftModifier):
                self.send_message()
                return True
        return super().eventFilter(source, event)

    def send_message(self):
        """
        入力メッセージを送信し、AIの応答を受け取る
        """
        message = self.input_edit.text().strip()
        if not message:
            return
        self.append_chat("ユーザー", message)
        self.input_edit.clear()

        api_provider = self.get_api_provider_func() if self.get_api_provider_func else API_PROVIDER_OLLAMA
        api_provider = normalize_api_provider(api_provider)
        api_url = self.get_api_url_func() if self.get_api_url_func else provider_default_api_url(api_provider)
        model_name = self.get_model_func() if self.get_model_func else "chat"

        try:
            payload = build_ai_request_payload(
                api_provider=api_provider,
                model=model_name,
                prompt=message,
            )
        except Exception as e:
            self.append_chat("エラー", str(e))
            return

        self.send_button.setEnabled(False)
        self.append_chat("システム", "応答を待っています...")
        self.waiting_message_displayed = True

        self.chat_worker = ChatWorker(api_provider, api_url, payload)
        self.chat_worker.result_ready.connect(self.handle_chat_result)
        self.chat_worker.error_occurred.connect(self.handle_chat_error)
        self.chat_worker.finished.connect(lambda: self.send_button.setEnabled(True))
        self.chat_worker.start()

    def handle_chat_result(self, reply):
        """
        AIからの応答を受け取り、チャット欄に表示
        """
        self.remove_waiting_message()
        self.append_chat("AI", reply)

    def handle_chat_error(self, error):
        """
        エラー時にメッセージを表示
        """
        self.remove_waiting_message()
        self.append_chat("エラー", f"サーバへの接続またはAI処理に失敗しました: {error}")

    def remove_waiting_message(self):
        """
        "応答を待っています..." の行を削除する
        """
        if self.waiting_message_displayed:
            text = self.chat_display.toPlainText()
            lines = text.split("\n")
            if lines and lines[-1].startswith("システム: 応答を待っています"):
                lines.pop()
            self.chat_display.setPlainText("\n".join(lines))
            self.waiting_message_displayed = False

    def append_chat(self, sender, text):
        """
        チャット欄にメッセージを追加し、末尾までスクロール
        """
        self.chat_display.append(f"{sender}: {text}")
        self.chat_display.moveCursor(QTextCursor.End)

# ----------------- 画像タグ付けタブ -----------------
class ImageTaggingTab(QWidget):
    """
    画像タグ付け機能を提供するタブ
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        loaded_config = load_app_config()
        self.model_presets = load_model_presets()
        self.prompt_presets = load_prompt_presets()
        default_model = self.model_presets[0]["model"]
        # 設定を辞書にまとめて保持
        api_provider = normalize_api_provider(loaded_config.get("api_provider") or DEFAULT_API_PROVIDER)
        self.settings = {
            "model": loaded_config.get("model", default_model),
            "api_provider": api_provider,
            "api_url": loaded_config.get("api_url", provider_default_api_url(api_provider)),
            "use_japanese": loaded_config.get("use_japanese", False),
            "custom_prompt": loaded_config.get("custom_prompt", ""),
            "clean_response": loaded_config.get("clean_response", True),
            "detail_level": loaded_config.get("detail_level", "standard"),
            "prompt_preset_id": loaded_config.get("prompt_preset_id", ""),
            "clean_patterns": loaded_config.get("clean_patterns", {
                "initial": default_initial_patterns,
                "additional": default_additional_patterns
            })
        }
        self.analyzer = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        """
        タブ内のUIを初期化
        """
        main_layout = QVBoxLayout(self)

        # ツールバー
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setMovable(False)

        provider_label = QLabel("接続先:")
        toolbar.addWidget(provider_label)

        self.provider_combo = QComboBox()
        for provider_value, provider_label_text in API_PROVIDER_CHOICES:
            self.provider_combo.addItem(provider_label_text, provider_value)
        provider_index = self.provider_combo.findData(self.settings["api_provider"])
        self.provider_combo.setCurrentIndex(provider_index)
        self.provider_combo.currentIndexChanged.connect(self.on_api_provider_changed)
        toolbar.addWidget(self.provider_combo)
        toolbar.addSeparator()

        model_label = QLabel("AIモデル:")
        toolbar.addWidget(model_label)

        self.model_combo = QComboBox()
        for preset in self.model_presets:
            self.model_combo.addItem(preset["model"])
            index = self.model_combo.count() - 1
            tooltip = preset["label"]
            if "description" in preset:
                tooltip = f"{tooltip}\n{preset['description']}"
            self.model_combo.setItemData(index, tooltip, Qt.ToolTipRole)
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumWidth(150)
        self.model_combo.setCurrentText(self.settings["model"])
        toolbar.addWidget(self.model_combo)
        toolbar.addSeparator()

        url_label = QLabel("API URL:")
        toolbar.addWidget(url_label)

        self.url_edit = QLineEdit(self.settings["api_url"])
        self.url_edit.setMinimumWidth(250)
        toolbar.addWidget(self.url_edit)
        toolbar.addSeparator()

        self.japanese_check = QCheckBox("日本語で出力")
        self.japanese_check.setChecked(self.settings["use_japanese"])
        toolbar.addWidget(self.japanese_check)
        toolbar.addSeparator()

        self.settings_button = QPushButton("詳細設定")
        self.settings_button.clicked.connect(self.show_settings)
        toolbar.addWidget(self.settings_button)

        main_layout.addWidget(toolbar)

        # メインコンテンツ（スプリッターで左右分割）
        main_splitter = QSplitter(Qt.Horizontal)

        # 左ペイン（画像リスト）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        list_label = QLabel("画像リスト")
        list_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(list_label)

        self.image_list = DropListWidget()
        self.image_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.image_list.currentItemChanged.connect(self.on_image_selected)
        self.image_list.files_dropped.connect(self.add_dropped_files)
        left_layout.addWidget(self.image_list)

        button_layout = QHBoxLayout()
        self.add_folder_button = QPushButton("フォルダ追加")
        self.add_files_button = QPushButton("ファイル追加")
        self.remove_button = QPushButton("選択削除")
        self.add_folder_button.clicked.connect(self.add_folder)
        self.add_files_button.clicked.connect(self.add_files)
        self.remove_button.clicked.connect(self.remove_selected)
        button_layout.addWidget(self.add_folder_button)
        button_layout.addWidget(self.add_files_button)
        button_layout.addWidget(self.remove_button)
        left_layout.addLayout(button_layout)

        # 右ペイン（プレビュー + 結果表示）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_splitter = QSplitter(Qt.Vertical)

        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        preview_label = QLabel("画像プレビュー")
        preview_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        preview_layout.addWidget(preview_label)

        self.image_preview = ImagePreviewWidget()
        preview_layout.addWidget(self.image_preview)

        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setContentsMargins(0, 0, 0, 0)

        result_label = QLabel("分析結果")
        result_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        result_layout.addWidget(result_label)

        self.tag_display = TagDisplayWidget()
        result_layout.addWidget(self.tag_display)

        right_splitter.addWidget(preview_widget)
        right_splitter.addWidget(result_widget)
        right_splitter.setSizes([300, 300])
        right_layout.addWidget(right_splitter)

        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([300, 500])
        main_layout.addWidget(main_splitter, 1)

        # 下部操作ボタン
        bottom_layout = QHBoxLayout()

        status_layout = QVBoxLayout()
        self.status_label = QLabel("準備完了")
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        bottom_layout.addLayout(status_layout, 1)

        self.run_button = QPushButton("分析実行")
        self.run_button.setMinimumSize(120, 40)
        self.run_button.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; border-radius: 5px; }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.run_button.clicked.connect(self.run_analysis)

        self.stop_button = QPushButton("停止")
        self.stop_button.setMinimumSize(120, 40)
        self.stop_button.setStyleSheet("""
            QPushButton { background-color: #f44336; color: white; font-weight: bold; border-radius: 5px; }
            QPushButton:hover { background-color: #d32f2f; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)

        bottom_layout.addWidget(self.run_button)
        bottom_layout.addWidget(self.stop_button)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def get_api_provider(self):
        return normalize_api_provider(self.provider_combo.currentData())

    def on_api_provider_changed(self, index):
        api_provider = self.get_api_provider()
        current_url = self.url_edit.text().strip()
        if not current_url or current_url in DEFAULT_API_URLS.values():
            self.url_edit.setText(provider_default_api_url(api_provider))
        self.settings["api_provider"] = api_provider

    def show_settings(self):
        """
        詳細設定ダイアログを開き、プロンプトやクリーニング設定を編集する
        """
        dialog = SettingsDialog(self, prompt_presets=self.prompt_presets)
        dialog.set_settings({
            "custom_prompt": self.settings["custom_prompt"],
            "clean_response": self.settings["clean_response"],
            "detail_level": self.settings["detail_level"],
            "prompt_preset_id": self.settings["prompt_preset_id"]
        })
        if dialog.exec():
            new_settings = dialog.get_settings()
            self.settings["custom_prompt"] = new_settings["custom_prompt"]
            self.settings["clean_response"] = new_settings["clean_response"]
            self.settings["detail_level"] = new_settings["detail_level"]
            self.settings["prompt_preset_id"] = new_settings.get("prompt_preset_id", "")
            if "use_japanese" in new_settings:
                self.settings["use_japanese"] = new_settings["use_japanese"]
                self.japanese_check.setChecked(new_settings["use_japanese"])

    def add_folder(self):
        """
        フォルダを選択し、その配下の画像をリストに追加
        """
        folder_path = QFileDialog.getExistingDirectory(self, "フォルダを選択")
        if folder_path:
            self.add_images_from_folder(folder_path)

    def add_files(self):
        """
        複数ファイルを選択し、リストに追加
        """
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "画像ファイルを選択", "",
            "画像ファイル (*.jpg *.jpeg *.png *.gif *.bmp *.webp *.heic)"
        )
        if file_paths:
            self.add_images(file_paths)

    def add_dropped_files(self, file_paths):
        """
        ドラッグ＆ドロップされたファイル/フォルダを処理
        """
        image_paths = []
        folder_paths = []
        for path in file_paths:
            if os.path.isdir(path):
                folder_paths.append(path)
            elif os.path.isfile(path) and self._is_supported_image(path):
                image_paths.append(path)
        if image_paths:
            self.add_images(image_paths)
        for folder in folder_paths:
            self.add_images_from_folder(folder)

    def add_images_from_folder(self, folder_path):
        """
        指定フォルダ下のすべての画像をリストに追加
        """
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.avif'}
        image_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file_path)[1].lower() in supported_formats:
                    image_paths.append(file_path)
        if image_paths:
            self.add_images(image_paths)

    def add_images(self, image_paths):
        """
        ファイルパスをリストに追加（既存チェック込み）
        """
        existing = {str(self.image_list.item(i).image_path) for i in range(self.image_list.count())}
        added = 0
        for path in image_paths:
            if path not in existing:
                self.image_list.addItem(ImageListItem(path))
                added += 1
        if added:
            self.status_label.setText(f"{added}個の画像を追加")

    def _is_supported_image(self, file_path):
        """
        対応フォーマットか判定
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.avif'}

    def remove_selected(self):
        """
        選択された画像をリストから削除
        """
        selected = self.image_list.selectedItems()
        if not selected:
            return
        for item in selected:
            row = self.image_list.row(item)
            self.image_list.takeItem(row)
        self.status_label.setText(f"{len(selected)}個の画像を削除")
        if self.image_list.count():
            self.image_list.setCurrentRow(0)
        else:
            self.image_preview.set_image(None)
            self.tag_display.set_tags("")

    def on_image_selected(self, current, previous):
        """
        選択中の画像をプレビューと分析結果欄に表示
        """
        if not current:
            self.image_preview.set_image(None)
            self.tag_display.set_tags("")
            return
        image_path = current.image_path
        self.image_preview.set_image(str(image_path))

        text_path = image_path.with_suffix('.txt')
        if text_path.exists():
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.tag_display.set_tags(content)
            except Exception as e:
                logger.error(f"テキスト読み込みエラー: {str(e)}")
                self.tag_display.set_tags("")
        else:
            self.tag_display.set_tags("")

    def run_analysis(self):
        """
        画像をまとめて分析し、結果をtxtファイルとして保存
        """
        if self.image_list.count() == 0:
            QMessageBox.warning(self, "警告", "分析する画像がありません。")
            return

        selected = self.image_list.selectedItems()
        if selected and len(selected) < self.image_list.count():
            reply = QMessageBox.question(
                self, "確認",
                f"選択された{len(selected)}個のみ分析しますか？\n「No」で全画像分析",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Yes:
                image_paths = [str(item.image_path) for item in selected]
            else:
                image_paths = [str(self.image_list.item(i).image_path) for i in range(self.image_list.count())]
        else:
            image_paths = [str(self.image_list.item(i).image_path) for i in range(self.image_list.count())]

        # 設定を更新
        self.settings["model"] = self.model_combo.currentText()
        self.settings["api_provider"] = self.get_api_provider()
        self.settings["api_url"] = self.url_edit.text()
        self.settings["use_japanese"] = self.japanese_check.isChecked()

        # Ollama利用時のみアプリ側からローカルサーバ起動を試みる。
        main_window = self.window()
        if (
            self.settings["api_provider"] == API_PROVIDER_OLLAMA
            and hasattr(main_window, "ai_process")
            and main_window.ai_process is None
        ):
            main_window.start_ai_server()

        # アナライザ生成
        self.analyzer = ImageAnalyzer(
            model=self.settings["model"],
            use_japanese=self.settings["use_japanese"],
            detail_level=self.settings["detail_level"],
            custom_prompt=self.settings["custom_prompt"] if self.settings["custom_prompt"] else None,
            clean_custom_response=self.settings["clean_response"],
            api_provider=self.settings["api_provider"],
            api_url=self.settings["api_url"],
            clean_patterns=self.settings["clean_patterns"]
        )

        # ワーカースレッド開始
        self.worker = AnalysisWorker(self.analyzer, image_paths)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.analysis_complete.connect(self.on_analysis_complete)
        self.worker.analysis_error.connect(self.on_analysis_error)
        self.worker.image_analyzed.connect(self.on_image_analyzed)

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("分析開始...")

        self.worker.start()

    def stop_analysis(self):
        """
        分析処理を中断
        """
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("分析停止中...")
            self.stop_button.setEnabled(False)

    def update_progress(self, value):
        """
        進捗バー更新
        """
        self.progress_bar.setValue(int(value))

    def on_analysis_complete(self, processed, errors):
        """
        分析完了後の処理
        """
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if errors:
            self.status_label.setText(f"分析完了: {processed}件処理, {errors}件エラー")
        else:
            self.status_label.setText(f"分析完了: {processed}件処理")

        current = self.image_list.currentItem()
        if current:
            self.on_image_selected(current, None)

    def on_analysis_error(self, error_message):
        """
        分析中に発生したエラーを表示
        """
        QMessageBox.critical(self, "エラー", f"分析エラー:\n{error_message}")
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("エラー発生")

    def on_image_analyzed(self, image_path, result):
        """
        各画像の分析結果をリアルタイムで反映
        """
        current = self.image_list.currentItem()
        if current and str(current.image_path) == image_path:
            self.tag_display.set_tags(result)

# ----------------- ファイル移動タブ -----------------
class FileMoveTab(QWidget):
    """
    検索キーワードでtxtを絞り込み、該当する画像をコピーまたは移動するタブ
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.search_worker = None
        self.operation_worker = None
        self.matched_files = []
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 検索条件
        search_group = QGroupBox("検索条件")
        search_layout = QVBoxLayout(search_group)

        folder_layout = QGridLayout()
        folder_layout.addWidget(QLabel("ソースフォルダ:"), 0, 0)
        self.source_edit = QLineEdit()
        self.source_edit.setReadOnly(True)
        folder_layout.addWidget(self.source_edit, 0, 1)
        self.source_button = QPushButton("参照")
        self.source_button.clicked.connect(self.browse_source)
        folder_layout.addWidget(self.source_button, 0, 2)

        folder_layout.addWidget(QLabel("宛先フォルダ:"), 1, 0)
        self.target_edit = QLineEdit()
        self.target_edit.setReadOnly(True)
        folder_layout.addWidget(self.target_edit, 1, 1)
        self.target_button = QPushButton("参照")
        self.target_button.clicked.connect(self.browse_target)
        folder_layout.addWidget(self.target_button, 1, 2)

        search_layout.addLayout(folder_layout)

        search_layout.addWidget(QLabel("検索キーワード（カンマ区切り）:"))
        self.keyword_edit = QLineEdit()
        search_layout.addWidget(self.keyword_edit)

        option_layout = QHBoxLayout()
        self.search_mode_group = QGroupBox("検索モード")
        mode_layout = QHBoxLayout(self.search_mode_group)
        self.or_radio = QRadioButton("OR検索")
        self.and_radio = QRadioButton("AND検索")
        self.or_radio.setChecked(True)
        mode_layout.addWidget(self.or_radio)
        mode_layout.addWidget(self.and_radio)
        self.regex_check = QCheckBox("正規表現を使用")
        option_layout.addWidget(self.search_mode_group)
        option_layout.addWidget(self.regex_check)
        option_layout.addStretch()
        search_layout.addLayout(option_layout)

        main_layout.addWidget(search_group)

        # 検索結果
        result_group = QGroupBox("検索結果")
        result_layout = QVBoxLayout(result_group)

        self.result_list = QListWidget()
        self.result_list.setIconSize(QSize(100, 100))
        self.result_list.setViewMode(QListWidget.IconMode)
        self.result_list.setResizeMode(QListWidget.Adjust)
        self.result_list.setSpacing(10)
        self.result_list.setSelectionMode(QListWidget.ExtendedSelection)
        result_layout.addWidget(self.result_list)

        selection_layout = QHBoxLayout()
        self.select_all_button = QPushButton("すべて選択")
        self.select_all_button.clicked.connect(self.select_all)
        self.select_none_button = QPushButton("選択解除")
        self.select_none_button.clicked.connect(self.select_none)
        self.selection_label = QLabel("0 / 0 個選択中")
        selection_layout.addWidget(self.select_all_button)
        selection_layout.addWidget(self.select_none_button)
        selection_layout.addStretch()
        selection_layout.addWidget(self.selection_label)
        result_layout.addLayout(selection_layout)

        main_layout.addWidget(result_group, 1)

        # 下部操作ボタン
        bottom_layout = QHBoxLayout()

        status_layout = QVBoxLayout()
        self.status_label = QLabel("準備完了")
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        bottom_layout.addLayout(status_layout, 1)

        self.search_button = QPushButton("検索")
        self.search_button.setMinimumSize(100, 40)
        self.search_button.setStyleSheet("""
            QPushButton { background-color: #2196F3; color: white; font-weight: bold; border-radius: 5px; }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.search_button.clicked.connect(self.search_files)

        self.copy_button = QPushButton("コピー")
        self.copy_button.setMinimumSize(100, 40)
        self.copy_button.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; border-radius: 5px; }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.copy_button.clicked.connect(lambda: self.process_files("copy"))
        self.copy_button.setEnabled(False)

        self.move_button = QPushButton("移動")
        self.move_button.setMinimumSize(100, 40)
        self.move_button.setStyleSheet("""
            QPushButton { background-color: #FF9800; color: white; font-weight: bold; border-radius: 5px; }
            QPushButton:hover { background-color: #F57C00; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.move_button.clicked.connect(lambda: self.process_files("move"))
        self.move_button.setEnabled(False)

        bottom_layout.addWidget(self.search_button)
        bottom_layout.addWidget(self.copy_button)
        bottom_layout.addWidget(self.move_button)

        main_layout.addLayout(bottom_layout)
        self.result_list.itemSelectionChanged.connect(self.update_selection_count)

        self.setLayout(main_layout)

    def browse_source(self):
        """
        ソースフォルダの参照ダイアログを開き、選択されたパスを表示
        """
        folder = QFileDialog.getExistingDirectory(self, "ソースフォルダを選択")
        if folder:
            self.source_edit.setText(folder)

    def browse_target(self):
        """
        宛先フォルダの参照ダイアログを開き、選択されたパスを表示
        """
        folder = QFileDialog.getExistingDirectory(self, "宛先フォルダを選択")
        if folder:
            self.target_edit.setText(folder)

    def search_files(self):
        """
        指定キーワードでtxtファイルを検索
        """
        source_dir = self.source_edit.text()
        if not source_dir:
            QMessageBox.warning(self, "警告", "ソースフォルダを選択してください。")
            return

        keywords = self.keyword_edit.text().strip()
        if not keywords:
            QMessageBox.warning(self, "警告", "検索キーワードを入力してください。")
            return

        search_terms = [term.strip() for term in keywords.split(',') if term.strip()]
        search_mode = "AND" if self.and_radio.isChecked() else "OR"
        use_regex = self.regex_check.isChecked()

        self.search_button.setEnabled(False)
        self.copy_button.setEnabled(False)
        self.move_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("検索中...")

        self.result_list.clear()
        self.matched_files = []
        self.update_selection_count()

        self.search_worker = FileSearchWorker(source_dir, search_terms, search_mode, use_regex)
        self.search_worker.progress_updated.connect(self.update_progress)
        self.search_worker.search_complete.connect(self.on_search_complete)
        self.search_worker.search_error.connect(self.on_search_error)
        self.search_worker.start()

    def on_search_complete(self, matched_files):
        """
        検索完了時に呼ばれるコールバック
        """
        self.search_button.setEnabled(True)
        self.matched_files = matched_files
        if not matched_files:
            self.status_label.setText("検索結果: 一致するファイルが見つかりませんでした")
            return
        for file_path in matched_files:
            self.result_list.addItem(ImageListItem(file_path))
        self.status_label.setText(f"検索結果: {len(matched_files)}個のファイルが見つかりました")
        self.copy_button.setEnabled(len(self.result_list.selectedItems()) > 0)
        self.move_button.setEnabled(len(self.result_list.selectedItems()) > 0)
        self.update_selection_count()

    def on_search_error(self, error_message):
        """
        検索中に発生したエラーを表示
        """
        QMessageBox.critical(self, "エラー", f"検索エラー:\n{error_message}")
        self.search_button.setEnabled(True)
        self.status_label.setText("エラー発生")

    def select_all(self):
        """
        リスト内のすべてのアイテムを選択
        """
        self.result_list.selectAll()

    def select_none(self):
        """
        リストの選択を解除
        """
        self.result_list.clearSelection()

    def update_selection_count(self):
        """
        リスト内アイテムの選択数を表示
        """
        count = self.result_list.count()
        selected = len(self.result_list.selectedItems())
        self.selection_label.setText(f"{selected} / {count} 個選択中")
        self.copy_button.setEnabled(selected > 0)
        self.move_button.setEnabled(selected > 0)

    def process_files(self, operation):
        """
        選択したファイルをコピーまたは移動し、.txtファイルも同時処理
        """
        target_dir = self.target_edit.text()
        if not target_dir:
            QMessageBox.warning(self, "警告", "宛先フォルダを選択してください。")
            return

        selected = self.result_list.selectedItems()
        if not selected:
            QMessageBox.warning(self, "警告", "ファイルを選択してください。")
            return

        op_text = "コピー" if operation == "copy" else "移動"
        reply = QMessageBox.question(
            self, "確認",
            f"選択された{len(selected)}個のファイルを{op_text}しますか？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        file_paths = [item.image_path for item in selected]

        self.search_button.setEnabled(False)
        self.copy_button.setEnabled(False)
        self.move_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"{op_text}中...")

        self.operation_worker = FileOperationWorker(file_paths, target_dir, operation)
        self.operation_worker.progress_updated.connect(self.update_progress)
        self.operation_worker.operation_complete.connect(
            lambda count: self.on_operation_complete(count, operation)
        )
        self.operation_worker.operation_error.connect(self.on_operation_error)
        self.operation_worker.start()

    def on_operation_complete(self, count, operation):
        """
        ファイル操作完了時に呼ばれるコールバック
        """
        self.search_button.setEnabled(True)
        self.copy_button.setEnabled(True)
        self.move_button.setEnabled(True)
        op_text = "コピー" if operation == "copy" else "移動"
        self.status_label.setText(f"{op_text}完了: {count}個のファイルを処理")

        # 移動の場合、成功したファイルをリストから削除
        if operation == "move" and count > 0:
            indices = sorted([self.result_list.row(item) for item in self.result_list.selectedItems()], reverse=True)
            for i in indices:
                self.result_list.takeItem(i)
            self.update_selection_count()

    def on_operation_error(self, error_message):
        """
        ファイル操作中に発生したエラーを表示
        """
        QMessageBox.critical(self, "エラー", f"ファイル操作エラー:\n{error_message}")
        self.search_button.setEnabled(True)
        self.copy_button.setEnabled(True)
        self.move_button.setEnabled(True)
        self.status_label.setText("エラー発生")

    def update_progress(self, value):
        """
        進捗バー更新
        """
        self.progress_bar.setValue(int(value))

# ----------------- 分析結果編集タブ -----------------
class ResultEditTab(QWidget):
    """
    分析結果(txt)の一括編集機能を提供するタブ
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_folder = None
        self.image_files = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # フォルダ選択
        folder_layout = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_button = QPushButton("フォルダ選択")
        self.folder_button.clicked.connect(self.select_folder)

        folder_layout.addWidget(QLabel("編集対象フォルダ:"))
        folder_layout.addWidget(self.folder_edit)
        folder_layout.addWidget(self.folder_button)
        layout.addLayout(folder_layout)

        # スプリッターで左右分割
        main_splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.file_list = QListWidget()
        self.file_list.setIconSize(QSize(100, 100))
        self.file_list.setViewMode(QListWidget.IconMode)
        self.file_list.setResizeMode(QListWidget.Adjust)
        self.file_list.setSpacing(10)
        self.file_list.setSelectionMode(QListWidget.SingleSelection)
        self.file_list.currentItemChanged.connect(self.on_item_selected)
        left_layout.addWidget(QLabel("画像ファイル一覧"))
        left_layout.addWidget(self.file_list)

        main_splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.preview = ImagePreviewWidget()
        right_layout.addWidget(self.preview)

        self.text_edit = QTextEdit()
        right_layout.addWidget(self.text_edit)

        save_layout = QHBoxLayout()
        self.save_button = QPushButton("変更を保存")
        self.save_button.clicked.connect(self.save_current_text)
        save_layout.addStretch()
        save_layout.addWidget(self.save_button)
        right_layout.addLayout(save_layout)

        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([300, 500])
        layout.addWidget(main_splitter)

        op_group = QGroupBox("一括操作")
        op_layout = QGridLayout(op_group)

        self.delete_edit = QLineEdit()
        self.prefix_edit = QLineEdit()
        self.suffix_edit = QLineEdit()
        self.find_edit = QLineEdit()
        self.replace_edit = QLineEdit()

        op_layout.addWidget(QLabel("削除する文字列:"), 0, 0)
        op_layout.addWidget(self.delete_edit, 0, 1)

        op_layout.addWidget(QLabel("先頭に追加:"), 1, 0)
        op_layout.addWidget(self.prefix_edit, 1, 1)

        op_layout.addWidget(QLabel("末尾に追加:"), 2, 0)
        op_layout.addWidget(self.suffix_edit, 2, 1)

        op_layout.addWidget(QLabel("置換(検索):"), 3, 0)
        op_layout.addWidget(self.find_edit, 3, 1)
        op_layout.addWidget(QLabel("置換(新文字列):"), 4, 0)
        op_layout.addWidget(self.replace_edit, 4, 1)

        self.run_button = QPushButton("一括実行")
        self.run_button.clicked.connect(self.run_bulk_edit)
        op_layout.addWidget(self.run_button, 5, 0, 1, 2)

        layout.addWidget(op_group)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit)

        self.setLayout(layout)

    def select_folder(self):
        """
        編集対象フォルダを選択し、画像と対応するtxtを一覧に読み込む
        """
        folder = QFileDialog.getExistingDirectory(self, "編集対象フォルダを選択")
        if folder:
            self.current_folder = Path(folder)
            self.folder_edit.setText(folder)
            self.load_files(folder)

    def load_files(self, folder):
        """
        対応する画像ファイル（とtxt）の一覧を表示リストに追加
        """
        self.file_list.clear()
        self.image_files = []
        if not Path(folder).exists():
            return
        supported = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.heic', '.avif'}
        for f in Path(folder).glob("*.*"):
            if f.suffix.lower() in supported:
                item = ImageListItem(str(f))
                self.file_list.addItem(item)
                self.image_files.append(f)

    def on_item_selected(self, current, previous):
        """
        リスト選択項目が変わったらプレビューとテキストを更新
        """
        if not current:
            self.preview.set_image(None)
            self.text_edit.setText("")
            return
        image_path = current.image_path
        self.preview.set_image(str(image_path))

        txt_path = image_path.with_suffix('.txt')
        if txt_path.exists():
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    self.text_edit.setText(f.read())
            except Exception as e:
                logger.error(f"テキスト読み込みエラー: {str(e)}")
                self.text_edit.setText("")
        else:
            self.text_edit.setText("")

    def save_current_text(self):
        """
        現在のテキストエディタ内容を.txtに保存
        """
        current = self.file_list.currentItem()
        if not current:
            return
        image_path = current.image_path
        txt_path = image_path.with_suffix('.txt')
        content = self.text_edit.toPlainText()
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.log_edit.append(f"{txt_path.name} に変更を保存しました。")
        except Exception as e:
            self.log_edit.append(f"保存エラー: {str(e)}")

    def run_bulk_edit(self):
        """
        指定フォルダ内の全.txtに対して削除/置換などを一括実行
        """
        if not self.current_folder or not self.current_folder.exists():
            QMessageBox.warning(self, "警告", "編集対象フォルダを選択してください。")
            return

        txt_files = []
        supported = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.heic', '.avif'}
        for f in self.current_folder.glob("*.*"):
            if f.suffix.lower() in supported:
                txt_path = f.with_suffix('.txt')
                if txt_path.exists():
                    txt_files.append(txt_path)

        if not txt_files:
            QMessageBox.warning(self, "警告", "対応するtxtファイルが見つかりません。")
            return

        self.log_edit.append(f"{len(txt_files)}個のファイルを処理開始...")

        delete_str = self.delete_edit.text()
        prefix = self.prefix_edit.text()
        suffix = self.suffix_edit.text()
        find_str = self.find_edit.text()
        replace_str = self.replace_edit.text()

        for txt_file in txt_files:
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # 削除処理
                if delete_str:
                    content = content.replace(delete_str, "")
                # 先頭追加
                if prefix:
                    content = prefix + content
                # 末尾追加
                if suffix:
                    content = content + suffix
                # 文字列置換
                if find_str:
                    content = content.replace(find_str, replace_str)

                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(content)
                self.log_edit.append(f"{txt_file.name}: 処理完了")

            except Exception as e:
                self.log_edit.append(f"{txt_file.name}: エラー {str(e)}")

        self.log_edit.append("一括編集完了。")

# ----------------- テキスト変換タブ -----------------
class TextTransformTab(QWidget):
    """
    既存の画像同名txtを翻訳・タグ化・自然文プロンプト化するタブ。
    """
    def __init__(self, model_presets=None, parent=None):
        super().__init__(parent)
        self.model_presets = model_presets if model_presets is not None else load_model_presets()
        self.transform_presets = []
        self.preset_load_error = None
        try:
            self.transform_presets = load_transform_presets()
        except Exception as e:
            self.preset_load_error = str(e)

        loaded_config = load_app_config()
        self.initial_api_provider = normalize_api_provider(loaded_config.get("api_provider") or DEFAULT_API_PROVIDER)
        self.initial_api_url = loaded_config.get("api_url", provider_default_api_url(self.initial_api_provider))
        self.preview_worker = None
        self.transform_worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        folder_group = QGroupBox("ソース")
        folder_layout = QGridLayout(folder_group)
        folder_layout.addWidget(QLabel("ソースフォルダ:"), 0, 0)
        self.source_edit = QLineEdit()
        self.source_edit.setReadOnly(True)
        folder_layout.addWidget(self.source_edit, 0, 1)
        self.source_button = QPushButton("参照")
        self.source_button.clicked.connect(self.browse_source)
        folder_layout.addWidget(self.source_button, 0, 2)

        self.include_subfolders_check = QCheckBox("サブフォルダを含める")
        self.include_subfolders_check.stateChanged.connect(self.reload_candidates_if_ready)
        folder_layout.addWidget(self.include_subfolders_check, 1, 1)
        self.target_ext_label = QLabel("対象: 画像ファイル + 同名 image.txt")
        folder_layout.addWidget(self.target_ext_label, 1, 2)
        main_layout.addWidget(folder_group)

        settings_group = QGroupBox("変換設定")
        settings_layout = QGridLayout(settings_group)

        settings_layout.addWidget(QLabel("変換モード:"), 0, 0)
        self.mode_combo = QComboBox()
        for preset in self.transform_presets:
            self.mode_combo.addItem(preset["label"], preset)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        settings_layout.addWidget(self.mode_combo, 0, 1)

        settings_layout.addWidget(QLabel("接続先:"), 0, 2)
        self.provider_combo = QComboBox()
        for provider_value, provider_label_text in API_PROVIDER_CHOICES:
            self.provider_combo.addItem(provider_label_text, provider_value)
        provider_index = self.provider_combo.findData(self.initial_api_provider)
        self.provider_combo.setCurrentIndex(provider_index)
        self.provider_combo.currentIndexChanged.connect(self.on_api_provider_changed)
        settings_layout.addWidget(self.provider_combo, 0, 3)

        settings_layout.addWidget(QLabel("使用モデル:"), 1, 0)
        self.model_combo = QComboBox()
        for preset in self.model_presets:
            self.model_combo.addItem(preset["model"])
            index = self.model_combo.count() - 1
            tooltip = preset["label"]
            if "description" in preset:
                tooltip = f"{tooltip}\n{preset['description']}"
            self.model_combo.setItemData(index, tooltip, Qt.ToolTipRole)
        self.model_combo.setEditable(True)
        settings_layout.addWidget(self.model_combo, 1, 1)

        settings_layout.addWidget(QLabel("API URL:"), 1, 2)
        self.api_url_edit = QLineEdit(self.initial_api_url)
        settings_layout.addWidget(self.api_url_edit, 1, 3)

        settings_layout.addWidget(QLabel("保存先サフィックス:"), 2, 0)
        self.suffix_label = QLabel("-")
        settings_layout.addWidget(self.suffix_label, 2, 1)

        self.overwrite_check = QCheckBox("既存サイドカーの上書きを許可")
        self.overwrite_check.setChecked(False)
        settings_layout.addWidget(self.overwrite_check, 2, 3)

        main_layout.addWidget(settings_group)

        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("対象ファイル"))
        self.file_list = QListWidget()
        self.file_list.setIconSize(QSize(100, 100))
        self.file_list.setViewMode(QListWidget.IconMode)
        self.file_list.setResizeMode(QListWidget.Adjust)
        self.file_list.setSpacing(10)
        self.file_list.setSelectionMode(QListWidget.SingleSelection)
        self.file_list.currentItemChanged.connect(self.on_file_selected)
        left_layout.addWidget(self.file_list)
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        text_splitter = QSplitter(Qt.Vertical)
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.addWidget(QLabel("変換前テキスト"))
        self.input_text = QTextEdit()
        self.input_text.setReadOnly(True)
        input_layout.addWidget(self.input_text)

        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(QLabel("変換後プレビュー"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_text)

        text_splitter.addWidget(input_widget)
        text_splitter.addWidget(output_widget)
        text_splitter.setSizes([260, 260])
        right_layout.addWidget(text_splitter)
        splitter.addWidget(right_widget)
        splitter.setSizes([320, 680])
        main_layout.addWidget(splitter, 1)

        bottom_layout = QHBoxLayout()
        status_layout = QVBoxLayout()
        self.status_label = QLabel("準備完了")
        status_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        bottom_layout.addLayout(status_layout, 1)

        self.preview_button = QPushButton("選択ファイルをプレビュー変換")
        self.preview_button.clicked.connect(self.preview_transform)
        self.batch_button = QPushButton("一括変換")
        self.batch_button.clicked.connect(self.run_batch_transform)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_batch_transform)
        self.stop_button.setEnabled(False)
        bottom_layout.addWidget(self.preview_button)
        bottom_layout.addWidget(self.batch_button)
        bottom_layout.addWidget(self.stop_button)
        main_layout.addLayout(bottom_layout)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(130)
        main_layout.addWidget(self.log_edit)

        if self.preset_load_error:
            self.append_log(f"変換プリセット読み込みエラー: {self.preset_load_error}")
            self.mode_combo.setEnabled(False)
        else:
            self.on_mode_changed(self.mode_combo.currentIndex())

        self._update_action_buttons()

    def get_api_provider(self):
        return normalize_api_provider(self.provider_combo.currentData())

    def on_api_provider_changed(self, index):
        api_provider = self.get_api_provider()
        current_url = self.api_url_edit.text().strip()
        if not current_url or current_url in DEFAULT_API_URLS.values():
            self.api_url_edit.setText(provider_default_api_url(api_provider))

    def browse_source(self):
        folder = QFileDialog.getExistingDirectory(self, "ソースフォルダを選択")
        if folder:
            self.source_edit.setText(folder)
            self.load_candidates()

    def reload_candidates_if_ready(self, state=None):
        if self.source_edit.text():
            self.load_candidates()

    def load_candidates(self):
        folder = Path(self.source_edit.text())
        if not folder.exists():
            QMessageBox.warning(self, "警告", "ソースフォルダが存在しません。")
            return

        self.file_list.clear()
        self.input_text.clear()
        self.output_text.clear()
        iterator = folder.rglob("*") if self.include_subfolders_check.isChecked() else folder.glob("*")
        image_files = sorted(
            path for path in iterator
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        with_txt = 0
        without_txt = 0
        for image_path in image_files:
            if image_path.with_suffix(".txt").exists():
                self.file_list.addItem(ImageListItem(str(image_path)))
                with_txt += 1
            else:
                without_txt += 1

        self.status_label.setText(f"対象: {with_txt}件 / txtなし: {without_txt}件")
        self.append_log(f"{folder} から image.txt を持つ画像 {with_txt} 件を読み込みました。")
        if self.file_list.count():
            self.file_list.setCurrentRow(0)
        self._update_action_buttons()

    def on_mode_changed(self, index):
        preset = self.get_current_preset(show_error=False)
        if not preset:
            self.suffix_label.setText("-")
            return
        self.suffix_label.setText(preset["output_suffix"])
        self.model_combo.setCurrentText(preset["recommended_model"])
        self.refresh_existing_output_preview()

    def on_file_selected(self, current, previous):
        self.input_text.clear()
        self.output_text.clear()
        if not current:
            self._update_action_buttons()
            return

        source_path = current.image_path.with_suffix(".txt")
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                self.input_text.setPlainText(f.read())
        except Exception as e:
            self.append_log(f"{source_path.name}: 読み込みエラー {e}")
        self.refresh_existing_output_preview()
        self._update_action_buttons()

    def refresh_existing_output_preview(self):
        current = self.file_list.currentItem()
        preset = self.get_current_preset(show_error=False)
        if not current or not preset:
            return
        output_path = current.image_path.with_name(f"{current.image_path.stem}{preset['output_suffix']}")
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    self.output_text.setPlainText(f.read())
            except Exception as e:
                self.append_log(f"{output_path.name}: 読み込みエラー {e}")
        else:
            self.output_text.clear()

    def preview_transform(self):
        current = self.file_list.currentItem()
        if not current:
            QMessageBox.warning(self, "警告", "プレビュー変換するファイルを選択してください。")
            return
        preset = self.get_current_preset()
        if not preset:
            return
        model = self.model_combo.currentText().strip()
        api_provider = self.get_api_provider()
        api_url = self.api_url_edit.text().strip()
        if not model or not api_url:
            QMessageBox.warning(self, "警告", "モデル名とAPI URLを入力してください。")
            return
        text = self.input_text.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "警告", "変換元テキストが空です。")
            return

        self.preview_button.setEnabled(False)
        self.batch_button.setEnabled(False)
        self.status_label.setText("プレビュー変換中...")
        self.preview_worker = TextTransformPreviewWorker(text, preset, api_provider, api_url, model)
        self.preview_worker.result_ready.connect(self.on_preview_ready)
        self.preview_worker.error_occurred.connect(self.on_preview_error)
        self.preview_worker.finished.connect(self.on_preview_finished)
        self.preview_worker.start()

    def on_preview_ready(self, result):
        self.output_text.setPlainText(result)
        self.append_log("プレビュー変換が完了しました。")

    def on_preview_error(self, error_message):
        self.append_log(f"プレビュー変換エラー: {error_message}")
        QMessageBox.critical(self, "エラー", f"プレビュー変換エラー:\n{error_message}")

    def on_preview_finished(self):
        self.preview_worker = None
        self.status_label.setText("準備完了")
        self._update_action_buttons()

    def run_batch_transform(self):
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "警告", "一括変換する対象ファイルがありません。")
            return
        preset = self.get_current_preset()
        if not preset:
            return
        model = self.model_combo.currentText().strip()
        api_provider = self.get_api_provider()
        api_url = self.api_url_edit.text().strip()
        if not model or not api_url:
            QMessageBox.warning(self, "警告", "モデル名とAPI URLを入力してください。")
            return

        image_paths = [str(self.file_list.item(i).image_path) for i in range(self.file_list.count())]
        self.progress_bar.setValue(0)
        self.status_label.setText("一括変換中...")
        self.preview_button.setEnabled(False)
        self.batch_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.transform_worker = TextTransformWorker(
            image_paths=image_paths,
            preset=preset,
            api_provider=api_provider,
            api_url=api_url,
            model=model,
            overwrite=self.overwrite_check.isChecked(),
        )
        self.transform_worker.progress_updated.connect(self.update_progress)
        self.transform_worker.log_message.connect(self.append_log)
        self.transform_worker.item_completed.connect(self.on_item_completed)
        self.transform_worker.item_failed.connect(self.on_item_failed)
        self.transform_worker.transform_complete.connect(self.on_batch_complete)
        self.transform_worker.start()

    def stop_batch_transform(self):
        if self.transform_worker and self.transform_worker.isRunning():
            self.transform_worker.stop()
            self.stop_button.setEnabled(False)
            self.status_label.setText("停止中...")

    def on_item_completed(self, image_path, output_path):
        current = self.file_list.currentItem()
        if current and str(current.image_path) == image_path:
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    self.output_text.setPlainText(f.read())
            except Exception as e:
                self.append_log(f"{Path(output_path).name}: 読み込みエラー {e}")

    def on_item_failed(self, image_path, error_message):
        logger.warning(f"テキスト変換失敗: {image_path}: {error_message}")

    def on_batch_complete(self, success, failed):
        self.stop_button.setEnabled(False)
        self.transform_worker = None
        self.status_label.setText(f"一括変換完了: 成功 {success} 件 / 失敗・スキップ {failed} 件")
        self._update_action_buttons()

    def update_progress(self, value):
        self.progress_bar.setValue(int(value))

    def append_log(self, message):
        self.log_edit.append(message)
        self.log_edit.moveCursor(QTextCursor.End)

    def get_current_preset(self, show_error=True):
        if self.preset_load_error:
            if show_error:
                QMessageBox.critical(self, "エラー", f"変換プリセット読み込みエラー:\n{self.preset_load_error}")
            return None
        preset = self.mode_combo.currentData()
        if not preset and show_error:
            QMessageBox.warning(self, "警告", "変換モードを選択してください。")
        return preset

    def _update_action_buttons(self):
        has_preset = self.preset_load_error is None and self.mode_combo.count() > 0
        has_current = self.file_list.currentItem() is not None
        batch_running = self.transform_worker is not None and self.transform_worker.isRunning()
        preview_running = self.preview_worker is not None and self.preview_worker.isRunning()
        self.preview_button.setEnabled(has_preset and has_current and not batch_running and not preview_running)
        self.batch_button.setEnabled(has_preset and self.file_list.count() > 0 and not batch_running and not preview_running)

# ----------------- メインウィンドウ -----------------
class MainWindow(QMainWindow):
    """
    アプリケーションのメインウィンドウ
    AIサーバーの起動（停止機能は削除済み）および各タブの管理を行う
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("画像タグ付け・ファイル移動ツール")
        self.setMinimumSize(1000, 700)

        loaded_config = load_app_config()

        # ollamaプロセスを保持する変数（Noneの場合、未起動）
        self.ai_process = None

        self.theme = normalize_theme(loaded_config.get("theme") or DEFAULT_THEME)
        self.theme_action_group = None
        self.apply_theme(self.theme, save=False)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        self.tagging_tab = ImageTaggingTab()
        self.move_tab = FileMoveTab()
        self.result_edit_tab = ResultEditTab()
        self.chat_tab = ChatTab(
            get_api_provider_func=lambda: self.tagging_tab.get_api_provider(),
            get_api_url_func=lambda: self.tagging_tab.url_edit.text(),
            get_model_func=lambda: self.tagging_tab.model_combo.currentText()
        )
        self.text_transform_tab = TextTransformTab(model_presets=self.tagging_tab.model_presets)

        self.tabs.addTab(self.tagging_tab, "画像タグ付け")
        self.tabs.addTab(self.move_tab, "ファイル移動")
        self.tabs.addTab(self.result_edit_tab, "分析結果編集")
        self.tabs.addTab(self.chat_tab, "AIチャット")
        self.tabs.addTab(self.text_transform_tab, "テキスト変換 / Text Transform")
        main_layout.addWidget(self.tabs)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("準備完了")

        menubar = self.menuBar()

        # 「設定」メニュー
        settings_menu = menubar.addMenu("設定")
        deletion_action = QAction("削除パターン設定", self)
        deletion_action.triggered.connect(self.open_deletion_pattern_dialog)
        settings_menu.addAction(deletion_action)
        theme_menu = settings_menu.addMenu("テーマ")
        self.theme_action_group = QActionGroup(self)
        self.theme_action_group.setExclusive(True)
        light_theme_action = QAction("ライトモード", self)
        light_theme_action.setCheckable(True)
        light_theme_action.setData("light")
        dark_theme_action = QAction("ダークモード", self)
        dark_theme_action.setCheckable(True)
        dark_theme_action.setData("dark")
        self.theme_action_group.addAction(light_theme_action)
        self.theme_action_group.addAction(dark_theme_action)
        theme_menu.addAction(light_theme_action)
        theme_menu.addAction(dark_theme_action)
        self.theme_action_group.triggered.connect(self.on_theme_action_triggered)
        self.update_theme_actions()

        # 「AIサーバ」メニュー
        ai_menu = menubar.addMenu("AIサーバ")
        stop_ai_action = QAction("Ollamaサーバ停止", self)
        stop_ai_action.triggered.connect(self.stop_ai_server)
        ai_menu.addAction(stop_ai_action)
        ai_start_action = QAction("Ollamaサーバ起動", self)
        ai_start_action.triggered.connect(self.start_ai_server)
        ai_menu.addAction(ai_start_action)

    def open_deletion_pattern_dialog(self):
        """
        削除パターン設定ダイアログを開く
        """
        current_config = load_app_config().get("clean_patterns", {
            "initial": default_initial_patterns,
            "additional": default_additional_patterns
        })
        dialog = DeletionPatternDialog(self, current_patterns=current_config)
        if dialog.exec():
            new_patterns = dialog.get_patterns()
            cfg = load_app_config()
            cfg["clean_patterns"] = new_patterns
            save_app_config(cfg)
            self.tagging_tab.settings["clean_patterns"] = new_patterns

    
    def stop_ai_server(self):
        """ollama serve を終了"""
        if self.ai_process and self.ai_process.poll() is None:
            self.ai_process.terminate()
            self.ai_process.wait(timeout=10)
        self.ai_process = None
        self.statusBar.showMessage("Ollamaサーバを停止しました", 5000)
    
    def start_ai_server(self):
        """
        ollama serve を起動する
        Windowsの場合は新しいコンソールを開いて起動し、
        macOS/Linuxではバックグラウンドで起動。
        """
        if self.ai_process is not None:
            QMessageBox.information(self, "情報", "既にAIサーバが起動しています。")
            return
        model = self.tagging_tab.model_combo.currentText()
        try:
            current_os = platform.system()
            if current_os == "Windows":
                # Windows環境で新しいコンソールを開いて起動
                CREATE_NEW_CONSOLE = 0x00000010
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                self.ai_process = subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP
                )
            else:
                # macOS/Linuxではバックグラウンドで起動
                self.ai_process = subprocess.Popen(["ollama", "serve"])

            self.statusBar.showMessage(f"Ollamaサーバ起動中: {model}")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"Ollamaサーバ起動エラー: {str(e)}")
            self.ai_process = None

    def on_theme_action_triggered(self, action):
        self.apply_theme(action.data(), save=True)

    def update_theme_actions(self):
        if not self.theme_action_group:
            return
        for action in self.theme_action_group.actions():
            action.setChecked(action.data() == self.theme)

    def apply_theme(self, theme, save=False):
        theme = normalize_theme(theme)
        self.theme = theme
        if theme == "dark":
            self.set_dark_theme()
        else:
            self.set_light_theme()
        self.update_theme_actions()

        if save:
            cfg = load_app_config()
            cfg["theme"] = theme
            save_app_config(cfg)
            if isinstance(getattr(self, "statusBar", None), QStatusBar):
                self.statusBar.showMessage(f"テーマを{self.theme_label(theme)}に変更しました", 5000)

    def theme_label(self, theme):
        theme = normalize_theme(theme)
        return "ダークモード" if theme == "dark" else "ライトモード"

    def set_light_theme(self):
        """
        シンプルなライトテーマを適用
        """
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, QColor(225, 225, 225))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.Highlight, QColor(76, 163, 220))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        self.setPalette(palette)
        app = QApplication.instance()
        if app:
            app.setPalette(palette)
            app.setStyleSheet("""
                QTextEdit#tagDisplay {
                    background-color: #ffffff;
                    color: #111111;
                    border: 1px solid #c8c8c8;
                    border-radius: 5px;
                }
            """)

    def set_dark_theme(self):
        """
        アプリ全体にダークテーマを適用する。
        """
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(32, 34, 37))
        palette.setColor(QPalette.WindowText, QColor(232, 234, 237))
        palette.setColor(QPalette.Base, QColor(21, 23, 26))
        palette.setColor(QPalette.AlternateBase, QColor(43, 46, 51))
        palette.setColor(QPalette.ToolTipBase, QColor(45, 48, 54))
        palette.setColor(QPalette.ToolTipText, QColor(232, 234, 237))
        palette.setColor(QPalette.Text, QColor(232, 234, 237))
        palette.setColor(QPalette.Button, QColor(47, 52, 58))
        palette.setColor(QPalette.ButtonText, QColor(232, 234, 237))
        palette.setColor(QPalette.Highlight, QColor(55, 115, 220))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        palette.setColor(QPalette.PlaceholderText, QColor(140, 145, 153))
        self.setPalette(palette)

        app = QApplication.instance()
        if app:
            app.setPalette(palette)
            app.setStyleSheet("""
                QWidget {
                    background-color: #202225;
                    color: #e8eaed;
                }
                QMainWindow, QDialog {
                    background-color: #202225;
                }
                QLineEdit, QTextEdit, QListWidget, QComboBox {
                    background-color: #15171a;
                    color: #e8eaed;
                    border: 1px solid #3c4043;
                    border-radius: 4px;
                    selection-background-color: #3773dc;
                    selection-color: #ffffff;
                }
                QLineEdit, QComboBox {
                    min-height: 24px;
                    padding: 2px 6px;
                }
                QTextEdit#tagDisplay {
                    background-color: #15171a;
                    color: #e8eaed;
                    border: 1px solid #3c4043;
                    border-radius: 5px;
                }
                QListWidget::item:selected {
                    background-color: #294f8f;
                    color: #ffffff;
                }
                QGroupBox {
                    border: 1px solid #3c4043;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 12px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 8px;
                    padding: 0 4px;
                    background-color: #202225;
                }
                QToolBar {
                    background-color: #202225;
                    border-bottom: 1px solid #3c4043;
                    spacing: 6px;
                }
                QMenuBar, QMenu {
                    background-color: #202225;
                    color: #e8eaed;
                }
                QMenuBar::item:selected, QMenu::item:selected {
                    background-color: #294f8f;
                }
                QStatusBar {
                    background-color: #202225;
                    color: #e8eaed;
                    border-top: 1px solid #3c4043;
                }
                QTabWidget::pane {
                    border: 1px solid #3c4043;
                }
                QTabBar::tab {
                    background-color: #2b2e33;
                    color: #e8eaed;
                    border: 1px solid #3c4043;
                    padding: 6px 12px;
                }
                QTabBar::tab:selected {
                    background-color: #15171a;
                    border-bottom: 2px solid #66a3ff;
                }
                QPushButton {
                    background-color: #2f343a;
                    color: #e8eaed;
                    border: 1px solid #4b525b;
                    border-radius: 4px;
                    padding: 4px 10px;
                }
                QPushButton:hover {
                    background-color: #3a4149;
                }
                QPushButton:disabled {
                    background-color: #2a2d31;
                    color: #7d838c;
                    border-color: #3c4043;
                }
                QProgressBar {
                    background-color: #15171a;
                    color: #e8eaed;
                    border: 1px solid #3c4043;
                    border-radius: 4px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #3773dc;
                    border-radius: 3px;
                }
                QSplitter::handle {
                    background-color: #3c4043;
                }
            """)

    def closeEvent(self, event):
        self.stop_ai_server()
        event.accept()

def main():
    """
    アプリケーションのエントリーポイント
    """
    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
