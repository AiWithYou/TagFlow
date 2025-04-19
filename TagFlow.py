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
from pathlib import Path
from PIL import Image, ImageQt
from threading import Thread

# PySide6インポート
from PySide6.QtGui import (
    QAction, QPixmap, QIcon, QImage, QColor, QPalette,
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

def apply_clean_patterns(text, patterns):
    """
    与えられたテキストに対して、初期および追加パターンを適用してクリーニングを行う
    """
    text = text.strip()
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
        self.image_label.setStyleSheet("background-color: #2a2a2a; border-radius: 5px;")
        layout.addWidget(self.image_label)

    def set_image(self, image_path):
        """
        指定されたパスの画像をラベルに表示する
        """
        self.image_path = image_path
        if not image_path or not Path(image_path).exists():
            self.image_label.setText("画像なし")
            self.pixmap = None
            return
        try:
            self.pixmap = QPixmap(image_path)
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
        self.text_display.setStyleSheet("background-color: #ffffff; border-radius: 5px; color: black;")
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
        model="gemma3:27b",
        use_japanese=False,
        detail_level="standard",
        custom_prompt=None,
        clean_custom_response=True,
        api_url="http://localhost:11434/api/generate",
        clean_patterns=None
    ):
        """
        :param model: 使用するモデル名
        :param use_japanese: Trueの場合、日本語で説明させる
        :param detail_level: 'brief', 'standard', 'detailed' の3段階
        :param custom_prompt: カスタムプロンプト文字列
        :param clean_custom_response: Trueの場合、余計な前置きを自動的に削除
        :param api_url: APIエンドポイントのURL
        :param clean_patterns: 削除・置換パターン辞書
        """
        self.model = model
        self.use_japanese = use_japanese
        self.detail_level = detail_level
        self.custom_prompt = custom_prompt
        self.clean_custom_response = clean_custom_response
        self.api_url = api_url
        # HEIC対応のため、拡張子を追加
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.avif'}
        self.clean_patterns = clean_patterns or {
            "initial": default_initial_patterns,
            "additional": default_additional_patterns
        }

    def encode_image(self, image_path):
        """
        Pillowで画像を開き、Base64にエンコードして返す
        """
        try:
            with Image.open(image_path) as img:
                img_buffer = io.BytesIO()
                save_format = img.format if img.format else "PNG"
            if save_format.upper() == "HEIF": save_format = "PNG"
            img.save(img_buffer, format=save_format)
            return base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"画像エンコードエラー: {str(e)}")
            raise

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
            base64_image = self.encode_image(image_path)
            payload = {
                "model": self.model,
                "prompt": self.get_prompt(),
                "stream": False,
                "images": [base64_image]
            }
            response = requests.post(self.api_url, json=payload)
            if response.status_code != 200:
                logger.error(f"APIエラー: status_code={response.status_code}, text={response.text}")
                raise Exception(f"APIエラー: {response.status_code} - {response.text}")

            result = response.json()
            response_text = result.get('response', 'No analysis available')
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
    result_ready = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, api_url, payload):
        super().__init__()
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
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("詳細設定")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)

        # カスタムプロンプト設定
        prompt_group = QGroupBox("カスタムプロンプト")
        prompt_layout = QVBoxLayout()
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
                self.detail_level = config["detail_level"]
                if self.detail_level == "brief":
                    self.brief_radio.setChecked(True)
                elif self.detail_level == "standard":
                    self.standard_radio.setChecked(True)
                else:
                    self.detailed_radio.setChecked(True)
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
        return {
            "custom_prompt": self.custom_prompt.toPlainText().strip(),
            "clean_response": self.clean_response.isChecked(),
            "detail_level": self.detail_level
        }

    def set_settings(self, settings):
        """
        外部から与えられた設定をフォームに反映
        """
        if "custom_prompt" in settings:
            self.custom_prompt.setPlainText(settings["custom_prompt"])
        if "clean_response" in settings:
            self.clean_response.setChecked(settings["clean_response"])
        if "detail_level" in settings:
            self.detail_level = settings["detail_level"]
            if self.detail_level == "brief":
                self.brief_radio.setChecked(True)
            elif self.detail_level == "standard":
                self.standard_radio.setChecked(True)
            else:
                self.detailed_radio.setChecked(True)

# ----------------- AIチャットタブ -----------------
class ChatTab(QWidget):
    """
    AIチャット用タブ
    """
    def __init__(self, parent=None, get_api_url_func=None, get_model_func=None):
        super().__init__(parent)
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

        api_url = self.get_api_url_func() if self.get_api_url_func else "http://localhost:11434/api/generate"
        model_name = self.get_model_func() if self.get_model_func else "chat"

        payload = {
            "model": model_name,
            "prompt": message,
            "stream": False,
            "images": []
        }

        self.send_button.setEnabled(False)
        self.append_chat("システム", "応答を待っています...")
        self.waiting_message_displayed = True

        self.chat_worker = ChatWorker(api_url, payload)
        self.chat_worker.result_ready.connect(self.handle_chat_result)
        self.chat_worker.error_occurred.connect(self.handle_chat_error)
        self.chat_worker.finished.connect(lambda: self.send_button.setEnabled(True))
        self.chat_worker.start()

    def handle_chat_result(self, result):
        """
        AIからの応答を受け取り、チャット欄に表示
        """
        self.remove_waiting_message()
        reply = result.get("response", "No reply")
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
        # 設定を辞書にまとめて保持
        self.settings = {
            "model": loaded_config.get("model", "gemma3:27b"),
            "api_url": loaded_config.get("api_url", "http://localhost:11434/api/generate"),
            "use_japanese": loaded_config.get("use_japanese", False),
            "custom_prompt": loaded_config.get("custom_prompt", ""),
            "clean_response": loaded_config.get("clean_response", True),
            "detail_level": loaded_config.get("detail_level", "standard"),
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

        model_label = QLabel("Ollamaモデル:")
        toolbar.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["gemma3:27b", "gemma3:4b"])
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumWidth(150)
        self.model_combo.setCurrentText(self.settings["model"])
        toolbar.addWidget(self.model_combo)
        toolbar.addSeparator()

        url_label = QLabel("Ollama URL:")
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

    def show_settings(self):
        """
        詳細設定ダイアログを開き、プロンプトやクリーニング設定を編集する
        """
        dialog = SettingsDialog(self)
        dialog.set_settings({
            "custom_prompt": self.settings["custom_prompt"],
            "clean_response": self.settings["clean_response"],
            "detail_level": self.settings["detail_level"]
        })
        if dialog.exec():
            new_settings = dialog.get_settings()
            self.settings["custom_prompt"] = new_settings["custom_prompt"]
            self.settings["clean_response"] = new_settings["clean_response"]
            self.settings["detail_level"] = new_settings["detail_level"]

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
                f"選択された{len(selected)}個のみ分析しますか？\n「いいえ」で全画像分析",
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

        # メインウィンドウでAIサーバ未起動の場合は起動（機能はあるが停止は削除済み）
        main_window = self.window()
        if hasattr(main_window, "ai_process") and main_window.ai_process is None:
            main_window.start_ai_server()

        # 設定を更新
        self.settings["model"] = self.model_combo.currentText()
        self.settings["api_url"] = self.url_edit.text()
        self.settings["use_japanese"] = self.japanese_check.isChecked()

        # アナライザ生成
        self.analyzer = ImageAnalyzer(
            model=self.settings["model"],
            use_japanese=self.settings["use_japanese"],
            detail_level=self.settings["detail_level"],
            custom_prompt=self.settings["custom_prompt"] if self.settings["custom_prompt"] else None,
            clean_custom_response=self.settings["clean_response"],
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
        supported = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.heic'}
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
        supported = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.heic'}
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

        # ollamaプロセスを保持する変数（Noneの場合、未起動）
        self.ai_process = None

        self.set_light_theme()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        self.tagging_tab = ImageTaggingTab()
        self.move_tab = FileMoveTab()
        self.result_edit_tab = ResultEditTab()
        self.chat_tab = ChatTab(
            get_api_url_func=lambda: self.tagging_tab.url_edit.text(),
            get_model_func=lambda: self.tagging_tab.model_combo.currentText()
        )

        self.tabs.addTab(self.tagging_tab, "画像タグ付け")
        self.tabs.addTab(self.move_tab, "ファイル移動")
        self.tabs.addTab(self.result_edit_tab, "分析結果編集")
        self.tabs.addTab(self.chat_tab, "AIチャット")
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

        # 「AIサーバ」メニュー
        ai_menu = menubar.addMenu("AIサーバ")
        stop_ai_action = QAction("AIサーバ停止", self)
        stop_ai_action.triggered.connect(self.stop_ai_server)
        ai_menu.addAction(stop_ai_action)
        ai_start_action = QAction("AIサーバ起動", self)
        ai_start_action.triggered.connect(self.start_ai_server)
        ai_menu.addAction(ai_start_action)
        # 停止機能は削除済み

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
        self.statusBar.showMessage("AIサーバを停止しました", 5000)
    
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

            self.statusBar.showMessage(f"AIサーバ起動中: {model}")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"AIサーバ起動エラー: {str(e)}")
            self.ai_process = None

    def set_light_theme(self):
        """
        シンプルなライトテーマを適用（任意）
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
