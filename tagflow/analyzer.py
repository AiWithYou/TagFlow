import base64
import io
import logging

import requests
from PIL import Image

from .utils import (
    default_initial_patterns,
    default_additional_patterns,
    apply_clean_patterns,
)

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """Ollama API を用いた画像分析クラス"""

    def __init__(
        self,
        model="gemma3:27b",
        use_japanese=False,
        detail_level="standard",
        custom_prompt=None,
        clean_custom_response=True,
        api_url="http://localhost:11434/api/generate",
        clean_patterns=None,
    ):
        self.model = model
        self.use_japanese = use_japanese
        self.detail_level = detail_level
        self.custom_prompt = custom_prompt
        self.clean_custom_response = clean_custom_response
        self.api_url = api_url
        self.supported_formats = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".heic",
            ".avif",
        }
        self.clean_patterns = clean_patterns or {
            "initial": default_initial_patterns,
            "additional": default_additional_patterns,
        }

    def encode_image(self, image_path):
        """Pillowで画像を開きBase64にエンコード"""
        try:
            with Image.open(image_path) as img:
                img_buffer = io.BytesIO()
                save_format = img.format if img.format else "PNG"
            if save_format.upper() == "HEIF":
                save_format = "PNG"
            img.save(img_buffer, format=save_format)
            return base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"画像エンコードエラー: {e}")
            raise

    def get_prompt(self):
        """プロンプトを生成"""
        if self.custom_prompt:
            prompt = self.custom_prompt.strip()
            if self.clean_custom_response:
                if self.use_japanese:
                    prompt += " 主要な要素や行動に焦点を当て、余計な前置きは不要です。"
                else:
                    prompt += " Focus on key elements and actions, and omit unnecessary introductory phrases."
            return prompt

        if self.use_japanese:
            if self.detail_level == "brief":
                return "この画像を1文で簡潔に説明してください。余計な前置きは不要です。"
            if self.detail_level == "standard":
                return "この画像を2〜3文で説明してください。主要な要素や行動に焦点を当て、余計な前置きは不要です。"
            return "この画像を4〜5文で詳しく説明してください。視覚的な要素、行動、雰囲気などを含めて説明し、余計な前置きは不要です。"
        else:
            if self.detail_level == "brief":
                return "Describe this image in a single concise sentence, without any introductory phrases."
            if self.detail_level == "standard":
                return "Describe this image in 2-3 sentences, focusing on key elements and actions. No introductory phrases."
            return "Describe this image in 4-5 sentences, including visual elements, actions, and atmosphere. No introductory phrases."

    def clean_response_text(self, response):
        """API レスポンスから不要な語句を除去"""
        return apply_clean_patterns(response, self.clean_patterns)

    def analyze_image(self, image_path):
        """画像を API に送りテキストを取得"""
        try:
            base64_image = self.encode_image(image_path)
            payload = {
                "model": self.model,
                "prompt": self.get_prompt(),
                "stream": False,
                "images": [base64_image],
            }
            response = requests.post(self.api_url, json=payload)
            if response.status_code != 200:
                logger.error(
                    f"APIエラー: status_code={response.status_code}, text={response.text}"
                )
                raise Exception(f"APIエラー: {response.status_code} - {response.text}")

            result = response.json()
            response_text = result.get("response", "No analysis available")
            if not self.clean_custom_response:
                return response_text
            return self.clean_response_text(response_text)
        except requests.exceptions.RequestException as e:
            logger.error(f"API通信エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"画像分析エラー: {e}")
            raise

