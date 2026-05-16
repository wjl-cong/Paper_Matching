"""
期刊论文语义匹配系统 - 翻译服务
====================================

功能说明：
- 提供中英文互译功能（可选）
- 用于跨语言检索增强
- 翻译失败时不影响系统运行（BGE-M3 本身支持跨语言）

翻译策略：
1. LibreTranslate（开源免费）
2. 备用：有道翻译
3. 降级：翻译失败时跳过（系统仍可正常运行）

注意：由于 BGE-M3 本身就支持跨语言检索，翻译是"锦上添花"而非必需。
即使翻译完全失败，系统仍可通过向量空间直接匹配中英文。

作者：AI Assistant
日期：2026-05-16
"""

import time
import re
from typing import Optional

# Google Translate（免费，无需 API Key）
GOOGLE_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"

# LibreTranslate 后端（需要 API Key）
LIBRE_TRANSLATE_URL = "https://libretranslate.com"

# 有道翻译 API
YOUDAO_URL = "https://fanyi.youdao.com/translate"


class TranslationService:
    """
    翻译服务类

    提供中英文互译功能，支持多种翻译后端。
    翻译失败时不影响系统运行（BGE-M3 本身支持跨语言检索）。
    """

    def __init__(self):
        """初始化翻译服务"""
        self._enabled = True  # 可以通过配置禁用翻译

    def set_enabled(self, enabled: bool):
        """设置是否启用翻译"""
        self._enabled = enabled

    def translate(self, text: str, source_lang: str = "auto",
                  target_lang: str = "en") -> Optional[str]:
        """
        翻译文本

        Args:
            text: 待翻译文本
            source_lang: 源语言（"auto"表示自动检测）
            target_lang: 目标语言（"en"英文，"zh"中文）

        Returns:
            翻译后的文本，失败返回 None
        """
        if not self._enabled:
            return None

        if not text or len(text.strip()) == 0:
            return None

        text = text.strip()
        if len(text) > 5000:
            text = text[:5000]

        # 标准化语言代码
        src_code = self._normalize_lang_code(source_lang)
        tgt_code = self._normalize_lang_code(target_lang)

        if src_code == tgt_code:
            return None

        # 尝试 Google Translate（免费，无需 API Key）
        result = self._try_google(text, src_code, tgt_code)
        if result:
            return result

        # 尝试 LibreTranslate
        result = self._try_libretranslate(text, src_code, tgt_code)
        if result:
            return result

        # 尝试有道翻译
        result = self._try_youdao(text, src_code, tgt_code)
        if result:
            return result

        print(f"翻译失败: {text[:50]}... ({src_code} -> {tgt_code})")
        return None

    def _normalize_lang_code(self, code: str) -> str:
        """标准化语言代码"""
        if code == "auto":
            return "auto"
        if code == "zh":
            return "zh"
        if code in ("en", "eng"):
            return "en"
        return code

    def _try_google(self, text: str, src: str, tgt: str) -> Optional[str]:
        """尝试 Google Translate（免费，无需 API Key）"""
        import requests
        import urllib.parse
        import warnings

        # 禁用 SSL 警告
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")

        try:
            # Google Translate 参数
            params = {
                "client": "gtx",
                "sl": src if src != "auto" else "auto",
                "tl": tgt,
                "dt": "t",
                "q": text
            }

            url = f"{GOOGLE_TRANSLATE_URL}?{urllib.parse.urlencode(params)}"

            response = requests.get(url, timeout=30, verify=False)

            if response.status_code == 200:
                data = response.json()
                # Google Translate 返回格式:
                # [[translations], null, src_lang, ...]
                # 其中 translations = [[translated_text, original_text, ...], ...]
                if data and isinstance(data, list) and len(data) > 0:
                    translations = data[0]
                    if translations and isinstance(translations, list) and len(translations) > 0:
                        first_translation = translations[0]
                        if isinstance(first_translation, list) and len(first_translation) > 0:
                            return first_translation[0]
        except requests.exceptions.Timeout:
            print(f"Google Translate timeout")
        except requests.exceptions.SSLError as e:
            print(f"Google Translate SSL error: {e}")
        except Exception as e:
            print(f"Google Translate error: {e}")
        return None

    def _try_libretranslate(self, text: str, src: str, tgt: str) -> Optional[str]:
        """尝试 LibreTranslate"""
        try:
            import requests
            response = requests.post(
                f"{LIBRE_TRANSLATE_URL}/translate",
                json={
                    "q": text,
                    "source": src if src != "auto" else "auto",
                    "target": tgt,
                    "format": "text"
                },
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("translatedText")
        except Exception:
            pass
        return None

    def _try_youdao(self, text: str, src: str, tgt: str) -> Optional[str]:
        """尝试有道翻译"""
        try:
            import requests
            import json
            import hashlib
            import random

            # 有道翻译参数
            appid = "202306160016xxxxxx"  # 需要替换为真实 appid
            secret = "xxxxxx"  # 需要替换为真实 secret

            salt = str(random.randint(1, 65536))
            sign_str = appid + text + salt + secret
            sign = hashlib.md5(sign_str.encode()).hexdigest()

            params = {
                "q": text,
                "from": "zh-CHS" if src == "zh" else "en",
                "to": "zh-CHS" if tgt == "zh" else "en",
                "appKey": appid,
                "salt": salt,
                "sign": sign,
                "signType": "v3",
                "timeout": 10
            }

            response = requests.get(
                "https://openapi.youdao.com/api",
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("errorCode") == "0":
                    return data.get("translation", [None])[0]
        except Exception:
            pass
        return None

    def zh_to_en(self, text: str) -> Optional[str]:
        """中文翻译成英文"""
        return self.translate(text, source_lang="zh", target_lang="en")

    def en_to_zh(self, text: str) -> Optional[str]:
        """英文翻译成中文"""
        return self.translate(text, source_lang="en", target_lang="zh")

    def detect_language(self, text: str) -> str:
        """
        检测文本语言

        Args:
            text: 待检测文本

        Returns:
            "zh" 表示中文，"en" 表示英文
        """
        if not text:
            return "en"

        # CJK 统一汉字范围检测
        cjk_count = 0
        for char in text:
            code = ord(char)
            # CJK 统一汉字范围：CJK 统一汉字、CJK 扩展A、CJK 扩展B
            if ((0x4E00 <= code <= 0x9FFF) or
                (0x3400 <= code <= 0x4DBF) or
                (0x20000 <= code <= 0x2A6DF)):
                cjk_count += 1

        # 如果超过 10% 的字符是 CJK，判定为中文
        if cjk_count > len(text) * 0.1:
            return "zh"

        return "en"


# 全局单例
_translation_service: Optional[TranslationService] = None


def get_translation_service() -> TranslationService:
    """获取翻译服务全局单例"""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service
