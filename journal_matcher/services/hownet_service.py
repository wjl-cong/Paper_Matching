"""
期刊论文语义匹配系统 - OpenHowNet 中文语义增强服务
====================================

功能说明：
- 基于HowNet/OpenHowNet义原知识库增强中文语义理解
- 提供中文词语的义原标注
- 支持基于义原的词相似度计算
- 辅助中文文本分词和语义扩展

OpenHowNet 核心能力：
- 237,973个概念，中英文词语及短语
- 每个概念包含义原标注（Sememe）
- 支持基于义原的语义相似度计算

义原（Sememe）是HowNet的核心概念，代表最小的语义单位。
例如"苹果"这个词语有多个义原：水果、食物、公司等。

作者：AI Assistant
日期：2026-05-16
"""

import logging
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WordSememe:
    """词语的义原标注"""
    word: str
    sense_count: int  # 概念数量
    sememes: List[str]  # 义原列表（英文）
    sememes_zh: List[str]  # 义原列表（中文）
    definition: str  # 义原定义


class HowNetService:
    """
    OpenHowNet 语义增强服务

    封装OpenHowNet API，提供中文语义增强能力

    使用示例：
        service = HowNetService()
        sememes = service.get_word_sememes("人工智能")
        similar = service.get_similar_words("深度学习", top_k=10)
    """

    def __init__(self, init_sim: bool = True):
        """
        初始化HowNet服务

        Args:
            init_sim: 是否初始化相似度计算（需要下载数据，第一次较慢）
        """
        self._initialized = False
        self._hownet_dict = None
        self._init_sim = init_sim
        self._init_failed = False

    def _lazy_init(self):
        """延迟初始化：只在第一次使用时才加载"""
        if self._initialized:
            return True

        if self._init_failed:
            return False

        try:
            import OpenHowNet
            print("正在初始化 OpenHowNet（首次运行需要下载数据，约200MB）...")

            self._hownet_dict = OpenHowNet.HowNetDict(init_sim=self._init_sim)
            self._initialized = True
            print("OpenHowNet 初始化完成")
            return True

        except ImportError:
            print("警告：OpenHowNet 未安装。使用 pip install OpenHowNet 安装")
            self._init_failed = True
            return False
        except Exception as e:
            print(f"警告：OpenHowNet 初始化失败: {e}")
            self._init_failed = True
            return False

    def is_available(self) -> bool:
        """检查服务是否可用"""
        return self._lazy_init()

    def get_word_sememes(self, word: str, max_senses: int = 3) -> List[WordSememe]:
        """
        获取词语的义原标注

        Args:
            word: 中文或英文词语
            max_senses: 最多返回的概念数量

        Returns:
            WordSememe列表，每个概念一个义原标注
        """
        if not self._lazy_init():
            return []

        if not word or not word.strip():
            return []

        results = []
        try:
            senses = self._hownet_dict.get_sense(word)

            for i, sense in enumerate(senses[:max_senses]):
                if sense is None:
                    continue

                # 获取义原列表
                sememe_list = sense.get_sememe_list()
                sememes_en = []
                sememes_zh = []
                for sememe in sememe_list:
                    if hasattr(sememe, 'en') and sememe.en:
                        sememes_en.append(sememe.en)
                    if hasattr(sememe, 'zh') and sememe.zh:
                        sememes_zh.append(sememe.zh)

                # 获取义原定义
                definition = ""
                if hasattr(sense, 'Def'):
                    definition = sense.Def

                results.append(WordSememe(
                    word=word,
                    sense_count=len(senses),
                    sememes=sememes_en,
                    sememes_zh=sememes_zh,
                    definition=definition
                ))

        except Exception as e:
            logger.debug(f"获取义原失败 '{word}': {e}")

        return results

    def expand_query_with_sememes(self, query: str, max_words: int = 20) -> List[str]:
        """
        使用义原扩展查询词

        通过获取查询词的义原，找到相关的上位词/下位词

        Args:
            query: 原始查询
            max_words: 最多返回的扩展词数量

        Returns:
            扩展后的词语列表
        """
        if not self._lazy_init():
            return []

        expanded = set()
        expanded.add(query)  # 保留原始查询

        # 分词提取关键词
        try:
            import jieba
            words = [w for w in jieba.cut(query) if len(w) >= 2]
        except ImportError:
            import re
            words = re.findall(r'[\u4e00-\u9fff]{2,4}', query)

        for word in words:
            # 获取该词的义原
            sememes = self.get_word_sememes(word)
            for sememe_data in sememes:
                # 添加义原作为扩展
                for sem in sememe_data.sememes[:3]:  # 最多3个义原
                    if len(sem) >= 2:
                        expanded.add(sem)
                for sem in sememe_data.sememes_zh[:3]:
                    if len(sem) >= 2:
                        expanded.add(sem)

        # 返回最多max_words个词
        return list(expanded)[:max_words]

    def get_all_sememes(self) -> List:
        """
        获取所有义原

        Returns:
            义原列表
        """
        if not self._lazy_init():
            return []

        try:
            return self._hownet_dict.get_all_sememes()
        except Exception as e:
            logger.debug(f"获取所有义原失败: {e}")
            return []

    def get_sememe(self, sememe: str, language: str = "en") -> List:
        """
        根据名称获取义原

        Args:
            sememe: 义原名称
            language: 语言 ('en' 或 'zh')

        Returns:
            匹配的义原列表
        """
        if not self._lazy_init():
            return []

        try:
            return self._hownet_dict.get_sememe(sememe, language=language)
        except Exception as e:
            logger.debug(f"获取义原失败 '{sememe}': {e}")
            return []

    def batch_get_sememes(self, words: List[str]) -> Dict[str, List[WordSememe]]:
        """
        批量获取多个词语的义原

        Args:
            words: 词语列表

        Returns:
            词语 -> 义原列表的字典
        """
        results = {}
        for word in words:
            sememes = self.get_word_sememes(word)
            if sememes:
                results[word] = sememes
        return results


# =============================================================================
# 中文停用词（用于分词和检索）
# =============================================================================

# 常见中文停用词
CHINESE_STOPWORDS: Set[str] = {
    "的", "在", "是", "了", "和", "与", "对", "及", "为", "以", "于",
    "上", "下", "中", "内", "外", "前", "后", "左", "右", "之间",
    "一个", "一些", "这个", "那个", "其", "其他", "另", "另一个",
    "我", "我们", "你", "您", "他", "她", "它", "他们", "她们", "它们",
    "有", "没有", "不是", "是", "而是", "而且", "并且", "或者", "还是",
    "这", "那", "这些", "那些", "此", "该", "本", "各", "每",
    "也", "都", "很", "非常", "更", "最", "比较", "相当", "十分", "极",
    "把", "被", "让", "使", "由", "从", "到", "向", "往",
    "关于", "对于", "由于", "通过", "根据", "按照", "随着", "沿着",
    "等", "等等", "之类", "什么的",
    "可以", "能够", "会", "能", "要", "想", "应该", "必须", "需要",
    "可能", "或许", "也许", "大概", "难道", "难道说",
    "不", "没", "无", "非", "别", "莫", "勿", "休",
    "再", "又", "还", "已", "已经", "曾", "曾经", "正在", "将", "将要",
    "然后", "接着", "于是", "因此", "所以", "从而", "只要", "只有",
    "如果", "假如", "倘若", "要是", "万一", "即使", "尽管", "虽然", "虽然说",
}

# 学术论文常用停用词
ACADEMIC_STOPWORDS: Set[str] = {
    "研究", "分析", "方法", "问题", "理论", "实践", "应用", "发展", "影响", "作用",
    "过程", "结果", "结论", "讨论", "说明", "表明", "认为", "指出", "提出", "认为",
    "本文", "本论文", "本研究", "本文研究", "本文认为", "作者", "本文作者",
    "基于", "根据", "通过", "利用", "采用", "使用", "运用", "结合", "综合",
    "包括", "涉及", "关于", "针对", "对于", "围绕", "鉴于", "由于",
    "第一", "第二", "第三", "首先", "其次", "最后", "一方面", "另一方面",
    "总之", "总之", "因此", "所以", "因而", "故", "故而",
    "目前", "现阶段", "当今", "当代", "现代", "当前", "近年来", "最近",
    "主要", "重要", "关键", "核心", "基本", "根本", "必要", "必需",
    "一定", "部分", "整体", "全部", "所有", "任何", "每个",
    "不同", "各种", "多种", "各类", "各项", "各种各样",
}

# 合并所有停用词
ALL_CHINESE_STOPWORDS: Set[str] = CHINESE_STOPWORDS | ACADEMIC_STOPWORDS


# =============================================================================
# 全局单例
# =============================================================================

_hownet_service: Optional[HowNetService] = None


def get_hownet_service(init_sim: bool = False) -> HowNetService:
    """
    获取HowNet服务的全局单例

    Args:
        init_sim: 是否初始化相似度计算（默认False，避免首次加载过慢）

    Returns:
        HowNetService实例
    """
    global _hownet_service
    if _hownet_service is None:
        _hownet_service = HowNetService(init_sim=init_sim)
    return _hownet_service


def reset_hownet_service():
    """重置HowNet服务（用于重新初始化）"""
    global _hownet_service
    _hownet_service = None
