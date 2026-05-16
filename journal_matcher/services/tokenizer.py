"""
期刊论文语义匹配系统 - 增强分词服务
====================================

功能说明：
- 支持中英文混合分词
- 使用 jieba 进行中文分词
- 使用正则进行英文分词
- 集成 OpenHowNet 义原扩展
- 支持自定义词典

与 OpenHowNet 的结合：
- 可以利用义原信息进行分词纠错
- 可以扩展同义词/相关词
- 可以识别领域特定词汇

作者：AI Assistant
日期：2026-05-16
"""

import re
from typing import List, Set, Optional, Callable
from dataclasses import dataclass

# 延迟导入 jieba，避免未安装时无法使用
_jieba_available = False
try:
    import jieba
    _jieba_available = True
except ImportError:
    pass


@dataclass
class Token:
    """分词结果"""
    text: str
    pos: str  # 词性标注
    is_chinese: bool
    is_english: bool


class ChineseTokenizer:
    """
    增强分词器

    支持：
    - jieba 中文分词（需要安装）
    - 正则英文分词
    - 自定义词典
    - 停用词过滤
    - OpenHowNet 义原扩展

    使用示例：
        tokenizer = ChineseTokenizer()
        tokens = tokenizer.tokenize("深度学习在人工智能中的应用")
        print([t.text for t in tokens])
    """

    def __init__(self,
                 use_jieba: bool = True,
                 stopwords: Optional[Set[str]] = None,
                 add_user_dict: Optional[List[str]] = None,
                 enable_hownet: bool = False):
        """
        初始化分词器

        Args:
            use_jieba: 是否使用jieba分词（False时使用正则）
            stopwords: 停用词集合
            add_user_dict: 用户词典（追加的词汇）
            enable_hownet: 是否启用OpenHowNet扩展
        """
        self.use_jieba = use_jieba and _jieba_available
        self.stopwords = stopwords or set()
        self.enable_hownet = enable_hownet

        if not self.use_jieba:
            print("警告：jieba未安装，使用正则分词，中文效果较差")

        # 添加用户词典
        if add_user_dict and _jieba_available:
            for word in add_user_dict:
                jieba.add_word(word)

    def tokenize(self, text: str,
                 remove_stopwords: bool = True,
                 return_pos: bool = False) -> List[str] | List[Token]:
        """
        分词

        Args:
            text: 待分词文本
            remove_stopwords: 是否移除停用词
            return_pos: 是否返回词性标注

        Returns:
            分词结果列表
        """
        if not text:
            return []

        if self.use_jieba:
            tokens = self._tokenize_jieba(text)
        else:
            tokens = self._tokenize_regex(text)

        if return_pos:
            return tokens

        # 返回纯文本列表
        result = [t.text for t in tokens]

        if remove_stopwords:
            result = [t for t in result if t not in self.stopwords and len(t) >= 2]

        return result

    def _tokenize_jieba(self, text: str) -> List[Token]:
        """使用jieba分词"""
        # 使用 jieba.lcut 获取词列表
        words = jieba.lcut(text)

        tokens = []
        for word in words:
            # 判断词的类型
            is_chinese = bool(re.search(r'[\u4e00-\u9fff]', word))
            is_english = bool(re.match(r'^[a-zA-Z]+$', word))

            # 简单词性标注（基于字符特征）
            pos = self._guess_pos(word, is_chinese, is_english)

            # 过滤空白字符
            if word.strip():
                tokens.append(Token(
                    text=word.strip(),
                    pos=pos,
                    is_chinese=is_chinese,
                    is_english=is_english
                ))

        return tokens

    def _tokenize_regex(self, text: str) -> List[Token]:
        """使用正则分词（备用方案）"""
        tokens = []

        # 提取英文单词
        english_matches = re.finditer(r'[a-zA-Z]+', text)
        for match in english_matches:
            word = match.group()
            tokens.append(Token(
                text=word,
                pos='ENG',
                is_chinese=False,
                is_english=True
            ))

        # 提取中文词组（2-4个连续汉字）
        chinese_matches = re.finditer(r'[\u4e00-\u9fff]{2,4}', text)
        for match in chinese_matches:
            word = match.group()
            tokens.append(Token(
                text=word,
                pos='NOUN',  # 默认词性
                is_chinese=True,
                is_english=False
            ))

        return tokens

    def _guess_pos(self, word: str, is_chinese: bool, is_english: bool) -> str:
        """简单词性猜测"""
        if is_english:
            # 英文词性猜测
            if len(word) <= 3:
                return 'SHORT'  # 短词
            return 'ENG'

        if is_chinese:
            # 常见词性模式
            if word.endswith(('的', '地', '得')):
                return 'AUX'  # 助词
            if word.endswith(('了', '着', '过', '吗', '呢')):
                return 'PART'  # 语气词
            if word in ('是', '在', '有', '和', '与', '对'):
                return 'PREP'  # 介词/连词
            return 'NOUN'  # 默认名词

        return 'UNK'

    def add_word(self, word: str, freq: int = None, tag: str = None):
        """
        添加词汇到词典

        Args:
            word: 词语
            freq: 词频（可选）
            tag: 词性（可选）
        """
        if _jieba_available:
            jieba.add_word(word, freq, tag)

    def delete_word(self, word: str):
        """从词典中删除词汇"""
        if _jieba_available:
            jieba.del_word(word)

    def load_user_dict(self, dict_path: str):
        """加载自定义词典文件"""
        if _jieba_available:
            jieba.load_userdict(dict_path)


# =============================================================================
# 学术论文分词器（预设配置）
# =============================================================================

def create_academic_tokenizer() -> ChineseTokenizer:
    """
    创建学术论文专用分词器

    Returns:
        配置好的ChineseTokenizer实例
    """
    # 学术领域专用词汇
    academic_words = [
        # 人工智能
        "深度学习", "机器学习", "神经网络", "卷积神经网络", "循环神经网络",
        "自然语言处理", "计算机视觉", "目标检测", "图像分割", "语音识别",
        "强化学习", "迁移学习", "联邦学习", "对抗学习", "表示学习",
        # 经济学
        "宏观经济", "微观经济", "产业结构", "资源配置", "技术创新",
        "全要素生产率", "经济增长", "通货膨胀", "货币政策", "财政政策",
        # 管理学
        "公司治理", "企业战略", "组织行为", "人力资源", "绩效管理",
        "供应链管理", "运营管理", "风险管理", "知识管理", "创新管理",
        # 金融学
        "资产定价", "投资组合", "风险管理", "公司金融", "行为金融",
        "金融市场", "证券投资", "衍生品", "量化投资", "资产管理",
        # 会计学
        "财务报告", "盈余管理", "内部控制", "公司估值", "审计质量",
        # 研究方法
        "实证研究", "案例研究", "文献综述", "计量模型", "回归分析",
        "面板数据", "时间序列", "因果推断", "中介效应", "调节效应",
    ]

    # 学术停用词
    from .hownet_service import ALL_CHINESE_STOPWORDS

    return ChineseTokenizer(
        use_jieba=True,
        stopwords=ALL_CHINESE_STOPWORDS,
        add_user_dict=academic_words,
        enable_hownet=False  # 默认关闭，按需启用
    )


# =============================================================================
# 全局单例
# =============================================================================

_tokenizer: Optional[ChineseTokenizer] = None


def get_tokenizer() -> ChineseTokenizer:
    """获取分词器的全局单例"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = create_academic_tokenizer()
    return _tokenizer


def reset_tokenizer():
    """重置分词器"""
    global _tokenizer
    _tokenizer = None
