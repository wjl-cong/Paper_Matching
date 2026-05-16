"""
期刊论文语义匹配系统 - 向量化服务
====================================

功能说明：
- 负责将文本（论文标题、摘要、用户查询）转换为向量表示
- 支持多语言向量化（中文和英文）
- 提供向量缓存机制避免重复计算

核心模型：
- 主模型：BGE-M3（多语言向量化模型）
- 特点：支持100+语言，包括中英文，无需翻译即可跨语言检索

作者：AI Assistant
日期：2026-05-16
"""

import os
import ssl
import time
from typing import List, Optional, Union
from functools import lru_cache

# 必须在导入 sentence_transformers 之前设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import config

# 本地模型路径（如果设置了 LOCAL_MODEL_PATH，则从本地加载）
LOCAL_MODEL_PATH = os.environ.get('LOCAL_MODEL_PATH', '').strip()

# 自动检测本地模型
def _find_local_model():
    """自动检测本地模型目录"""
    if LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    # 检测默认位置: embedding.py 在 services/ 下，往上两级是 journal_matcher/
    default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bge-m3")
    if os.path.isdir(default_path):
        return default_path
    return ""


class EmbeddingService:
    """
    向量化服务类

    负责文本的向量化表示，支持：
    1. 单条文本向量化
    2. 批量文本向量化
    3. 向量归一化
    4. 模型加载和缓存

    设计思路：
    - 使用 sentence-transformers 库加载预训练模型
    - 模型在首次使用时加载，之后缓存在内存中
    - 支持GPU加速（如可用）
    """

    def __init__(self):
        """
        初始化向量化服务

        加载预训练模型。首次初始化会下载模型（约1GB），请确保网络连接。
        """
        self.model = None  # 向量化模型
        self.model_name = config.embedding.MODEL_NAME  # 模型名称
        self.embedding_dim = config.embedding.EMBEDDING_DIM  # 向量维度
        self._is_ready = False  # 模型是否已加载
        self._load_model()  # 加载模型

    def _load_model(self):
        """
        加载向量化模型

        加载策略：
        1. 自动检测本地模型目录
        2. 尝试加载主模型（BGE-M3）
        3. 如果失败，尝试加载轻量级备选模型
        4. 如果仍然失败，抛出异常

        备选模型说明：
        - paraphrase-multilingual-MiniLM-L12-v2 是一个12层的小模型
        - 虽然效果不如BGE-M3，但加载快、占用内存少
        """
        # 自动检测本地模型
        local_path = _find_local_model()
        
        try:
            # 优先使用本地模型路径
            if local_path:
                print(f"正在从本地加载模型: {local_path}...")
                model_path = local_path
            else:
                print(f"正在加载向量化模型: {self.model_name}...")
                model_path = self.model_name
            
            # 使用sentence-transformers加载模型
            # device='cpu' 强制使用CPU（确保兼容性）
            # 可以改为 device='cuda' 在有GPU的环境加速
            self.model = SentenceTransformer(model_path, device='cpu')
            self._is_ready = True
            print(f"模型加载成功！向量维度: {self.embedding_dim}")

        except Exception as e:
            print(f"加载主模型失败: {e}")
            print(f"尝试加载备选模型: {config.embedding.MODEL_NAME_FALLBACK}")

            try:
                fallback_path = local_path if local_path else config.embedding.MODEL_NAME_FALLBACK
                self.model = SentenceTransformer(fallback_path, device='cpu')
                self.embedding_dim = 384  # MiniLM模型的维度
                self._is_ready = True
                print(f"备选模型加载成功！向量维度: {self.embedding_dim}")

            except Exception as e2:
                print(f"加载备选模型也失败: {e2}")
                self._is_ready = False
                raise RuntimeError("无法加载任何向量化模型，请检查网络连接或手动下载模型")

    def encode(self, texts: Union[str, List[str]],
               normalize: bool = True,
               batch_size: int = None) -> np.ndarray:
        """
        将文本转换为向量

        这是核心方法，将文本（标题、摘要、查询等）转换为固定维度的向量。
        向量之间的相似度可以通过余弦相似度或内积计算。

        Args:
            texts: 单个文本字符串或文本列表
            normalize: 是否对向量进行L2归一化
                     归一化后，向量的内积 = 余弦相似度，计算更快
            batch_size: 批处理大小，默认使用配置值

        Returns:
            numpy数组，每行对应一个文本的向量

        示例：
            # 单文本
            vector = service.encode("深度学习在医学影像中的应用")
            print(vector.shape)  # (1024,)

            # 多文本
            vectors = service.encode(["论文1标题", "论文2标题"])
            print(vectors.shape)  # (2, 1024)
        """
        if not self._is_ready:
            raise RuntimeError("向量化模型未加载")

        # 处理单文本输入
        if isinstance(texts, str):
            texts = [texts]

        # 设置批处理大小
        if batch_size is None:
            batch_size = config.embedding.BATCH_SIZE

        # 限制文本长度，防止超长文本
        # 策略：截断而非报错
        max_length = config.embedding.MAX_INPUT_LENGTH
        texts = [text[:max_length * 10] if text else "" for text in texts]

        # 调用模型的encode方法
        # show_progress_bar=False 禁用进度条（避免日志混乱）
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return embeddings

    def encode_paper(self, title: str, abstract: str = None,
                      keywords: List[str] = None) -> np.ndarray:
        """
        将论文转换为向量

        将论文的标题、摘要、关键词组合成检索文本，然后向量化。

        组合策略说明：
        - 按信息量优先级组合：[标题] | [摘要] | [关键词]
        - 使用 " | " 分隔，便于模型识别不同部分
        - 这种格式在BGE-M3训练数据中常见，有助于模型理解

        Args:
            title: 论文标题
            abstract: 论文摘要（可选）
            keywords: 关键词列表（可选）

        Returns:
            论文的向量表示（1D numpy数组）
        """
        # 构建检索文本
        parts = [title]

        if abstract:
            parts.append(abstract)

        if keywords:
            # 关键词用逗号连接
            keywords_str = ", ".join(keywords)
            parts.append(f"Keywords: {keywords_str}")

        search_text = " | ".join(parts)

        # 向量化
        vectors = self.encode([search_text], normalize=True)

        return vectors[0]

    def encode_query(self, query: str) -> np.ndarray:
        """
        将用户查询转换为向量

        用户输入的主题描述，直接向量化即可。
        无需特殊处理，模型会自动理解语义。

        Args:
            query: 用户输入的检索主题

        Returns:
            查询向量（1D numpy数组）
        """
        return self.encode([query], normalize=True)[0]

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度

        如果向量已经归一化，直接用内积即可。

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            相似度分数（0.0 ~ 1.0）
        """
        # 归一化
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        vec1 = vec1 / norm1
        vec2 = vec2 / norm2

        # 内积 = 余弦相似度（因为已经归一化）
        similarity = np.dot(vec1, vec2)

        # 限制范围，防止浮点误差
        return float(np.clip(similarity, 0.0, 1.0))

    def batch_compute_similarity(self, query_vec: np.ndarray,
                                  doc_vecs: np.ndarray) -> np.ndarray:
        """
        批量计算查询向量与多个文档向量的相似度

        比逐个调用 compute_similarity 快很多（利用矩阵运算）

        Args:
            query_vec: 查询向量 (dim,)
            doc_vecs: 文档向量矩阵 (n, dim)

        Returns:
            相似度数组 (n,)
        """
        # 归一化
        query_vec = query_vec / np.linalg.norm(query_vec)
        doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

        # 批量计算内积
        similarities = np.dot(doc_vecs, query_vec)

        return np.clip(similarities, 0.0, 1.0)

    def is_ready(self) -> bool:
        """检查模型是否已就绪"""
        return self._is_ready

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "is_ready": self._is_ready,
            "max_length": config.embedding.MAX_INPUT_LENGTH
        }


# =============================================================================
# 全局单例（延迟加载）
# =============================================================================

# 使用延迟加载模式：只在首次使用时初始化模型
# 这样可以加快应用启动速度
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    获取向量化服务的全局单例

    使用单例模式避免重复加载模型（模型加载较慢）

    Returns:
        EmbeddingService实例
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
