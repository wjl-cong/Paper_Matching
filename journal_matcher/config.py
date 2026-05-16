"""
期刊论文语义匹配系统 - 配置文件
====================================

功能说明：
- 集中管理系统所有配置参数
- 包含向量化模型、API、数据库、检索阈值等配置
- 支持环境变量覆盖默认值

作者：AI Assistant
日期：2026-05-16
"""

import os
from pathlib import Path
from typing import Dict, List

# =============================================================================
# 项目路径配置
# =============================================================================

# 项目根目录
BASE_DIR = Path(__file__).parent.parent
# 数据存储目录
DATA_DIR = BASE_DIR / "data"
# SQLite数据库文件路径
CACHE_DB_PATH = DATA_DIR / "papers.db"
# FAISS向量索引文件路径
VECTOR_INDEX_PATH = DATA_DIR / "faiss_index.bin"

# 确保数据目录存在
DATA_DIR.mkdir(exist_ok=True)

# =============================================================================
# 向量化模型配置
# =============================================================================

class EmbeddingConfig:
    """向量化模型配置类"""

    # 主模型：BGE-M3，多语言向量化模型，支持100+语言
    # 特点：
    # 1. 多语言：原生支持中文和英文，无需翻译
    # 2. 多功能：同时支持稠密检索、多向量检索、稀疏检索
    # 3. 多粒度：支持从短句到8192 token的长文档
    # 下载地址：https://huggingface.co/BAAI/bge-m3
    MODEL_NAME = "BAAI/bge-m3"

    # 备选模型：轻量级多语言模型，适合快速演示
    # 如果BGE-M3加载失败或性能不足，使用此备选
    MODEL_NAME_FALLBACK = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # 向量维度：BGE-M3生成1024维向量
    # 注意：这个维度必须与FAISS索引维度一致
    EMBEDDING_DIM = 1024

    # 向量归一化：归一化后的向量余弦相似度等价于点积
    NORMALIZE_EMBEDDINGS = True

    # 最大输入长度：防止超长文本导致内存问题
    MAX_INPUT_LENGTH = 512

    # 批处理大小：每次向量化处理的文档数量
    # 较大的批次可以提高速度，但会占用更多内存
    BATCH_SIZE = 32


# =============================================================================
# 检索参数配置
# =============================================================================

class RetrievalConfig:
    """检索相关配置"""

    # 召回阶段返回数量：第一阶段向量检索返回的候选论文数
    # 选择50的原因：保证召回率，跨语言检索需要更多候选
    TOP_K_RECALL = 50

    # 重排后返回数量：最终返回给用户的每期刊论文数
    # 5是一个经验值：足够展示结果，又不会信息过载
    TOP_N_RERANK = 5

    # 相似度阈值：判断论文是否"相似"的最低分数
    # 范围：0.0 ~ 1.0
    # - 高阈值(>0.7)：返回高度相似的论文，假阳性低
    # - 低阈值(<0.4)：返回更多相关论文，假阳性高
    # 跨语言检索默认使用较低阈值
    SIMILARITY_THRESHOLD = 0.1

    # Cross-Encoder重排序模型
    # 用于第二阶段精细化打分，比向量相似度更准确
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

    # 是否启用混合检索（向量 + BM25）
    # 混合检索可以兼顾语义相似度和词汇匹配
    ENABLE_HYBRID_SEARCH = False

    # 混合检索权重：向量分数的权重
    # (1 - HYBRID_ALPHA)是BM25分数的权重
    HYBRID_ALPHA = 0.7


# =============================================================================
# API配置
# =============================================================================

class APIConfig:
    """外部API配置"""

    # OpenAlex API：主要的学术文献数据源
    # 优势：数据全面、免费、字段丰富、支持多语言
    # 文档：https://docs.openalex.org
    # API Key: 已配置（用户提供的Key）
    OPENALEX_BASE_URL = "https://api.openalex.org"
    OPENALEX_API_KEY = "wgpECpeGjw91ACAZe4wqo6"  # 用户提供的API Key

    # CrossRef API：补充数据源
    # 优势：免注册、响应快、元数据标准
    # 文档：https://www.crossref.org/documentation/retrieve-metadata/rest-api/
    CROSSREF_BASE_URL = "https://api.crossref.org"
    # 使用邮箱标识请求来源，这是CrossRef的推荐做法
    # 建议更改为您的真实邮箱，避免被限流
    CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL", "your_email@example.com")

    # Semantic Scholar API：用于补全缺失数据
    # 优势：提供论文摘要、引用关系、推荐等功能
    # 文档：https://api.semanticscholar.org/
    SEMANTICSCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
    SEMANTICSCHOLAR_API_KEY = os.getenv("SEMANTICSCHOLAR_API_KEY", "")

    # CNKI国际版API：国内期刊数据源
    # 优势：覆盖中国学术期刊
    # 注意：可能需要机构授权
    CNKI_BASE_URL = "https://oversea.cnki.net"

    # 请求配置
    REQUEST_TIMEOUT = 10  # 单次请求超时时间（秒）
    MAX_RETRIES = 3       # 失败重试次数
    RETRY_DELAY = 2       # 重试间隔（秒）
    # 请求间隔：避免API限流（请求/秒）
    RATE_LIMIT_DELAY = 0.5


# =============================================================================
# 数据库配置
# =============================================================================

class DatabaseConfig:
    """数据库配置"""

    # SQLite数据库配置
    DB_PATH = str(CACHE_DB_PATH)

    # 期刊信息缓存有效期（小时）
    # 超过这个时间，期刊论文数据会被视为过期
    JOURNAL_CACHE_TTL_HOURS = 24

    # 论文向量缓存有效期（天）
    # 向量计算是CPU密集型操作，缓存可以避免重复计算
    VECTOR_CACHE_TTL_DAYS = 7

    # 每次抓取论文的最大数量
    # 扩大规模以获取足够多的中英文文献
    MAX_PAPERS_PER_JOURNAL = 200


# =============================================================================
# 默认期刊列表
# =============================================================================

# 示例期刊配置：国内外各5本
# 注意：实际面试时会由面试官指定具体期刊
DEFAULT_JOURNALS: Dict[str, List[Dict[str, str]]] = {
    # 国内期刊（中文为主）
    "CN": [
        {"name": "计算机学报", "issn": "0254-4164", "description": "中国计算机学会会刊"},
        {"name": "软件学报", "issn": "1000-9825", "description": "中国科学院软件研究所主办"},
        {"name": "自动化学报", "issn": "0254-4156", "description": "中国自动化学会主办"},
        {"name": "电子学报", "issn": "0372-2112", "description": "中国电子学会主办"},
        {"name": "通信学报", "issn": "0216-383X", "description": "中国通信学会主办"},
    ],
    # 国际期刊（英文为主）
    "INT": [
        {"name": "Nature", "issn": "0028-0836", "description": "国际顶级综合期刊"},
        {"name": "Science", "issn": "0036-8075", "description": "AAAS主办的综合期刊"},
        {"name": "IEEE TPAMI", "issn": "0162-8828", "description": "IEEE模式分析与机器智能汇刊"},
        {"name": "ACM Computing Surveys", "issn": "0360-0300", "description": "ACM计算综述"},
        {"name": "The Lancet", "issn": "0140-6736", "description": "国际顶级医学期刊"},
    ],
}


# =============================================================================
# 日志配置
# =============================================================================

class LogConfig:
    """日志配置"""

    # 日志级别：DEBUG/INFO/WARNING/ERROR
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # 日志格式
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 是否输出到文件
    LOG_TO_FILE = True
    LOG_FILE_PATH = DATA_DIR / "app.log"


# =============================================================================
# 应用配置（汇总）
# =============================================================================

class Config:
    """全局配置汇总类"""

    # 各子配置实例
    embedding = EmbeddingConfig()
    retrieval = RetrievalConfig()
    api = APIConfig()
    database = DatabaseConfig()
    log = LogConfig()

    # 顶层路径配置（从模块顶层导入）
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    CACHE_DB_PATH = CACHE_DB_PATH
    VECTOR_INDEX_PATH = VECTOR_INDEX_PATH

    # 默认期刊配置
    DEFAULT_JOURNALS = DEFAULT_JOURNALS

    def get_safe_index_path(self) -> Path:
        """获取安全的FAISS索引路径，避免中文路径问题"""
        path = self.VECTOR_INDEX_PATH
        # 如果路径包含非ASCII字符，使用备用路径
        if str(path).encode('ascii', 'replace').decode('ascii') != str(path):
            import tempfile
            safe_dir = Path(tempfile.gettempdir()) / "journal_matcher_data"
            safe_dir.mkdir(exist_ok=True)
            return safe_dir / "faiss_index.bin"
        return path

    # 版本信息
    VERSION = "1.0.0"
    AUTHOR = "AI Assistant"
    DATE = "2026-05-16"


# 全局配置实例
config = Config()
