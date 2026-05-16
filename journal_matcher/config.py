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

    # ====================================
    # 中文增强相关配置
    # ====================================

    # 是否使用 jieba 中文分词
    # 启用后，中文分词更精准，BM25效果更好
    # 需要安装：pip install jieba
    USE_JIEBA_TOKENIZER = True

    # 是否使用 OpenHowNet 义原扩展
    # 启用后，可以利用义原知识库增强中文语义理解
    # 需要安装：pip install OpenHowNet
    # 注意：首次运行会下载数据（约200MB）
    USE_HOWNET_EXPANSION = False


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

    # 每次抓取论文的最大数量（每个期刊只抓最新一期1篇）
    MAX_PAPERS_PER_JOURNAL = 1


# =============================================================================
# 经管类期刊配置（中文5本 + UTD 24本）
# =============================================================================

# 中文经管类期刊（ISSN可能有误，失败时会用名称搜索）
CN_JOURNALS_MANAGEMENT = [
    {"name": "管理世界", "issn": "1000-5935", "description": "管理学顶级期刊"},
    {"name": "经济研究", "issn": "0577-9154", "description": "经济学顶级期刊"},
    {"name": "管理科学学报", "issn": "1005-2542", "description": "管理科学"},
    {"name": "中国工业经济", "issn": "1002-5502", "description": "产业经济学"},
    {"name": "会计研究", "issn": "1003-2886", "description": "会计学"},
]

# UTD 24本期刊列表
UTD_24_JOURNALS = [
    # 金融 (4本)
    {"name": "Journal of Finance", "issn": "0022-1082", "category": "Finance", "description": "JF"},
    {"name": "Journal of Financial Economics", "issn": "0304-405X", "category": "Finance", "description": "JFE"},
    {"name": "Review of Financial Studies", "issn": "0893-9454", "category": "Finance", "description": "RFS"},
    {"name": "Journal of Financial and Quantitative Analysis", "issn": "0022-1090", "category": "Finance", "description": "JFQA"},
    # 会计 (4本)
    {"name": "Accounting Review", "issn": "0001-4826", "category": "Accounting", "description": "AR"},
    {"name": "Journal of Accounting and Economics", "issn": "0165-4101", "category": "Accounting", "description": "JAE"},
    {"name": "Journal of Accounting Research", "issn": "0021-8456", "category": "Accounting", "description": "JAR"},
    {"name": "Accounting, Organizations and Society", "issn": "0361-3682", "category": "Accounting", "description": "AOS"},
    # 管理科学 (4本)
    {"name": "Management Science", "issn": "0025-1909", "category": "Management", "description": "MS"},
    {"name": "Administrative Science Quarterly", "issn": "0001-8392", "category": "Management", "description": "ASQ"},
    {"name": "Academy of Management Journal", "issn": "0001-4273", "category": "Management", "description": "AMJ"},
    {"name": "Strategic Management Journal", "issn": "0143-2095", "category": "Management", "description": "SMJ"},
    # 运营管理 (2本)
    {"name": "Operations Research", "issn": "0030-364X", "category": "Operations", "description": "OR"},
    {"name": "Manufacturing & Service Operations Management", "issn": "1523-4614", "category": "Operations", "description": "M&SOM"},
    # 信息系统 (2本)
    {"name": "MIS Quarterly", "issn": "0276-7783", "category": "IS", "description": "MISQ"},
    {"name": "Information Systems Research", "issn": "1047-7047", "category": "IS", "description": "ISR"},
    # 市场营销 (3本)
    {"name": "Journal of Marketing", "issn": "0022-2429", "category": "Marketing", "description": "JM"},
    {"name": "Journal of Marketing Research", "issn": "0022-2437", "category": "Marketing", "description": "JMR"},
    {"name": "Journal of Consumer Research", "issn": "0093-5301", "category": "Marketing", "description": "JCR"},
    # 组织行为 (2本)
    {"name": "Academy of Management Review", "issn": "0363-7425", "category": "OB", "description": "AMR"},
    {"name": "Organization Science", "issn": "1047-7039", "category": "OB", "description": "OS"},
    # 经济学 (3本)
    {"name": "American Economic Review", "issn": "0002-8282", "category": "Economics", "description": "AER"},
    {"name": "Quarterly Journal of Economics", "issn": "0033-5533", "category": "Economics", "description": "QJE"},
    {"name": "Journal of Political Economy", "issn": "0022-3808", "category": "Economics", "description": "JPE"},
]

# 默认期刊列表（保持兼容）
DEFAULT_JOURNALS: Dict[str, List[Dict[str, str]]] = {
    # 国内期刊（中文为主）
    "CN": CN_JOURNALS_MANAGEMENT,
    # 国际期刊（英文为主）
    "INT": UTD_24_JOURNALS,
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
