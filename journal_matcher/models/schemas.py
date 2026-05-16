"""
期刊论文语义匹配系统 - 数据模型定义
====================================

功能说明：
- 定义论文、期刊、检索结果等核心数据结构的 Pydantic 模型
- 提供数据验证、序列化/反序列化功能
- 确保API请求和响应的数据类型安全

作者：AI Assistant
日期：2026-05-16
"""

from datetime import datetime, date
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# 期刊相关模型
# =============================================================================

class Journal(BaseModel):
    """
    期刊数据模型

    代表一本学术期刊的基本信息

    属性说明：
    - issn: 国际标准连续出版物号，期刊的唯一标识符
    - name: 期刊名称
    - country: 国家代码，"CN"表示中国，"INT"表示国际
    - description: 期刊描述/简介
    - last_fetched: 最后一次抓取论文的时间
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    issn: str = Field(..., description="ISSN国际标准连续出版物号")
    name: str = Field(..., description="期刊名称")
    country: str = Field(default="INT", description="国家代码，CN=中国，INT=国际")
    description: Optional[str] = Field(None, description="期刊简介")
    last_fetched: Optional[datetime] = Field(None, description="最后抓取时间")


class JournalCreate(BaseModel):
    """创建期刊的请求模型"""
    issn: str = Field(..., description="ISSN")
    name: str = Field(..., description="期刊名称")
    country: str = Field(default="INT", description="国家代码")
    description: Optional[str] = None


# =============================================================================
# 论文相关模型
# =============================================================================

class Paper(BaseModel):
    """
    论文数据模型

    代表一篇学术论文的完整元数据

    属性说明：
    - doi: 数字对象唯一标识符，是论文的全球唯一标识
    - title: 论文标题
    - abstract: 论文摘要，可能为空
    - authors: 作者列表，JSON格式存储
    - journal_issn: 发表期刊的ISSN
    - journal_name: 发表期刊的名称（冗余存储，方便展示）
    - published_date: 发表日期
    - url: 论文链接
    - keywords: 关键词列表
    - language: 论文语言，"zh"表示中文，"en"表示英文
    - indexed_at: 被系统索引的时间
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    doi: str = Field(..., description="DOI数字对象唯一标识符")
    title: str = Field(..., description="论文标题")
    abstract: Optional[str] = Field(None, description="论文摘要")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    journal_issn: str = Field(..., description="期刊ISSN")
    journal_name: str = Field(..., description="期刊名称")
    published_date: Optional[date] = Field(None, description="发表日期")
    url: Optional[str] = Field(None, description="论文链接")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    language: str = Field(default="en", description="语言，zh=中文，en=英文")
    indexed_at: datetime = Field(default_factory=datetime.now, description="索引时间")

    def get_searchable_text(self) -> str:
        """
        获取论文的可搜索文本

        用于构建向量化检索的输入文本。
        按优先级组合标题、摘要、关键词，以最大化语义信息量。

        Returns:
            str: 拼接后的可搜索文本
        """
        parts = [self.title]

        # 优先使用英文摘要（跨语言检索效果更好）
        if self.abstract:
            parts.append(self.abstract)

        # 添加关键词增强检索
        if self.keywords:
            parts.append("Keywords: " + ", ".join(self.keywords))

        return " | ".join(parts)

    def is_complete(self) -> bool:
        """
        检查论文数据是否完整

        论文数据完整性影响向量化效果：
        - 有摘要的论文向量化效果最好
        - 无摘要但有标题和关键词的论文可接受
        - 只有标题的论文向量化效果有限

        Returns:
            bool: 数据是否完整（有摘要）
        """
        return bool(self.abstract and len(self.abstract) > 50)


class PaperSummary(BaseModel):
    """
    论文摘要模型（用于API响应）

    只包含论文的关键信息，减小响应体积
    """
    doi: str
    title: str
    journal_name: str
    published_date: Optional[date] = None
    url: Optional[str] = None

    @classmethod
    def from_paper(cls, paper: Paper) -> "PaperSummary":
        """从Paper模型创建PaperSummary"""
        return cls(
            doi=paper.doi,
            title=paper.title,
            journal_name=paper.journal_name,
            published_date=paper.published_date,
            url=paper.url
        )


# =============================================================================
# 检索相关模型
# =============================================================================

class SearchQuery(BaseModel):
    """
    检索请求模型

    用户输入的检索查询

    属性说明：
    - query: 用户输入的检索主题（自然语言）
    - top_k: 每期刊返回的最大论文数
    - threshold: 相似度阈值，低于此值的论文不返回
    """
    query: str = Field(..., description="检索主题/查询语句", min_length=1, max_length=500)
    top_k: int = Field(default=5, description="每期刊返回的最大论文数", ge=1, le=20)
    threshold: float = Field(default=0.5, description="相似度阈值", ge=0.0, le=1.0)


class SimilarPaper(BaseModel):
    """
    相似论文模型

    代表一篇与用户查询相似的论文

    属性说明：
    - paper: 论文摘要信息
    - score: 相似度分数，0.0~1.0，越高越相似
    - abstract_snippet: 摘要片段（便于用户快速浏览）
    """
    paper: PaperSummary
    score: float = Field(..., description="相似度分数，0.0~1.0")
    abstract_snippet: Optional[str] = Field(None, description="摘要片段")

    def __init__(self, **data):
        """重写初始化，支持从Paper对象直接创建"""
        if "paper" in data and isinstance(data["paper"], Paper):
            data["paper"] = PaperSummary.from_paper(data["paper"])
        super().__init__(**data)


class JournalSearchResult(BaseModel):
    """
    期刊检索结果模型

    某个期刊的检索结果

    属性说明：
    - journal: 期刊基本信息
    - found: 是否找到相似论文
    - similar_papers: 相似论文列表
    - total_papers: 该期刊的总论文数
    """
    journal: Journal
    found: bool = Field(..., description="是否找到相似论文")
    similar_papers: List[SimilarPaper] = Field(default_factory=list, description="相似论文列表")
    total_papers: int = Field(default=0, description="该期刊总论文数")


# =============================================================================
# API响应模型
# =============================================================================

class InitRequest(BaseModel):
    """
    系统初始化请求模型

    指定要初始化的期刊列表
    """
    journal_configs: Optional[List[Dict[str, str]]] = Field(
        None,
        description="期刊配置列表，格式：[{'issn': 'xxx', 'name': 'xxx'}, ...]"
    )
    use_default: bool = Field(default=True, description="是否使用默认期刊列表")


class InitResponse(BaseModel):
    """
    系统初始化响应模型

    返回初始化结果统计
    """
    status: str = Field(..., description="状态：success/failed/partial")
    message: str = Field(..., description="状态消息")
    journals_count: int = Field(default=0, description="配置的期刊数量")
    papers_count: int = Field(default=0, description="抓取的论文总数")
    vectorized_count: int = Field(default=0, description="已完成向量化的论文数")
    errors: List[str] = Field(default_factory=list, description="错误信息列表")


class SearchResponse(BaseModel):
    """
    检索响应模型

    返回完整的检索结果
    """
    status: str = Field(..., description="状态：success/failed")
    query: str = Field(..., description="原始查询")
    total_journals: int = Field(default=0, description="检索的期刊总数")
    total_papers: int = Field(default=0, description="检索的论文总数")
    found_count: int = Field(default=0, description="找到相似论文的期刊数")
    results: List[JournalSearchResult] = Field(default_factory=list, description="各期刊检索结果")
    search_time_ms: float = Field(default=0.0, description="检索耗时（毫秒）")


class JournalListResponse(BaseModel):
    """期刊列表响应模型"""
    journals: List[Journal] = Field(default_factory=list, description="期刊列表")
    total: int = Field(default=0, description="期刊总数")


class PaperListResponse(BaseModel):
    """期刊论文列表响应模型"""
    journal: Journal
    papers: List[PaperSummary] = Field(default_factory=list, description="论文列表")
    total: int = Field(default=0, description="论文总数")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="版本号")
    database_connected: bool = Field(..., description="数据库是否连接")
    vector_index_ready: bool = Field(..., description="向量索引是否就绪")
    embedding_model_ready: bool = Field(..., description="向量化模型是否就绪")


# =============================================================================
# 错误处理模型
# =============================================================================

class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")
