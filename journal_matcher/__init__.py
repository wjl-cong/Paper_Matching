"""
期刊论文语义匹配系统 - 包初始化文件
====================================

导出主要组件，提供统一的导入接口

作者：AI Assistant
日期：2026-05-16
"""

from .config import config, Config
from .models.schemas import (
    Journal, JournalCreate,
    Paper, PaperSummary,
    SearchQuery, SearchResponse, JournalSearchResult, SimilarPaper,
    InitRequest, InitResponse,
    ErrorResponse, HealthResponse
)
from .services.database import DatabaseService
from .services.embedding import EmbeddingService, get_embedding_service
from .services.journal import JournalService, get_journal_service
from .services.search import SearchService, get_search_service

__version__ = "1.0.0"
__all__ = [
    # 配置
    "config",
    "Config",
    # 数据模型
    "Journal",
    "JournalCreate",
    "Paper",
    "PaperSummary",
    "SearchQuery",
    "SearchResponse",
    "JournalSearchResult",
    "SimilarPaper",
    "InitRequest",
    "InitResponse",
    "ErrorResponse",
    "HealthResponse",
    # 服务
    "DatabaseService",
    "EmbeddingService",
    "get_embedding_service",
    "JournalService",
    "get_journal_service",
    "SearchService",
    "get_search_service",
]
