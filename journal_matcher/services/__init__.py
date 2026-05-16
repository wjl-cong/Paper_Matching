"""
服务层初始化文件
"""

from .database import DatabaseService
from .embedding import EmbeddingService, get_embedding_service
from .journal import JournalService, get_journal_service
from .search import SearchService, get_search_service

# 中文增强模块（可选）
try:
    from .hownet_service import HowNetService, get_hownet_service
    from .tokenizer import ChineseTokenizer, get_tokenizer, create_academic_tokenizer
    from .chinese_journal import ChineseJournalService, get_chinese_journal_service
    CHINESE_SUPPORT = True
except ImportError:
    CHINESE_SUPPORT = False

__all__ = [
    "DatabaseService",
    "EmbeddingService",
    "get_embedding_service",
    "JournalService",
    "get_journal_service",
    "SearchService",
    "get_search_service",
    # 中文增强模块
    "CHINESE_SUPPORT",
]
