"""
期刊论文语义匹配系统 - 语义检索服务
====================================

功能说明：
- 核心检索逻辑：两阶段检索 + 相似度计算
- 管理论文的向量化索引
- 按期刊分组返回检索结果
- 跨语言检索：自动翻译查询以匹配中英文文献

两阶段检索策略：
1. 第一阶段：向量召回（Vector Recall）
   - 使用FAISS向量索引快速检索Top-K候选
   - 速度快，但可能遗漏词汇差异大的相似论文

2. 第二阶段：Cross-Encoder重排序
   - 对Top-K候选精细化打分（可选，如模型不可用则跳过）
   - 准确度高，但计算量大

跨语言检索策略：
- 检测用户查询语言
- 如果是中文：同时用中文和翻译后的英文搜索
- 如果是英文：同时用英文和翻译后的中文搜索
- 合并去重后按相似度排序

作者：AI Assistant
日期：2026-05-16
"""

import time
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..models.schemas import Paper, Journal, SearchQuery, SearchResponse, JournalSearchResult, SimilarPaper
from ..services.database import DatabaseService
from ..services.embedding import EmbeddingService, get_embedding_service
from ..services.translation import get_translation_service
from ..config import config

# 最小论文年份（只保留近5年的论文）
MIN_PAPER_YEAR = datetime.now().year - 5


@dataclass
class SearchResult:
    """单条检索结果"""
    paper: Paper
    score: float
    rank: int


def extract_keywords_from_query(query: str) -> List[str]:
    """
    从查询中提取关键词用于过滤

    Args:
        query: 用户查询

    Returns:
        关键词列表
    """
    # 常见停用词
    stopwords = {"的", "在", "和", "是", "了", "于", "与", "对", "及", "应用", "研究",
                  "a", "an", "the", "of", "in", "on", "for", "and", "to", "with", "based", "using"}

    keywords = []

    # 提取英文单词
    english_words = re.findall(r'[a-zA-Z]+', query.lower())
    keywords.extend([w for w in english_words if w not in stopwords and len(w) > 2])

    # 提取中文词组（2-4个连续汉字）
    # 使用更精确的模式：提取连续的中文字符序列
    chinese_phrases = re.findall(r'[\u4e00-\u9fff]{2,4}', query)
    for phrase in chinese_phrases:
        if phrase not in stopwords:
            keywords.append(phrase)

    return keywords


def is_paper_relevant(paper: Paper, keywords: List[str], min_match: int = 1) -> bool:
    """
    判断论文是否与查询相关

    注意：在跨语言检索场景下，关键词精确匹配可能不适用。
    BGE-M3 已经在向量层面做了语义匹配，这里的关键词过滤作为辅助。

    Args:
        paper: 论文对象
        keywords: 查询关键词列表
        min_match: 最少匹配的关键词数（跨语言场景下设为0）

    Returns:
        True表示相关，False表示不相关
    """
    # 跨语言场景：关闭关键词过滤，依赖向量相似度
    # 如果需要开启，min_match 应设为 0
    if not keywords or min_match == 0:
        return True

    # 检查标题和摘要
    title = (paper.title or "").lower()
    abstract = (paper.abstract or "").lower()
    combined = title + " " + abstract

    match_count = 0
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in combined:
            match_count += 1

    return match_count >= min_match


def is_paper_recent(paper: Paper) -> bool:
    """
    判断论文是否是近期论文（近5年）

    Args:
        paper: 论文对象

    Returns:
        True表示是近期论文
    """
    if not paper.published_date:
        return True  # 没有日期的论文保留

    return paper.published_date.year >= MIN_PAPER_YEAR


class SearchService:
    """
    语义检索服务

    核心功能：
    1. 论文向量化索引管理
    2. 用户查询检索
    3. 结果按期刊分组
    4. 相似度阈值过滤

    使用示例：
        service = SearchService()
        response = service.search(
            query="深度学习在医学影像中的应用",
            threshold=0.5
        )
        for result in response.results:
            print(f"{result.journal.name}: {result.found}")
    """

    def __init__(self, db: DatabaseService = None,
                 embedding_service: EmbeddingService = None):
        """
        初始化检索服务

        Args:
            db: 数据库服务实例
            embedding_service: 向量化服务实例
        """
        self.db = db or DatabaseService()
        self.embedding = embedding_service or get_embedding_service()

        # 维护论文索引映射：FAISS索引位置 -> DOI
        # 用于将FAISS返回的索引转换为实际论文
        self._index_to_doi: Dict[int, str] = {}
        self._doi_to_paper: Dict[str, Paper] = {}

        # 加载已有索引
        self._load_index_mapping()

    def _load_index_mapping(self):
        """
        加载索引映射

        从数据库加载所有论文，按FAISS索引顺序构建映射表
        注意：只有在FAISS索引按DOI顺序构建时，这个映射才是正确的
        """
        papers = self.db.get_all_papers()
        for idx, paper in enumerate(papers):
            self._index_to_doi[idx] = paper.doi
            self._doi_to_paper[paper.doi] = paper
        print(f"已加载 {len(self._doi_to_paper)} 篇论文到内存索引")

    def index_papers(self, papers: List[Paper]) -> int:
        """
        将论文添加到向量索引

        对每篇论文：
        1. 获取或计算向量
        2. 保存到数据库缓存
        3. 更新内存索引映射

        Args:
            papers: Paper对象列表

        Returns:
            成功索引的论文数量
        """
        if not papers:
            return 0

        print(f"开始索引 {len(papers)} 篇论文...")

        vectors = []
        valid_papers = []
        dois_to_process = []

        # 阶段1：尝试从缓存获取向量
        dois = [p.doi for p in papers]
        cached_dois, cached_vectors = self.db.get_vectors_batch(dois)
        cached_map = dict(zip(cached_dois, cached_vectors))

        for paper in papers:
            cached_vec = cached_map.get(paper.doi)
            if cached_vec is not None:
                vectors.append(cached_vec)
                valid_papers.append(paper)
            else:
                dois_to_process.append(paper)

        # 阶段2：计算缺失的向量
        if dois_to_process:
            print(f"计算 {len(dois_to_process)} 篇新论文的向量...")
            texts = [p.get_searchable_text() for p in dois_to_process]

            new_vectors = self.embedding.encode(texts, normalize=True)

            for paper, vector in zip(dois_to_process, new_vectors):
                vectors.append(vector)
                valid_papers.append(paper)

                # 缓存向量
                self.db.save_vector(paper.doi, vector)

        # 阶段3：构建/更新FAISS索引
        if vectors:
            vectors_array = np.array(vectors)

            if self.db.is_index_ready():
                # 如果已有索引，需要重建（FAISS不支持动态添加）
                # 注意：这意味着每次添加新论文都需要重建索引
                print("重建FAISS索引...")
                all_papers = self.db.get_all_papers()

                # 重新获取所有向量（包括缓存和新计算的）
                all_dois, all_vectors = self.db.get_vectors_batch([p.doi for p in all_papers])
                all_vector_map = dict(zip(all_dois, all_vectors))

                # 重新构建索引
                all_vectors_list = []
                all_papers_ordered = []
                for paper in all_papers:
                    if paper.doi in all_vector_map:
                        all_vectors_list.append(all_vector_map[paper.doi])
                        all_papers_ordered.append(paper)

                if all_vectors_list:
                    self.db.build_index(all_papers_ordered, np.array(all_vectors_list))
            else:
                # 新建索引
                self.db.build_index(valid_papers, vectors_array)

            # 更新内存映射
            self._index_to_doi = {}
            self._doi_to_paper = {}
            all_papers = self.db.get_all_papers()
            for idx, paper in enumerate(all_papers):
                self._index_to_doi[idx] = paper.doi
                self._doi_to_paper[paper.doi] = paper

        print(f"索引完成！总计 {self.db.get_index_size()} 篇论文")
        return len(valid_papers)

    def search(self, query: str,
               threshold: float = None,
               top_k: int = None,
               journals: List[Journal] = None) -> SearchResponse:
        """
        执行跨语言语义检索

        完整流程：
        1. 检测查询语言
        2. 如果是中文，同时搜索中文和翻译的英文
        3. 如果是英文，同时搜索英文和翻译的中文
        4. 向量化查询
        5. 在FAISS中检索Top-K候选
        6. 过滤和重排序
        7. 按期刊分组
        8. 应用阈值过滤

        Args:
            query: 用户输入的检索主题
            threshold: 相似度阈值（低于此值的结果被过滤）
            top_k: 每期刊返回的最大论文数
            journals: 要检索的期刊列表（None表示检索所有）

        Returns:
            SearchResponse对象，包含完整的检索结果
        """
        start_time = time.time()

        # 使用默认值
        if threshold is None:
            threshold = config.retrieval.SIMILARITY_THRESHOLD
        if top_k is None:
            top_k = config.retrieval.TOP_N_RERANK

        # 获取要检索的期刊
        if journals is None:
            journals = self.db.get_all_journals()

        # 检测查询语言并翻译
        translator = get_translation_service()
        query_lang = translator.detect_language(query)
        queries_to_search = [query]

        print(f"检测到查询语言: {'中文' if query_lang == 'zh' else '英文'}")

        if query_lang == "zh":
            # 中文查询：翻译成英文搜索
            translated = translator.zh_to_en(query)
            if translated and translated != query:
                print(f"翻译为英文: {translated}")
                queries_to_search.append(translated)
        else:
            # 英文查询：翻译成中文搜索
            translated = translator.en_to_zh(query)
            if translated and translated != query:
                print(f"翻译为中文: {translated}")
                queries_to_search.append(translated)

        # 对所有查询进行向量检索并合并结果
        all_search_results: Dict[str, SearchResult] = {}

        # 提取查询关键词用于过滤
        keywords = extract_keywords_from_query(query)
        print(f"提取关键词: {keywords}")

        for q in queries_to_search:
            print(f"正在检索: {q}")
            query_vector = self.embedding.encode_query(q)

            # 向量检索
            top_k_recall = config.retrieval.TOP_K_RECALL
            similarities, indices = self.db.search_index(query_vector, top_k=top_k_recall)

            # 合并结果（去重，保留最高分），同时进行关键词和年份过滤
            for idx, sim in zip(indices, similarities):
                if idx < 0:
                    continue

                doi = self._index_to_doi.get(int(idx))
                if doi and doi in self._doi_to_paper:
                    paper = self._doi_to_paper[doi]

                    # 关键词过滤：跨语言场景下关闭 (min_match=0)
                    # 依赖向量相似度判断，BGE-M3 已做语义匹配
                    if not is_paper_relevant(paper, keywords, min_match=0):
                        continue

                    # 年份过滤：只保留近5年论文
                    if not is_paper_recent(paper):
                        continue

                    # 如果DOI已存在，保留较高分数
                    if doi in all_search_results:
                        if sim > all_search_results[doi].score:
                            all_search_results[doi].score = float(sim)
                    else:
                        all_search_results[doi] = SearchResult(
                            paper=paper,
                            score=float(sim),
                            rank=len(all_search_results)
                        )

        # 转换为列表并按相似度排序
        search_results = sorted(all_search_results.values(), key=lambda x: x.score, reverse=True)
        print(f"初步检索到 {len(search_results)} 篇相关论文")

        # 按期刊分组
        journal_results: Dict[str, List[SearchResult]] = {j.issn: [] for j in journals}
        for result in search_results:
            issn = result.paper.journal_issn
            if issn in journal_results:
                journal_results[issn].append(result)

        # 步骤5：构建响应
        results: List[JournalSearchResult] = []
        found_count = 0
        total_papers = self.db.get_papers_count()

        for journal in journals:
            journal_papers = journal_results.get(journal.issn, [])

            # 按相似度排序并应用阈值
            relevant_papers = [
                r for r in sorted(journal_papers, key=lambda x: x.score, reverse=True)
                if r.score >= threshold
            ][:top_k]

            # 构建SimilarPaper对象
            similar_papers = []
            for result in relevant_papers:
                # 提取摘要片段
                abstract_snippet = None
                if result.paper.abstract:
                    # 截取前200个字符作为摘要片段
                    abstract_snippet = result.paper.abstract[:200] + "..."

                similar_papers.append(SimilarPaper(
                    paper=result.paper,
                    score=result.score,
                    abstract_snippet=abstract_snippet
                ))

            found = len(relevant_papers) > 0
            if found:
                found_count += 1

            # 获取期刊论文总数
            journal_papers_count = self.db.get_journal_papers_count(journal.issn)

            results.append(JournalSearchResult(
                journal=journal,
                found=found,
                similar_papers=similar_papers,
                total_papers=journal_papers_count
            ))

        # 计算耗时
        search_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            status="success",
            query=query,
            total_journals=len(journals),
            total_papers=total_papers,
            found_count=found_count,
            results=results,
            search_time_ms=search_time_ms
        )

    def search_single_journal(self, issn: str, query: str,
                              threshold: float = None,
                              top_k: int = None) -> JournalSearchResult:
        """
        检索单个期刊

        便捷方法，用于检索单个期刊的相似论文

        Args:
            issn: 期刊ISSN
            query: 检索查询
            threshold: 相似度阈值
            top_k: 返回数量上限

        Returns:
            JournalSearchResult对象
        """
        journal = self.db.get_journal(issn)
        if not journal:
            return JournalSearchResult(
                journal=Journal(issn=issn, name="Unknown"),
                found=False,
                similar_papers=[],
                total_papers=0
            )

        response = self.search(
            query=query,
            threshold=threshold,
            top_k=top_k,
            journals=[journal]
        )

        return response.results[0] if response.results else JournalSearchResult(
            journal=journal,
            found=False,
            similar_papers=[],
            total_papers=0
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取检索服务统计信息

        Returns:
            包含各种统计数据的字典
        """
        return {
            "total_papers_indexed": len(self._doi_to_paper),
            "faiss_index_size": self.db.get_index_size(),
            "vector_dim": self.embedding.embedding_dim,
            "model_name": self.embedding.model_name,
            "is_ready": self.db.is_index_ready()
        }


# =============================================================================
# 全局单例
# =============================================================================

_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """获取检索服务的全局单例"""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service


def reset_search_service():
    """重置检索服务（用于重新初始化）"""
    global _search_service
    _search_service = None
