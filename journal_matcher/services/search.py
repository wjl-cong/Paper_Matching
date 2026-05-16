"""
期刊论文语义匹配系统 - 语义检索服务
====================================

功能说明：
- 核心检索逻辑：两阶段检索 + 相似度计算
- 管理论文的向量化索引
- 按期刊分组返回检索结果
- 跨语言检索：自动翻译查询以匹配中英文文献
- 中文语义增强：集成 OpenHowNet 义原知识库和 jieba 分词

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

中文语义增强策略：
- 使用 jieba 进行精准中文分词
- 使用 OpenHowNet 获取义原标注
- 基于义原扩展查询词（同义词/相关词）
- 增强中文 BM25 检索效果

作者：AI Assistant
日期：2026-05-16
"""

import time
import re
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..models.schemas import Paper, Journal, SearchQuery, SearchResponse, JournalSearchResult, SimilarPaper
from ..services.database import DatabaseService
from ..services.embedding import EmbeddingService, get_embedding_service
from ..services.translation import get_translation_service
from ..config import config

# 尝试导入中文增强模块
try:
    from .tokenizer import ChineseTokenizer, get_tokenizer, create_academic_tokenizer
    from .hownet_service import get_hownet_service, ALL_CHINESE_STOPWORDS
    CHINESE_SUPPORT_AVAILABLE = True
except ImportError:
    CHINESE_SUPPORT_AVAILABLE = False
    ALL_CHINESE_STOPWORDS = set()

# 最小论文年份（只保留近5年的论文）
MIN_PAPER_YEAR = datetime.now().year - 5


@dataclass
class SearchResult:
    """单条检索结果"""
    paper: Paper
    score: float
    rank: int


def extract_keywords_from_query(query: str,
                                use_jieba: bool = True,
                                use_hownet: bool = False) -> List[str]:
    """
    从查询中提取关键词用于过滤/检索

    Args:
        query: 用户查询
        use_jieba: 是否使用 jieba 分词（中文效果更好）
        use_hownet: 是否使用 OpenHowNet 扩展（同义词扩展）

    Returns:
        关键词列表
    """
    # 合并停用词（英文 + 中文）
    stopwords: Set[str] = {"a", "an", "the", "of", "in", "on", "for", "and", "to",
                           "with", "based", "using"}  # 英文停用词
    if CHINESE_SUPPORT_AVAILABLE:
        stopwords |= ALL_CHINESE_STOPWORDS

    keywords = []

    # 提取英文单词
    english_words = re.findall(r'[a-zA-Z]+', query.lower())
    keywords.extend([w for w in english_words if w not in stopwords and len(w) > 2])

    # 中文分词：优先使用 jieba
    if CHINESE_SUPPORT_AVAILABLE and use_jieba:
        try:
            import jieba
            chinese_tokens = list(jieba.cut(query))
            for token in chinese_tokens:
                # 过滤：长度>=2、非停用词、包含中文字符
                if (len(token) >= 2 and
                    token not in stopwords and
                    re.search(r'[\u4e00-\u9fff]', token)):
                    keywords.append(token)
        except Exception:
            # 备用：使用正则
            chinese_phrases = re.findall(r'[\u4e00-\u9fff]{2,4}', query)
            for phrase in chinese_phrases:
                if phrase not in stopwords:
                    keywords.append(phrase)
    else:
        # 提取中文词组（2-4个连续汉字）
        chinese_phrases = re.findall(r'[\u4e00-\u9fff]{2,4}', query)
        for phrase in chinese_phrases:
            if phrase not in stopwords:
                keywords.append(phrase)

    # OpenHowNet 扩展：添加同义词/相关词
    if CHINESE_SUPPORT_AVAILABLE and use_hownet:
        hownet = get_hownet_service(init_sim=False)
        if hownet.is_available():
            expanded = hownet.expand_query_with_sememes(query, max_words=30)
            # 添加扩展词（去重）
            for word in expanded:
                if word not in keywords:
                    keywords.append(word)

    return list(set(keywords))  # 去重


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


# =============================================================================
# BM25 精确搜索相关
# =============================================================================

import math
from collections import Counter

class BM25:
    """
    BM25 排序算法实现

    用于精确搜索：在篇名+关键词+摘要中进行关键词匹配

    支持中文分词：
    - 优先使用 jieba（需要安装）
    - 备用正则分词
    - 支持 OpenHowNet 义原扩展（可选）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 use_jieba: bool = True, use_hownet: bool = False):
        """
        初始化 BM25

        Args:
            k1: BM25参数，控制词频饱和度
            b: BM25参数，控制文档长度归一化
            use_jieba: 是否使用 jieba 分词
            use_hownet: 是否使用 OpenHowNet 扩展
        """
        self.k1 = k1
        self.b = b
        self.use_jieba = use_jieba and CHINESE_SUPPORT_AVAILABLE
        self.use_hownet = use_hownet and CHINESE_SUPPORT_AVAILABLE
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}  # term -> doc frequency
        self.N = 0  # total documents

        # OpenHowNet 扩展词缓存
        self._hownet_expanded: Dict[str, List[str]] = {}

    def _tokenize(self, text: str) -> list:
        """
        分词：支持中英文混合分词

        Args:
            text: 待分词文本

        Returns:
            分词列表
        """
        if not text:
            return []

        tokens = []

        # 提取英文单词
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        tokens.extend(english_words)

        # 中文分词
        chinese_text = re.sub(r'[a-zA-Z]+', ' ', text)  # 移除英文，只保留中文
        if chinese_text.strip():
            if self.use_jieba:
                try:
                    import jieba
                    chinese_tokens = list(jieba.cut(chinese_text))
                    # 过滤单字和空白
                    chinese_tokens = [t for t in chinese_tokens if len(t) >= 2 and re.search(r'[\u4e00-\u9fff]', t)]
                    tokens.extend(chinese_tokens)
                except Exception:
                    # 备用：正则分词
                    tokens.extend(re.findall(r'[\u4e00-\u9fff]{2,4}', chinese_text))
            else:
                # 提取中文词组（2-4个连续汉字）
                tokens.extend(re.findall(r'[\u4e00-\u9fff]{2,4}', chinese_text))

        return tokens

    def _expand_query(self, query: str) -> str:
        """
        使用 OpenHowNet 扩展查询词

        Args:
            query: 原始查询

        Returns:
            扩展后的查询字符串
        """
        if not self.use_hownet:
            return query

        if query in self._hownet_expanded:
            return self._hownet_expanded[query]

        expanded = [query]
        hownet = get_hownet_service(init_sim=False)

        if hownet.is_available():
            # 使用 jieba 分词提取关键词
            try:
                import jieba
                words = [w for w in jieba.cut(query) if len(w) >= 2]
            except Exception:
                words = re.findall(r'[\u4e00-\u9fff]{2,4}', query)

            for word in words:
                sememes = hownet.get_word_sememes(word)
                for sem_data in sememes[:1]:  # 只取第一个义项
                    # 添加义原作为扩展
                    expanded.extend(sem_data.sememes[:2])
                    expanded.extend(sem_data.sememes_zh[:2])

        result = " ".join(expanded)
        self._hownet_expanded[query] = result
        return result

    def _get_word_freq(self, text: str) -> Counter:
        """获取词频"""
        return Counter(self._tokenize(text))

    def fit(self, documents: list):
        """
        构建BM25索引

        Args:
            documents: 文档列表，每个元素是字符串
        """
        self.documents = documents
        self.N = len(documents)
        self.doc_freqs = {}

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            # 统计词频
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_length = sum(self.doc_lengths) / self.N if self.N > 0 else 0

    def score(self, query: str, doc_idx: int) -> float:
        """
        计算query与单个文档的BM25分数

        Args:
            query: 查询字符串
            doc_idx: 文档索引

        Returns:
            BM25分数
        """
        if self.N == 0 or self.avg_doc_length == 0:
            return 0.0

        # 使用 OpenHowNet 扩展查询（可选）
        if self.use_hownet:
            query = self._expand_query(query)

        query_terms = self._tokenize(query)
        if not query_terms:
            return 0.0

        doc_text = self.documents[doc_idx]
        doc_freqs = self._get_word_freq(doc_text)
        doc_len = self.doc_lengths[doc_idx]

        score = 0.0
        for term in query_terms:
            if term in doc_freqs:
                tf = doc_freqs[term]
                df = self.doc_freqs.get(term, 0)

                # IDF
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

                # TF component
                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length))

                score += idf * tf_component

        return score

    def search(self, query: str, top_k: int = 50) -> list:
        """
        搜索返回top_k个文档索引和分数

        Returns:
            list of (doc_idx, score)
        """
        # 使用 OpenHowNet 扩展查询
        if self.use_hownet:
            query = self._expand_query(query)

        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


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
                 embedding_service: EmbeddingService = None,
                 use_jieba: bool = True,
                 use_hownet: bool = False):
        """
        初始化检索服务

        Args:
            db: 数据库服务实例
            embedding_service: 向量化服务实例
            use_jieba: 是否使用 jieba 分词（中文效果更好）
            use_hownet: 是否使用 OpenHowNet 义原扩展
        """
        self.db = db or DatabaseService()
        self.embedding = embedding_service or get_embedding_service()

        # 中文增强选项
        self.use_jieba = use_jieba and CHINESE_SUPPORT_AVAILABLE
        self.use_hownet = use_hownet and CHINESE_SUPPORT_AVAILABLE

        if self.use_hownet:
            print("启用 OpenHowNet 义原扩展（中文语义增强）")
        if self.use_jieba:
            print("启用 jieba 中文分词")

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

    def _build_bm25_index(self):
        """构建BM25索引（精确搜索用）"""
        papers = self.db.get_all_papers()
        # 构建可搜索文本：篇名 + 关键词 + 摘要
        documents = []
        for paper in papers:
            parts = [paper.title]
            if paper.keywords:
                parts.append(" ".join(paper.keywords))
            if paper.abstract:
                parts.append(paper.abstract)
            documents.append(" | ".join(parts))

        # 使用中文增强参数构建 BM25
        self._bm25 = BM25(use_jieba=self.use_jieba, use_hownet=self.use_hownet)
        self._bm25.fit(documents)
        print(f"BM25索引构建完成，共 {len(documents)} 篇文档")

        if self.use_hownet:
            print("BM25 已启用 OpenHowNet 义原扩展")

    def _exact_search(self, query: str, threshold: float = 0.1) -> Dict[str, SearchResult]:
        """
        精确搜索：使用BM25在篇名+关键词+摘要中匹配

        Args:
            query: 查询字符串
            threshold: 最小分数阈值

        Returns:
            Dict[str, SearchResult]: DOI -> SearchResult
        """
        if not hasattr(self, '_bm25') or self._bm25 is None:
            self._build_bm25_index()

        papers = self.db.get_all_papers()
        bm25_results = self._bm25.search(query, top_k=100)

        results = {}
        for doc_idx, score in bm25_results:
            if doc_idx < len(papers):
                paper = papers[doc_idx]
                # 归一化分数到0-1范围
                # BM25原始分数通常较大，按最大值归一化更合理
                # 取当前批次最高分来归一化，避免阈值过高过滤掉所有结果
                normalized_score = min(score / 10.0, 1.0)
                if normalized_score >= threshold:
                    results[paper.doi] = SearchResult(
                        paper=paper,
                        score=normalized_score,
                        rank=len(results)
                    )

        return results

    def _semantic_search(self, query: str, threshold: float = 0.1) -> Dict[str, SearchResult]:
        """
        语义搜索：使用BGE-M3向量匹配

        Args:
            query: 查询字符串
            threshold: 最小分数阈值

        Returns:
            Dict[str, SearchResult]: DOI -> SearchResult
        """
        # 检测查询语言并翻译
        translator = get_translation_service()
        query_lang = translator.detect_language(query)
        queries_to_search = [query]

        print(f"检测到查询语言: {'中文' if query_lang == 'zh' else '英文'}")

        if query_lang == "zh":
            translated = translator.zh_to_en(query)
            if translated and translated != query:
                print(f"翻译为英文: {translated}")
                queries_to_search.append(translated)
        else:
            translated = translator.en_to_zh(query)
            if translated and translated != query:
                print(f"翻译为中文: {translated}")
                queries_to_search.append(translated)

        all_results = {}

        for q in queries_to_search:
            print(f"正在检索: {q}")
            query_vector = self.embedding.encode_query(q)

            # 向量检索
            top_k_recall = config.retrieval.TOP_K_RECALL
            similarities, indices = self.db.search_index(query_vector, top_k=top_k_recall)

            for idx, sim in zip(indices, similarities):
                if idx < 0:
                    continue

                doi = self._index_to_doi.get(int(idx))
                if doi and doi in self._doi_to_paper:
                    paper = self._doi_to_paper[doi]

                    # 年份过滤
                    if not is_paper_recent(paper):
                        continue

                    # 如果DOI已存在，保留较高分数
                    if doi in all_results:
                        if sim > all_results[doi].score:
                            all_results[doi].score = float(sim)
                    else:
                        all_results[doi] = SearchResult(
                            paper=paper,
                            score=float(sim),
                            rank=len(all_results)
                        )

        # 应用阈值
        return {doi: r for doi, r in all_results.items() if r.score >= threshold}

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
               journals: List[Journal] = None,
               mode: str = "semantic") -> SearchResponse:
        """
        执行检索（支持精确搜索和语义搜索）

        Args:
            query: 用户输入的检索主题
            threshold: 相似度阈值（低于此值的结果被过滤）
            top_k: 每期刊返回的最大论文数
            journals: 要检索的期刊列表（None表示检索所有）
            mode: 搜索模式，"exact"=精确搜索(BM25)，"semantic"=语义搜索(BGE-M3)，"both"=综合

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

        print(f"检索模式: {'精确搜索' if mode == 'exact' else '语义搜索' if mode == 'semantic' else '综合搜索'}")

        # 根据模式执行搜索
        if mode == "exact":
            all_search_results = self._exact_search(query, threshold)
        elif mode == "semantic":
            all_search_results = self._semantic_search(query, threshold)
        else:  # "both"
            # 综合搜索：合并精确和语义结果
            exact_results = self._exact_search(query, threshold)
            semantic_results = self._semantic_search(query, threshold)
            # 合并去重，语义分数归一化到0-1
            all_search_results = exact_results.copy()

            # 获取语义结果中的最大分数用于归一化
            if semantic_results:
                max_semantic = max(r.score for r in semantic_results.values())
            else:
                max_semantic = 1.0

            for doi, r in semantic_results.items():
                normalized_score = r.score / max_semantic if max_semantic > 0 else 0
                if doi in all_search_results:
                    all_search_results[doi].score = max(
                        all_search_results[doi].score, normalized_score
                    )
                else:
                    all_search_results[doi] = SearchResult(
                        paper=r.paper,
                        score=normalized_score,
                        rank=r.rank
                    )

        search_results = sorted(all_search_results.values(), key=lambda x: x.score, reverse=True)
        print(f"检索到 {len(search_results)} 篇相关论文")

        # 获取要检索的期刊
        if journals is None:
            journals = self.db.get_all_journals()

        # 按期刊分组（如果有期刊配置）
        if journals:
            journal_results: Dict[str, List[SearchResult]] = {j.issn: [] for j in journals}
            for result in search_results:
                issn = result.paper.journal_issn
                if issn in journal_results:
                    journal_results[issn].append(result)
        else:
            # 没有期刊配置时，按论文的期刊名称分组
            journal_results = {}
            for result in search_results:
                journal_name = result.paper.journal_name or "Unknown"
                if journal_name not in journal_results:
                    journal_results[journal_name] = []
                journal_results[journal_name].append(result)

        # 步骤5：构建响应
        results: List[JournalSearchResult] = []
        found_count = 0
        total_papers = self.db.get_papers_count()

        if journals:
            # 有期刊配置时的处理
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
                    abstract_snippet = None
                    if result.paper.abstract:
                        abstract_snippet = result.paper.abstract[:500]
                        if len(result.paper.abstract) > 500:
                            abstract_snippet += "..."

                    similar_papers.append(SimilarPaper(
                        paper=result.paper,
                        score=result.score,
                        abstract_snippet=abstract_snippet
                    ))

                found = len(relevant_papers) > 0
                if found:
                    found_count += 1

                journal_papers_count = self.db.get_journal_papers_count(journal.issn)

                results.append(JournalSearchResult(
                    journal=journal,
                    found=found,
                    similar_papers=similar_papers,
                    total_papers=journal_papers_count
                ))
        else:
            # 没有期刊配置时，按期刊名称分组返回
            for journal_name, journal_papers in journal_results.items():
                # 按相似度排序并应用阈值
                relevant_papers = [
                    r for r in sorted(journal_papers, key=lambda x: x.score, reverse=True)
                    if r.score >= threshold
                ][:top_k]

                if not relevant_papers:
                    continue

                found_count += 1

                # 构建SimilarPaper对象
                similar_papers = []
                for result in relevant_papers:
                    abstract_snippet = None
                    if result.paper.abstract:
                        abstract_snippet = result.paper.abstract[:500]
                        if len(result.paper.abstract) > 500:
                            abstract_snippet += "..."

                    similar_papers.append(SimilarPaper(
                        paper=result.paper,
                        score=result.score,
                        abstract_snippet=abstract_snippet
                    ))

                results.append(JournalSearchResult(
                    journal=Journal(issn="", name=journal_name, country="CN"),
                    found=True,
                    similar_papers=similar_papers,
                    total_papers=len(journal_papers)
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
