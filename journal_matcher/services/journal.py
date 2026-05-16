"""
期刊论文语义匹配系统 - 期刊论文获取服务
====================================

功能说明：
- 从多个数据源（OpenAlex、CrossRef、Semantic Scholar）获取期刊论文
- 处理数据格式差异，统一输出格式
- 实现多层降级策略，保证数据获取的可靠性

核心挑战：
- 国内期刊：CNKI需要机构授权，使用OpenAlex作为替代
- 国外期刊：通过OpenAlex API获取主流国际期刊
- 数据补全：多源交叉验证，确保字段完整

作者：AI Assistant
日期：2026-05-16
"""

import time
import random
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import quote
import requests
import logging

from ..models.schemas import Paper, Journal
from ..config import config

# 只获取近N年的论文，提高相关性
MIN_PAPER_YEAR = datetime.now().year - 5

# API 请求配置
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒
REQUEST_DELAY = 1  # 请求间隔（秒）

logger = logging.getLogger(__name__)


class JournalService:
    """
    期刊论文获取服务

    从多个学术数据库API获取期刊论文数据

    数据源优先级：
    1. OpenAlex（主数据源，数据全面、更新及时）
    2. CrossRef（备选数据源，元数据标准）
    3. Semantic Scholar（补全数据源，提供摘要等）

    设计思路：
    - 统一抽象各数据源的接口
    - 自动处理API限流和错误
    - 缓存已获取的数据避免重复请求
    """

    def __init__(self):
        """初始化期刊服务"""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"Research-Tool/1.0 (mailto:{config.api.CROSSREF_EMAIL})",
            "Accept": "application/json"
        })
        self._last_request_time = {}  # 记录每个API上次请求时间

    def _throttled_request(self, api_name: str, func, *args, **kwargs):
        """
        带限流和重试的请求封装

        Args:
            api_name: API名称（用于区分不同API的限流）
            func: 请求函数
            *args, **kwargs: 传递给func的参数

        Returns:
            func的返回值
        """
        # 检查距离上次请求的时间
        if api_name in self._last_request_time:
            elapsed = time.time() - self._last_request_time[api_name]
            if elapsed < REQUEST_DELAY:
                time.sleep(REQUEST_DELAY - elapsed)

        # 重试机制
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                result = func(*args, **kwargs)
                self._last_request_time[api_name] = time.time()
                return result
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"  请求失败，{wait_time}秒后重试 ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                else:
                    print(f"  请求最终失败: {e}")

        raise last_error if last_error else Exception("Unknown error")

    # =========================================================================
    # OpenAlex API 相关方法
    # =========================================================================

    def fetch_papers_from_openalex(self, issn: str, journal_name: str,
                                   max_papers: int = 1,
                                   language: str = None) -> List[Paper]:
        """
        从OpenAlex获取期刊论文

        OpenAlex是目前最全面的学术文献数据库之一，包含：
        - 2亿+学术作品
        - 丰富的元数据（作者、期刊、机构等）
        - 支持按ISSN、关键词等多种方式检索
        - 支持按语言筛选（language:zh 获取中文论文）
        - 提供免费API

        Args:
            issn: 期刊ISSN
            journal_name: 期刊名称（用于填充journal_name字段）
            max_papers: 最大获取论文数量
            language: 语言筛选（如"zh"获取中文论文，"en"获取英文论文）

        Returns:
            Paper对象列表
        """
        papers = []
        page = 1
        per_page = min(200, max_papers)  # OpenAlex每页最大200条

        while len(papers) < max_papers:
            # 构建请求URL
            # OpenAlex filter语法：
            # - 同一字段多个值用逗号分隔（OR关系）
            # - 不同字段用分号分隔（AND关系）
            # 注意：host_venue.issn 可能需要改为 source.issn 或其他格式

            import datetime
            current_year = datetime.datetime.now().year
            # 扩大年份范围：优先当年，其次前1年，再前2年
            # 中文期刊在OpenAlex上可能尚未收录2026年论文
            year_filters = []
            for year_offset in range(3):
                year = current_year - year_offset
                year_filters.append(f"publication_year:{year}")
            year_filter = ",".join(year_filters)  # OR关系：当年或前两年

            filters = [
                f"locations.source.issn:{issn}",
                "type:article",
                year_filter,  # 扩大范围
            ]
            # 如果指定了语言，添加语言筛选
            if language:
                filters.append(f"language:{language}")

            # 用分号连接不同字段过滤器
            filter_str = ";".join(filters)

            url = (
                f"https://api.openalex.org/works?"
                f"filter={filter_str}&"
                f"per_page={per_page}&"
                f"page={page}&"
                f"sort=publication_date:desc"
            )

            # 添加API Key（如果有）
            if config.api.OPENALEX_API_KEY:
                url += f"&api_key={config.api.OPENALEX_API_KEY}"

            try:
                response = self._make_request(url)
                if response is None:
                    break

                data = response.json()
                results = data.get("results", [])

                if not results:
                    break  # 没有更多结果

                for item in results:
                    paper = self._parse_openalex_work(item, journal_name, issn)
                    if paper:
                        papers.append(paper)

                        # 达到最大数量则停止
                        if len(papers) >= max_papers:
                            break

                # 检查是否还有下一页
                meta = data.get("meta", {})
                if page >= meta.get("page_count", 1):
                    break

                page += 1
                time.sleep(config.api.RATE_LIMIT_DELAY)  # 避免API限流

            except Exception as e:
                print(f"OpenAlex请求失败: {e}")
                break

        return papers[:max_papers]

    def _parse_openalex_work(self, work: Dict[str, Any],
                            default_journal_name: str,
                            journal_issn: str = "") -> Optional[Paper]:
        """
        解析OpenAlex的work对象为Paper模型

        OpenAlex的字段映射：
        - id -> doi (需要提取DOI部分)
        - display_name -> title
        - abstract_inverted_index -> abstract (需要反转为原文)
        - authorships -> authors
        - publication_date -> published_date
        - doi -> url
        - keywords -> keywords
        - primary_location -> source info

        Args:
            work: OpenAlex返回的work对象
            default_journal_name: 默认期刊名称

        Returns:
            Paper对象或None
        """
        try:
            # 提取DOI作为唯一标识
            # OpenAlex的id格式：https://openalex.org/W2893546377
            # DOI格式：10.xxx/xxxxx
            raw_id = work.get("id", "")
            doi = self._extract_doi(raw_id)

            if not doi:
                # 如果没有DOI，尝试用其他方式生成唯一ID
                doi = f"openalex:{raw_id.split('/')[-1]}" if raw_id else None

            if not doi:
                return None

            # 提取标题
            title = work.get("display_name", "")
            if not title:
                return None

            # 提取和反转摘要
            # OpenAlex将摘要存储为倒排索引格式
            # {"word1": [0, 5], "word2": [1, 6]} 表示word1出现在位置0和5
            abstract = self._reconstruct_abstract(
                work.get("abstract_inverted_index")
            )

            # 提取作者列表
            authorships = work.get("authorships", [])
            authors = []
            for authorship in authorships:
                author = authorship.get("author", {})
                author_name = author.get("display_name", "")
                if author_name:
                    authors.append(author_name)

            # 提取期刊信息
            primary_location = work.get("primary_location", {})
            source = primary_location.get("source", {})
            journal_name = source.get("display_name", default_journal_name)

            # 提取发表日期（过滤未来日期）
            pub_date_str = work.get("publication_date")
            published_date = None
            if pub_date_str:
                try:
                    # 尝试标准格式 YYYY-MM-DD
                    parsed_date = date.fromisoformat(pub_date_str)
                    if parsed_date <= datetime.now().date():
                        published_date = parsed_date
                except ValueError:
                    try:
                        # 尝试 YYYY-MM 格式
                        parts = pub_date_str.split("-")
                        year, month = int(parts[0]), int(parts[1]) if len(parts) > 1 and parts[1] else 1
                        parsed_date = date(year, month, 1)
                        if parsed_date <= datetime.now().date():
                            published_date = parsed_date
                    except Exception:
                        pass

            # 提取URL
            url = work.get("doi") or primary_location.get("landing_page_url")

            # 提取关键词：优先从 concepts 取（OpenAlex主要字段），其次从 keywords 取
            keywords = []
            # 方式1：从 concepts 取（这是OpenAlex的主要概念字段）
            for concept in work.get("concepts", [])[:10]:
                if isinstance(concept, dict):
                    level = concept.get("level", 99)
                    # 只取具体概念（level 1-2 较具体，level 0 较宽泛）
                    if level <= 2:
                        kw = concept.get("display_name", "")
                        if kw:
                            keywords.append(kw)
            # 方式2：如果 concepts 为空，从 keywords 取
            if not keywords:
                for kw in work.get("keywords", [])[:10]:
                    if isinstance(kw, dict):
                        keywords.append(kw.get("display_name", ""))
                    elif isinstance(kw, str):
                        keywords.append(kw)
            # 去重
            keywords = list(dict.fromkeys(keywords))[:10]

            # 判断语言：从标题检测（中文标点或CJK字符）
            title = work.get("display_name", "")
            language = self._detect_language(title)

            # 过滤：只保留近年论文（相关性更高）
            if published_date and published_date.year < MIN_PAPER_YEAR:
                return None

            return Paper(
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                journal_issn=journal_issn,
                journal_name=journal_name,
                published_date=published_date,
                url=url,
                keywords=keywords,
                language=language
            )

        except Exception as e:
            print(f"解析OpenAlex work失败: {e}")
            return None

    def _extract_doi(self, openalex_id: str) -> Optional[str]:
        """
        从OpenAlex ID提取DOI

        OpenAlex有时会包含DOI信息在id字段中
        """
        if not openalex_id:
            return None

        # 尝试直接从id中提取DOI
        # 格式示例：https://doi.org/10.1038/nature12373
        if "doi.org" in openalex_id:
            return openalex_id.split("doi.org/")[-1]

        # 如果没有DOI，使用OpenAlex ID作为唯一标识
        return None

    def _reconstruct_abstract(self, inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
        """
        将OpenAlex的倒排索引格式还原为原始摘要文本

        倒排索引格式：
        {
            "deep": [0],
            "learning": [1],
            "has": [2, 7],
            ...
        }
        需要按位置重新拼接
        """
        if not inverted_index:
            return None

        try:
            # 找出最大位置
            max_pos = max(pos for positions in inverted_index.values() for pos in positions)
            words = [""] * (max_pos + 1)

            for word, positions in inverted_index.items():
                for pos in positions:
                    words[pos] = word

            abstract = " ".join(words)
            return abstract if len(abstract) > 20 else None

        except Exception:
            return None

    def _detect_language(self, text: str) -> str:
        """
        检测文本语言

        通过检测中文字符（CJK统一汉字范围）来判断是否为中文

        Args:
            text: 待检测文本

        Returns:
            "zh" 表示中文，"en" 表示英文
        """
        if not text:
            return "en"

        # CJK统一汉字范围：4E00-9FFF
        # CJK统一汉字扩展A：3400-4DBF
        # 判断是否包含中文字符
        for char in text:
            code = ord(char)
            if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF):
                return "zh"

        return "en"

    # =========================================================================
    # CrossRef API 相关方法
    # =========================================================================

    def fetch_papers_from_crossref(self, issn: str, journal_name: str,
                                   max_papers: int = 50) -> List[Paper]:
        """
        从CrossRef获取期刊论文

        CrossRef是DOI的官方注册机构，数据权威可靠。
        特点：
        - 免注册即可使用（有速率限制）
        - 元数据标准规范
        - 覆盖全球主要期刊

        Args:
            issn: 期刊ISSN
            journal_name: 期刊名称
            max_papers: 最大获取数量

        Returns:
            Paper对象列表
        """
        papers = []
        rows = min(200, max_papers)  # CrossRef每页最大100条，OpenAlex支持200

        for offset in range(0, max_papers, rows):
            # CrossRef API参数说明：
            # issn: 按ISSN筛选
            # rows: 每页数量
            # offset: 偏移量（分页）
            # sort: 排序方式
            # 不加时间筛选，先确保能获取到数据
            url = (
                f"https://api.crossref.org/journals/{issn}/works"
                f"?rows={rows}&"
                f"offset={offset}&"
                f"sort=published-online&"
                f"order=desc"
            )

            try:
                response = self._make_request(url)
                if response is None:
                    break

                data = response.json()
                items = data.get("message", {}).get("items", [])

                if not items:
                    break

                for item in items:
                    paper = self._parse_crossref_work(item, journal_name, issn)
                    if paper:
                        papers.append(paper)

                time.sleep(config.api.RATE_LIMIT_DELAY)

            except Exception as e:
                print(f"CrossRef请求失败: {e}")
                break

        return papers[:max_papers]

    def _parse_crossref_work(self, work: Dict[str, Any],
                            default_journal_name: str,
                            issn: str) -> Optional[Paper]:
        """
        解析CrossRef的work对象为Paper模型

        CrossRef字段说明：
        - DOI: 数字对象唯一标识符
        - title: 论文标题（数组，取第一个）
        - abstract: 摘要（有时有XML标签，需要清理）
        - author: 作者列表
        - published: 发表日期
        - URL: 论文链接
        - container-title: 期刊名称
        """
        try:
            # 提取DOI
            doi = work.get("DOI", "")
            if not doi:
                return None

            # 提取标题
            titles = work.get("title", [])
            title = titles[0] if titles else ""
            if not title:
                return None

            # 提取摘要
            abstract = work.get("abstract", "")
            if abstract:
                # 清理XML标签
                import re
                abstract = re.sub(r'<[^>]+>', '', abstract)

            # 提取作者
            authors = []
            for author in work.get("author", []):
                name_parts = []
                if author.get("given"):
                    name_parts.append(author["given"])
                if author.get("family"):
                    name_parts.append(author["family"])
                if name_parts:
                    authors.append(" ".join(name_parts))

            # 提取期刊名称
            container_titles = work.get("container-title", [])
            journal_name = container_titles[0] if container_titles else default_journal_name

            # 提取发表日期（过滤未来日期，预发表论文不可靠）
            published_date = None
            date_parts = work.get("published-print", work.get("published-online", {}))
            if date_parts:
                parts = date_parts.get("date-parts", [[]])[0]
                if len(parts) >= 1:
                    year = parts[0]
                    month = parts[1] if len(parts) > 1 else 1
                    day = parts[2] if len(parts) > 2 else 1
                    try:
                        parsed_date = date(year, month, day)
                        # 过滤未来日期（预发表/提前上线论文不可靠）
                        if parsed_date <= datetime.now().date():
                            published_date = parsed_date
                    except ValueError:
                        pass

            # 提取URL
            url = work.get("URL", f"https://doi.org/{doi}")

            # 提取关键词（有时有）
            keywords = []
            for subject in work.get("subject", [])[:10]:
                if isinstance(subject, str):
                    keywords.append(subject)

            # 判断语言
            language = self._detect_language(title)

            # 过滤：只保留近年论文（相关性更高）
            if published_date and published_date.year < MIN_PAPER_YEAR:
                return None

            return Paper(
                doi=doi,
                title=title,
                abstract=abstract if len(abstract) > 20 else None,
                authors=authors,
                journal_issn=issn,
                journal_name=journal_name,
                published_date=published_date,
                url=url,
                keywords=keywords,
                language=language
            )

        except Exception as e:
            print(f"解析CrossRef work失败: {e}")
            return None

    # =========================================================================
    # CrossRef 期刊名称搜索
    # =========================================================================

    def fetch_papers_from_crossref_by_name(self, journal_name: str,
                                           max_papers: int = 50) -> List[Paper]:
        """
        从CrossRef通过期刊名称获取论文

        Args:
            journal_name: 期刊名称
            max_papers: 最大获取数量

        Returns:
            Paper对象列表
        """
        papers = []
        rows = min(100, max_papers)

        for offset in range(0, max_papers, rows):
            # CrossRef 期刊查询接口
            url = (
                f"https://api.crossref.org/journals"
                f"?query.title={quote(journal_name)}"
                f"&rows={rows}&offset={offset}"
            )

            try:
                response = self._make_request(url)
                if response is None:
                    break

                data = response.json()
                items = data.get("message", {}).get("items", [])

                if not items:
                    break

                # 从期刊列表获取 ISSN
                for item in items:
                    # 检查是否是匹配的期刊
                    titles = item.get("title", [])
                    if not titles or journal_name.lower() not in titles[0].lower():
                        continue

                    # 获取期刊的ISSN
                    issns = item.get("ISSN", [])
                    if not issns:
                        continue

                    issn = issns[0]
                    # 使用这个ISSN获取论文
                    journal_papers = self.fetch_papers_from_crossref(issn, journal_name, max_papers)
                    papers.extend(journal_papers)
                    break  # 找到匹配的期刊就停止

                time.sleep(config.api.RATE_LIMIT_DELAY)

            except Exception as e:
                print(f"CrossRef期刊名称搜索失败: {e}")
                break

        return papers[:max_papers]

    # =========================================================================
    # Semantic Scholar API
    # =========================================================================

    def fetch_papers_from_semantic_scholar(self, issn: str, journal_name: str,
                                           max_papers: int = 50) -> List[Paper]:
        """
        从Semantic Scholar获取期刊论文

        Semantic Scholar 特点：
        - AI驱动的学术搜索引擎
        - 收录范围广，包括很多中文期刊
        - 提供论文引用关系等信息

        Args:
            issn: 期刊ISSN
            journal_name: 期刊名称
            max_papers: 最大获取数量

        Returns:
            Paper对象列表
        """
        papers = []
        offset = 0
        per_page = 100

        while len(papers) < max_papers:
            # Semantic Scholar Graph API
            # 使用 venue 字段按期刊筛选
            query = {
                "query": journal_name,
                "filter": f"venue:{journal_name}",
                "offset": offset,
                "limit": per_page,
                "fields": "title,abstract,year,authors,journal,venue,externalIds,url"
            }

            try:
                # Semantic Scholar 搜索接口
                url = "https://api.semanticscholar.org/graph/v1/paper/search"

                response = self.session.get(
                    url,
                    params=query,
                    timeout=30
                )

                if response.status_code != 200:
                    print(f"Semantic Scholar HTTP错误: {response.status_code}")
                    break

                data = response.json()
                results = data.get("data", [])

                if not results:
                    break

                for item in results:
                    # 验证期刊名称匹配
                    venue = item.get("venue", "")
                    if journal_name.lower() not in venue.lower():
                        continue

                    # 解析外部ID
                    external_ids = item.get("externalIds", {})
                    doi = external_ids.get("DOI", "")

                    # 提取标题
                    title = item.get("title", "")
                    if not title:
                        continue

                    # 提取摘要
                    abstract = item.get("abstract", "") or ""

                    # 提取作者
                    authors_data = item.get("authors", [])
                    authors = [a.get("name", "") for a in authors_data]

                    # 提取年份
                    year = item.get("year")
                    published_date = None
                    if year:
                        published_date = date(year, 1, 1)

                    paper = Paper(
                        doi=doi,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        published_date=published_date,
                        journal_name=journal_name,
                        journal_issn=issn,
                        url=item.get("url", ""),
                        keywords=[],
                        language="zh" if issn.startswith("7") else "en"
                    )
                    papers.append(paper)

                total = data.get("total", 0)
                if offset + len(results) >= total:
                    break

                offset += per_page
                time.sleep(config.api.RATE_LIMIT_DELAY)

            except Exception as e:
                print(f"Semantic Scholar请求失败: {e}")
                break

        return papers[:max_papers]

    # =========================================================================
    # 综合获取方法（降级策略）
    # =========================================================================

    def fetch_papers(self, journal: Journal,
                     max_papers: int = 50,
                     language: str = None) -> Tuple[List[Paper], str]:
        """
        综合获取期刊论文（带降级策略）

        尝试从多个数据源获取论文，按优先级自动降级：
        1. OpenAlex（数据最全面，按ISSN搜索）
        2. CrossRef（作为备选，按ISSN搜索）
        3. 期刊名称搜索（兜底策略，适用于中文期刊）

        Args:
            journal: Journal对象
            max_papers: 最大获取数量
            language: 指定语言（zh/en），None表示根据期刊类型自动选择

        Returns:
            (Paper列表, 使用的数据源名称)
        """
        issn = journal.issn
        name = journal.name
        is_chinese_journal = journal.country == "CN"

        # 传递 language 参数，由调用方决定：
        # - cli.py 调用时 language=None，获取该期刊最新1篇论文（不区分语言）
        # - 其他场景可传入 "zh" 或 "en" 限制语言

        # 首先尝试OpenAlex按ISSN搜索
        print(f"尝试从OpenAlex获取 (ISSN: {issn})...")
        try:
            papers = self._throttled_request(
                "openalex",
                self.fetch_papers_from_openalex,
                issn, name, max_papers, language
            )
        except Exception as e:
            print(f"  OpenAlex请求失败: {e}")
            papers = []

        # 如果需要补充获取（language=None 时不补充，保持只获取最新论文）
        if len(papers) < max_papers and language is not None:
            remaining = max_papers - len(papers)
            print(f"  获取到 {len(papers)} 篇，补充获取 {remaining} 篇...")
            try:
                more_papers = self._throttled_request(
                    "openalex",
                    self.fetch_papers_from_openalex,
                    issn, name, remaining
                )
                papers.extend(more_papers)
            except Exception:
                pass

        if len(papers) > 0:
            for paper in papers:
                paper.journal_issn = issn
            return papers[:max_papers], "OpenAlex"

        # OpenAlex失败，跳过限流的API，直接尝试期刊名称搜索（对于中文期刊更有效）
        if is_chinese_journal:
            print(f"中文期刊，尝试期刊名称搜索: {name}")
            papers = self._fetch_papers_by_journal_name(name, max_papers)
            if len(papers) > 0:
                return papers, "NameSearch"

        # OpenAlex失败，尝试CrossRef按ISSN搜索
        print(f"OpenAlex获取失败，尝试CrossRef (ISSN: {issn})...")
        try:
            papers = self._throttled_request(
                "crossref",
                self.fetch_papers_from_crossref,
                issn, name, max_papers
            )
            if len(papers) > 0:
                return papers, "CrossRef"
        except Exception as e:
            print(f"  CrossRef请求失败: {e}")

        # 最后尝试：使用期刊名称搜索（兜底策略）
        print(f"尝试期刊名称搜索: {name}")
        papers = self._fetch_papers_by_journal_name(name, max_papers)
        if len(papers) > 0:
            return papers, "NameSearch"

        # 全部失败
        print(f"无法获取期刊 '{name}' 的论文")
        return [], "None"

    def _fetch_papers_by_journal_name(self, journal_name: str,
                                      max_papers: int = 50,
                                      year_range: int = 5) -> List[Paper]:
        """
        通过期刊名称搜索论文（兜底策略）

        当ISSN搜索失败时使用，搜索精度较低但覆盖更广。

        Args:
            journal_name: 期刊名称
            max_papers: 最大获取数量
            year_range: 往前搜索的年份范围（默认5年，覆盖近期论文）

        Returns:
            Paper对象列表
        """
        papers = []
        page = 1
        per_page = 50

        while len(papers) < max_papers:
            # URL编码期刊名称
            encoded_name = quote(journal_name)
            # 构建过滤器 - 扩大年份范围，中文顶刊数据量有限
            import datetime
            current_year = datetime.datetime.now().year
            year_filters = [f"publication_year:{current_year - y}" for y in range(year_range)]
            year_filter = ",".join(year_filters)

            # 使用正确的 OpenAlex 字段名
            # display_name.search 用于全文搜索期刊名称
            filters = [
                f"display_name.search:{encoded_name}",
                "type:article",
                year_filter,
            ]
            filter_str = ";".join(filters)

            url = (
                f"https://api.openalex.org/works?"
                f"filter={filter_str}&"
                f"per_page={per_page}&"
                f"page={page}&"
                f"sort=publication_date:desc"
            )

            try:
                response = self._make_request(url)
                if response is None:
                    break

                data = response.json()
                results = data.get("results", [])

                if not results:
                    break

                for item in results:
                    # 检查来源名称是否匹配（更严格）
                    source = item.get("primary_location", {}).get("source", {})
                    source_name = source.get("display_name", "")
                    # 使用宽松的匹配：期刊名包含在来源名中，或来源名包含期刊名
                    if not (journal_name.lower() in source_name.lower() or 
                            source_name.lower() in journal_name.lower()):
                        continue

                    paper = self._parse_openalex_work(item, journal_name, "")
                    if paper:
                        papers.append(paper)

                    if len(papers) >= max_papers:
                        break

                meta = data.get("meta", {})
                if page >= meta.get("page_count", 1):
                    break

                page += 1
                time.sleep(config.api.RATE_LIMIT_DELAY)

            except Exception as e:
                print(f"期刊名称搜索失败: {e}")
                break

        return papers[:max_papers]

    def _make_request(self, url: str, timeout: int = None) -> Optional[requests.Response]:
        """
        发送HTTP请求（带重试和错误处理）

        Args:
            url: 请求URL
            timeout: 超时时间（秒）

        Returns:
            Response对象或None（失败时）
        """
        if timeout is None:
            timeout = config.api.REQUEST_TIMEOUT

        max_retries = config.api.MAX_RETRIES

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=timeout)

                # 检查HTTP状态码
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    # API限流，等待后重试
                    wait_time = (attempt + 1) * 2
                    print(f"API限流，等待{wait_time}秒...")
                    time.sleep(wait_time)
                else:
                    # 详细日志：打印完整URL和状态码，便于调试
                    print(f"HTTP错误: {response.status_code}")
                    print(f"请求URL: {url[:100]}..." if len(url) > 100 else f"请求URL: {url}")
                    if response.status_code == 400:
                        print("提示: 400错误通常是ISSN不存在或过滤器参数错误")
                        # 打印响应内容以便调试
                        try:
                            error_data = response.json()
                            if "error" in error_data:
                                print(f"API错误信息: {error_data['error']}")
                            if "message" in error_data:
                                print(f"API消息: {error_data['message']}")
                        except:
                            print(f"响应内容: {response.text[:200]}")
                    return None

            except requests.exceptions.Timeout:
                print(f"请求超时（第{attempt + 1}次尝试）")
                time.sleep(config.api.RETRY_DELAY)
            except requests.exceptions.RequestException as e:
                print(f"请求异常: {e}")
                break

        return None

    # =========================================================================
    # 辅助方法
    # =========================================================================

    @staticmethod
    def detect_language(text: str) -> str:
        """
        简单语言检测

        基于字符集判断是中英文：
        - 包含大量中文字符 -> "zh"
        - 否则 -> "en"
        """
        if not text:
            return "en"

        # 统计中文字符数量
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len([c for c in text if c.isalpha()])

        if total_chars == 0:
            return "en"

        # 如果中文字符占比超过30%，认为是中文
        if chinese_chars / total_chars > 0.3:
            return "zh"

        return "en"


# =============================================================================
# 全局单例
# =============================================================================

_journal_service: Optional[JournalService] = None


def get_journal_service() -> JournalService:
    """获取期刊服务的全局单例"""
    global _journal_service
    if _journal_service is None:
        _journal_service = JournalService()
    return _journal_service
