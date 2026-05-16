"""
期刊论文语义匹配系统 - 中文文献获取服务
====================================

功能说明：
- 专门处理中文期刊论文的获取
- 支持多个中文数据源
- 优化中文文献的检索策略

中文文献获取策略：
1. 优先从 OpenAlex 获取（扩大年份范围）
2. 使用中文期刊名称搜索（而非 ISSN）
3. 降级到 CrossRef 和其他数据源
4. 支持 ISSN 校验和修正

作者：AI Assistant
日期：2026-05-16
"""

import re
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import requests

from ..models.schemas import Paper, Journal


class ChineseJournalService:
    """
    中文期刊论文获取服务

    专门优化中文期刊论文的获取，解决以下问题：
    1. ISSN 不准确或缺失
    2. OpenAlex 对中文期刊收录有限
    3. 中文文献标题、摘要、关键词获取失败

    使用示例：
        service = ChineseJournalService()
        papers = service.fetch_chinese_papers("管理世界")
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Research-Tool/1.0",
            "Accept": "application/json"
        })

    def fetch_chinese_papers(self, journal_name: str,
                            max_papers: int = 10,
                            year_range: int = 5) -> Tuple[List[Paper], str]:
        """
        获取中文期刊论文

        策略（按优先级）：
        1. Semantic Scholar（AI驱动的学术搜索，收录广泛）
        2. OpenAlex 期刊名称搜索（扩大年份范围）
        3. CrossRef 搜索
        4. OpenAlex ISSN 搜索

        Args:
            journal_name: 期刊名称
            max_papers: 最大获取数量
            year_range: 往前搜索的年份范围

        Returns:
            (Paper列表, 数据源名称)
        """
        # 策略1：Semantic Scholar（最推荐）
        print(f"尝试从 Semantic Scholar 获取: {journal_name}")
        papers = self._fetch_from_semantic_scholar(journal_name, max_papers)
        if papers:
            return papers, "SemanticScholar"

        # 策略2：OpenAlex 期刊名称搜索
        print(f"尝试从 OpenAlex 获取: {journal_name}")
        papers = self._fetch_from_openalex_by_name(journal_name, max_papers, year_range)
        if papers:
            return papers, "OpenAlex_NameSearch"

        # 策略3：CrossRef
        print(f"尝试从 CrossRef 获取: {journal_name}")
        papers = self._fetch_from_crossref(journal_name, max_papers)
        if papers:
            return papers, "CrossRef"

        return [], "None"

    def _fetch_from_semantic_scholar(self, journal_name: str,
                                      max_papers: int = 10) -> List[Paper]:
        """
        通过 Semantic Scholar 获取论文

        Semantic Scholar 的优势：
        - AI驱动的学术搜索引擎
        - 收录范围广，包括很多中文期刊
        - 提供更完整的元数据

        注意：需要 API Key 来避免限流
        申请地址：https://www.semanticscholar.org/product/api#api-key-form
        """
        papers = []
        offset = 0
        per_page = min(100, max_papers)

        # Semantic Scholar API
        # 使用 venue 字段匹配期刊名称
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": journal_name,
                "filter": f"venue:{journal_name}",
                "offset": offset,
                "limit": per_page,
                "fields": "title,abstract,year,authors,journal,venue,externalIds,url"
            }

            # 检查是否有 API key
            headers = {}
            api_key = self._get_semantic_scholar_api_key()
            if api_key:
                headers["x-api-key"] = api_key

            response = self.session.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 429:
                print("    Semantic Scholar API 限流，请申请 API Key")
                return []
            if response.status_code != 200:
                return []

            data = response.json()
            results = data.get("data", [])
            total = data.get("total", 0)

            if not results:
                return []

            for item in results:
                # 验证期刊名称匹配
                venue = item.get("venue", "")
                if not self._is_journal_match(journal_name, venue):
                    continue

                paper = self._parse_semantic_scholar_paper(item)
                if paper:
                    papers.append(paper)

                if len(papers) >= max_papers:
                    break

            # 如果需要更多，尝试分页
            while len(papers) < max_papers and offset + len(results) < total:
                offset += per_page
                params["offset"] = offset
                response = self.session.get(url, params=params, headers=headers, timeout=30)
                if response.status_code != 200:
                    break

                data = response.json()
                results = data.get("data", [])
                if not results:
                    break

                for item in results:
                    if not self._is_journal_match(journal_name, item.get("venue", "")):
                        continue

                    paper = self._parse_semantic_scholar_paper(item)
                    if paper:
                        papers.append(paper)

                    if len(papers) >= max_papers:
                        break

        except Exception as e:
            print(f"    Semantic Scholar 请求失败: {e}")

        return papers[:max_papers]

    def _get_semantic_scholar_api_key(self) -> Optional[str]:
        """获取 Semantic Scholar API Key"""
        import os
        # 优先从环境变量获取
        api_key = os.getenv("SEMANTICSCHOLAR_API_KEY", "")
        if api_key:
            return api_key

        # 其次从配置文件获取
        try:
            from ..config import config
            api_key = config.api.SEMANTICSCHOLAR_API_KEY
            if api_key:
                return api_key
        except:
            pass

        return None

    def _parse_semantic_scholar_paper(self, item: Dict[str, Any]) -> Optional[Paper]:
        """解析 Semantic Scholar 论文"""
        try:
            # 提取 DOI
            external_ids = item.get("externalIds", {})
            doi = external_ids.get("DOI", "")

            # 提取标题
            title = item.get("title", "")
            if not title:
                return None

            # 提取摘要
            abstract = item.get("abstract", "") or ""

            # 提取作者
            authors_data = item.get("authors", [])
            authors = [a.get("name", "") for a in authors_data]

            # 提取期刊信息
            journal_info = item.get("journal", {})
            journal_name = journal_info.get("name", "") if isinstance(journal_info, dict) else str(journal_info)

            # 提取年份
            year = item.get("year")
            published_date = None
            if year:
                from datetime import date
                published_date = date(year, 1, 1)

            # 提取 URL
            url = item.get("url", "") or f"https://doi.org/{doi}" if doi else ""

            # 判断语言
            language = "zh" if re.search(r'[\u4e00-\u9fff]', title) else "en"

            return Paper(
                doi=doi,
                title=title,
                abstract=abstract if len(abstract) > 20 else None,
                authors=authors,
                journal_issn="",
                journal_name=journal_name,
                published_date=published_date,
                url=url,
                keywords=[],
                language=language
            )

        except Exception as e:
            print(f"解析 Semantic Scholar 论文失败: {e}")
            return None

    def _fetch_from_openalex_by_name(self, journal_name: str,
                                      max_papers: int = 10,
                                      year_range: int = 5) -> List[Paper]:
        """
        通过期刊名称从 OpenAlex 获取论文

        关键改进：
        - 扩大年份范围（默认5年）
        - 移除严格的时间限制
        - 使用 fuzzy 匹配期刊名称
        """
        papers = []
        page = 1
        per_page = 50

        # URL 编码期刊名称
        encoded_name = self._url_encode(journal_name)

        # 扩大年份范围
        current_year = datetime.now().year
        year_filters = [f"publication_year:{current_year - y}" for y in range(year_range)]
        year_filter = ",".join(year_filters)

        while len(papers) < max_papers:
            # 使用 display_name.search 搜索期刊名称
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
                response = self.session.get(url, timeout=30)
                if response.status_code != 200:
                    break

                data = response.json()
                results = data.get("results", [])
                if not results:
                    break

                for item in results:
                    # 验证期刊名称匹配（更严格的验证）
                    source = item.get("primary_location", {}).get("source", {})
                    source_name = source.get("display_name", "")

                    # 检查来源是否匹配（忽略大小写和常见差异）
                    if not self._is_journal_match(journal_name, source_name):
                        continue

                    paper = self._parse_openalex_work(item, journal_name)
                    if paper:
                        papers.append(paper)

                    if len(papers) >= max_papers:
                        break

                # 检查是否还有更多页
                meta = data.get("meta", {})
                if page >= meta.get("page_count", 1):
                    break

                page += 1
                time.sleep(0.5)

            except Exception as e:
                print(f"OpenAlex 请求失败: {e}")
                break

        return papers[:max_papers]

    def _fetch_from_crossref(self, journal_name: str,
                             max_papers: int = 10) -> List[Paper]:
        """通过 CrossRef 获取中文期刊论文"""
        papers = []
        rows = min(100, max_papers)

        try:
            # CrossRef 期刊搜索
            url = (
                f"https://api.crossref.org/journals"
                f"?query.title={self._url_encode(journal_name)}"
                f"&rows=5"
            )

            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return []

            data = response.json()
            items = data.get("message", {}).get("items", [])

            # 找到匹配的期刊
            matched_issn = None
            for item in items:
                titles = item.get("title", [])
                if titles and self._is_journal_match(journal_name, titles[0]):
                    issns = item.get("ISSN", [])
                    if issns:
                        matched_issn = issns[0]
                        break

            if not matched_issn:
                return []

            # 通过 ISSN 获取论文
            url = (
                f"https://api.crossref.org/journals/{matched_issn}/works"
                f"?rows={rows}&sort=published-online&order=desc"
            )

            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return []

            data = response.json()
            items = data.get("message", {}).get("items", [])

            for item in items:
                paper = self._parse_crossref_work(item, journal_name, matched_issn)
                if paper:
                    papers.append(paper)

                if len(papers) >= max_papers:
                    break

        except Exception as e:
            print(f"CrossRef 请求失败: {e}")

        return papers[:max_papers]

    def _parse_openalex_work(self, work: Dict[str, Any],
                            default_journal_name: str) -> Optional[Paper]:
        """解析 OpenAlex work 对象"""
        try:
            # 提取 DOI
            raw_id = work.get("id", "")
            doi = None
            if "doi.org" in raw_id:
                doi = raw_id.split("doi.org/")[-1]
            if not doi:
                doi = f"openalex:{raw_id.split('/')[-1]}"

            if not doi:
                return None

            # 提取标题
            title = work.get("display_name", "")
            if not title:
                return None

            # 提取摘要
            abstract_inv = work.get("abstract_inverted_index")
            abstract = self._reconstruct_abstract(abstract_inv)

            # 提取作者
            authors = []
            for auth in work.get("authorships", []):
                author = auth.get("author", {})
                name = author.get("display_name", "")
                if name:
                    authors.append(name)

            # 提取期刊信息
            primary_loc = work.get("primary_location", {})
            source = primary_loc.get("source", {})
            journal_name = source.get("display_name", default_journal_name)

            # 提取发表日期
            pub_date_str = work.get("publication_date")
            published_date = None
            if pub_date_str:
                try:
                    from datetime import date
                    published_date = date.fromisoformat(pub_date_str)
                except:
                    pass

            # 提取关键词
            keywords = []
            for concept in work.get("concepts", [])[:10]:
                if isinstance(concept, dict) and concept.get("level", 99) <= 2:
                    kw = concept.get("display_name", "")
                    if kw:
                        keywords.append(kw)

            # 判断语言
            language = "zh" if re.search(r'[\u4e00-\u9fff]', title) else "en"

            return Paper(
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                journal_issn="",
                journal_name=journal_name,
                published_date=published_date,
                url=work.get("doi"),
                keywords=keywords,
                language=language
            )

        except Exception as e:
            print(f"解析 OpenAlex work 失败: {e}")
            return None

    def _parse_crossref_work(self, work: Dict[str, Any],
                            journal_name: str,
                            issn: str) -> Optional[Paper]:
        """解析 CrossRef work 对象"""
        try:
            doi = work.get("DOI", "")
            if not doi:
                return None

            titles = work.get("title", [])
            title = titles[0] if titles else ""
            if not title:
                return None

            # 提取摘要
            abstract = work.get("abstract", "")
            if abstract:
                abstract = re.sub(r'<[^>]+>', '', abstract)

            # 提取作者
            authors = []
            for author in work.get("author", []):
                parts = []
                if author.get("given"):
                    parts.append(author["given"])
                if author.get("family"):
                    parts.append(author["family"])
                if parts:
                    authors.append(" ".join(parts))

            # 提取期刊名称
            container_titles = work.get("container-title", [])
            actual_journal = container_titles[0] if container_titles else journal_name

            # 提取发表日期
            published_date = None
            date_parts = work.get("published-print", work.get("published-online", {}))
            if date_parts:
                parts = date_parts.get("date-parts", [[]])[0]
                if len(parts) >= 1:
                    try:
                        from datetime import date
                        year = parts[0]
                        month = parts[1] if len(parts) > 1 else 1
                        day = parts[2] if len(parts) > 2 else 1
                        published_date = date(year, month, day)
                    except:
                        pass

            # 提取关键词
            keywords = []
            for subject in work.get("subject", [])[:10]:
                if isinstance(subject, str):
                    keywords.append(subject)

            # 判断语言
            language = "zh" if re.search(r'[\u4e00-\u9fff]', title) else "en"

            return Paper(
                doi=doi,
                title=title,
                abstract=abstract if len(abstract) > 20 else None,
                authors=authors,
                journal_issn=issn,
                journal_name=actual_journal,
                published_date=published_date,
                url=work.get("URL", f"https://doi.org/{doi}"),
                keywords=keywords,
                language=language
            )

        except Exception as e:
            print(f"解析 CrossRef work 失败: {e}")
            return None

    def _reconstruct_abstract(self, inverted_index: Optional[Dict]) -> Optional[str]:
        """将倒排索引还原为原文"""
        if not inverted_index:
            return None

        try:
            max_pos = max(pos for positions in inverted_index.values() for pos in positions)
            words = [""] * (max_pos + 1)

            for word, positions in inverted_index.items():
                for pos in positions:
                    words[pos] = word

            abstract = " ".join(words)
            return abstract if len(abstract) > 20 else None
        except:
            return None

    def _is_journal_match(self, name1: str, name2: str) -> bool:
        """判断两个期刊名称是否匹配"""
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()

        # 完全匹配
        if n1 == n2:
            return True

        # 一个包含另一个
        if n1 in n2 or n2 in n1:
            return True

        # 移除常见后缀后比较
        suffixes = ["学报", "杂志", "期刊", "研究"]
        for suffix in suffixes:
            if n1.replace(suffix, "") == n2.replace(suffix, ""):
                return True

        return False

    def _get_alternative_names(self, journal_name: str) -> List[str]:
        """获取期刊的替代名称"""
        alternatives = {
            "管理世界": ["管理世界", "管理世界杂志"],
            "经济研究": ["经济研究", "经济学研究"],
            "管理科学学报": ["管理科学学报", "管理科学"],
            "中国工业经济": ["中国工业经济", "工业经济研究"],
            "会计研究": ["会计研究", "会计研究杂志"],
        }
        return alternatives.get(journal_name, [])

    @staticmethod
    def _url_encode(text: str) -> str:
        """URL 编码"""
        from urllib.parse import quote
        return quote(text)


# =============================================================================
# 全局单例
# =============================================================================

_chinese_service: Optional[ChineseJournalService] = None


def get_chinese_journal_service() -> ChineseJournalService:
    """获取中文期刊服务的全局单例"""
    global _chinese_service
    if _chinese_service is None:
        _chinese_service = ChineseJournalService()
    return _chinese_service
