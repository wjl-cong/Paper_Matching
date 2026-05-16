"""
期刊论文语义匹配系统 - 命令行界面 (CLI)
====================================

功能说明：
- 提供命令行操作接口
- 支持初始化、检索、查看状态等操作
- 适合面试现场快速演示

使用方法：
    python -m journal_matcher.cli init              # 使用默认期刊初始化
    python -m journal_matcher.cli init -j journals.json  # 使用自定义配置文件
    python -m journal_matcher.cli search "深度学习在医学影像中的应用"
    python -m journal_matcher.cli status  # 查看状态
    python -m journal_matcher.cli serve   # 启动API服务

作者：AI Assistant
日期：2026-05-16
"""

import sys
import json
import argparse
import io
from pathlib import Path
from typing import Optional, List

# Windows 控制台编码修复
if sys.platform == 'win32':
    import os
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from journal_matcher.config import config
from journal_matcher.services.database import DatabaseService
from journal_matcher.services.embedding import get_embedding_service
from journal_matcher.services.journal import get_journal_service
from journal_matcher.services.search import SearchService
from journal_matcher.models.schemas import Journal


def load_journals_from_json(json_path: str) -> list:
    """
    从JSON配置文件加载期刊列表

    Args:
        json_path: JSON配置文件路径

    Returns:
        期刊配置列表

    JSON格式示例：
    {
        "journals": [
            {"name": "Nature", "issn": "0028-0836", "country": "INT"},
            {"name": "计算机学报", "issn": "0254-4164", "country": "CN"}
        ]
    }
    """
    path = Path(json_path)
    if not path.exists():
        print(f"[ERROR] 配置文件不存在: {json_path}")
        print("请创建 journals.json 文件，格式参考：")
        print('{"journals": [{"name": "期刊名", "issn": "ISSN号", "country": "CN/INT"}]}')
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    journals = data.get("journals", [])
    if not journals:
        print("[ERROR] 配置文件中没有期刊数据")
        sys.exit(1)

    return journals


class CLI:
    """
    命令行界面类

    提供交互式命令行操作
    """

    def __init__(self):
        """初始化CLI"""
        self.db: Optional[DatabaseService] = None
        self.search_service: Optional[SearchService] = None

    def _input_journals_interactive(self) -> List[Journal]:
        """
        交互式输入10本期刊

        Returns:
            Journal对象列表
        """
        journals = []
        print("\n请输入您的10本期刊（国内5本 + 国际5本）:")
        print("-" * 60)

        for i in range(1, 11):
            country = "CN" if i <= 5 else "INT"
            region = "国内" if i <= 5 else "国际"

            print(f"\n[{i}/10] {region}期刊")
            name = input("  期刊名称: ").strip()
            issn = input("  ISSN (如 0254-4164): ").strip()

            if name and issn:
                journal = Journal(
                    issn=issn,
                    name=name,
                    country=country,
                    description=f"{region}期刊"
                )
                journals.append(journal)
                self.db.save_journal(journal)

        print(f"\n共配置了 {len(journals)} 本期刊")
        return journals

    def init(self, use_default: bool = True,
             journal_configs: list = None):
        """
        初始化系统

        Args:
            use_default: 是否使用默认期刊列表
            journal_configs: 自定义期刊配置
        """
        print("=" * 60)
        print("期刊论文语义匹配系统 - 初始化")
        print("=" * 60)

        # 初始化数据库
        self.db = DatabaseService()
        print("[1/5] 数据库连接成功")

        # 加载向量化模型
        print("[2/5] 正在加载向量化模型...")
        embedding = get_embedding_service()
        print(f"      模型: {embedding.model_name}")
        print(f"      维度: {embedding.embedding_dim}")

        # 获取期刊服务
        journal_service = get_journal_service()

        # 询问用户是否输入自定义期刊
        if journal_configs:
            # 使用命令行传入的自定义配置
            print("[3/5] 使用自定义期刊配置...")
            journals = []
            for j in journal_configs:
                journal = Journal(
                    issn=j["issn"],
                    name=j["name"],
                    country=j.get("country", "INT"),
                    description=j.get("description", "")
                )
                journals.append(journal)
                self.db.save_journal(journal)
            print(f"      配置了 {len(journals)} 本期刊")
        elif use_default:
            # 询问用户是否使用默认配置
            print("[3/5] 期刊配置...")
            choice = input("是否输入自己的10本期刊？(y/N): ").strip().lower()
            if choice == 'y':
                # 让用户输入期刊
                journals = self._input_journals_interactive()
            else:
                # 使用默认配置
                print("      使用默认期刊配置...")
                journals = []
                for country, journal_list in config.DEFAULT_JOURNALS.items():
                    for j in journal_list:
                        journal = Journal(
                            issn=j["issn"],
                            name=j["name"],
                            country=country,
                            description=j.get("description", "")
                        )
                        journals.append(journal)
                        self.db.save_journal(journal)
                print(f"      配置了 {len(journals)} 本期刊")
        else:
            print("[ERROR] 既没有自定义配置，也没有默认配置")
            return

        # 抓取论文
        print("[4/5] 开始抓取论文...")
        journals = self.db.get_all_journals()
        all_papers = []
        total_papers = 0

        for i, journal in enumerate(journals):
            print(f"      [{i+1}/{len(journals)}] {journal.name}...")
            try:
                # 动态获取距离如今最近最新的一期论文，不区分语言，每期刊只取1篇
                papers, source = journal_service.fetch_papers(journal, max_papers=1, language=None)

                if papers:
                    # 去重（根据DOI）
                    seen_dois = set()
                    unique_papers = []
                    for p in papers:
                        if p.doi and p.doi not in seen_dois:
                            seen_dois.add(p.doi)
                            unique_papers.append(p)
                        elif not p.doi:  # 没有DOI的保留
                            unique_papers.append(p)

                    self.db.save_papers_batch(unique_papers)
                    all_papers.extend(unique_papers)
                    total_papers += len(unique_papers)
                    print(f"        [{source}] 获取到 {len(unique_papers)} 篇")

                    # 显示最新论文信息
                    for p in unique_papers:
                        pub_date = p.published_date.strftime("%Y-%m") if p.published_date else "未知日期"
                        print(f"        最新论文: {p.title[:50]}{'...' if len(p.title) > 50 else ''}")
                        print(f"        发表日期: {pub_date}")
                else:
                    print("        无数据")
            except Exception as e:
                print(f"        失败: {e}")

        print(f"\n共抓取 {total_papers} 篇论文")

        # 向量化
        if all_papers:
            print("[5/5] 正在向量化...")
            self.search_service = SearchService(self.db, embedding)
            indexed = self.search_service.index_papers(all_papers)
            print(f"向量化完成: {indexed} 篇")

        print("=" * 60)
        print("初始化完成！")
        print("=" * 60)

        # 生成初始化报告
        self._generate_init_report(journals, total_papers, all_papers)

    def search(self, query: str, mode: str = "semantic",
               threshold: float = None, top_k: int = 5,
               show_details: bool = True):
        """
        执行检索

        Args:
            query: 检索查询
            mode: 搜索模式，"exact"=精确搜索，"semantic"=语义搜索，"both"=综合
            threshold: 相似度阈值
            top_k: 每期刊返回数量
            show_details: 是否显示详细信息
        """
        # 确保系统已初始化
        self._ensure_initialized()

        print("=" * 60)
        print(f"检索主题: {query}")
        print(f"搜索模式: {'精确搜索' if mode == 'exact' else '语义搜索' if mode == 'semantic' else '综合搜索'}")
        print("=" * 60)

        # 根据搜索模式自动设置阈值（在调用search之前）
        if threshold is None:
            if mode == "exact":
                threshold = 0.01   # BM25 归一化后通常很小，设为极低
            elif mode == "semantic":
                threshold = 0.35  # 语义相似度通常在 0.4-0.8
            else:  # both
                threshold = 0.1
        print(f"相似度阈值: {threshold}")

        # 执行检索
        response = self.search_service.search(
            query=query,
            mode=mode,
            threshold=threshold,
            top_k=top_k
        )

        print(f"\n检索耗时: {response.search_time_ms:.2f}ms")
        print(f"检索期刊数: {response.total_journals}")
        print(f"论文总数: {response.total_papers}")
        print(f"找到相似论文的期刊数: {response.found_count}")
        print()

        found_total = 0
        for result in response.results:
            status = "[OK]" if result.found else "[FAIL]"
            print(f"[{result.journal.name}] {status}")
            print(f"    期刊论文总数: {result.total_papers}")

            if result.found and show_details:
                for i, paper in enumerate(result.similar_papers):
                    found_total += 1
                    pub_date = paper.paper.published_date.strftime("%Y-%m") if paper.paper.published_date else "未知"
                    print(f"\n    === Paper {i+1} ===")
                    print(f"    Score: {paper.score:.4f}")
                    print(f"    Date: {pub_date}")
                    print(f"    Title: {paper.paper.title}")
                    # Keywords
                    if paper.paper.keywords:
                        print(f"    Keywords: {', '.join(paper.paper.keywords[:8])}")
                    else:
                        print(f"    Keywords: N/A")
                    # Abstract
                    display_abstract = paper.abstract_snippet or paper.paper.abstract
                    if display_abstract:
                        print(f"    Abstract: {display_abstract[:200]}...")
                    else:
                        print(f"    Abstract: N/A")
                    # Authors
                    if paper.paper.authors:
                        authors_str = ", ".join(paper.paper.authors[:3])
                        if len(paper.paper.authors) > 3:
                            authors_str += " et al."
                        print(f"    Authors: {authors_str}")
                    # Link
                    if paper.paper.url:
                        print(f"    Link: {paper.paper.url}")
            print()

        print("=" * 60)
        if found_total > 0:
            print(f"共找到 {found_total} 篇相似论文")
            # 询问是否生成详细报告
            choice = input("\n是否生成详细分析报告？(y/N): ").strip().lower()
            if choice == 'y':
                self.generate_report(query, response, mode=mode)
        else:
            print("未找到任何相似论文")
        print("=" * 60)

    def _generate_init_report(self, journals: List, total_papers: int, all_papers: List):
        """
        生成初始化报告

        Args:
            journals: 期刊列表
            total_papers: 总论文数
            all_papers: 所有论文列表
        """
        print("\n" + "=" * 60)
        print("初始化数据分析报告")
        print("=" * 60)

        # 按期刊分组统计
        journal_stats = {}
        cn_count = 0
        int_count = 0

        for journal in journals:
            papers_in_journal = [p for p in all_papers if p.journal_issn == journal.issn]
            count = len(papers_in_journal)

            if journal.country == "CN":
                cn_count += 1
            else:
                int_count += 1

            # 统计论文语言分布
            zh_count = sum(1 for p in papers_in_journal if p.language == "zh")
            en_count = sum(1 for p in papers_in_journal if p.language == "en")

            # 有摘要的论文比例
            has_abstract = sum(1 for p in papers_in_journal if p.abstract)
            abstract_ratio = has_abstract / count * 100 if count > 0 else 0

            journal_stats[journal.issn] = {
                "name": journal.name,
                "country": journal.country,
                "count": count,
                "zh_count": zh_count,
                "en_count": en_count,
                "abstract_ratio": abstract_ratio
            }

        print(f"\n【数据概览】")
        print(f"  期刊总数: {len(journals)} 本")
        print(f"    - 国内期刊: {cn_count} 本")
        print(f"    - 国际期刊: {int_count} 本")
        print(f"  论文总数: {total_papers} 篇")

        print(f"\n【期刊详情】")
        for issn, stats in journal_stats.items():
            region = "国内" if stats["country"] == "CN" else "国际"
            print(f"  {stats['name']} ({region})")
            print(f"    论文数: {stats['count']} 篇 (中文{stats['zh_count']}篇/英文{stats['en_count']}篇)")
            print(f"    有摘要比例: {stats['abstract_ratio']:.1f}%")

        # 语言分布
        zh_total = sum(s["zh_count"] for s in journal_stats.values())
        en_total = sum(s["en_count"] for s in journal_stats.values())

        print(f"\n【语言分布】")
        print(f"  中文论文: {zh_total} 篇 ({zh_total/total_papers*100:.1f}%)" if total_papers > 0 else "  中文论文: 0 篇")
        print(f"  英文论文: {en_total} 篇 ({en_total/total_papers*100:.1f}%)" if total_papers > 0 else "  英文论文: 0 篇")

        # 论文年份分布
        year_stats = {}
        for paper in all_papers:
            if paper.published_date:
                year = paper.published_date.year
                year_stats[year] = year_stats.get(year, 0) + 1

        if year_stats:
            print(f"\n【年份分布】")
            for year in sorted(year_stats.keys(), reverse=True)[:5]:
                print(f"  {year}年: {year_stats[year]} 篇")

        # 论文分类（基于关键词）
        print(f"\n【论文分类】(按关键词统计)")
        keyword_counts = {}
        for paper in all_papers:
            # 使用标题中的关键词
            if paper.keywords:
                for kw in paper.keywords[:3]:  # 每篇最多取3个关键词
                    keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

        if keyword_counts:
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"  Top 10 关键词:")
            for kw, count in sorted_keywords:
                bar = "█" * min(count, 20)
                print(f"    {kw}: {count}篇 {bar}")
        else:
            # 基于标题词频分类
            title_words = ["深度学习", "机器学习", "神经网络", "图像处理", "目标检测",
                          "语义分割", "Transformer", "注意力机制", "医学", "诊断",
                          "分类", "预测", "优化", "识别", "分割"]
            word_stats = {}
            for paper in all_papers:
                title = paper.title.lower()
                for word in title_words:
                    if word.lower() in title:
                        word_stats[word] = word_stats.get(word, 0) + 1

            if word_stats:
                sorted_words = sorted(word_stats.items(), key=lambda x: x[1], reverse=True)
                print(f"  Top 关键词:")
                for word, count in sorted_words[:10]:
                    bar = "█" * min(count, 20)
                    print(f"    {word}: {count}篇 {bar}")

        print("\n" + "=" * 60)

    def generate_report(self, query: str, response=None, mode: str = "semantic"):
        """
        生成论文匹配度分析报告

        显示每篇论文的：篇名、关键词、摘要、匹配度

        Args:
            query: 检索查询
            response: 可选的已有检索结果
            mode: 搜索模式
        """
        self._ensure_initialized()

        print("\n" + "=" * 70)
        print("  论文匹配度分析报告")
        print("=" * 70)

        # 如果没有传入结果，重新执行检索（低阈值获取所有论文）
        if response is None:
            response = self.search_service.search(
                query=query,
                mode=mode,
                threshold=0.0,  # 低阈值，获取所有论文
                top_k=50
            )

        print(f"\n【检索主题】{query}")
        print(f"【搜索模式】{'精确搜索' if mode == 'exact' else '语义搜索' if mode == 'semantic' else '综合搜索'}")
        print(f"【检索期刊】{response.total_journals} 本")
        print(f"【论文总数】{response.total_papers} 篇")

        # 收集所有论文并排序
        all_papers_with_scores = []
        for result in response.results:
            journal = result.journal
            for paper in result.similar_papers:
                p = paper.paper
                all_papers_with_scores.append({
                    "journal": journal.name,
                    "country": "国内" if journal.country == "CN" else "国际",
                    "title": p.title,
                    "keywords": p.keywords,
                    "abstract": p.abstract,
                    "score": paper.score,
                    "published_date": p.published_date,
                    "authors": p.authors,
                    "url": p.url,
                })

        # 按匹配度排序
        all_papers_with_scores.sort(key=lambda x: x["score"], reverse=True)

        # 打印每篇论文详情
        print("\n" + "-" * 70)
        print(f"  论文详情（共 {len(all_papers_with_scores)} 篇，按匹配度排序）")
        print("-" * 70)

        for i, paper in enumerate(all_papers_with_scores, 1):
            pub_date = paper['published_date'].strftime("%Y-%m") if paper['published_date'] else "未知"
            print(f"\n【{i}】{paper['journal']} ({paper['country']})")
            print(f"    匹配度: {paper['score']:.4f}")
            print(f"    发表日期: {pub_date}")
            print(f"    篇名: {paper['title']}")
            print(f"    关键词: {', '.join(paper['keywords'][:10]) if paper['keywords'] else '无'}")
            abstract = paper['abstract'][:500] + "..." if paper['abstract'] and len(paper['abstract']) > 500 else (paper['abstract'] or '无')
            print(f"    摘要: {abstract}")
            if paper['authors']:
                authors_str = ", ".join(paper['authors'][:3])
                if len(paper['authors']) > 3:
                    authors_str += " 等"
                print(f"    作者: {authors_str}")
            if paper['url']:
                print(f"    链接: {paper['url']}")
            print()

        print("=" * 70)
        print("报告结束")
        print("=" * 70)

    def status(self):
        """查看系统状态"""
        self._ensure_initialized()

        print("=" * 60)
        print("系统状态")
        print("=" * 60)

        stats = self.search_service.get_statistics()
        print(f"向量化模型: {stats['model_name']}")
        print(f"向量维度: {stats['vector_dim']}")
        print(f"索引论文数: {stats['total_papers_indexed']}")
        print(f"FAISS索引大小: {stats['faiss_index_size']}")
        print()

        # 期刊列表
        journals = self.db.get_all_journals()
        print(f"已配置期刊: {len(journals)} 本")
        for journal in journals:
            count = self.db.get_journal_papers_count(journal.issn)
            print(f"  - {journal.name} ({journal.issn}): {count} 篇")

        print("=" * 60)

    def _ensure_initialized(self):
        """确保系统已初始化"""
        if self.db is None:
            self.db = DatabaseService()

        if self.search_service is None:
            embedding = get_embedding_service()
            self.search_service = SearchService(self.db, embedding)

        if self.db.get_papers_count() == 0:
            print("系统未初始化，请先运行 'init' 命令")
            sys.exit(1)


def main():
    """CLI入口函数"""
    parser = argparse.ArgumentParser(
        description="期刊论文语义匹配系统 - 面试现场AI编程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用流程:
  1. 修改 journals.json 配置面试官提供的10本期刊
  2. python main.py init -j journals.json    # 初始化
  3. python main.py search "你的研究主题"    # 检索

示例:
  # 使用自定义期刊配置（面试现场）
  python main.py init -j journals.json

  # 使用默认配置（测试用）
  python main.py init

  # 检索相似论文
  python main.py search "深度学习在医学影像中的应用"

  # 查看系统状态
  python main.py status

  # 启动Web API服务
  python main.py serve

提示:
  - 使用 -j 参数指定期刊配置文件
  - 使用 -j journals.json 从当前目录读取配置
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # init 命令
    init_parser = subparsers.add_parser("init", help="初始化系统")
    init_parser.add_argument("-j", "--journals", type=str, default=None,
                            help="期刊配置文件路径（JSON格式）")
    init_parser.add_argument("--no-default", action="store_true",
                            help="不使用默认期刊列表（必须指定-j）")

    # search 命令
    search_parser = subparsers.add_parser("search", help="检索相似论文")
    search_parser.add_argument("query", type=str, help="检索主题")
    search_parser.add_argument("--mode", type=str, choices=["exact", "semantic", "both"],
                              default="both", help="搜索模式: exact=精确搜索, semantic=语义搜索, both=综合")
    search_parser.add_argument("--threshold", type=float, default=None,
                              help="相似度阈值 (0.0-1.0)，默认根据模式自动设置")
    search_parser.add_argument("--top-k", type=int, default=5,
                              help="每期刊返回数量")
    search_parser.add_argument("--no-details", action="store_true",
                              help="不显示详细信息")

    # status 命令
    subparsers.add_parser("status", help="查看系统状态")

    # serve 命令
    serve_parser = subparsers.add_parser("serve", help="启动API服务")
    serve_parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    serve_parser.add_argument("--port", type=int, default=8000, help="监听端口")

    # import 命令
    import_parser = subparsers.add_parser("import", help="导入论文数据")
    import_parser.add_argument("file", type=str, nargs="?", default=None, help="要导入的文件路径 (JSON/CSV)")
    import_parser.add_argument("--sample", action="store_true", help="导入示例中文论文数据")

    args = parser.parse_args()

    # 创建CLI实例
    cli = CLI()

    # 执行命令
    if args.command == "init":
        journal_configs = None

        # 如果指定了期刊配置文件，加载它
        if args.journals:
            print(f"加载期刊配置: {args.journals}")
            journal_configs = load_journals_from_json(args.journals)
            cli.init(use_default=False, journal_configs=journal_configs)
        elif args.no_default:
            print("[ERROR] 必须指定 -j 参数提供期刊配置文件")
            print("示例: python main.py init -j journals.json")
            sys.exit(1)
        else:
            # 使用默认配置
            cli.init(use_default=True, journal_configs=None)

    elif args.command == "search":
        cli.search(
            query=args.query,
            mode=args.mode,
            threshold=args.threshold,
            top_k=args.top_k,
            show_details=not args.no_details
        )

    elif args.command == "status":
        cli.status()

    elif args.command == "serve":
        import uvicorn
        print(f"启动API服务: http://{args.host}:{args.port}")
        uvicorn.run(
            "journal_matcher.api.main:app",
            host=args.host,
            port=args.port,
            reload=False
        )

    elif args.command == "import":
        from journal_matcher.services.import_papers import (
            import_papers_from_json, import_papers_from_csv,
            create_sample_chinese_papers, export_papers_to_json
        )

        if args.sample:
            # 导入示例中文论文
            print("导入示例中文论文数据...")
            papers = create_sample_chinese_papers()
            cli.db = DatabaseService()

            # 保存到数据库
            cli.db.save_papers_batch(papers)
            print(f"成功导入 {len(papers)} 篇示例论文")

            # 向量化
            from journal_matcher.services.embedding import get_embedding_service
            embedding = get_embedding_service()
            cli.search_service = SearchService(cli.db, embedding)
            cli.search_service.index_papers(papers)
            print("向量化完成！")

            # 显示导入的论文
            print("\n导入的论文:")
            for p in papers:
                print(f"  - [{p.journal_name}] {p.title}")

        elif args.file:
            # 从文件导入
            file_path = args.file
            print(f"导入文件: {file_path}")

            if file_path.endswith('.json'):
                papers = import_papers_from_json(file_path)
            elif file_path.endswith('.csv'):
                papers = import_papers_from_csv(file_path)
            else:
                print("[ERROR] 不支持的文件格式，请使用 JSON 或 CSV")
                sys.exit(1)

            if papers:
                cli.db = DatabaseService()
                cli.db.save_papers_batch(papers)
                print(f"成功导入 {len(papers)} 篇论文")

                # 向量化
                from journal_matcher.services.embedding import get_embedding_service
                embedding = get_embedding_service()
                cli.search_service = SearchService(cli.db, embedding)
                cli.search_service.index_papers(papers)
                print("向量化完成！")
        else:
            print("[ERROR] 请指定要导入的文件或使用 --sample 参数")
            print("用法:")
            print("  python main.py import --sample  # 导入示例数据")
            print("  python main.py import data.json  # 从文件导入")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
