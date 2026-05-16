"""
手动导入中文论文数据
====================

由于中文期刊在国际数据库（OpenAlex、CrossRef、Semantic Scholar）中收录有限，
提供手动导入功能，允许用户从其他渠道获取数据后导入系统。

支持格式：
1. JSON 格式（推荐）
2. CSV 格式（简化）

作者：AI Assistant
日期：2026-05-16
"""

import json
import csv
import re
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..models.schemas import Paper, Journal


def parse_paper_from_dict(data: Dict[str, Any]) -> Optional[Paper]:
    """
    从字典解析 Paper 对象

    支持的字段：
    - title: 论文标题（必填）
    - authors: 作者列表
    - abstract: 摘要
    - published_date: 发表日期（YYYY-MM-DD 或 YYYY）
    - doi: DOI
    - url: 链接
    - keywords: 关键词列表
    - journal_name: 期刊名称
    - journal_issn: 期刊ISSN
    """
    try:
        title = data.get("title")
        if not title:
            return None

        # 解析日期
        pub_date = None
        pub_date_str = data.get("published_date") or data.get("date") or data.get("year")
        if pub_date_str:
            if isinstance(pub_date_str, int):
                pub_date = date(pub_date_str, 1, 1)
            elif isinstance(pub_date_str, str):
                pub_date_str = pub_date_str.strip()
                if re.match(r'^\d{4}$', pub_date_str):
                    pub_date = date(int(pub_date_str), 1, 1)
                elif '-' in pub_date_str:
                    try:
                        pub_date = date.fromisoformat(pub_date_str)
                    except:
                        pass

        # 解析作者
        authors = data.get("authors", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(',')]

        # 解析关键词
        keywords = data.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',')]

        # 判断语言
        language = data.get("language")
        if not language:
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', title))
            language = "zh" if has_chinese else "en"

        # 生成 DOI（如果没有）
        doi = data.get("doi") or f"manual:{hash(title)}"

        return Paper(
            doi=doi,
            title=title,
            abstract=data.get("abstract") or data.get("summary"),
            authors=authors,
            journal_issn=data.get("journal_issn") or data.get("issn", ""),
            journal_name=data.get("journal_name") or data.get("journal", "Unknown"),
            published_date=pub_date,
            url=data.get("url") or data.get("link", ""),
            keywords=keywords,
            language=language
        )

    except Exception as e:
        print(f"解析论文失败: {e}")
        return None


def import_papers_from_json(file_path: str) -> List[Paper]:
    """
    从 JSON 文件导入论文

    JSON 格式示例：
    {
        "papers": [
            {
                "title": "论文标题",
                "authors": ["作者1", "作者2"],
                "abstract": "摘要内容",
                "published_date": "2024-01-15",
                "doi": "10.xxxx/xxxxx",
                "keywords": ["关键词1", "关键词2"],
                "journal_name": "管理世界",
                "journal_issn": "1000-5935"
            }
        ]
    }

    或直接是论文数组：
    [
        {"title": "...", ...},
        {"title": "...", ...}
    ]
    """
    papers = []
    path = Path(file_path)

    if not path.exists():
        print(f"文件不存在: {file_path}")
        return papers

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 支持两种格式
        if isinstance(data, list):
            paper_list = data
        elif isinstance(data, dict):
            paper_list = data.get("papers", [data])
        else:
            print("JSON 格式错误")
            return papers

        for item in paper_list:
            paper = parse_paper_from_dict(item)
            if paper:
                papers.append(paper)

        print(f"成功导入 {len(papers)} 篇论文")

    except Exception as e:
        print(f"导入失败: {e}")

    return papers


def import_papers_from_csv(file_path: str) -> List[Paper]:
    """
    从 CSV 文件导入论文

    CSV 格式（第一行是表头）：
    title,authors,abstract,published_date,doi,keywords,journal_name,journal_issn
    论文标题,"作者1,作者2",摘要内容,2024-01-15,10.xxxx/xxxxx,"关键词1,关键词2",期刊名,ISSN
    """
    papers = []
    path = Path(file_path)

    if not path.exists():
        print(f"文件不存在: {file_path}")
        return papers

    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                paper = parse_paper_from_dict(row)
                if paper:
                    papers.append(paper)

        print(f"成功导入 {len(papers)} 篇论文")

    except Exception as e:
        print(f"导入失败: {e}")

    return papers


def export_papers_to_json(papers: List[Paper], file_path: str):
    """导出论文到 JSON 文件"""
    data = []
    for paper in papers:
        data.append({
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "published_date": str(paper.published_date) if paper.published_date else None,
            "doi": paper.doi,
            "url": paper.url,
            "keywords": paper.keywords,
            "journal_name": paper.journal_name,
            "journal_issn": paper.journal_issn,
            "language": paper.language
        })

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已导出 {len(papers)} 篇论文到 {file_path}")


def create_sample_chinese_papers() -> List[Paper]:
    """
    创建示例中文论文数据

    这些是模拟数据，用于演示系统功能
    实际使用时，请替换为真实数据
    """
    sample_papers = [
        {
            "title": "数字化转型对企业绩效的影响机制研究",
            "authors": ["张三", "李四", "王五"],
            "abstract": "本文基于资源基础观和动态能力理论，探讨数字化转型对企业绩效的影响机制。研究发现，数字化转型通过提升企业创新能力、优化运营效率两条路径显著促进企业绩效的提升。",
            "published_date": "2024-03-15",
            "doi": "10.3969/j.issn.1000-5935.2024.03.001",
            "keywords": ["数字化转型", "企业绩效", "创新能力", "运营效率"],
            "journal_name": "管理世界",
            "journal_issn": "1000-5935"
        },
        {
            "title": "绿色技术创新与环境污染治理的协同效应",
            "authors": ["赵六", "钱七"],
            "abstract": "本研究利用中国上市公司数据，实证检验绿色技术创新对环境污染治理的影响。研究表明，绿色技术创新能够显著降低企业污染排放，且这种效应在政府环境规制较强的地区更为明显。",
            "published_date": "2024-02-20",
            "doi": "10.3969/j.issn.0577-9154.2024.02.001",
            "keywords": ["绿色技术", "环境治理", "企业创新", "环境规制"],
            "journal_name": "经济研究",
            "journal_issn": "0577-9154"
        },
        {
            "title": "CEO特征与企业社会责任履行研究",
            "authors": ["孙八", "周九", "吴十"],
            "abstract": "基于高层梯队理论，本文研究CEO个人特征对企业社会责任履行的影。本研究丰富了企业社会责任的前因研究，为企业高层管理者选拔提供参考。",
            "published_date": "2023-12-10",
            "doi": "10.3969/j.issn.1003-2886.2023.12.001",
            "keywords": ["CEO特征", "企业社会责任", "高层梯队理论", "公司治理"],
            "journal_name": "会计研究",
            "journal_issn": "1003-2886"
        },
        {
            "title": "产业集聚与区域创新能力提升研究",
            "authors": ["郑一", "王二"],
            "abstract": "本文考察产业集聚对区域创新能力的影响及其作用机制。研究发现，产业集料通过知识溢出效应和资源共享效应促进区域创新能力的提升。",
            "published_date": "2024-01-05",
            "doi": "10.3969/j.issn.1005-2542.2024.01.001",
            "keywords": ["产业集聚", "区域创新", "知识溢出", "资源配置"],
            "journal_name": "管理科学学报",
            "journal_issn": "1005-2542"
        },
        {
            "title": "数字经济对制造业升级的影响研究",
            "authors": ["冯十二", "陈十三"],
            "abstract": "本文利用省级面板数据，实证分析数字经济对制造业升级的影响。研究结果表明，数字经济发展显著促进了制造业结构升级，这种促进作用在东部地区更为显著。",
            "published_date": "2024-04-18",
            "doi": "10.3969/j.issn.1002-5502.2024.04.001",
            "keywords": ["数字经济", "制造业升级", "产业结构", "区域差异"],
            "journal_name": "中国工业经济",
            "journal_issn": "1002-5502"
        },
    ]

    papers = []
    for item in sample_papers:
        paper = parse_paper_from_dict(item)
        if paper:
            papers.append(paper)

    return papers


# =============================================================================
# 命令行工具
# =============================================================================

def main():
    """命令行入口"""
    import sys

    if len(sys.argv) < 2:
        print("""
中文论文手动导入工具
====================

用法:
    python import_papers.py import <文件路径>     # 导入论文
    python import_papers.py sample                # 生成示例数据
    python import_papers.py template              # 生成模板文件

示例:
    # 从 JSON 导入
    python import_papers.py import papers.json

    # 从 CSV 导入
    python import_papers.py import papers.csv

    # 生成示例数据
    python import_papers.py sample
""")
        return

    command = sys.argv[1]

    if command == "import":
        if len(sys.argv) < 3:
            print("请指定要导入的文件路径")
            return

        file_path = sys.argv[2]
        if file_path.endswith('.json'):
            papers = import_papers_from_json(file_path)
        elif file_path.endswith('.csv'):
            papers = import_papers_from_csv(file_path)
        else:
            print("不支持的文件格式，请使用 JSON 或 CSV")
            return

        if papers:
            print(f"\n成功导入 {len(papers)} 篇论文:")
            for p in papers[:5]:
                print(f"  - {p.title}")

    elif command == "sample":
        print("生成示例中文论文数据...")
        papers = create_sample_chinese_papers()
        output_path = "sample_chinese_papers.json"
        export_papers_to_json(papers, output_path)
        print(f"\n示例数据已保存到: {output_path}")

    elif command == "template":
        template = {
            "papers": [
                {
                    "title": "论文标题（必填）",
                    "authors": ["作者1", "作者2"],
                    "abstract": "摘要内容",
                    "published_date": "2024-01-01",
                    "doi": "10.xxxx/xxxxx（可选）",
                    "keywords": ["关键词1", "关键词2"],
                    "journal_name": "期刊名称",
                    "journal_issn": "ISSN（可选）"
                }
            ]
        }
        output_path = "papers_template.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        print(f"模板文件已保存到: {output_path}")


if __name__ == "__main__":
    main()
