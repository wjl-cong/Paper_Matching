"""
中文文献获取测试脚本
====================

用于测试中文期刊论文的获取是否正常

使用方法：
    python test_chinese_papers.py

作者：AI Assistant
日期：2026-05-16
"""

import sys


def test_chinese_paper_fetch():
    """测试中文期刊论文获取"""
    print("=" * 60)
    print("测试中文期刊论文获取")
    print("=" * 60)

    try:
        from journal_matcher.services.chinese_journal import get_chinese_journal_service

        service = get_chinese_journal_service()

        # 测试期刊列表
        test_journals = [
            "管理世界",
            "经济研究",
            "管理科学学报",
            "中国工业经济",
            "会计研究",
        ]

        for journal_name in test_journals:
            print(f"\n{'='*40}")
            print(f"获取期刊: {journal_name}")
            print('='*40)

            papers, source = service.fetch_chinese_papers(journal_name, max_papers=3)

            if papers:
                print(f"✅ 成功获取 {len(papers)} 篇论文 (来源: {source})")
                for i, paper in enumerate(papers[:2]):
                    print(f"\n  [{i+1}] {paper.title}")
                    print(f"      作者: {', '.join(paper.authors[:3]) if paper.authors else 'N/A'}")
                    print(f"      日期: {paper.published_date}")
                    if paper.abstract:
                        print(f"      摘要: {paper.abstract[:100]}...")
                    if paper.keywords:
                        print(f"      关键词: {', '.join(paper.keywords[:5])}")
            else:
                print(f"❌ 无法获取该期刊的论文")

            # 避免请求过快
            import time
            time.sleep(1)

        print("\n" + "=" * 60)
        print("✅ 中文期刊论文获取测试完成！")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_journal_issn_lookup():
    """测试 ISSN 查找"""
    print("\n" + "=" * 60)
    print("测试 ISSN 查找（通过期刊名称）")
    print("=" * 60)

    try:
        import requests

        journals = [
            "管理世界",
            "经济研究",
        ]

        for name in journals:
            print(f"\n查找: {name}")

            url = f"https://api.crossref.org/journals?query.title={name}&rows=3"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                items = data.get("message", {}).get("items", [])

                for item in items:
                    titles = item.get("title", [])
                    if titles:
                        title = titles[0]
                        issns = item.get("ISSN", [])
                        print(f"  找到: {title}")
                        print(f"    ISSN: {', '.join(issns) if issns else 'N/A'}")

            import time
            time.sleep(0.5)

        return True

    except Exception as e:
        print(f"❌ ISSN 查找失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("中文文献获取测试套件")
    print("=" * 60)

    results = {}

    # 测试中文论文获取
    results["paper_fetch"] = test_chinese_paper_fetch()

    # 测试 ISSN 查找（可选）
    print("\n是否测试 ISSN 查找？(可能会修改配置文件)")
    response = input("输入 'y' 继续，其他跳过: ").strip().lower()
    if response == 'y':
        results["issn_lookup"] = test_journal_issn_lookup()

        if results.get("issn_lookup"):
            print("\n是否更新 config.py 中的 ISSN？")
            print("(这将修正中文期刊的 ISSN)")
    else:
        results["issn_lookup"] = None

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results.items():
        if passed is None:
            status = "⏭️ 跳过"
        elif passed:
            status = "✅ 通过"
        else:
            status = "❌ 失败"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
