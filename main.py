"""
期刊论文语义匹配系统 - 交互式主入口
====================================

使用方法：
  python main.py              # 交互式模式
  python main.py search 主题  # 直接搜索

作者：AI Assistant
日期：2026-05-16
"""

import sys
import os
from pathlib import Path

# 设置 huggingface 镜像
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def parse_journal_input() -> list:
    """
    交互式输入10本期刊
    格式：期刊名,ISSN,国家(CN/INT)
    例如：计算机学报,0254-4164,CN
    """
    print("请输入10本期刊（格式：期刊名,ISSN,国家），每行一本")
    print("国家：C=国内，I=国际")
    print("输入空行结束输入，不足10本将用默认期刊补充")
    print("-" * 50)

    journals = []
    while len(journals) < 10:
        line = input(f"[{len(journals)+1}/10] > ").strip()
        if not line:
            break

        parts = line.split(',')
        if len(parts) >= 3:
            name, issn, country = parts[0].strip(), parts[1].strip(), parts[2].strip().upper()
            # 简化输入：C -> CN, I -> INT
            if country == 'C':
                country = 'CN'
            elif country == 'I':
                country = 'INT'
            journals.append({
                "name": name,
                "issn": issn,
                "country": country
            })
        else:
            print("格式错误，请用：期刊名,ISSN,国家")

    return journals


def interactive_mode():
    """交互式模式"""
    from journal_matcher.cli import CLI
    from journal_matcher.config import config

    cli = CLI()

    print("=" * 60)
    print("期刊论文语义匹配系统")
    print("=" * 60)
    print()

    # 询问是否自定义期刊
    print("请选择期刊配置方式：")
    print("  1) 使用默认10本期刊（直接回车）")
    print("  2) 手动输入10本期刊")
    print("> ", end="")
    choice = input().strip()

    if choice == "2":
        # 用户自定义期刊
        custom_journals = parse_journal_input()
        if custom_journals:
            print(f"\n已输入 {len(custom_journals)} 本期刊")
            cli.init(use_default=False, journal_configs=custom_journals)
        else:
            print("\n未输入期刊，使用默认配置")
            cli.init(use_default=True)
    else:
        # 使用默认
        print("\n使用默认10本期刊（国内5本 + 国际5本）")
        cli.init(use_default=True)

    print()

    # 询问搜索主题
    print("-" * 50)
    print("请输入论文主题（直接回车使用默认示例）")
    print("> ", end="")
    query = input().strip()

    if not query:
        query = "深度学习在医学影像中的应用"

    print()

    # 执行搜索
    cli.search(query=query, mode="both", threshold=None, top_k=5, show_details=True)

    # 询问是否继续搜索
    while True:
        print("-" * 50)
        print("是否继续搜索？(y/n)")
        print("> ", end="")
        again = input().strip().lower()
        if again != 'y':
            print("再见！")
            break

        print("\n请输入新的论文主题：")
        print("> ", end="")
        query = input().strip()
        if not query:
            print("主题不能为空")
            continue

        cli.search(query=query, mode="both", threshold=None, top_k=5, show_details=True)


def main():
    """主入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="期刊论文语义匹配系统")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # search 命令
    search_parser = subparsers.add_parser("search", help="检索相似论文")
    search_parser.add_argument("query", type=str, nargs="?", help="检索主题")
    search_parser.add_argument("--mode", type=str, choices=["exact", "semantic", "both"],
                              default="both", help="搜索模式: exact=精确搜索, semantic=语义搜索, both=综合")
    search_parser.add_argument("--threshold", type=float, default=None,
                              help="相似度阈值 (0.0-1.0)，默认根据模式自动设置")
    search_parser.add_argument("--top-k", type=int, default=5)

    # 其他命令
    init_parser = subparsers.add_parser("init", help="初始化系统")
    init_parser.add_argument("-j", "--journals", type=str, default=None)

    subparsers.add_parser("status", help="查看系统状态")

    serve_parser = subparsers.add_parser("serve", help="启动API服务")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    from journal_matcher.cli import CLI, load_journals_from_json

    # 无参数时进入交互式模式
    if args.command is None or args.command == "search":
        if args.command == "search" and args.query:
            # 命令行直接搜索
            cli = CLI()
            cli._ensure_initialized()
            cli.search(args.query, mode=args.mode, threshold=args.threshold,
                      top_k=args.top_k, show_details=True)
        else:
            # 交互式模式
            interactive_mode()
        return

    cli = CLI()

    if args.command == "init":
        if args.journals:
            journal_configs = load_journals_from_json(args.journals)
            cli.init(use_default=False, journal_configs=journal_configs)
        else:
            cli.init(use_default=True)

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


if __name__ == "__main__":
    main()
