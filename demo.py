"""
期刊论文语义匹配系统 - 演示脚本
====================================

这个脚本展示了系统的基本使用方法。
建议按顺序运行各个函数。

使用方法：
    python demo.py

作者：AI Assistant
日期：2026-05-16
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from journal_matcher import (
    DatabaseService,
    get_embedding_service,
    get_journal_service,
    SearchService,
    Journal,
    Paper
)


def demo_step1_overview():
    """演示1：系统概览"""
    print("=" * 70)
    print("步骤1：系统概览")
    print("=" * 70)
    print("""
本系统实现以下功能：
1. 从多个学术数据库（OpenAlex、CrossRef）获取期刊论文
2. 使用BGE-M3多语言向量化模型处理文本
3. 通过FAISS向量索引实现快速语义检索
4. 支持中英文跨语言检索（无需翻译）

核心难点解决方案：
- 国外期刊访问困难：使用开放API而非直接爬取网站
- 跨语言检索：使用多语言向量化模型
- 论文数据不完整：多源交叉补全
    """)


def demo_step2_config():
    """演示2：配置查看"""
    print("\n" + "=" * 70)
    print("步骤2：查看系统配置")
    print("=" * 70)

    from journal_matcher.config import config

    print(f"向量化模型: {config.embedding.MODEL_NAME}")
    print(f"向量维度: {config.embedding.EMBEDDING_DIM}")
    print(f"检索召回数: {config.retrieval.TOP_K_RECALL}")
    print(f"相似度阈值: {config.retrieval.SIMILARITY_THRESHOLD}")
    print()
    print("默认期刊配置:")
    for country, journals in config.DEFAULT_JOURNALS.items():
        country_name = "国内" if country == "CN" else "国际"
        print(f"  [{country_name}]")
        for j in journals:
            print(f"    - {j['name']} (ISSN: {j['issn']})")


def demo_step3_database():
    """演示3：数据库操作"""
    print("\n" + "=" * 70)
    print("步骤3：数据库操作")
    print("=" * 70)

    db = DatabaseService()
    print(f"数据库路径: {db.db_path}")
    print(f"数据库连接: OK")

    # 清空现有数据（演示用）
    print("\n清空现有数据...")
    db.clear_all_data()
    print("已清空")

    # 创建期刊
    print("\n添加期刊...")
    journal = Journal(
        issn="0028-0836",
        name="Nature",
        country="INT",
        description="国际顶级综合期刊"
    )
    db.save_journal(journal)
    print(f"已添加: {journal.name}")

    # 查询期刊
    print("\n查询期刊...")
    saved_journal = db.get_journal("0028-0836")
    if saved_journal:
        print(f"  ISSN: {saved_journal.issn}")
        print(f"  名称: {saved_journal.name}")
        print(f"  国家: {saved_journal.country}")


def demo_step4_embedding():
    """演示4：向量化服务"""
    print("\n" + "=" * 70)
    print("步骤4：向量化服务")
    print("=" * 70)

    print("加载向量化模型（首次可能需要下载，约1GB）...")
    embedding = get_embedding_service()

    print(f"模型: {embedding.model_name}")
    print(f"维度: {embedding.embedding_dim}")
    print(f"状态: {'就绪' if embedding.is_ready() else '未就绪'}")

    # 测试向量化
    print("\n测试向量化...")

    # 中文文本
    chinese_text = "深度学习在医学影像诊断中的应用研究"
    chinese_vec = embedding.encode(chinese_text)
    print(f"  中文文本: {chinese_text}")
    print(f"  向量形状: {chinese_vec.shape}")
    print(f"  向量范数: {float((chinese_vec ** 2).sum() ** 0.5):.4f}")

    # 英文文本
    english_text = "Deep learning for medical image diagnosis"
    english_vec = embedding.encode(english_text)
    print(f"\n  英文文本: {english_text}")
    print(f"  向量形状: {english_vec.shape}")

    # 计算相似度
    similarity = embedding.compute_similarity(chinese_vec, english_vec)
    print(f"\n  中英文相似度: {similarity:.4f}")
    print("  （由于使用多语言模型，语义相似的文本相似度较高）")


def demo_step5_data_models():
    """演示5：数据模型"""
    print("\n" + "=" * 70)
    print("步骤5：数据模型")
    print("=" * 70)

    # 创建论文对象
    paper = Paper(
        doi="10.1038/nature12373",
        title="Deep learning for medical image analysis",
        abstract="This paper presents a deep learning approach for automated medical image analysis. We demonstrate that convolutional neural networks can achieve comparable performance to human experts in detecting various diseases from radiographic images.",
        authors=["John Smith", "Jane Doe", "Wei Zhang"],
        journal_issn="0028-0836",
        journal_name="Nature",
        published_date=None,
        url="https://doi.org/10.1038/nature12373",
        keywords=["deep learning", "medical imaging", "neural networks"],
        language="en"
    )

    print("论文对象示例:")
    print(f"  DOI: {paper.doi}")
    print(f"  标题: {paper.title}")
    print(f"  作者: {', '.join(paper.authors)}")
    print(f"  关键词: {', '.join(paper.keywords)}")
    print(f"  数据完整性: {'完整' if paper.is_complete() else '不完整'}")
    print()
    print("可搜索文本:")
    print(f"  {paper.get_searchable_text()}")


def demo_step6_search_flow():
    """演示6：检索流程"""
    print("\n" + "=" * 70)
    print("步骤6：检索流程说明")
    print("=" * 70)

    print("""
检索流程分为两个阶段：

【第一阶段：向量召回（Vector Recall）】
┌─────────────────────────────────────────────────────────────────┐
│  用户查询: "深度学习在医学影像中的应用"                          │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────────────────────────┐                   │
│  │  BGE-M3 向量化                           │                   │
│  │  转换为 1024 维向量                       │                   │
│  └──────────────────────────────────────────┘                   │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────────────────────────┐                   │
│  │  FAISS 向量索引检索                      │                   │
│  │  快速找到 Top-20 候选论文                │                   │
│  └──────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘

【第二阶段：结果处理】
┌─────────────────────────────────────────────────────────────────┐
│  对 Top-20 候选论文：                                          │
│  1. 按期刊分组                                                  │
│  2. 应用相似度阈值过滤（默认 0.5）                              │
│  3. 按相似度排序                                                │
│  4. 返回每期刊 Top-5                                           │
└─────────────────────────────────────────────────────────────────┘

【输出示例】
{
  "total_journals": 10,
  "found_count": 3,
  "results": [
    {
      "journal": {"name": "Nature", ...},
      "found": true,
      "similar_papers": [
        {
          "title": "Deep learning in medical imaging",
          "score": 0.823,
          "abstract_snippet": "..."
        }
      ]
    },
    ...
  ]
}
    """)


def demo_step7_api_usage():
    """演示7：API使用示例"""
    print("\n" + "=" * 70)
    print("步骤7：API使用示例")
    print("=" * 70)

    print("""
【启动服务】
    uvicorn journal_matcher.api.main:app --host 0.0.0.0 --port 8000

【API端点】
1. 健康检查
   GET /health

2. 初始化系统（抓取论文+向量化）
   POST /api/init
   Body: {"use_default": true}

3. 语义检索
   POST /api/search
   Body: {
     "query": "深度学习在医学影像中的应用",
     "top_k": 5,
     "threshold": 0.5
   }

4. 获取期刊列表
   GET /api/journals

5. 获取期刊论文
   GET /api/journals/{issn}/papers

【Swagger文档】
启动服务后访问：http://localhost:8000/docs
    """)


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              期刊论文语义匹配系统 - 演示脚本                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # 运行演示
    demo_step1_overview()
    demo_step2_config()
    demo_step3_database()
    demo_step4_embedding()
    demo_step5_data_models()
    demo_step6_search_flow()
    demo_step7_api_usage()

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("""
下一步建议：
1. 运行 'python main.py init' 初始化系统并抓取论文
2. 运行 'python main.py search "你的研究主题"' 进行检索
3. 运行 'python main.py serve' 启动Web API服务
4. 访问 http://localhost:8000/docs 查看API文档
    """)


if __name__ == "__main__":
    main()
