#!/usr/bin/env python
"""测试经管类期刊检索系统"""
import sys
sys.path.insert(0, '.')

print("测试导入...")
from journal_matcher.services.search import SearchService, BM25, extract_keywords_from_query
from journal_matcher.config import config, CN_JOURNALS_MANAGEMENT, UTD_24_JOURNALS

print("✓ 导入成功")

print("\n期刊配置验证:")
print(f"  中文经管期刊: {len(CN_JOURNALS_MANAGEMENT)} 本")
print(f"  UTD期刊: {len(UTD_24_JOURNALS)} 本")

print("\nBM25测试:")
bm25 = BM25()
bm25.fit(["深度学习在医学影像中的应用", "Machine learning for medical imaging"])
scores = bm25.search("深度学习", top_k=2)
print(f"  搜索'深度学习': {scores}")

print("\n关键词提取测试:")
kw = extract_keywords_from_query("企业创新与数字化转型")
print(f"  查询: {kw}")

print("\n✓ 所有测试通过")
