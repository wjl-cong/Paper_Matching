"""
Chinese Enhancement Test - Quick Version
"""
import sys
import os

print("=" * 60)
print("Chinese Enhancement Test Suite")
print("=" * 60)

# Test 1: jieba
print("\n[1] Testing jieba...")
try:
    import jieba
    test_text = "Deep learning medical diagnosis"
    seg_list = jieba.lcut(test_text)
    print(f"    Input: {test_text}")
    print(f"    Result: {' / '.join(seg_list)}")
    print("    [OK] jieba PASSED")
    jieba_ok = True
except Exception as e:
    print(f"    [FAIL] jieba failed: {e}")
    jieba_ok = False

# Test 2: OpenHowNet (quick)
print("\n[2] Testing OpenHowNet...")
try:
    import OpenHowNet
    hownet_dict = OpenHowNet.HowNetDict(init_sim=False)
    senses = hownet_dict.get_sense("AI")
    print(f"    Query 'AI': Found {len(senses)} senses")
    if senses:
        print(f"    First sense: {senses[0].en_word}")
    print("    [OK] OpenHowNet PASSED")
    hownet_ok = True
except Exception as e:
    print(f"    [FAIL] OpenHowNet failed: {e}")
    hownet_ok = False

# Test 3: SearchService
print("\n[3] Testing SearchService integration...")
try:
    from journal_matcher.services.search import extract_keywords_from_query, CHINESE_SUPPORT_AVAILABLE
    print(f"    Chinese support available: {CHINESE_SUPPORT_AVAILABLE}")
    test_query = "Deep learning AI medical"
    kw = extract_keywords_from_query(test_query, use_jieba=True)
    print(f"    Query: {test_query}")
    print(f"    Keywords: {kw[:10]}")
    print("    [OK] SearchService PASSED")
    search_ok = True
except Exception as e:
    print(f"    [FAIL] SearchService failed: {e}")
    search_ok = False

# Summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print(f"  jieba:       {'[OK]' if jieba_ok else '[FAIL]'}")
print(f"  OpenHowNet:  {'[OK]' if hownet_ok else '[FAIL]'}")
print(f"  SearchSvc:   {'[OK]' if search_ok else '[FAIL]'}")

print("\n" + "=" * 60)
print("Status")
print("=" * 60)
if jieba_ok:
    print("[OK] Chinese tokenization available")
    print("     - BM25 search will use jieba for better Chinese results")
    print("     - config.USE_JIEBA_TOKENIZER = True")

if hownet_ok:
    print("[OK] OpenHowNet available")
    print("     - Can enable semantic expansion for Chinese")
    print("     - Set config.USE_HOWNET_EXPANSION = True")

print("\nNext step: Run 'python main.py init' to initialize the system")
