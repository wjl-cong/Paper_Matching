"""Quick test for Chinese paper fetch"""
from journal_matcher.services.chinese_journal import get_chinese_journal_service
import time

service = get_chinese_journal_service()

print("="*60)
print("Test Chinese Journal Paper Fetch")
print("="*60)

# Test with both Chinese and English journal names
test_journals = [
    ("管理世界", "Management World"),  # Chinese with English alias
    ("经济研究", "Economic Research Journal"),
    ("Journal of Finance", None),
]

for cn_name, en_name in test_journals:
    name_to_use = en_name if en_name else cn_name
    print(f"\n{'='*50}")
    print(f"Fetching: {cn_name} ({name_to_use})")
    print('='*50)
    
    papers, source = service.fetch_chinese_papers(name_to_use, max_papers=3)
    
    if papers:
        print(f"SUCCESS: Got {len(papers)} papers (source: {source})")
        for i, p in enumerate(papers[:2]):
            print(f"\n  [{i+1}] {p.title}")
            print(f"      Authors: {', '.join(p.authors[:3]) if p.authors else 'N/A'}")
            print(f"      Date: {p.published_date}")
            if p.abstract:
                print(f"      Abstract: {p.abstract[:100]}...")
            if p.keywords:
                print(f"      Keywords: {', '.join(p.keywords[:5])}")
    else:
        print(f"FAILED: No papers found (source: {source})")
    
    time.sleep(1)

print("\n" + "="*60)
print("Test Complete")
print("="*60)
