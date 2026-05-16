"""Test OpenAlex journal name search fix"""
import sys
import requests

print("=" * 60)
print("Testing OpenAlex Journal Name Search Fix")
print("=" * 60)

# Test journal name search
test_journals = [
    "管理世界",
    "会计研究",
    "经济研究",
]

for name in test_journals:
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print('='*50)
    
    from urllib.parse import quote
    encoded_name = quote(name)
    
    # Method 1: display_name.search
    filters = f"display_name.search:{encoded_name};type:article"
    url = f"https://api.openalex.org/works?filter={filters}&per_page=5&sort=publication_date:desc"
    
    print(f"URL: {url[:80]}...")
    
    try:
        r = requests.get(url, timeout=30)
        print(f"Status: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            total = data.get("meta", {}).get("count", 0)
            print(f"Total results: {total}")
            
            if results:
                for item in results[:3]:
                    title = item.get("display_name", "")[:60]
                    source = item.get("primary_location", {}).get("source", {})
                    source_name = source.get("display_name", "N/A")
                    print(f"  - [{source_name}] {title}...")
        else:
            print(f"Error: {r.text[:200]}")
            
    except Exception as e:
        print(f"Exception: {e}")
    
    import time
    time.sleep(0.5)

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
