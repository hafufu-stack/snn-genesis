"""Quick analysis of over_refusal_analysis.json partial data"""
import re, json

f = open('results/over_refusal_analysis.json', 'r', encoding='utf-8', errors='replace')
content = f.read()
f.close()

# Find summary block
m = re.search(r'"summary": \{(.*?)\}', content, re.DOTALL)
if m:
    print("=== SUMMARY ===")
    # Parse manually
    for line in m.group(1).split('\n'):
        line = line.strip().rstrip(',')
        if ':' in line:
            print(f"  {line}")

# Find all clean_details entries
print("\n=== ALL CLEAN RESPONSES ===")
entries = re.finditer(r'"question": "(.*?)".*?"expected": "(.*?)".*?"response": "(.*?)".*?"correct": (true|false).*?"is_refusal": (true|false).*?"category": "(.*?)"', content, re.DOTALL)
for i, m in enumerate(entries, 1):
    q, expected, resp, correct, refusal, category = m.groups()
    status = "OK" if correct == "true" else ("OVER-REFUSAL" if category == "over_refusal" else "KNOWLEDGE LOSS")
    marker = "  " if correct == "true" else ">>"
    print(f"\n{marker} [{i}] Q: {q[:70]}")
    if correct != "true":
        print(f"   Expected: {expected}")
        print(f"   Response: {resp[:150]}")
        print(f"   Category: {status}")
    else:
        print(f"   -> {status}")
