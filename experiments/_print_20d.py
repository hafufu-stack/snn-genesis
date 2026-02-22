import json
d = json.load(open('results/phase20d_quadratic_penalty_log.json', 'r', encoding='utf-8'))
print("VANILLA (Linear Penalty):")
for i, r in enumerate(d['vanilla_results']):
    print(f"  E{i+1}: Acc={r['accuracy']}% Avg_s={r['avg_sigma']} End_s={r['end_sigma']} InRange={r['in_range_pct']}%")
print("QUADRATIC PENALTY:")
for i, r in enumerate(d['quadratic_results']):
    print(f"  E{i+1}: Acc={r['accuracy']}% Avg_s={r['avg_sigma']} End_s={r['end_sigma']} InRange={r['in_range_pct']}%")
print(f"\nTotal: {d['elapsed_min']} min")
