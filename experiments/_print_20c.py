import json
d = json.load(open('results/phase20c_exploration_bonus_log.json', 'r', encoding='utf-8'))
print("VANILLA:")
for i, r in enumerate(d['vanilla_results']):
    print(f"  E{i+1}: Acc={r['accuracy']}% Avg_s={r['avg_sigma']} End_s={r['end_sigma']}")
print("EXPLORATION:")
for i, r in enumerate(d['exploration_results']):
    print(f"  E{i+1}: Acc={r['accuracy']}% Avg_s={r['avg_sigma']} End_s={r['end_sigma']}")
print(f"\nTotal: {d['elapsed_min']} min")
