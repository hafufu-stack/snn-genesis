import json

d = json.load(open("results/phase8_dpo_n100_log.json"))
print(f"Total time: {d.get('total_time_minutes', '?')} min")
print(f"Started: {d['config']['started']}")
print(f"Finished: {d.get('finished', '?')}")

print("\n=== DPO ===")
for r in d["dpo"]:
    loss_str = f", loss={r['loss']:.4f}" if r.get('loss') else ""
    print(f"  R{r['round']}: clean={r['clean_acc']:.1f}%, nm={r['nightmare_acc']:.1f}%{loss_str}")

print("\n=== SFT ===")
for r in d["sft"]:
    loss_str = f", loss={r['loss']:.4f}" if r.get('loss') else ""
    print(f"  R{r['round']}: clean={r['clean_acc']:.1f}%, nm={r['nightmare_acc']:.1f}%{loss_str}")

print("\n=== Statistics ===")
stats = d.get("statistics", {})
if stats:
    print(json.dumps(stats, indent=2))
else:
    print("  No statistics computed (binom_test error)")

print("\n=== Config ===")
cfg = d.get("config", {})
print(f"  n_test_clean: {cfg.get('n_test_clean')}")
print(f"  n_test_nightmare: {cfg.get('n_test_nightmare')}")
print(f"  n_train_clean: {cfg.get('n_train_clean')}")
print(f"  rounds: {cfg.get('rounds')}")
