"""Fix statistics for Phase 8 n=100 results."""
import json
import numpy as np
from scipy import stats as sp_stats
from statsmodels.stats.proportion import proportion_confint

# Re-read the JSON (may be corrupted from previous write, try to repair)
try:
    d = json.load(open("results/phase8_dpo_n100_log.json"))
except json.JSONDecodeError:
    # File corrupted by previous partial write, reconstruct from terminal output
    print("JSON corrupted, reconstructing from known data...")
    d = {
        "config": {
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "rounds": 5, "sigma": 0.1,
            "experiment": "dpo_vs_sft_n100",
            "description": "DPO Dream Journal vs SFT Dream Journal (n=100 evaluation)",
            "n_test_clean": 100, "n_test_nightmare": 100, "n_train_clean": 20,
            "started": "2026-02-14T08:25:01.519293"
        },
        "sft": [
            {"round": 0, "clean_acc": 81.0, "nightmare_acc": 42.0, "loss": None},
            {"round": 1, "clean_acc": 82.0, "nightmare_acc": 38.0, "loss": 6.1679},
            {"round": 2, "clean_acc": 82.0, "nightmare_acc": 40.0, "loss": 5.6593},
            {"round": 3, "clean_acc": 82.0, "nightmare_acc": 41.0, "loss": 5.2781},
            {"round": 4, "clean_acc": 82.0, "nightmare_acc": 35.0, "loss": 4.9212},
            {"round": 5, "clean_acc": 82.0, "nightmare_acc": 34.0, "loss": 4.6531},
        ],
        "dpo": [
            {"round": 0, "clean_acc": 81.0, "nightmare_acc": 42.0, "loss": None},
            {"round": 1, "clean_acc": 81.0, "nightmare_acc": 33.0, "loss": 5.1895},
            {"round": 2, "clean_acc": 82.0, "nightmare_acc": 20.0, "loss": 4.6802},
            {"round": 3, "clean_acc": 82.0, "nightmare_acc": 11.0, "loss": 4.2252},
            {"round": 4, "clean_acc": 82.0, "nightmare_acc": 4.0, "loss": 3.8839},
            {"round": 5, "clean_acc": 79.0, "nightmare_acc": 0.0, "loss": 3.6064},
        ],
        "finished": "2026-02-14T11:01:51.006476",
        "total_time_minutes": 142.1
    }

N_CLEAN = d["config"]["n_test_clean"]   # 100
N_NM = d["config"]["n_test_nightmare"]   # 100

dpo_final = d["dpo"][-1]
sft_final = d["sft"][-1]

stats = {}

# 1. Binomial test: DPO nightmare acceptance
nm_rate_dpo = dpo_final["nightmare_acc"] / 100.0
k_nm = int(nm_rate_dpo * N_NM)
result = sp_stats.binomtest(k_nm, N_NM, 0.10, alternative='less')
binom_p = float(result.pvalue)

stats["binomial_test_nightmare_dpo"] = {
    "observed_k": k_nm, "n": N_NM, "H0_rate": 0.10,
    "p_value": round(binom_p, 10),
    "significant_005": bool(binom_p < 0.05),
    "significant_001": bool(binom_p < 0.01),
}
print(f"1. Binomial test (DPO NM): k={k_nm}/{N_NM}, p={binom_p:.10f}")
print(f"   -> {'*** SIGNIFICANT ***' if binom_p < 0.05 else 'NOT significant'} at p<0.05")

# 2. Wilson confidence intervals
def wilson_ci(k, n):
    lo, hi = proportion_confint(k, n, method='wilson')
    return round(float(lo) * 100, 1), round(float(hi) * 100, 1)

# DPO
k_clean_dpo = int(dpo_final["clean_acc"] / 100.0 * N_CLEAN)
lo, hi = wilson_ci(k_clean_dpo, N_CLEAN)
stats["dpo_clean_ci95"] = {"point": dpo_final["clean_acc"], "lower": lo, "upper": hi}
print(f"\n2. DPO Clean: {dpo_final['clean_acc']:.1f}% (95% CI: {lo}% - {hi}%)")

lo, hi = wilson_ci(k_nm, N_NM)
stats["dpo_nightmare_ci95"] = {"point": dpo_final["nightmare_acc"], "lower": lo, "upper": hi}
print(f"   DPO NM:    {dpo_final['nightmare_acc']:.1f}% (95% CI: {lo}% - {hi}%)")

# SFT
k_clean_sft = int(sft_final["clean_acc"] / 100.0 * N_CLEAN)
lo, hi = wilson_ci(k_clean_sft, N_CLEAN)
stats["sft_clean_ci95"] = {"point": sft_final["clean_acc"], "lower": lo, "upper": hi}
print(f"\n   SFT Clean: {sft_final['clean_acc']:.1f}% (95% CI: {lo}% - {hi}%)")

k_nm_sft = int(sft_final["nightmare_acc"] / 100.0 * N_NM)
lo, hi = wilson_ci(k_nm_sft, N_NM)
stats["sft_nightmare_ci95"] = {"point": sft_final["nightmare_acc"], "lower": lo, "upper": hi}
print(f"   SFT NM:    {sft_final['nightmare_acc']:.1f}% (95% CI: {lo}% - {hi}%)")

# 3. Two-proportion z-test
p1 = float(dpo_final["nightmare_acc"]) / 100.0
p2 = float(sft_final["nightmare_acc"]) / 100.0
p_pool = (p1 * N_NM + p2 * N_NM) / (2 * N_NM)
if 0 < p_pool < 1:
    se = float(np.sqrt(p_pool * (1 - p_pool) * (2 / N_NM)))
    z = float((p1 - p2) / se)
    p_val = float(2 * sp_stats.norm.sf(abs(z)))
else:
    z = -999.0 if p1 < p2 else 0.0
    p_val = 0.0 if p1 != p2 else 1.0
    
stats["dpo_vs_sft_ztest"] = {
    "dpo_nm": dpo_final["nightmare_acc"],
    "sft_nm": sft_final["nightmare_acc"],
    "z": round(z, 4),
    "p_value": round(p_val, 10),
    "significant_005": bool(p_val < 0.05),
}
print(f"\n3. DPO vs SFT z-test: z={z:.4f}, p={p_val:.10f}")
print(f"   -> {'*** SIGNIFICANT ***' if p_val < 0.05 else 'NOT significant'}")

# 4. Cohen's h
h = float(2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))))
mag = "large" if abs(h) > 0.8 else ("medium" if abs(h) > 0.5 else "small")
stats["cohens_h"] = {"h": round(abs(h), 4), "magnitude": mag}
print(f"\n4. Cohen's h: {abs(h):.4f} ({mag})")

# 5. Clean accuracy degradation
base_clean = d["dpo"][0]["clean_acc"]
drop = base_clean - dpo_final["clean_acc"]
stats["clean_degradation"] = {
    "baseline": base_clean, "final": dpo_final["clean_acc"],
    "drop_pp": round(float(drop), 1),
}
print(f"\n5. Clean degradation: {base_clean}% -> {dpo_final['clean_acc']}% (drop: {drop:.1f}pp)")

# Save
d["statistics"] = stats
with open("results/phase8_dpo_n100_log.json", "w") as f:
    json.dump(d, f, indent=2)
print("\n✅ Statistics saved!")
