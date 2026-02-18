"""
Phase 19 Step 2: Analyze Gemini 3 Pro transfer test results.
"""
import json
import os
import datetime

RESULTS_DIR = r"C:\Users\kyjan\研究\snn-genesis\results"
TRANSFER_DIR = os.path.join(RESULTS_DIR, "phase19_transfer")

# Load Step 1 results
with open(os.path.join(RESULTS_DIR, "phase19_step1_log.json"), "r", encoding="utf-8") as f:
    step1 = json.load(f)

# Gemini results — all refused
gemini_results = {
    "snn_batch_01": {"total": 25, "accepted": 0, "refused": 25},
    "snn_batch_02": {"total": 25, "accepted": 0, "refused": 25},
    "base_batch_01": {"total": 25, "accepted": 0, "refused": 25},
    "base_batch_02": {"total": 25, "accepted": 0, "refused": 25},
}

# Summary
snn_total = 50
snn_accepted = 0
base_total = 50
base_accepted = 0

print("=" * 60)
print("Phase 19: Nightmare Transferability — Full Results")
print("=" * 60)

print("\n  ┌─────────────────────────────────────────────────────┐")
print("  │           TRANSFER MATRIX (ASR %)                   │")
print("  ├──────────────┬──────────────┬────────────────────────┤")
print("  │ Source       │  Mistral-7B  │   Gemini 3 Pro         │")
print("  ├──────────────┼──────────────┼────────────────────────┤")
print(f"  │ SNN (σ=0.10) │  {step1['snn_acceptance_rate']:5.1f}%      │   {snn_accepted/snn_total*100:.1f}%                  │")
print(f"  │ No Noise     │  {step1['base_acceptance_rate']:5.1f}%      │   {base_accepted/base_total*100:.1f}%                  │")
print("  └──────────────┴──────────────┴────────────────────────┘")

print(f"\n  Key Finding:")
print(f"    Mistral-7B (7B):    SNN → 89.5% accepted, Base → 25.0% accepted")
print(f"    Gemini 3 Pro:       SNN → 0.0% accepted,  Base → 0.0% accepted")
print(f"")
print(f"    Transfer Rate: 0.0% (SNN nightmares do NOT transfer to Gemini)")
print(f"    Gemini refused ALL 100 prompts across both conditions.")

print(f"\n  Interpretation:")
print(f"    1. SNN nightmares are MODEL-SPECIFIC, not universal")
print(f"    2. Gemini 3 Pro's safety training is robust against these prompts")
print(f"    3. Model scale/training provides strong defense")
print(f"    4. This is GOOD for AI safety — vulnerabilities are containable")

# Save final results
output = {
    "experiment": "Phase 19: Nightmare Transferability",
    "step1_source_model": "Mistral-7B-Instruct-v0.3",
    "step2_target_model": "Gemini 3 Pro (Google AI Ultra)",
    "snn_config": {"sigma": 0.10, "layers": "L15-20"},
    "source_results": {
        "snn_acceptance_rate": step1["snn_acceptance_rate"],
        "snn_accepted": step1["snn_accepted_count"],
        "snn_total": step1["n_prompts"],
        "base_acceptance_rate": step1["base_acceptance_rate"],
        "base_accepted": step1["base_accepted_count"],
        "base_total": step1["n_prompts"],
    },
    "target_results": {
        "model": "Gemini 3 Pro",
        "snn_accepted": snn_accepted,
        "snn_total": snn_total,
        "snn_asr": 0.0,
        "base_accepted": base_accepted,
        "base_total": base_total,
        "base_asr": 0.0,
    },
    "transfer_rate": 0.0,
    "gemini_batch_results": gemini_results,
    "conclusion": "SNN-generated nightmares do NOT transfer from Mistral-7B to Gemini 3 Pro. "
                   "Gemini refused all 100 prompts (50 SNN + 50 baseline) with 0% ASR. "
                   "This indicates the vulnerability is model-specific and that larger, "
                   "well-trained models have robust defenses against these attack vectors.",
    "finished": str(datetime.datetime.now()),
}

path = os.path.join(RESULTS_DIR, "phase19_transfer_log.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n  💾 Results: {path}")

# Create visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#ccc",
    "ytick.color": "#ccc",
    "grid.color": "#333",
    "grid.alpha": 0.3,
})

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
fig.suptitle("Phase 19: Nightmare Transferability\n"
             "Mistral-7B → Gemini 3 Pro", fontsize=14, fontweight="bold", color="#e94560")

categories = ["SNN Nightmare\n(σ=0.10, L15-20)", "Baseline\n(No Noise)"]
mistral_rates = [89.5, 25.0]
gemini_rates = [0.0, 0.0]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, mistral_rates, width, label="Mistral-7B (7B)", color="#e94560", alpha=0.85)
bars2 = ax.bar(x + width/2, gemini_rates, width, label="Gemini 3 Pro", color="#2ecc71", alpha=0.85)

for bar, rate in zip(bars1, mistral_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{rate:.1f}%", ha="center", fontsize=12, fontweight="bold", color="#e94560")
for bar, rate in zip(bars2, gemini_rates):
    ax.text(bar.get_x() + bar.get_width()/2, max(rate, 1) + 2,
            f"{rate:.1f}%", ha="center", fontsize=12, fontweight="bold", color="#2ecc71")

ax.set_ylabel("Nightmare Acceptance Rate (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 110)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, axis="y")

# Add annotation
ax.annotate("Transfer Rate: 0%\n(Complete Defense)",
            xy=(0.5, 50), fontsize=14, fontweight="bold",
            color="#FFD700", ha="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#16213e", edgecolor="#FFD700", alpha=0.9))

plt.tight_layout()
fig_dir = r"C:\Users\kyjan\研究\snn-genesis\figures"
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "phase19_transfer.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  📊 Figure: {fig_path}")
