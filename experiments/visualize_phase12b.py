"""
visualize_phase12b.py — Phase 12b Results Visualization
"""
import json
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load results
with open(os.path.join(RESULTS_DIR, "phase12b_log.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data["summary"]

# ─── Style ───
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
    "font.family": "sans-serif",
})

# Color palette
COLORS = {
    "Baseline (T=0)":                    "#888888",
    "Temperature (T=1.5)":               "#FF6B35",
    "SNN σ=0.05 L15-20":                 "#FF4444",
    "SNN σ=0.03 L15-20":                 "#FFA500",
    "SNN σ=0.01 L5-10":                  "#44AA44",
    "SNN σ=0.005 L5-10":                 "#00D4AA",
    "Hybrid T=1.5 + σ=0.05 L15-20":     "#E040FB",
    "Hybrid T=1.5 + σ=0.03 L15-20":     "#7C4DFF",
    "Hybrid T=1.5 + σ=0.01 L15-20":     "#448AFF",
}

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle("Phase 12b: Extended Edge of Chaos (max_tokens=200)\nTruncation-Tolerant Evaluation",
             fontsize=15, fontweight="bold", color="#e94560", y=0.98)

# ─── Plot 1: Scatter (Coherence vs Creativity) ───
ax = axes[0, 0]
for cond, stats in summary.items():
    c = COLORS.get(cond, "#666")
    marker = "D" if "Hybrid" in cond else ("s" if "Baseline" in cond else ("^" if "Temperature" in cond else "o"))
    size = 150 if "Hybrid" in cond or cond in ["Baseline (T=0)", "Temperature (T=1.5)"] else 100
    ax.scatter(stats["coherence_mean"], stats["creativity_mean"],
              c=c, marker=marker, s=size, edgecolors="white", linewidth=0.8,
              label=cond, zorder=5)

golden = Rectangle((3.5, 2.5), 2, 3, linewidth=2, edgecolor='gold',
                   facecolor='gold', alpha=0.08, label='Target Zone')
ax.add_patch(golden)
ax.set_xlabel("Coherence (1-5)")
ax.set_ylabel("Creativity (1-5)")
ax.set_title("Coherence vs Creativity", fontweight="bold")
ax.set_xlim(1.5, 5.0)
ax.set_ylim(0.8, 3.5)
ax.axhline(y=2, color="gray", linestyle="--", alpha=0.3)
ax.axvline(x=3, color="gray", linestyle="--", alpha=0.3)
ax.grid(True)
ax.legend(fontsize=6, loc="upper left", framealpha=0.7)

# ─── Plot 2: Creative Score ranked bar chart ───
ax = axes[0, 1]
sorted_conds = sorted(summary.items(), key=lambda x: x[1]["creative_score"], reverse=True)
names = [c[0] for c in sorted_conds]
scores = [c[1]["creative_score"] for c in sorted_conds]
bar_colors = [COLORS.get(n, "#666") for n in names]

short_names = []
for n in names:
    if "Hybrid" in n:
        short_names.append(n.replace("Hybrid T=1.5 + ", "H+").replace("L15-20", "L15"))
    elif "SNN" in n:
        short_names.append(n.replace("SNN ", "").replace("L15-20", "L15").replace("L5-10", "L5"))
    else:
        short_names.append(n.split("(")[0].strip())

bars = ax.barh(range(len(names)), scores, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(short_names, fontsize=8)
ax.set_xlabel("Creative Score (√(Coh × Cre))")
ax.set_title("Ranked Creative Score", fontweight="bold")
ax.invert_yaxis()
for bar, val in zip(bars, scores):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=9, fontweight="bold")
ax.grid(True, axis="x")

# ─── Plot 3: Comparison with Phase 12 ───
ax = axes[1, 0]
# Phase 12 results (from previous experiment)
phase12_scores = {
    "Baseline (T=0)": {"coherence": 3.0, "creativity": 1.87},
    "Temperature (T=1.5)": {"coherence": 3.0, "creativity": 2.53},
    "SNN σ=0.05 L15-20": {"coherence": 2.80, "creativity": 1.73},
    "SNN σ=0.03 L15-20": {"coherence": 3.0, "creativity": 1.67},
}

common = [k for k in phase12_scores if k in summary]
x = np.arange(len(common))
width = 0.35

p12_vals = [np.sqrt(phase12_scores[k]["coherence"] * phase12_scores[k]["creativity"]) for k in common]
p12b_vals = [summary[k]["creative_score"] for k in common]

b1 = ax.bar(x - width/2, p12_vals, width, label="Phase 12 (max=100)", alpha=0.7, color="#e94560")
b2 = ax.bar(x + width/2, p12b_vals, width, label="Phase 12b (max=200)", alpha=0.7, color="#0f3460")

short = [k.split("(")[0].strip() if "SNN" not in k else k.replace("SNN ", "").replace("L15-20", "\nL15") for k in common]
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=7)
ax.set_ylabel("Creative Score")
ax.set_title("Phase 12 vs 12b Comparison", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, axis="y")
for b, v in zip(b1, p12_vals):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f"{v:.2f}", ha="center", fontsize=7)
for b, v in zip(b2, p12b_vals):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f"{v:.2f}", ha="center", fontsize=7)

# ─── Plot 4: Hybrid vs Pure (dual bars) ───
ax = axes[1, 1]
hybrid_names = ["σ=0.05", "σ=0.03", "σ=0.01"]
pure_snn_coh = [summary.get("SNN σ=0.05 L15-20", {}).get("coherence_mean", 0),
                summary.get("SNN σ=0.03 L15-20", {}).get("coherence_mean", 0),
                0]  # no pure SNN σ=0.01 L15-20 in this experiment
pure_snn_cre = [summary.get("SNN σ=0.05 L15-20", {}).get("creativity_mean", 0),
                summary.get("SNN σ=0.03 L15-20", {}).get("creativity_mean", 0),
                0]
hybrid_coh = [summary.get(f"Hybrid T=1.5 + σ={s} L15-20", {}).get("coherence_mean", 0) for s in ["0.05", "0.03", "0.01"]]
hybrid_cre = [summary.get(f"Hybrid T=1.5 + σ={s} L15-20", {}).get("creativity_mean", 0) for s in ["0.05", "0.03", "0.01"]]

x = np.arange(len(hybrid_names))
w = 0.2

ax.bar(x - 1.5*w, hybrid_coh, w, label="Hybrid Coh", alpha=0.8, color="#7C4DFF")
ax.bar(x - 0.5*w, hybrid_cre, w, label="Hybrid Cre", alpha=0.8, color="#E040FB")
ax.bar(x + 0.5*w, pure_snn_coh, w, label="Pure SNN Coh", alpha=0.8, color="#0f3460")
ax.bar(x + 1.5*w, pure_snn_cre, w, label="Pure SNN Cre", alpha=0.8, color="#e94560")

# Temperature reference lines
temp_coh = summary.get("Temperature (T=1.5)", {}).get("coherence_mean", 0)
temp_cre = summary.get("Temperature (T=1.5)", {}).get("creativity_mean", 0)
ax.axhline(y=temp_coh, color="#FF6B35", linestyle="--", alpha=0.7, label=f"Temp Coh ({temp_coh:.1f})")
ax.axhline(y=temp_cre, color="#FF6B35", linestyle=":", alpha=0.7, label=f"Temp Cre ({temp_cre:.1f})")

ax.set_xticks(x)
ax.set_xticklabels(hybrid_names)
ax.set_ylabel("Score (1-5)")
ax.set_title("Hybrid vs Pure SNN (L15-20)", fontweight="bold")
ax.legend(fontsize=6, loc="upper right")
ax.grid(True, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs(FIGURES_DIR, exist_ok=True)
fig_path = os.path.join(FIGURES_DIR, "phase12b_extended.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n📊 Figure saved: {fig_path}")

# Print key findings
print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)
print(f"  🏆 Winner: Temperature (T=1.5) — Score {summary['Temperature (T=1.5)']['creative_score']:.2f}")
print(f"  📊 Baseline Coherence improved: 3.0 → {summary['Baseline (T=0)']['coherence_mean']:.1f} (max_tokens effect)")
print(f"  🔬 Best Hybrid: σ=0.01 L15-20 — Score {summary['Hybrid T=1.5 + σ=0.01 L15-20']['creative_score']:.2f}")
print(f"  ⚠️  Hybrid did NOT beat pure Temperature")
print(f"  💡 SNN noise consistently reduces creativity scores")
