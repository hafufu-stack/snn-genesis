"""
visualize_phase12.py — Create comprehensive Phase 12 visualizations.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LOG_FILE = os.path.join(RESULTS_DIR, "phase12_edge_of_chaos_log.json")

with open(LOG_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data["summary"]

# ─── Color scheme ───
COLORS = {
    "Baseline (T=0)": "#6C757D",
    "Temperature (T=1.5)": "#FFC107",
    "SNN σ=0.05 L15-20 (language)": "#E53935",
    "SNN σ=0.03 L15-20 (language)": "#F4511E",
    "SNN σ=0.01 L15-20 (language)": "#FF7043",
    "SNN σ=0.005 L15-20 (language)": "#FFAB91",
    "SNN σ=0.05 L5-10 (concept)": "#1565C0",
    "SNN σ=0.03 L5-10 (concept)": "#1E88E5",
    "SNN σ=0.01 L5-10 (concept)": "#42A5F5",
    "SNN σ=0.005 L5-10 (concept)": "#90CAF9",
}

SHORT_NAMES = {
    "Baseline (T=0)": "Baseline",
    "Temperature (T=1.5)": "Temp T=1.5",
    "SNN σ=0.05 L15-20 (language)": "σ=0.05\nL15-20",
    "SNN σ=0.03 L15-20 (language)": "σ=0.03\nL15-20",
    "SNN σ=0.01 L15-20 (language)": "σ=0.01\nL15-20",
    "SNN σ=0.005 L15-20 (language)": "σ=0.005\nL15-20",
    "SNN σ=0.05 L5-10 (concept)": "σ=0.05\nL5-10",
    "SNN σ=0.03 L5-10 (concept)": "σ=0.03\nL5-10",
    "SNN σ=0.01 L5-10 (concept)": "σ=0.01\nL5-10",
    "SNN σ=0.005 L5-10 (concept)": "σ=0.005\nL5-10",
}

# ─── Figure ───
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle("Phase 12: Edge of Chaos — Finding the Golden Sigma",
             fontsize=22, fontweight='bold', color='white', y=0.97)
fig.text(0.5, 0.94, "SNN-Genesis v3  |  Mistral-7B + SNN Noise  |  Gemini LLM-as-Judge",
         fontsize=12, color='#888888', ha='center')

# ═══ Plot 1: Scatter (Coherence vs Creativity) ═══
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#16213e')

for cond, stats in summary.items():
    color = COLORS.get(cond, '#ffffff')
    ax1.scatter(stats["coherence_mean"], stats["creativity_mean"],
                s=200, c=color, edgecolors='white', linewidths=1.5,
                zorder=5, alpha=0.9)
    # Label
    offset_x = 0.03 if stats["coherence_mean"] > 1.5 else 0.05
    ax1.annotate(SHORT_NAMES.get(cond, cond).replace('\n', ' '),
                 (stats["coherence_mean"] + offset_x, stats["creativity_mean"]),
                 fontsize=7, color=color, fontweight='bold')

ax1.set_xlabel("Coherence (1-5)", color='white', fontsize=11)
ax1.set_ylabel("Creativity (1-5)", color='white', fontsize=11)
ax1.set_title("Coherence vs Creativity", color='white', fontsize=14, fontweight='bold')
ax1.set_xlim(0.5, 4.0)
ax1.set_ylim(0.5, 3.5)
ax1.tick_params(colors='white')
ax1.grid(True, alpha=0.2, color='white')
# Add danger zone
ax1.axhspan(0.5, 1.5, color='red', alpha=0.08)
ax1.axvspan(0.5, 1.5, color='red', alpha=0.08)
ax1.text(1.0, 0.7, "💀 DESTRUCTION ZONE", color='red', alpha=0.5, fontsize=9, ha='center')

# ═══ Plot 2: Creative Score bar chart ═══
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#16213e')

conditions_sorted = sorted(summary.items(), key=lambda x: x[1]["creative_score"], reverse=True)
names = [SHORT_NAMES.get(c[0], c[0]) for c in conditions_sorted]
scores = [c[1]["creative_score"] for c in conditions_sorted]
colors_sorted = [COLORS.get(c[0], '#ffffff') for c in conditions_sorted]

bars = ax2.barh(range(len(names)), scores, color=colors_sorted, edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(len(names)))
ax2.set_yticklabels(names, fontsize=8, color='white')
ax2.set_xlabel("Creative Score (√(Coh×Cre))", color='white', fontsize=11)
ax2.set_title("Creative Score Ranking", color='white', fontsize=14, fontweight='bold')
ax2.tick_params(colors='white')
ax2.set_xlim(0, 3.5)
ax2.invert_yaxis()

# Add value labels
for bar, score in zip(bars, scores):
    ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'{score:.2f}', va='center', color='white', fontsize=9, fontweight='bold')

# Highlight winner
ax2.text(scores[0] / 2, 0, "🏆", ha='center', va='center', fontsize=16)

# ═══ Plot 3: Heatmap (σ × Layer) ═══
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#16213e')

sigmas = [0.05, 0.03, 0.01, 0.005]
layers = ["L15-20\n(language)", "L5-10\n(concept)"]
layer_keys = ["L15-20 (language)", "L5-10 (concept)"]

heatmap_data = np.zeros((len(sigmas), len(layers)))
for i, sigma in enumerate(sigmas):
    for j, layer_key in enumerate(layer_keys):
        cond_name = f"SNN σ={sigma} {layer_key}"
        if cond_name in summary:
            heatmap_data[i, j] = summary[cond_name]["creative_score"]

im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=3.0)
ax3.set_xticks(range(len(layers)))
ax3.set_xticklabels(layers, fontsize=10, color='white')
ax3.set_yticks(range(len(sigmas)))
ax3.set_yticklabels([f"σ={s}" for s in sigmas], fontsize=10, color='white')
ax3.set_title("σ × Layer Heatmap", color='white', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(sigmas)):
    for j in range(len(layers)):
        val = heatmap_data[i, j]
        text_color = 'white' if val < 1.5 else 'black'
        ax3.text(j, i, f'{val:.2f}', ha='center', va='center',
                 fontsize=14, fontweight='bold', color=text_color)

plt.colorbar(im, ax=ax3, label='Creative Score', shrink=0.8)

# ═══ Plot 4: Dual bar (Coherence & Creativity) ═══
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#16213e')

all_conditions = list(summary.keys())
x = np.arange(len(all_conditions))
width = 0.35

coh_vals = [summary[c]["coherence_mean"] for c in all_conditions]
cre_vals = [summary[c]["creativity_mean"] for c in all_conditions]

bars1 = ax4.bar(x - width/2, coh_vals, width, label='Coherence',
                color='#26C6DA', edgecolor='white', linewidth=0.5)
bars2 = ax4.bar(x + width/2, cre_vals, width, label='Creativity',
                color='#FF7043', edgecolor='white', linewidth=0.5)

ax4.set_xticks(x)
ax4.set_xticklabels([SHORT_NAMES.get(c, c) for c in all_conditions],
                     fontsize=7, color='white', rotation=45, ha='right')
ax4.set_ylabel("Score (1-5)", color='white', fontsize=11)
ax4.set_title("Coherence & Creativity by Condition", color='white', fontsize=14, fontweight='bold')
ax4.tick_params(colors='white')
ax4.legend(facecolor='#16213e', edgecolor='white', labelcolor='white', fontsize=9)
ax4.set_ylim(0, 4.0)

# Add horizontal baseline
ax4.axhline(y=2.0, color='yellow', linestyle='--', alpha=0.3, linewidth=1)
ax4.text(len(all_conditions)-0.5, 2.1, "min useful", fontsize=7, color='yellow', alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.92])

output_path = os.path.join(RESULTS_DIR, "phase12_edge_of_chaos.png")
fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Saved: {output_path}")

# Also print key findings
print("\n" + "=" * 60)
print("KEY FINDINGS:")
print("=" * 60)
print(f"\n🏆 Best Creative Score: {conditions_sorted[0][0]} = {conditions_sorted[0][1]['creative_score']:.2f}")
print(f"\n📊 Layer Comparison:")
print(f"   L15-20 (language) avg score: {np.mean([summary[c]['creative_score'] for c in summary if 'L15-20' in c]):.2f}")
print(f"   L5-10  (concept)  avg score: {np.mean([summary[c]['creative_score'] for c in summary if 'L5-10' in c]):.2f}")
print(f"\n💥 L5-10 at σ≥0.03: TOTAL COLLAPSE → Coh=1.0, Cre=1.0")
print(f"🛡️  L15-20 at all σ: COHERENT → Coh≥2.8")
print(f"\n🔬 Key Insight: Layer selection >> σ value for output quality")
