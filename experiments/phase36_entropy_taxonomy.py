"""
Phase 36: Entropy Taxonomy — Is the Entropy Ordering Universal?
================================================================

Phase 34 discovered: Math(4.38) < Creative(4.92) < Factual(5.27) on Mistral-7B.
This experiment tests whether this ordering is universal across 3 architectures.

NO training. NO CfC. NO PPO. Just pure inference and entropy measurement.
Each model processes 60 prompts (20 per task) and we measure output entropy.

If Math < Creative < Factual holds across all 3 architectures →
  "Output entropy is a universal physical fingerprint of task type"

Models:
  1. Mistral-7B-Instruct-v0.3  (Meta)
  2. Qwen/Qwen2.5-7B-Instruct  (Alibaba)
  3. microsoft/Phi-3-mini-4k-instruct  (Microsoft)

Usage:
    python experiments/phase36_entropy_taxonomy.py
"""

import os, sys, json, time, random, gc
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scipy import stats as sp_stats

SEED = 2026
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS = [
    {"name": "mistralai/Mistral-7B-Instruct-v0.3", "short": "Mistral-7B", "org": "Meta"},
    {"name": "Qwen/Qwen2.5-7B-Instruct", "short": "Qwen2.5-7B", "org": "Alibaba"},
    {"name": "microsoft/Phi-3-mini-4k-instruct", "short": "Phi-3-mini", "org": "Microsoft"},
]

# ─── Same prompts as Phase 34 ───

FACTUAL_PROMPTS = [
    "What is the capital of France?", "Who wrote Romeo and Juliet?",
    "What year did World War II end?", "What is the chemical symbol for gold?",
    "Who painted the Mona Lisa?", "What is the largest planet in our solar system?",
    "What is the speed of light in km/s?", "Who developed the theory of relativity?",
    "What is the boiling point of water in Celsius?", "What is the smallest country in the world?",
    "Who invented the telephone?", "What is the chemical formula for water?",
    "What year was the Declaration of Independence signed?", "What is the tallest mountain in the world?",
    "Who was the first person to walk on the moon?", "What is the capital of Japan?",
    "What element has the atomic number 6?", "Who wrote the Origin of Species?",
    "What is the largest ocean on Earth?", "What year did the Berlin Wall fall?",
]

CREATIVE_PROMPTS = [
    "Describe a color that doesn't exist in the visible spectrum.",
    "Invent a new fundamental law of physics that could explain dark energy.",
    "Write a short poem about the sound of silence from a deaf musician's perspective.",
    "Design a holiday that celebrates the concept of entropy.",
    "Describe what music would look like if you could see sound waves.",
    "Invent a sport that could be played in zero gravity.",
    "Write a recipe for a dish that tastes like nostalgia.",
    "Create a new type of weather that has never existed.",
    "Describe the autobiography of a cloud.",
    "Invent a language that can only express emotions, not facts.",
    "Write a haiku about the taste of mathematics.",
    "Describe what happens when two different dreams collide.",
    "Invent a new musical instrument made from impossible materials.",
    "Write a love letter from one star to another.",
    "Describe a garden where the plants grow backwards in time.",
    "Invent a philosophical paradox about artificial consciousness.",
    "Write a news report from the year 3000.",
    "Write a business plan for a company that sells shadows.",
    "Imagine a language where words change meaning based on the listener's mood.",
    "Describe the autobiography of a single electron.",
]

MATH_PROMPTS = [
    "What is 17 + 28?", "What is 156 - 89?", "What is 12 × 7?",
    "What is 144 / 12?", "What is 23 + 45 + 12?", "What is 8 × 9?",
    "What is 200 - 137?", "What is 15 × 15?", "What is 1000 / 8?",
    "What is 99 + 101?", "What is 7 × 11?", "What is 500 - 267?",
    "What is 25 × 4?", "What is 360 / 9?", "What is 88 + 77?",
    "What is 13 × 17?", "What is 7! (7 factorial)?", "What is 1024 / 32?",
    "What is the next prime after 29?", "What is 13 × 13?",
]


def load_model(model_name):
    """Load model with 4-bit quantization"""
    print(f"\n📦 Loading {model_name}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    print(f"  ✅ Loaded")
    return model, tokenizer


def measure_entropy(model, tokenizer, prompt):
    """Measure output entropy (next-token probability distribution) for a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # Last token logits
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
    
    # Also measure top-k concentration
    top5 = torch.topk(probs, 5).values.sum().item()
    top10 = torch.topk(probs, 10).values.sum().item()
    top50 = torch.topk(probs, 50).values.sum().item()
    
    return {
        "entropy": entropy,
        "top5_prob": top5,
        "top10_prob": top10,
        "top50_prob": top50,
    }


def measure_model(model, tokenizer, model_short):
    """Measure entropy for all 60 prompts on one model"""
    print(f"\n{'═'*60}")
    print(f"  📊 Measuring entropy: {model_short}")
    print(f"{'═'*60}")
    
    results = {"factual": [], "creative": [], "math": []}
    
    for i, p in enumerate(FACTUAL_PROMPTS):
        m = measure_entropy(model, tokenizer, p)
        m["prompt"] = p
        results["factual"].append(m)
        if (i + 1) % 10 == 0:
            avg = np.mean([r["entropy"] for r in results["factual"]])
            print(f"  Factual [{i+1}/20] avg_entropy={avg:.3f}")
    
    for i, p in enumerate(CREATIVE_PROMPTS):
        m = measure_entropy(model, tokenizer, p)
        m["prompt"] = p
        results["creative"].append(m)
        if (i + 1) % 10 == 0:
            avg = np.mean([r["entropy"] for r in results["creative"]])
            print(f"  Creative [{i+1}/20] avg_entropy={avg:.3f}")
    
    for i, p in enumerate(MATH_PROMPTS):
        m = measure_entropy(model, tokenizer, f"Answer with just the number. {p}")
        m["prompt"] = p
        results["math"].append(m)
        if (i + 1) % 10 == 0:
            avg = np.mean([r["entropy"] for r in results["math"]])
            print(f"  Math [{i+1}/20] avg_entropy={avg:.3f}")
    
    # Compute stats
    ent_f = [r["entropy"] for r in results["factual"]]
    ent_c = [r["entropy"] for r in results["creative"]]
    ent_m = [r["entropy"] for r in results["math"]]
    
    mean_f, std_f = np.mean(ent_f), np.std(ent_f)
    mean_c, std_c = np.mean(ent_c), np.std(ent_c)
    mean_m, std_m = np.mean(ent_m), np.std(ent_m)
    
    # Statistical tests: Kruskal-Wallis (non-parametric ANOVA)
    kw_stat, kw_p = sp_stats.kruskal(ent_f, ent_c, ent_m)
    
    # Pairwise Mann-Whitney U tests
    mw_fc_stat, mw_fc_p = sp_stats.mannwhitneyu(ent_f, ent_c, alternative='two-sided')
    mw_fm_stat, mw_fm_p = sp_stats.mannwhitneyu(ent_f, ent_m, alternative='two-sided')
    mw_cm_stat, mw_cm_p = sp_stats.mannwhitneyu(ent_c, ent_m, alternative='two-sided')
    
    # Determine ordering
    means = {"factual": mean_f, "creative": mean_c, "math": mean_m}
    ordering = sorted(means.keys(), key=lambda k: means[k])
    ordering_str = " < ".join([f"{t}({means[t]:.2f})" for t in ordering])
    
    # Check if Phase 34 ordering (Math < Creative < Factual) holds
    phase34_ordering = (mean_m < mean_c < mean_f)
    
    print(f"\n  📊 {model_short} Results:")
    print(f"     Factual:  {mean_f:.3f} ± {std_f:.3f}")
    print(f"     Creative: {mean_c:.3f} ± {std_c:.3f}")
    print(f"     Math:     {mean_m:.3f} ± {std_m:.3f}")
    print(f"     Ordering: {ordering_str}")
    print(f"     Kruskal-Wallis H={kw_stat:.3f}, p={kw_p:.4f}")
    print(f"     Phase 34 ordering (M<C<F): {'✅ CONFIRMED' if phase34_ordering else '❌ DIFFERENT'}")
    
    return {
        "model_short": model_short,
        "factual": {"mean": round(mean_f, 4), "std": round(std_f, 4),
                    "values": [round(e, 4) for e in ent_f]},
        "creative": {"mean": round(mean_c, 4), "std": round(std_c, 4),
                     "values": [round(e, 4) for e in ent_c]},
        "math": {"mean": round(mean_m, 4), "std": round(std_m, 4),
                 "values": [round(e, 4) for e in ent_m]},
        "ordering": ordering_str,
        "phase34_ordering_holds": phase34_ordering,
        "statistics": {
            "kruskal_wallis": {"H": round(kw_stat, 4), "p": round(kw_p, 6)},
            "mann_whitney_FC": {"U": round(mw_fc_stat, 2), "p": round(mw_fc_p, 6)},
            "mann_whitney_FM": {"U": round(mw_fm_stat, 2), "p": round(mw_fm_p, 6)},
            "mann_whitney_CM": {"U": round(mw_cm_stat, 2), "p": round(mw_cm_p, 6)},
        },
        "top_k_concentration": {
            "factual_top5": round(np.mean([r["top5_prob"] for r in results["factual"]]), 4),
            "creative_top5": round(np.mean([r["top5_prob"] for r in results["creative"]]), 4),
            "math_top5": round(np.mean([r["top5_prob"] for r in results["math"]]), 4),
            "factual_top50": round(np.mean([r["top50_prob"] for r in results["factual"]]), 4),
            "creative_top50": round(np.mean([r["top50_prob"] for r in results["creative"]]), 4),
            "math_top50": round(np.mean([r["top50_prob"] for r in results["math"]]), 4),
        },
        "per_prompt": results,
    }


def visualize_entropy_taxonomy(all_results):
    """Create comprehensive comparison figure"""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Phase 36: Entropy Taxonomy — Is the Entropy Ordering Universal?\n"
                 "Pure inference, no training, no CfC. Just measuring LLM output entropy.",
                 fontsize=14, fontweight="bold")
    
    colors = {"factual": "#2ecc71", "creative": "#e74c3c", "math": "#3498db"}
    
    # ─── Panel 1: Box plots per model ───
    ax1 = fig.add_subplot(2, 2, 1)
    positions_base = np.arange(len(all_results)) * 4
    for i, res in enumerate(all_results):
        data = [res["factual"]["values"], res["creative"]["values"], res["math"]["values"]]
        pos = [positions_base[i], positions_base[i] + 1, positions_base[i] + 2]
        bp = ax1.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=True)
        for patch, color in zip(bp["boxes"], [colors["factual"], colors["creative"], colors["math"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    tick_positions = [p + 1 for p in positions_base]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([r["model_short"] for r in all_results])
    ax1.set_ylabel("Output Entropy")
    ax1.set_title("Entropy Distribution by Task & Model")
    ax1.legend([plt.Rectangle((0,0),1,1,fc=c,alpha=0.6) for c in colors.values()],
               colors.keys(), loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    
    # ─── Panel 2: Mean entropy comparison (grouped bar) ───
    ax2 = fig.add_subplot(2, 2, 2)
    x = np.arange(len(all_results))
    width = 0.25
    for i, (task, color) in enumerate(colors.items()):
        means = [r[task]["mean"] for r in all_results]
        stds = [r[task]["std"] for r in all_results]
        ax2.bar(x + i * width, means, width, yerr=stds, label=task,
                color=color, alpha=0.7, capsize=3)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([r["model_short"] for r in all_results])
    ax2.set_ylabel("Mean Entropy ± σ")
    ax2.set_title("Mean Entropy by Task & Model")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    
    # ─── Panel 3: Ordering consistency check ───
    ax3 = fig.add_subplot(2, 2, 3)
    for i, res in enumerate(all_results):
        y_pos = len(all_results) - 1 - i
        for task, color in colors.items():
            mean_val = res[task]["mean"]
            ax3.scatter(mean_val, y_pos, c=color, s=200, zorder=5, edgecolors="white", linewidth=2)
            ax3.annotate(f'{task[0].upper()}\n{mean_val:.2f}', (mean_val, y_pos),
                        ha='center', va='center', fontsize=7, fontweight='bold')
        # Draw connecting line
        means = [res[task]["mean"] for task in ["math", "creative", "factual"]]
        ax3.plot([min(means), max(means)], [y_pos, y_pos], 'k-', alpha=0.3, linewidth=1)
    
    ax3.set_yticks(range(len(all_results)))
    ax3.set_yticklabels([r["model_short"] for r in reversed(all_results)])
    ax3.set_xlabel("Output Entropy")
    ax3.set_title("Entropy Ordering Across Models\n(M=Math, C=Creative, F=Factual)")
    ax3.grid(True, alpha=0.3, axis="x")
    
    # ─── Panel 4: Verdict ───
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    
    # Count how many models match Phase 34 ordering
    matches = sum(1 for r in all_results if r["phase34_ordering_holds"])
    
    verdict_lines = ["ENTROPY TAXONOMY RESULTS", "=" * 40, ""]
    for r in all_results:
        marker = "✅" if r["phase34_ordering_holds"] else "❌"
        verdict_lines.append(f"{marker} {r['model_short']}: {r['ordering']}")
        kw = r["statistics"]["kruskal_wallis"]
        sig = "***" if kw["p"] < 0.001 else "**" if kw["p"] < 0.01 else "*" if kw["p"] < 0.05 else "n.s."
        verdict_lines.append(f"   Kruskal-Wallis H={kw['H']:.2f}, p={kw['p']:.4f} ({sig})")
        verdict_lines.append("")
    
    verdict_lines.append("=" * 40)
    if matches == 3:
        verdict_lines.append("🎆 UNIVERSAL ORDERING CONFIRMED!")
        verdict_lines.append("Math < Creative < Factual across 3 architectures")
        verdict_lines.append("→ Entropy is a physical fingerprint of task type")
    elif matches >= 2:
        verdict_lines.append(f"🔬 PARTIAL UNIVERSAL: {matches}/3 match")
        verdict_lines.append("→ Architecture-dependent variation exists")
    else:
        verdict_lines.append(f"❌ NOT UNIVERSAL: only {matches}/3 match")
        verdict_lines.append("→ Entropy ordering is model-specific")
    
    # Top-K concentration
    verdict_lines.append("")
    verdict_lines.append("Top-5 Token Concentration:")
    for r in all_results:
        tk = r["top_k_concentration"]
        verdict_lines.append(f"  {r['model_short']}: F={tk['factual_top5']:.3f} "
                           f"C={tk['creative_top5']:.3f} M={tk['math_top5']:.3f}")
    
    ax4.text(0.05, 0.95, "\n".join(verdict_lines), transform=ax4.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase36_entropy_taxonomy.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


def main():
    start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    
    all_results = []
    
    for model_info in MODELS:
        model, tokenizer = load_model(model_info["name"])
        result = measure_model(model, tokenizer, model_info["short"])
        result["organization"] = model_info["org"]
        all_results.append(result)
        
        # Free GPU memory before loading next model
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  🗑️ {model_info['short']} unloaded, GPU memory freed")
    
    # Visualization
    fig_path = visualize_entropy_taxonomy(all_results)
    
    # Cross-model analysis
    print(f"\n{'═'*60}")
    print(f"  🏆 GRAND SUMMARY: Entropy Taxonomy")
    print(f"{'═'*60}")
    
    matches = sum(1 for r in all_results if r["phase34_ordering_holds"])
    
    for r in all_results:
        print(f"\n  {r['model_short']} ({r['organization']}):")
        print(f"    Factual:  {r['factual']['mean']:.3f} ± {r['factual']['std']:.3f}")
        print(f"    Creative: {r['creative']['mean']:.3f} ± {r['creative']['std']:.3f}")
        print(f"    Math:     {r['math']['mean']:.3f} ± {r['math']['std']:.3f}")
        print(f"    Ordering: {r['ordering']}")
        marker = "✅" if r["phase34_ordering_holds"] else "❌"
        print(f"    Phase 34 ordering (M<C<F): {marker}")
    
    print(f"\n  Universal ordering: {matches}/3 architectures match")
    
    if matches == 3:
        print(f"  🎆🎆🎆 UNIVERSAL LAW: Entropy IS a task fingerprint!")
    elif matches >= 2:
        print(f"  🔬 Partial universality: {matches}/3")
    else:
        print(f"  ❌ Not universal: architecture-dependent")
    
    # Save results
    elapsed = time.time() - start
    output = {
        "experiment": "Phase 36: Entropy Taxonomy",
        "elapsed_minutes": round(elapsed / 60, 1),
        "models": [m["short"] for m in MODELS],
        "n_prompts_per_task": 20,
        "results": [{k: v for k, v in r.items() if k != "per_prompt"} for r in all_results],
        "per_prompt_data": {r["model_short"]: r["per_prompt"] for r in all_results},
        "cross_model_summary": {
            "phase34_ordering_matches": matches,
            "total_models": len(MODELS),
            "universal": matches == len(MODELS),
        },
        "figure_path": fig_path,
    }
    
    log_path = os.path.join(RESULTS_DIR, "phase36_entropy_taxonomy_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"  📁 Results: {log_path}")
    print(f"  📊 Figure: {fig_path}")


if __name__ == "__main__":
    main()
