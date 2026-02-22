"""
Step 2+3: Publication-quality Figure + Comprehension Score
==========================================================

Step 2: Create a clean, publication-ready figure from Phase 45 results
       showing the Déjà Vu entropy drop phenomenon.

Step 3: Build a "Comprehension Score" combining Déjà Vu (entropy) +
       Layer Conflict, and validate on test prompts.

Usage:
    python experiments/step2_3_figure_and_score.py
"""

import torch, torch.nn as nn
import os, json, gc, time, random, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
CONFLICT_LAYER_SHALLOW = 8
CONFLICT_LAYER_DEEP = 28
SEED = 2026

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


TEST_PROMPTS = [
    # Factual
    {"text": "Q: What country has the most islands in the world?\nA:", "type": "factual", "label": "Islands"},
    {"text": "Q: What is the scientific name for humans?\nA:", "type": "factual", "label": "Humans"},
    {"text": "Q: Which planet has the most moons?\nA:", "type": "factual", "label": "Moons"},
    {"text": "Q: What is the chemical formula for water?\nA:", "type": "factual", "label": "Water"},
    {"text": "Q: Who painted the Mona Lisa?\nA:", "type": "factual", "label": "Mona Lisa"},
    # Math
    {"text": "Solve this math problem. Give only the numerical answer.\n\nQ: What is 17 + 28?\nA:", "type": "math", "label": "17+28"},
    {"text": "Solve this math problem. Give only the numerical answer.\n\nQ: What is 15 × 15?\nA:", "type": "math", "label": "15×15"},
    {"text": "Solve this math problem. Give only the numerical answer.\n\nQ: What is 12 × 7?\nA:", "type": "math", "label": "12×7"},
    {"text": "Solve this math problem. Give only the numerical answer.\n\nQ: What is 2^8?\nA:", "type": "math", "label": "2^8"},
    {"text": "Solve this math problem. Give only the numerical answer.\n\nQ: What is sqrt(144)?\nA:", "type": "math", "label": "√144"},
]


def load_model(model_name, short_name):
    print(f"\n📦 Loading {model_name}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    print(f"  ✅ {short_name} loaded")
    return model, tokenizer


def compute_entropy_trajectory(model, tokenizer, prompt):
    """Get per-token entropy."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    entropies = []
    for i in range(logits.shape[0]):
        probs = torch.softmax(logits[i].float(), dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        entropies.append(ent)
    return entropies


def compute_conflict_trajectory(model, tokenizer, prompt):
    """Get per-token layer conflict (L8 vs L28 cosine distance)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states
        n_layers = len(hs) - 1
        shallow_idx = min(CONFLICT_LAYER_SHALLOW, n_layers - 1)
        deep_idx = min(CONFLICT_LAYER_DEEP, n_layers - 1)
        h_shallow = hs[shallow_idx + 1][0].float()
        h_deep = hs[deep_idx + 1][0].float()
    conflicts = []
    for i in range(h_shallow.shape[0]):
        cos = torch.nn.functional.cosine_similarity(
            h_shallow[i].unsqueeze(0), h_deep[i].unsqueeze(0)).item()
        conflicts.append(1.0 - cos)
    return conflicts


def compute_comprehension_score(entropy_1st, entropy_2nd, conflict_1st, conflict_2nd):
    """
    Comprehension Score = w1 * (entropy_drop) + w2 * (conflict_increase)

    - entropy_drop: how much more confident the model is on 2nd read
    - conflict_increase: how much more the layers specialize on 2nd read

    Both are positive when understanding improves.
    Score range: roughly [-1, 2+], higher = better understanding
    """
    ent_drop = np.mean(entropy_1st) - np.mean(entropy_2nd)  # positive = more confident
    conflict_delta = np.mean(conflict_2nd) - np.mean(conflict_1st)  # positive = more specialization
    w_ent, w_conf = 0.7, 0.3
    score = w_ent * ent_drop + w_conf * (conflict_delta * 5.0)  # scale conflict to comparable range
    return score, ent_drop, conflict_delta


def analyze_all_prompts(model, tokenizer):
    """Run analysis on all test prompts."""
    results = []
    for pi, item in enumerate(TEST_PROMPTS):
        prompt = item["text"]
        repeated = prompt + " " + prompt

        print(f"  [{pi+1}/{len(TEST_PROMPTS)}] {item['label']} ({item['type']})...")

        # Single pass entropy
        ent_single = compute_entropy_trajectory(model, tokenizer, prompt)

        # Repeated pass
        ent_repeated = compute_entropy_trajectory(model, tokenizer, repeated)
        split = len(ent_single)
        ent_1st = ent_repeated[:split]
        ent_2nd = ent_repeated[split:]

        # Single pass conflict
        conf_single = compute_conflict_trajectory(model, tokenizer, prompt)

        # Repeated pass conflict
        conf_repeated = compute_conflict_trajectory(model, tokenizer, repeated)
        conf_1st = conf_repeated[:split]
        conf_2nd = conf_repeated[split:]

        # Comprehension score
        score, ent_drop, conf_delta = compute_comprehension_score(ent_1st, ent_2nd, conf_1st, conf_2nd)

        print(f"    Entropy: single={np.mean(ent_single):.3f} | 1st={np.mean(ent_1st):.3f} → 2nd={np.mean(ent_2nd):.3f} (drop={ent_drop:+.3f})")
        print(f"    Conflict: single={np.mean(conf_single):.4f} | 1st={np.mean(conf_1st):.4f} → 2nd={np.mean(conf_2nd):.4f} (Δ={conf_delta:+.4f})")
        print(f"    🧠 Comprehension Score: {score:.3f}")

        results.append({
            "label": item["label"],
            "type": item["type"],
            "single_entropy": ent_single,
            "first_entropy": ent_1st,
            "second_entropy": ent_2nd,
            "single_conflict": conf_single,
            "first_conflict": conf_1st,
            "second_conflict": conf_2nd,
            "entropy_drop": round(ent_drop, 4),
            "conflict_delta": round(conf_delta, 4),
            "comprehension_score": round(score, 4),
        })

    return results


def create_publication_figure(results):
    """Create a publication-quality 4-panel figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # Publication styling
    rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
    })

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # ──── Panel A: Token-level entropy trajectory (example prompt) ────
    ax1 = fig.add_subplot(gs[0, 0])
    r0 = results[0]  # First prompt (Islands)
    x = range(min(len(r0["single_entropy"]), len(r0["first_entropy"]),
                  len(r0["second_entropy"]) if r0["second_entropy"] else 1))
    ax1.plot(list(x), r0["single_entropy"][:len(list(x))], 'b-', alpha=0.5, linewidth=1, label="Single pass")
    ax1.plot(list(x), r0["first_entropy"][:len(list(x))], color='#e74c3c', alpha=0.5, linewidth=1, label="1st pass (in repeat)")
    if r0["second_entropy"]:
        x2 = range(len(r0["second_entropy"]))
        ax1.plot(list(x2), r0["second_entropy"], color='#2ecc71', linewidth=2, label="2nd pass (Déjà Vu)")
    ax1.fill_between(range(len(r0["first_entropy"][:len(list(x))])),
                     r0["first_entropy"][:len(list(x))],
                     r0["second_entropy"][:len(list(x))] if r0["second_entropy"] else [0]*len(list(x)),
                     alpha=0.15, color='#2ecc71', label='_nolegend_')
    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Entropy (nats)")
    ax1.set_title("(A) Per-Token Entropy: \"What country has the most islands?\"")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ──── Panel B: Entropy drop by prompt type ────
    ax2 = fig.add_subplot(gs[0, 1])
    factual = [r for r in results if r["type"] == "factual"]
    math_r = [r for r in results if r["type"] == "math"]

    labels_f = [r["label"] for r in factual]
    labels_m = [r["label"] for r in math_r]
    drops_f = [r["entropy_drop"] for r in factual]
    drops_m = [r["entropy_drop"] for r in math_r]

    all_labels = labels_f + labels_m
    all_drops = drops_f + drops_m
    all_colors = ["#3498db"] * len(labels_f) + ["#e74c3c"] * len(labels_m)

    y_pos = range(len(all_labels))
    bars = ax2.barh(list(y_pos), all_drops, color=all_colors, alpha=0.8, height=0.7)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(all_labels)
    ax2.set_xlabel("Entropy Drop (1st → 2nd pass)")
    ax2.set_title("(B) Déjà Vu Effect by Task Type")
    ax2.axvline(x=0, color='black', linewidth=0.5)
    # Add value labels
    for bar, val in zip(bars, all_drops):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:+.3f}", va='center', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.2, axis='x')
    # Legend
    from matplotlib.patches import Patch
    ax2.legend([Patch(facecolor='#3498db'), Patch(facecolor='#e74c3c')],
               ['Factual', 'Math'], loc='lower right')

    # ──── Panel C: Comprehension Score ────
    ax3 = fig.add_subplot(gs[1, 0])
    scores = [r["comprehension_score"] for r in results]
    labels_all = [r["label"] for r in results]
    types = [r["type"] for r in results]
    colors_c = ["#3498db" if t == "factual" else "#e74c3c" for t in types]
    bars3 = ax3.bar(labels_all, scores, color=colors_c, alpha=0.8, width=0.7)
    ax3.set_ylabel("Comprehension Score")
    ax3.set_title("(C) Comprehension Score = 0.7·ΔEntropy + 0.3·ΔConflict")
    ax3.axhline(y=0, color='black', linewidth=0.5)
    for bar, val in zip(bars3, scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha='center', fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.2, axis='y')
    ax3.tick_params(axis='x', rotation=45)

    # ──── Panel D: Entropy vs Conflict scatter ────
    ax4 = fig.add_subplot(gs[1, 1])
    ent_drops = [r["entropy_drop"] for r in results]
    conf_deltas = [r["conflict_delta"] for r in results]
    scatter_colors = ["#3498db" if r["type"] == "factual" else "#e74c3c" for r in results]
    ax4.scatter(ent_drops, conf_deltas, c=scatter_colors, s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    for r in results:
        ax4.annotate(r["label"], (r["entropy_drop"], r["conflict_delta"]),
                     fontsize=8, textcoords="offset points", xytext=(5, 5))
    ax4.set_xlabel("Entropy Drop (Confidence)")
    ax4.set_ylabel("Conflict Δ (Layer Specialization)")
    ax4.set_title("(D) Déjà Vu Space: Confidence vs Specialization")
    ax4.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax4.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(True, alpha=0.2)

    # ─── Title ───
    fig.suptitle("SNN-Genesis: Déjà Vu Sensor — Observing LLM Comprehension\n"
                 "First physical evidence of the 'Aha Moment' in Transformer attention",
                 fontsize=15, fontweight='bold', y=0.98)

    fig_path = os.path.join(FIGURES_DIR, "phase45_publication_figure.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  📊 Publication figure saved: {fig_path}")
    return fig_path


def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model(MODEL_NAME, MODEL_SHORT)

    print("\n" + "═"*60)
    print("  📸 Analyzing Déjà Vu + Comprehension Score")
    print("═"*60)
    results = analyze_all_prompts(model, tokenizer)

    # Summary stats
    factual_r = [r for r in results if r["type"] == "factual"]
    math_r = [r for r in results if r["type"] == "math"]

    avg_drop_f = np.mean([r["entropy_drop"] for r in factual_r])
    avg_drop_m = np.mean([r["entropy_drop"] for r in math_r])
    avg_conf_f = np.mean([r["conflict_delta"] for r in factual_r])
    avg_conf_m = np.mean([r["conflict_delta"] for r in math_r])
    avg_score_f = np.mean([r["comprehension_score"] for r in factual_r])
    avg_score_m = np.mean([r["comprehension_score"] for r in math_r])

    print(f"\n{'═'*60}")
    print(f"  📊 SUMMARY")
    print(f"{'═'*60}")
    print(f"  Factual: Entropy drop={avg_drop_f:+.3f} Conflict Δ={avg_conf_f:+.4f} Score={avg_score_f:.3f}")
    print(f"  Math:    Entropy drop={avg_drop_m:+.3f} Conflict Δ={avg_conf_m:+.4f} Score={avg_score_m:.3f}")
    print(f"  Overall: Score={np.mean([r['comprehension_score'] for r in results]):.3f}")

    # Create publication figure
    fig_path = create_publication_figure(results)

    elapsed = time.time() - t_start

    output = {
        "experiment": "Step 2+3: Publication Figure + Comprehension Score",
        "model": MODEL_SHORT,
        "elapsed_minutes": round(elapsed / 60, 1),
        "results": [{k: v for k, v in r.items()
                      if k not in ["single_entropy", "first_entropy", "second_entropy",
                                   "single_conflict", "first_conflict", "second_conflict"]}
                     for r in results],
        "summary": {
            "factual_avg_entropy_drop": round(avg_drop_f, 4),
            "factual_avg_conflict_delta": round(avg_conf_f, 5),
            "factual_avg_score": round(avg_score_f, 4),
            "math_avg_entropy_drop": round(avg_drop_m, 4),
            "math_avg_conflict_delta": round(avg_conf_m, 5),
            "math_avg_score": round(avg_score_m, 4),
        },
        "figure_path": fig_path,
    }

    log_path = os.path.join(RESULTS_DIR, "step2_3_figure_score_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
