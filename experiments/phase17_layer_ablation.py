"""
phase17_layer_ablation.py — Layer Ablation: Where Does SNN Noise Actually Matter?
=================================================================================

Phase 17: Test SNN noise injection at DIFFERENT layer ranges to answer:
  "Why does MMLU show 0% alignment tax even at σ=0.10?"

Hypothesis: Knowledge (MMLU) is encoded in early layers; nuanced reasoning
(TruthfulQA) engages deeper layers where SNN L15-20 is injected.

Layer ranges tested on Mistral-7B (32 decoder layers):
  - L0-5     (very early: embedding processing)
  - L5-10    (early: basic features)
  - L10-15   (mid-low: knowledge retrieval?)
  - L15-20   (mid-high: current SNN target — reasoning?)
  - L20-25   (late: output formation)
  - L25-31   (very late: final decision)

Each range tested at σ=0.10 (to maximize visibility of effects).
Benchmarks: MMLU (8 subjects, 1600Q) and TruthfulQA MC1 (817Q).

Expected runtime: ~6 conditions × ~15 min each × 2 benchmarks ≈ 3 hours
"""

import torch
import os
import sys
import json
import gc
import time
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase5_scaleup import load_model, get_model_layers

# ─── Settings ───
SIGMA = 0.10  # High dose to maximize effect visibility

# Layer ranges to test (Mistral-7B has 32 layers: 0-31)
LAYER_RANGES = {
    "L0-5":   list(range(0, 6)),
    "L5-10":  list(range(5, 11)),
    "L10-15": list(range(10, 16)),
    "L15-20": list(range(15, 21)),   # Current SNN target
    "L20-25": list(range(20, 26)),
    "L25-31": list(range(25, 32)),
}

# MMLU subjects (same 8 as Phase 14b for direct comparison)
TARGET_SUBJECTS = [
    "elementary_mathematics",
    "high_school_mathematics",
    "conceptual_physics",
    "philosophy",
    "moral_scenarios",
    "security_studies",
    "clinical_knowledge",
    "professional_medicine",
]
MAX_PER_SUBJECT = 200

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


# ─── GPU SNN Hook ───
def make_gpu_snn_hook(sigma=0.10, seed=42):
    """GPU-native SNN-style noise hook (same as phase14b)."""
    def hook(module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * sigma
        return (hs + noise,) + args[1:]
    return hook


# ─── Dataset Loaders ───
def load_mmlu():
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    filtered = []
    subject_counts = {}
    for row in ds:
        subj = row["subject"]
        if subj in TARGET_SUBJECTS:
            cnt = subject_counts.get(subj, 0)
            if cnt < MAX_PER_SUBJECT:
                filtered.append(row)
                subject_counts[subj] = cnt + 1
    print(f"  ✅ MMLU: {len(filtered)} questions ({len(subject_counts)} subjects)")
    return filtered


def load_truthfulqa():
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    print(f"  ✅ TruthfulQA: {len(ds)} questions")
    return ds


# ─── Evaluation Functions ───
def compute_choice_logprob_mmlu(model, tokenizer, question, choices, choice_idx):
    """MMLU: compute log-prob for a specific answer choice (A/B/C/D)."""
    letters = ["A", "B", "C", "D"]
    prompt = f"Q: {question}\n"
    for i, c in enumerate(choices):
        prompt += f"{letters[i]}) {c}\n"
    prompt += "Answer: "
    answer = letters[choice_idx]
    full_text = prompt + answer

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
    total_logprob = 0.0
    n_tokens = 0
    for i in range(prompt_len - 1, full_ids.shape[1] - 1):
        next_token = full_ids[0, i + 1]
        total_logprob += log_probs[i, next_token].item()
        n_tokens += 1
    return total_logprob / max(n_tokens, 1)


def compute_choice_logprob_tqa(model, tokenizer, question, choice):
    """TruthfulQA: compute log-prob for a specific answer choice."""
    prompt = f"Q: {question}\nA: "
    full_text = prompt + choice
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
    total_logprob = 0.0
    n_tokens = 0
    for i in range(prompt_len - 1, full_ids.shape[1] - 1):
        next_token = full_ids[0, i + 1]
        total_logprob += log_probs[i, next_token].item()
        n_tokens += 1
    return total_logprob / max(n_tokens, 1)


def evaluate_mmlu(model, tokenizer, dataset, condition_name, hooks=None):
    """Evaluate MMLU accuracy with optional hooks."""
    print(f"\n{'='*60}")
    print(f"  MMLU | {condition_name}")
    print(f"  Questions: {len(dataset)}")
    print(f"{'='*60}")

    handles = []
    if hooks:
        layers = get_model_layers(model)
        for layer_idx, hook_fn in hooks:
            if layer_idx < len(layers):
                handle = layers[layer_idx].register_forward_pre_hook(hook_fn)
                handles.append(handle)

    correct = 0
    total = 0
    t0 = time.time()

    try:
        for idx, row in enumerate(dataset):
            question = row["question"]
            choices = row["choices"]
            answer = row["answer"]

            logprobs = []
            for ci in range(len(choices)):
                lp = compute_choice_logprob_mmlu(model, tokenizer, question, choices, ci)
                logprobs.append(lp)

            predicted = np.argmax(logprobs)
            if predicted == answer:
                correct += 1
            total += 1

            if (idx + 1) % 100 == 0:
                acc = correct / total * 100
                elapsed = time.time() - t0
                eta = elapsed / total * (len(dataset) - total)
                print(f"  [{idx+1}/{len(dataset)}] {acc:.1f}% ({correct}/{total}) ETA: {eta/60:.1f}min")

    finally:
        for h in handles:
            h.remove()

    accuracy = correct / total * 100 if total > 0 else 0
    elapsed = time.time() - t0
    print(f"  ✅ MMLU {condition_name}: {accuracy:.1f}% ({correct}/{total}) [{elapsed/60:.1f}min]")

    return {
        "benchmark": "MMLU",
        "condition": condition_name,
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "time_minutes": round(elapsed / 60, 1),
    }


def evaluate_truthfulqa(model, tokenizer, dataset, condition_name, hooks=None):
    """Evaluate TruthfulQA MC1 accuracy with optional hooks."""
    print(f"\n{'='*60}")
    print(f"  TruthfulQA MC1 | {condition_name}")
    print(f"  Questions: {len(dataset)}")
    print(f"{'='*60}")

    handles = []
    if hooks:
        layers = get_model_layers(model)
        for layer_idx, hook_fn in hooks:
            if layer_idx < len(layers):
                handle = layers[layer_idx].register_forward_pre_hook(hook_fn)
                handles.append(handle)

    correct = 0
    total = 0
    t0 = time.time()

    try:
        for idx, row in enumerate(dataset):
            question = row["question"]
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            correct_idx = labels.index(1)

            logprobs = []
            for choice in choices:
                lp = compute_choice_logprob_tqa(model, tokenizer, question, choice)
                logprobs.append(lp)

            predicted = np.argmax(logprobs)
            if predicted == correct_idx:
                correct += 1
            total += 1

            if (idx + 1) % 100 == 0:
                acc = correct / total * 100
                elapsed = time.time() - t0
                eta = elapsed / total * (len(dataset) - total)
                print(f"  [{idx+1}/{len(dataset)}] {acc:.1f}% ({correct}/{total}) ETA: {eta/60:.1f}min")

    finally:
        for h in handles:
            h.remove()

    accuracy = correct / total * 100 if total > 0 else 0
    elapsed = time.time() - t0
    print(f"  ✅ TruthfulQA {condition_name}: {accuracy:.1f}% ({correct}/{total}) [{elapsed/60:.1f}min]")

    return {
        "benchmark": "TruthfulQA MC1",
        "condition": condition_name,
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "time_minutes": round(elapsed / 60, 1),
    }


# ─── Visualization ───
def visualize(all_results):
    """Create dual heatmap: layer range × benchmark accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    # Organize data
    layer_names = ["Base"] + list(LAYER_RANGES.keys())

    mmlu_accs = []
    tqa_accs = []
    for name in layer_names:
        for r in all_results:
            if r["condition"] == name or (name == "Base" and r["condition"] == "Base (No Noise)"):
                if r["benchmark"] == "MMLU":
                    mmlu_accs.append(r["accuracy"])
                elif r["benchmark"] == "TruthfulQA MC1":
                    tqa_accs.append(r["accuracy"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Phase 17: Layer Ablation — Where Does SNN Noise Matter?\n"
                 f"Mistral-7B (32 layers) | σ={SIGMA}",
                 fontsize=14, fontweight="bold", color="#e94560")

    x = np.arange(len(layer_names))
    width = 0.6

    # MMLU subplot
    colors_mmlu = ["#888888"] + ["#00D4AA"] * len(LAYER_RANGES)
    bars1 = ax1.bar(x, mmlu_accs, width, color=colors_mmlu, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, acc in zip(bars1, mmlu_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylabel("MMLU Accuracy (%)", fontsize=12)
    ax1.set_title("MMLU (Factual Knowledge)", fontsize=12, color="#00D4AA")
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, fontsize=10)
    ax1.set_ylim(0, max(mmlu_accs) * 1.3 if mmlu_accs else 30)
    ax1.axhline(y=mmlu_accs[0] if mmlu_accs else 0, color="#e94560", linestyle="--", alpha=0.5, label="Baseline")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y")

    # Add tax annotations for MMLU
    if mmlu_accs:
        base = mmlu_accs[0]
        for i, acc in enumerate(mmlu_accs[1:], 1):
            delta = acc - base
            sign = "+" if delta >= 0 else ""
            color = "#00FF00" if abs(delta) < 1 else "#FF4444"
            ax1.text(i, max(mmlu_accs) * 1.15, f"Tax: {sign}{delta:.1f}%",
                     ha="center", fontsize=8, color=color, fontweight="bold")

    # TruthfulQA subplot
    colors_tqa = ["#888888"] + ["#FF6B35"] * len(LAYER_RANGES)
    bars2 = ax2.bar(x, tqa_accs, width, color=colors_tqa, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, acc in zip(bars2, tqa_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("MC1 Accuracy (%)", fontsize=12)
    ax2.set_title("TruthfulQA MC1 (Truthfulness / Reasoning)", fontsize=12, color="#FF6B35")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_names, fontsize=10)
    ax2.set_ylim(0, max(tqa_accs) * 1.3 if tqa_accs else 50)
    ax2.axhline(y=tqa_accs[0] if tqa_accs else 0, color="#e94560", linestyle="--", alpha=0.5, label="Baseline")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y")

    # Add tax annotations for TruthfulQA
    if tqa_accs:
        base = tqa_accs[0]
        for i, acc in enumerate(tqa_accs[1:], 1):
            delta = acc - base
            sign = "+" if delta >= 0 else ""
            color = "#00FF00" if abs(delta) < 1 else "#FF4444"
            ax2.text(i, max(tqa_accs) * 1.15, f"Tax: {sign}{delta:.1f}%",
                     ha="center", fontsize=8, color=color, fontweight="bold")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "phase17_layer_ablation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


# ─── Main ───
def main():
    print("=" * 60)
    print("Phase 17: Layer Ablation — Where Does SNN Noise Matter?")
    print("=" * 60)
    print(f"  σ = {SIGMA}")
    print(f"  Layer ranges: {list(LAYER_RANGES.keys())}")
    print(f"  Benchmarks: MMLU (8 subj × 200Q) + TruthfulQA MC1 (817Q)")
    t0 = time.time()

    # Load datasets
    print("\n📚 Loading datasets...")
    mmlu_data = load_mmlu()
    tqa_data = load_truthfulqa()

    # Load model
    print("\n📦 Loading Mistral-7B...")
    model, tokenizer = load_model()

    num_layers = len(get_model_layers(model))
    print(f"  ✅ {num_layers} decoder layers found")

    all_results = []

    # ─── Baseline (no noise) ───
    print("\n" + "▶" * 60)
    print("  BASELINE (No Noise)")
    print("▶" * 60)

    r_mmlu_base = evaluate_mmlu(model, tokenizer, mmlu_data, "Base (No Noise)")
    all_results.append(r_mmlu_base)

    r_tqa_base = evaluate_truthfulqa(model, tokenizer, tqa_data, "Base (No Noise)")
    all_results.append(r_tqa_base)

    # Save intermediate
    save_intermediate(all_results, t0)

    # ─── Layer ablation conditions ───
    for range_name, layer_indices in LAYER_RANGES.items():
        print(f"\n{'▶'*60}")
        print(f"  CONDITION: σ={SIGMA} @ {range_name} (layers {layer_indices})")
        print(f"{'▶'*60}")

        hook_fn = make_gpu_snn_hook(sigma=SIGMA, seed=2026)
        hooks = [(idx, hook_fn) for idx in layer_indices if idx < num_layers]

        r_mmlu = evaluate_mmlu(model, tokenizer, mmlu_data, range_name, hooks=hooks)
        all_results.append(r_mmlu)

        r_tqa = evaluate_truthfulqa(model, tokenizer, tqa_data, range_name, hooks=hooks)
        all_results.append(r_tqa)

        # Save intermediate after each condition
        save_intermediate(all_results, t0)

    # ─── Cleanup ───
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ─── Visualize ───
    print("\n🎨 Creating visualization...")
    visualize(all_results)

    # ─── Save final results ───
    output = {
        "experiment": "Phase 17: Layer Ablation",
        "sigma": SIGMA,
        "layer_ranges": {k: v for k, v in LAYER_RANGES.items()},
        "results": all_results,
        "finished": str(datetime.datetime.now()),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "phase17_layer_ablation_log.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Final results: {results_path}")

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("PHASE 17 COMPLETE — LAYER ABLATION")
    print("=" * 70)

    mmlu_base = r_mmlu_base["accuracy"]
    tqa_base = r_tqa_base["accuracy"]

    print(f"\n  {'Condition':15s} | {'MMLU':>8s} {'Tax':>8s} | {'TruthfulQA':>10s} {'Tax':>8s}")
    print(f"  {'-'*15}-+-{'-'*8}-{'-'*8}-+-{'-'*10}-{'-'*8}")

    for name in ["Base (No Noise)"] + list(LAYER_RANGES.keys()):
        mmlu_r = next((r for r in all_results if r["condition"] == name and r["benchmark"] == "MMLU"), None)
        tqa_r = next((r for r in all_results if r["condition"] == name and r["benchmark"] == "TruthfulQA MC1"), None)

        if mmlu_r and tqa_r:
            m_tax = mmlu_r["accuracy"] - mmlu_base
            t_tax = tqa_r["accuracy"] - tqa_base
            m_sign = "+" if m_tax >= 0 else ""
            t_sign = "+" if t_tax >= 0 else ""
            m_tax_str = "---" if name == "Base (No Noise)" else f"{m_sign}{m_tax:.1f}%"
            t_tax_str = "---" if name == "Base (No Noise)" else f"{t_sign}{t_tax:.1f}%"
            print(f"  {name:15s} | {mmlu_r['accuracy']:7.1f}% {m_tax_str:>8s} | {tqa_r['accuracy']:9.1f}% {t_tax_str:>8s}")

    total_min = (time.time() - t0) / 60
    print(f"\n  ⏱ Total time: {total_min:.1f} min ({total_min/60:.1f} hours)")

    # Beep
    try:
        import winsound
        winsound.Beep(800, 200)
        winsound.Beep(1000, 200)
        winsound.Beep(800, 200)
    except Exception:
        print("\a")


def save_intermediate(results, t0):
    """Save intermediate results (crash-safe)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "experiment": "Phase 17: Layer Ablation (in progress)",
        "sigma": SIGMA,
        "results": results,
        "elapsed_minutes": round((time.time() - t0) / 60, 1),
        "last_updated": str(datetime.datetime.now()),
    }
    path = os.path.join(RESULTS_DIR, "phase17_intermediate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  💾 Intermediate saved: {path}")


if __name__ == "__main__":
    main()
