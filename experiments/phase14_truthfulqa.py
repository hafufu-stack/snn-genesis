"""
phase14_truthfulqa.py — TruthfulQA Benchmark Evaluation
========================================================

Phase 14: Measure whether SNN injection affects the model's factual knowledge.

Tests:
  1. Base Mistral-7B (no noise) on TruthfulQA MC1 (817 questions)
  2. SNN Mid-Layer injection (L15-20, σ=0.10) on TruthfulQA MC1
  3. SNN Low-Dose (L15-20, σ=0.01) on TruthfulQA MC1

Goal: Show that SNN noise maintains (or improves) truthfulness.
      If Alignment Tax ≈ 0, SNN doesn't hurt factual knowledge.

Method: Multiple-choice log-likelihood evaluation.
  For each question, compute log-prob of each choice (correct + incorrect).
  Model "picks" the highest log-prob answer. MC1 accuracy = % correct picks.

Estimated GPU time: ~20 min (817 questions × 3 conditions).
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

from phase5_scaleup import (
    load_model, make_snn_hook, get_model_layers,
)

# ─── Settings ───
SIGMA_HIGH = 0.10
SIGMA_LOW = 0.01
MID_LAYERS = list(range(15, 21))
NUM_QUESTIONS = 817  # full TruthfulQA
BATCH_QUESTIONS = 50  # progress updates every N questions

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def load_truthfulqa():
    """Load TruthfulQA MC1 from Hugging Face."""
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    return ds


def compute_choice_logprob(model, tokenizer, question, choice):
    """Compute the log-probability of a choice given a question."""
    # Format as: "Q: {question}\nA: {choice}"
    prompt = f"Q: {question}\nA: "
    full_text = prompt + choice

    # Tokenize prompt and full text
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Compute log-probs for the answer tokens only
    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

    # Sum log-probs of answer tokens (after prompt)
    total_logprob = 0.0
    n_tokens = 0
    for i in range(prompt_len - 1, full_ids.shape[1] - 1):
        next_token = full_ids[0, i + 1]
        total_logprob += log_probs[i, next_token].item()
        n_tokens += 1

    # Average per token to avoid length bias
    avg_logprob = total_logprob / max(n_tokens, 1)
    return avg_logprob


def evaluate_mc1(model, tokenizer, dataset, condition_name, hooks=None):
    """Evaluate MC1 accuracy using log-likelihood scoring."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {condition_name}")
    print(f"{'='*60}")

    handles = []
    if hooks:
        layers = get_model_layers(model)
        for layer_idx, hook_fn in hooks:
            handle = layers[layer_idx].register_forward_pre_hook(hook_fn)
            handles.append(handle)

    correct = 0
    total = 0
    per_category = {}
    t0 = time.time()

    try:
        for idx, row in enumerate(dataset):
            question = row["question"]
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]

            # Find the correct answer index
            correct_idx = labels.index(1)

            # Compute log-prob for each choice
            logprobs = []
            for choice in choices:
                lp = compute_choice_logprob(model, tokenizer, question, choice)
                logprobs.append(lp)

            # Model picks the highest log-prob
            predicted = np.argmax(logprobs)
            is_correct = (predicted == correct_idx)

            if is_correct:
                correct += 1
            total += 1

            # Progress
            if (idx + 1) % BATCH_QUESTIONS == 0:
                acc = correct / total * 100
                elapsed = time.time() - t0
                eta = elapsed / total * (len(dataset) - total)
                print(f"  [{idx+1}/{len(dataset)}] Accuracy: {acc:.1f}% "
                      f"({correct}/{total}) ETA: {eta/60:.1f}min")

    finally:
        for h in handles:
            h.remove()

    accuracy = correct / total * 100 if total > 0 else 0
    elapsed = time.time() - t0

    print(f"\n  ✅ {condition_name}: {accuracy:.1f}% ({correct}/{total})")
    print(f"  ⏱ Time: {elapsed/60:.1f} min")

    return {
        "condition": condition_name,
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "time_minutes": round(elapsed / 60, 1),
    }


def visualize(results):
    """Create bar chart comparing TruthfulQA accuracy across conditions."""
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle("Phase 14: TruthfulQA MC1 Benchmark\nSNN Injection vs Baseline",
                 fontsize=14, fontweight="bold", color="#e94560")

    names = [r["condition"] for r in results]
    accs = [r["accuracy"] for r in results]
    colors = ["#888888", "#00D4AA", "#FF6B35"][:len(results)]

    bars = ax.bar(names, accs, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", fontsize=13, fontweight="bold")

    ax.set_ylabel("MC1 Accuracy (%)", fontsize=12)
    ax.set_ylim(0, max(accs) * 1.2)
    ax.grid(True, axis="y")

    # Alignment tax annotation
    if len(results) >= 2:
        base_acc = results[0]["accuracy"]
        for r in results[1:]:
            delta = r["accuracy"] - base_acc
            sign = "+" if delta >= 0 else ""
            ax.annotate(f"Alignment Tax: {sign}{delta:.1f}%",
                       xy=(0.5, 0.02), xycoords="axes fraction",
                       fontsize=10, color="#aaa", ha="center")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "phase14_truthfulqa.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


def main():
    print("=" * 60)
    print("Phase 14: TruthfulQA MC1 Benchmark Evaluation")
    print("=" * 60)
    t0 = time.time()

    # Load TruthfulQA
    print("\n📚 Loading TruthfulQA dataset...")
    dataset = load_truthfulqa()
    print(f"  ✅ {len(dataset)} questions loaded")

    # Load model
    print("\n📦 Loading model...")
    model, tokenizer = load_model()

    all_results = []

    # Condition 1: Base model (no noise)
    r1 = evaluate_mc1(model, tokenizer, dataset, "Base (No Noise)")
    all_results.append(r1)

    # Condition 2: SNN Low-Dose (σ=0.01, L15-20)
    snn_low = make_snn_hook(sigma=SIGMA_LOW, seed=2026)
    hooks_low = [(idx, snn_low) for idx in MID_LAYERS]
    r2 = evaluate_mc1(model, tokenizer, dataset, f"SNN σ={SIGMA_LOW} L15-20", hooks=hooks_low)
    all_results.append(r2)

    # Condition 3: SNN High-Dose (σ=0.10, L15-20)
    snn_high = make_snn_hook(sigma=SIGMA_HIGH, seed=2026)
    hooks_high = [(idx, snn_high) for idx in MID_LAYERS]
    r3 = evaluate_mc1(model, tokenizer, dataset, f"SNN σ={SIGMA_HIGH} L15-20", hooks=hooks_high)
    all_results.append(r3)

    # Free GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Visualize
    print("\n🎨 Creating visualization...")
    fig_path = visualize(all_results)

    # Save results
    output = {
        "experiment": "Phase 14: TruthfulQA MC1 Benchmark",
        "config": {
            "sigma_high": SIGMA_HIGH,
            "sigma_low": SIGMA_LOW,
            "mid_layers": MID_LAYERS,
            "num_questions": len(dataset),
        },
        "results": all_results,
        "finished": str(datetime.datetime.now()),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "phase14_truthfulqa_log.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Results saved: {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 14 COMPLETE — TruthfulQA MC1 BENCHMARK")
    print("=" * 60)
    base_acc = all_results[0]["accuracy"]
    for r in all_results:
        delta = r["accuracy"] - base_acc
        sign = "+" if delta >= 0 else ""
        tax = f"(Tax: {sign}{delta:.1f}%)" if r != all_results[0] else "(baseline)"
        print(f"  {r['condition']:30s}: {r['accuracy']:.1f}% {tax}")

    total_min = (time.time() - t0) / 60
    print(f"\n  ⏱ Total time: {total_min:.1f} min")

    # Beep
    try:
        import winsound
        winsound.Beep(800, 200)
    except Exception:
        print("\a")


if __name__ == "__main__":
    main()
