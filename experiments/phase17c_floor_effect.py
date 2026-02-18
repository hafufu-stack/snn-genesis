"""
phase17c_floor_effect.py — Floor Effect Validation
====================================================

Phase 17c: Runs the SAME Layer Ablation as Phase 17, but on benchmarks
where Mistral-7B achieves HIGH baseline accuracy (HellaSwag ~75%, ARC-Easy ~70%).

PURPOSE: Rule out the floor effect criticism:
  "MMLU at 21% is below chance (25%), so 0% tax is trivially expected."

If HellaSwag (baseline ~75%) also shows 0% tax across all layers,
the Distributed Knowledge Hypothesis is bulletproof.

Conditions (same as Phase 17):
  - Base (no noise)
  - L0-5, L5-10, L10-15, L15-20, L20-25, L25-31
  σ=0.10 for all noisy conditions
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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# ─── Settings ───
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
SIGMA = 0.10
N_HELLASWAG = 1000   # HellaSwag samples (full=~10k, 1000 sufficient)
N_ARC = None          # ARC-Easy: use all (~2.4k)

LAYER_RANGES = {
    "L0-5":   list(range(0, 6)),
    "L5-10":  list(range(5, 11)),
    "L10-15": list(range(10, 16)),
    "L15-20": list(range(15, 21)),
    "L20-25": list(range(20, 26)),
    "L25-31": list(range(25, 32)),
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─── Hook ───
def make_gpu_snn_hook(sigma=0.10):
    """GPU-native SNN noise hook (same as Phase 17)."""
    def hook(module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * sigma
        return (hs + noise,) + args[1:]
    return hook


# ─── Model loading ───
def load_model():
    print(f"\n📦 Loading {MODEL_NAME}...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def get_layers(model):
    """Extract decoder layers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find decoder layers")


# ─── HellaSwag evaluation ───
def compute_completion_logprob(model, tokenizer, context, completion):
    """Compute log-probability of completion given context."""
    full_text = context + completion
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # (seq_len, vocab)

    # Only score the completion tokens
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    completion_start = len(ctx_ids)

    total_lp = 0.0
    n_tokens = 0
    for i in range(completion_start, len(full_ids)):
        token_id = full_ids[i]
        total_lp += log_probs[i - 1, token_id].item()
        n_tokens += 1

    return total_lp / max(n_tokens, 1)  # normalized


def evaluate_hellaswag(model, tokenizer, dataset, condition_name, hooks=None):
    """Evaluate on HellaSwag (sentence completion)."""
    layers = get_layers(model)
    handles = []
    if hooks:
        for layer_idx, hook_fn in hooks:
            if layer_idx < len(layers):
                h = layers[layer_idx].register_forward_pre_hook(hook_fn)
                handles.append(h)

    correct = 0
    total = 0
    t0 = time.time()

    try:
        for idx, row in enumerate(dataset):
            ctx = row["ctx"]
            endings = row["endings"]
            label = int(row["label"])

            # Format context
            activity = row.get("activity_label", "")
            if activity:
                context = f"{activity}: {ctx}"
            else:
                context = ctx

            # Score each ending
            logprobs = []
            for ending in endings:
                lp = compute_completion_logprob(model, tokenizer, context, " " + ending)
                logprobs.append(lp)

            predicted = np.argmax(logprobs)
            if predicted == label:
                correct += 1
            total += 1

            if total % 100 == 0:
                acc = correct / total * 100
                elapsed = (time.time() - t0) / 60
                eta = elapsed / total * (len(dataset) - total)
                print(f"  [{total}/{len(dataset)}] {acc:.1f}% ({correct}/{total}) ETA: {eta:.1f}min")
    finally:
        for h in handles:
            h.remove()

    accuracy = correct / total * 100 if total > 0 else 0
    elapsed = (time.time() - t0) / 60
    print(f"  ✅ HellaSwag | {condition_name}: {accuracy:.2f}% ({correct}/{total}) [{elapsed:.1f}min]")

    return {
        "benchmark": "HellaSwag",
        "condition": condition_name,
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "time_minutes": round(elapsed, 1),
    }


# ─── ARC-Easy evaluation ───
def evaluate_arc_easy(model, tokenizer, dataset, condition_name, hooks=None):
    """Evaluate on ARC-Easy (multiple choice QA)."""
    layers = get_layers(model)
    handles = []
    if hooks:
        for layer_idx, hook_fn in hooks:
            if layer_idx < len(layers):
                h = layers[layer_idx].register_forward_pre_hook(hook_fn)
                handles.append(h)

    correct = 0
    total = 0
    t0 = time.time()

    try:
        for idx, row in enumerate(dataset):
            question = row["question"]
            choices = row["choices"]["text"]
            labels = row["choices"]["label"]
            answer_key = row["answerKey"]

            # Find correct index
            try:
                correct_idx = labels.index(answer_key)
            except ValueError:
                continue

            # Format as MCQ
            prompt = f"Question: {question}\n"
            for i, (lbl, choice) in enumerate(zip(labels, choices)):
                prompt += f"{lbl}. {choice}\n"
            prompt += "Answer:"

            # Score each choice
            logprobs = []
            for choice in choices:
                lp = compute_completion_logprob(model, tokenizer, prompt, " " + choice)
                logprobs.append(lp)

            predicted = np.argmax(logprobs)
            if predicted == correct_idx:
                correct += 1
            total += 1

            if total % 100 == 0:
                acc = correct / total * 100
                elapsed = (time.time() - t0) / 60
                eta = elapsed / total * (len(dataset) - total)
                print(f"  [{total}/{len(dataset)}] {acc:.1f}% ({correct}/{total}) ETA: {eta:.1f}min")
    finally:
        for h in handles:
            h.remove()

    accuracy = correct / total * 100 if total > 0 else 0
    elapsed = (time.time() - t0) / 60
    print(f"  ✅ ARC-Easy | {condition_name}: {accuracy:.2f}% ({correct}/{total}) [{elapsed:.1f}min]")

    return {
        "benchmark": "ARC-Easy",
        "condition": condition_name,
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "time_minutes": round(elapsed, 1),
    }


# ─── Visualization ───
def visualize(all_results):
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

    benchmarks = sorted(set(r["benchmark"] for r in all_results))
    fig, axes = plt.subplots(len(benchmarks), 1, figsize=(14, 6 * len(benchmarks)))
    if len(benchmarks) == 1:
        axes = [axes]

    fig.suptitle("Phase 17c: Floor Effect Validation\n"
                 f"Mistral-7B (32 layers) | σ={SIGMA}",
                 fontsize=14, fontweight="bold", color="#e94560")

    conditions = ["Base (No Noise)", "L0-5", "L5-10", "L10-15", "L15-20", "L20-25", "L25-31"]
    colors = ["#888888", "#00D4AA", "#00D4AA", "#00D4AA", "#00D4AA", "#00D4AA", "#00D4AA"]

    for ax, bench in zip(axes, benchmarks):
        bench_results = [r for r in all_results if r["benchmark"] == bench]
        accs = []
        names = []
        for cond in conditions:
            r = next((x for x in bench_results if x["condition"] == cond), None)
            if r:
                accs.append(r["accuracy"])
                names.append(cond.replace("Base (No Noise)", "Base"))

        if not accs:
            continue

        x = np.arange(len(names))
        bars = ax.bar(x, accs, color=colors[:len(names)], alpha=0.85,
                      edgecolor="white", linewidth=0.5)

        base_acc = accs[0]
        for i, (bar, acc) in enumerate(zip(bars, accs)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{acc:.1f}%", ha="center", fontsize=10, fontweight="bold")
            if i > 0:
                tax = acc - base_acc
                sign = "+" if tax >= 0 else ""
                color = "#00FF00" if abs(tax) < 1 else "#FF4444"
                ax.text(i, max(accs) * 1.08, f"Tax: {sign}{tax:.1f}%",
                        ha="center", fontsize=9, color=color, fontweight="bold")

        ax.set_title(f"{bench} (Baseline: {base_acc:.1f}%)", fontsize=13, color="#00D4AA")
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.axhline(y=base_acc, color="#e94560", linestyle="--", alpha=0.5)
        ax.set_ylim(0, max(accs) * 1.2)
        ax.grid(True, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase17c_floor_effect.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")


def save_intermediate(results, t0):
    output = {
        "experiment": "Phase 17c: Floor Effect Validation (in progress)",
        "sigma": SIGMA,
        "results": results,
        "elapsed_minutes": round((time.time() - t0) / 60, 1),
        "last_updated": str(datetime.datetime.now()),
    }
    path = os.path.join(RESULTS_DIR, "phase17c_intermediate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 60)
    print("Phase 17c: Floor Effect Validation")
    print("=" * 60)
    print(f"  σ = {SIGMA}")
    print(f"  HellaSwag samples: {N_HELLASWAG}")
    print(f"  ARC-Easy: full dataset")
    t0 = time.time()

    # Load datasets
    print("\n📚 Loading datasets...")
    hellaswag_full = load_dataset("Rowan/hellaswag", split="validation")
    if N_HELLASWAG and N_HELLASWAG < len(hellaswag_full):
        hellaswag = hellaswag_full.shuffle(seed=42).select(range(N_HELLASWAG))
    else:
        hellaswag = hellaswag_full
    print(f"  ✅ HellaSwag: {len(hellaswag)} samples")

    arc_easy = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    print(f"  ✅ ARC-Easy: {len(arc_easy)} samples")

    # Load model
    model, tokenizer = load_model()
    num_layers = len(get_layers(model))
    print(f"  ✅ {num_layers} decoder layers found")

    all_results = []

    # ─── Baselines ───
    print(f"\n{'▶'*60}")
    print("  BASELINE (No Noise)")
    print(f"{'▶'*60}")

    r = evaluate_hellaswag(model, tokenizer, hellaswag, "Base (No Noise)")
    all_results.append(r)
    save_intermediate(all_results, t0)

    r = evaluate_arc_easy(model, tokenizer, arc_easy, "Base (No Noise)")
    all_results.append(r)
    save_intermediate(all_results, t0)

    # ─── Layer ablation ───
    for range_name, layer_indices in LAYER_RANGES.items():
        print(f"\n{'▶'*60}")
        print(f"  CONDITION: σ={SIGMA} @ {range_name} (layers {layer_indices})")
        print(f"{'▶'*60}")

        hook_fn = make_gpu_snn_hook(sigma=SIGMA)
        hooks = [(idx, hook_fn) for idx in layer_indices if idx < num_layers]

        r = evaluate_hellaswag(model, tokenizer, hellaswag, range_name, hooks=hooks)
        all_results.append(r)
        save_intermediate(all_results, t0)

        r = evaluate_arc_easy(model, tokenizer, arc_easy, range_name, hooks=hooks)
        all_results.append(r)
        save_intermediate(all_results, t0)

    # ─── Cleanup ───
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ─── Visualize ───
    print("\n🎨 Creating visualization...")
    visualize(all_results)

    # ─── Save final ───
    output = {
        "experiment": "Phase 17c: Floor Effect Validation",
        "sigma": SIGMA,
        "layer_ranges": {k: v for k, v in LAYER_RANGES.items()},
        "results": all_results,
        "finished": str(datetime.datetime.now()),
    }
    path = os.path.join(RESULTS_DIR, "phase17c_floor_effect_log.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Final results: {path}")

    # ─── Summary ───
    print("\n" + "=" * 60)
    print("PHASE 17c COMPLETE — FLOOR EFFECT VALIDATION")
    print("=" * 60)

    for bench in ["HellaSwag", "ARC-Easy"]:
        bench_results = [r for r in all_results if r["benchmark"] == bench]
        if not bench_results:
            continue
        base = next((r for r in bench_results if r["condition"] == "Base (No Noise)"), None)
        if not base:
            continue
        print(f"\n  {bench} (Base: {base['accuracy']:.2f}%)")
        print(f"  {'Condition':20s} | {'Accuracy':>8s} | {'Tax':>8s}")
        print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}")
        for r in bench_results:
            tax = r["accuracy"] - base["accuracy"]
            sign = "+" if tax >= 0 else ""
            tax_str = "—" if r["condition"] == "Base (No Noise)" else f"{sign}{tax:.2f}%"
            print(f"  {r['condition']:20s} | {r['accuracy']:>6.2f}% | {tax_str:>8s}")

    total_min = (time.time() - t0) / 60
    print(f"\n  ⏱ Total time: {total_min:.1f} min")

    # Beep
    try:
        import winsound
        winsound.Beep(800, 200)
        winsound.Beep(1000, 200)
        winsound.Beep(1200, 200)
    except Exception:
        print("\a")


if __name__ == "__main__":
    main()
