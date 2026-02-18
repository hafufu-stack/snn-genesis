"""
phase17d_low_dose.py — Low-Dose Safety Validation
===================================================

Phase 17d: Tests σ=0.01 (v3.1 operational dose) on high-accuracy
benchmarks (HellaSwag, ARC-Easy) to confirm that v3.1's "near-zero
alignment tax" claim holds on tasks where floor effects don't apply.

Also tests σ=0.05 as an intermediate data point for dose-response.

Conditions:
  - Base (no noise)
  - σ=0.01 @ L15-20  (v3.1 operational dose)
  - σ=0.05 @ L15-20  (intermediate)
  - σ=0.10 @ L15-20  (Phase 17c reference, for comparison)
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
TARGET_LAYERS = list(range(15, 21))  # L15-20 only
SIGMAS = [0.01, 0.05, 0.10]
N_HELLASWAG = 1000
N_ARC = None  # full

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def make_gpu_snn_hook(sigma=0.10):
    def hook(module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * sigma
        return (hs + noise,) + args[1:]
    return hook


def load_model():
    print(f"\n📦 Loading {MODEL_NAME}...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
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
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find decoder layers")


def compute_completion_logprob(model, tokenizer, context, completion):
    full_text = context + completion
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    completion_start = len(ctx_ids)
    total_lp = 0.0
    n_tokens = 0
    for i in range(completion_start, len(full_ids)):
        total_lp += log_probs[i - 1, full_ids[i]].item()
        n_tokens += 1
    return total_lp / max(n_tokens, 1)


def evaluate_hellaswag(model, tokenizer, dataset, condition_name, hooks=None):
    layers = get_layers(model)
    handles = []
    if hooks:
        for layer_idx, hook_fn in hooks:
            if layer_idx < len(layers):
                handles.append(layers[layer_idx].register_forward_pre_hook(hook_fn))

    correct = total = 0
    t0 = time.time()
    try:
        for idx, row in enumerate(dataset):
            ctx = row["ctx"]
            endings = row["endings"]
            label = int(row["label"])
            activity = row.get("activity_label", "")
            context = f"{activity}: {ctx}" if activity else ctx
            logprobs = [compute_completion_logprob(model, tokenizer, context, " " + e) for e in endings]
            if np.argmax(logprobs) == label:
                correct += 1
            total += 1
            if total % 200 == 0:
                print(f"  [{total}/{len(dataset)}] {correct/total*100:.1f}% ETA: {(time.time()-t0)/total*(len(dataset)-total)/60:.1f}min")
    finally:
        for h in handles:
            h.remove()
    acc = correct / total * 100
    elapsed = (time.time() - t0) / 60
    print(f"  ✅ HellaSwag | {condition_name}: {acc:.2f}% ({correct}/{total}) [{elapsed:.1f}min]")
    return {"benchmark": "HellaSwag", "condition": condition_name,
            "accuracy": round(acc, 2), "correct": correct, "total": total,
            "time_minutes": round(elapsed, 1)}


def evaluate_arc_easy(model, tokenizer, dataset, condition_name, hooks=None):
    layers = get_layers(model)
    handles = []
    if hooks:
        for layer_idx, hook_fn in hooks:
            if layer_idx < len(layers):
                handles.append(layers[layer_idx].register_forward_pre_hook(hook_fn))

    correct = total = 0
    t0 = time.time()
    try:
        for idx, row in enumerate(dataset):
            question = row["question"]
            choices = row["choices"]["text"]
            labels_list = row["choices"]["label"]
            answer_key = row["answerKey"]
            try:
                correct_idx = labels_list.index(answer_key)
            except ValueError:
                continue
            prompt = f"Question: {question}\n"
            for lbl, choice in zip(labels_list, choices):
                prompt += f"{lbl}. {choice}\n"
            prompt += "Answer:"
            logprobs = [compute_completion_logprob(model, tokenizer, prompt, " " + c) for c in choices]
            if np.argmax(logprobs) == correct_idx:
                correct += 1
            total += 1
            if total % 500 == 0:
                print(f"  [{total}/{len(dataset)}] {correct/total*100:.1f}% ETA: {(time.time()-t0)/total*(len(dataset)-total)/60:.1f}min")
    finally:
        for h in handles:
            h.remove()

    acc = correct / total * 100 if total > 0 else 0
    elapsed = (time.time() - t0) / 60
    print(f"  ✅ ARC-Easy | {condition_name}: {acc:.2f}% ({correct}/{total}) [{elapsed:.1f}min]")

    return {"benchmark": "ARC-Easy", "condition": condition_name,
            "accuracy": round(acc, 2), "correct": correct, "total": total,
            "time_minutes": round(elapsed, 1)}


def visualize(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#ccc", "ytick.color": "#ccc",
        "grid.color": "#333", "grid.alpha": 0.3,
    })

    benchmarks = ["HellaSwag", "ARC-Easy"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 17d: Low-Dose Safety Validation\n"
                 "Mistral-7B L15-20 | σ=0.01 vs 0.05 vs 0.10",
                 fontsize=14, fontweight="bold", color="#e94560")

    conditions = ["Base (No Noise)", "σ=0.01", "σ=0.05", "σ=0.10"]
    colors = ["#888888", "#00D4AA", "#FFA726", "#e94560"]

    for ax, bench in zip(axes, benchmarks):
        bench_results = [r for r in all_results if r["benchmark"] == bench]
        accs, names = [], []
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
                    f"{acc:.1f}%", ha="center", fontsize=11, fontweight="bold")
            if i > 0:
                tax = acc - base_acc
                color = "#00FF00" if abs(tax) < 3 else ("#FFA726" if abs(tax) < 10 else "#FF4444")
                ax.text(i, max(accs) * 1.08, f"Tax: {'+' if tax >= 0 else ''}{tax:.1f}%",
                        ha="center", fontsize=10, color=color, fontweight="bold")

        ax.set_title(f"{bench} (Baseline: {base_acc:.1f}%)", fontsize=13, color="#00D4AA")
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.axhline(y=base_acc, color="#e94560", linestyle="--", alpha=0.5)
        ax.set_ylim(0, max(accs) * 1.2)
        ax.grid(True, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase17d_low_dose.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")


def main():
    print("=" * 60)
    print("Phase 17d: Low-Dose Safety Validation")
    print("=" * 60)
    print(f"  Sigmas: {SIGMAS}")
    print(f"  Target layers: L15-20")
    t0 = time.time()

    # Load datasets
    print("\n📚 Loading datasets...")
    hs_full = load_dataset("Rowan/hellaswag", split="validation")
    hellaswag = hs_full.shuffle(seed=42).select(range(N_HELLASWAG)) if N_HELLASWAG else hs_full
    print(f"  ✅ HellaSwag: {len(hellaswag)}")

    arc_easy = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    print(f"  ✅ ARC-Easy: {len(arc_easy)}")

    model, tokenizer = load_model()
    num_layers = len(get_layers(model))
    print(f"  ✅ {num_layers} layers")

    all_results = []

    # Baseline
    print(f"\n{'▶'*60}\n  BASELINE\n{'▶'*60}")
    all_results.append(evaluate_hellaswag(model, tokenizer, hellaswag, "Base (No Noise)"))
    all_results.append(evaluate_arc_easy(model, tokenizer, arc_easy, "Base (No Noise)"))

    # Each sigma
    for sigma in SIGMAS:
        print(f"\n{'▶'*60}\n  σ={sigma} @ L15-20\n{'▶'*60}")
        hook_fn = make_gpu_snn_hook(sigma=sigma)
        hooks = [(idx, hook_fn) for idx in TARGET_LAYERS if idx < num_layers]
        cond_name = f"σ={sigma}"

        all_results.append(evaluate_hellaswag(model, tokenizer, hellaswag, cond_name, hooks=hooks))
        all_results.append(evaluate_arc_easy(model, tokenizer, arc_easy, cond_name, hooks=hooks))

        # Save intermediate
        intermediate = {
            "experiment": "Phase 17d: Low-Dose (in progress)",
            "results": all_results,
            "last_updated": str(datetime.datetime.now()),
        }
        with open(os.path.join(RESULTS_DIR, "phase17d_intermediate.json"), "w", encoding="utf-8") as f:
            json.dump(intermediate, f, indent=2, ensure_ascii=False)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Visualize
    print("\n🎨 Creating visualization...")
    visualize(all_results)

    # Save final
    output = {
        "experiment": "Phase 17d: Low-Dose Safety Validation",
        "sigmas": SIGMAS,
        "target_layers": TARGET_LAYERS,
        "results": all_results,
        "finished": str(datetime.datetime.now()),
    }
    path = os.path.join(RESULTS_DIR, "phase17d_low_dose_log.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Results: {path}")

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 17d COMPLETE — LOW-DOSE VALIDATION")
    print("=" * 60)
    for bench in ["HellaSwag", "ARC-Easy"]:
        br = [r for r in all_results if r["benchmark"] == bench]
        base = next((r for r in br if r["condition"] == "Base (No Noise)"), None)
        if not base:
            continue
        print(f"\n  {bench} (Base: {base['accuracy']:.2f}%)")
        print(f"  {'Condition':20s} | {'Accuracy':>8s} | {'Tax':>8s}")
        print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}")
        for r in br:
            tax = r["accuracy"] - base["accuracy"]
            tax_str = "—" if r["condition"] == "Base (No Noise)" else f"{'+' if tax >= 0 else ''}{tax:.2f}%"
            print(f"  {r['condition']:20s} | {r['accuracy']:>6.2f}% | {tax_str:>8s}")

    total_min = (time.time() - t0) / 60
    print(f"\n  ⏱ Total: {total_min:.1f} min")

    try:
        import winsound
        winsound.Beep(800, 200)
        winsound.Beep(1000, 200)
        winsound.Beep(1200, 200)
    except Exception:
        print("\a")


if __name__ == "__main__":
    main()
