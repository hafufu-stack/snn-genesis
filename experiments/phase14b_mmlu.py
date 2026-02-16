"""
phase14b_mmlu.py — MMLU Benchmark Evaluation (v3 — GPU-native hooks)
=====================================================================

v3 Fix: Uses GPU-native noise hooks instead of CPU ChaoticReservoir.
The Reservoir approach is too slow for 1600-question benchmarks (CPU bottleneck).

For alignment tax measurement, what matters is the noise MAGNITUDE (σ),
not the exact chaotic correlation structure. We keep the SNN's key property
(structured, correlated noise) but compute it on GPU with PyTorch.

Reuses Base result from first run: 21.2% (339/1600).
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
SIGMA_HIGH = 0.10
SIGMA_LOW = 0.01
MID_LAYERS = list(range(15, 21))

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
BATCH_PROGRESS = 50

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def make_gpu_snn_hook(sigma=0.10, seed=42):
    """
    GPU-native SNN-style noise hook.
    Uses spatially-correlated noise (smoothed Gaussian) to approximate
    the chaotic reservoir's structured perturbation, but runs entirely on GPU.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    def hook(module, args):
        hs = args[0]
        # Generate base noise on GPU
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        # Add temporal correlation: mix with a low-freq component
        # This approximates the SNN reservoir's correlated output
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq  # structured component
        noise = noise * sigma
        return (hs + noise,) + args[1:]

    return hook


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
    print(f"  ✅ {len(filtered)} questions loaded ({len(subject_counts)} subjects)")
    return filtered


def compute_choice_logprob(model, tokenizer, question, choices, choice_idx):
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


def evaluate_mmlu(model, tokenizer, dataset, condition_name, hooks=None):
    """Evaluate MMLU accuracy using log-likelihood."""
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
    per_subject = {}
    t0 = time.time()

    try:
        for idx, row in enumerate(dataset):
            question = row["question"]
            choices = row["choices"]
            answer = row["answer"]
            subject = row["subject"]

            logprobs = []
            for ci in range(len(choices)):
                lp = compute_choice_logprob(model, tokenizer, question, choices, ci)
                logprobs.append(lp)

            predicted = np.argmax(logprobs)
            is_correct = (predicted == answer)

            if is_correct:
                correct += 1
            total += 1

            if subject not in per_subject:
                per_subject[subject] = {"correct": 0, "total": 0}
            per_subject[subject]["total"] += 1
            if is_correct:
                per_subject[subject]["correct"] += 1

            if (idx + 1) % BATCH_PROGRESS == 0:
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

    per_subject_acc = {}
    for subj, stats in sorted(per_subject.items()):
        s_acc = stats["correct"] / stats["total"] * 100
        per_subject_acc[subj] = round(s_acc, 1)
        print(f"    {subj:30s}: {s_acc:5.1f}% ({stats['correct']}/{stats['total']})")

    print(f"\n  ✅ {condition_name}: {accuracy:.1f}% ({correct}/{total})")
    print(f"  ⏱ Time: {elapsed/60:.1f} min")

    return {
        "condition": condition_name,
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "per_subject": per_subject_acc,
        "time_minutes": round(elapsed / 60, 1),
    }


def visualize(results):
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 14b: MMLU Benchmark — SNN Injection vs Baseline",
                 fontsize=14, fontweight="bold", color="#e94560")

    ax = axes[0]
    names = [r["condition"] for r in results]
    accs = [r["accuracy"] for r in results]
    colors = ["#888888", "#00D4AA", "#FF6B35"][:len(results)]

    bars = ax.bar(names, accs, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", fontsize=13, fontweight="bold")

    ax.set_ylabel("MMLU Accuracy (%)", fontsize=12)
    ax.set_ylim(0, max(accs) * 1.3)
    ax.set_title("Overall MMLU Accuracy")
    ax.grid(True, axis="y")

    base_acc = results[0]["accuracy"]
    tax_texts = []
    for r in results[1:]:
        delta = r["accuracy"] - base_acc
        sign = "+" if delta >= 0 else ""
        tax_texts.append(f"{r['condition']}: Tax {sign}{delta:.1f}%")
    if tax_texts:
        ax.annotate("\n".join(tax_texts), xy=(0.5, 0.02),
                   xycoords="axes fraction", fontsize=9, color="#aaa", ha="center")

    ax = axes[1]
    subjects = sorted(results[0]["per_subject"].keys())
    x = np.arange(len(subjects))
    width = 0.25

    for i, r in enumerate(results):
        subject_accs = [r["per_subject"].get(s, 0) for s in subjects]
        ax.barh(x + i * width, subject_accs, width, color=colors[i],
                alpha=0.8, label=r["condition"])

    short_names = [s.replace("_", " ").replace("high school ", "HS ")[:20]
                   for s in subjects]
    ax.set_yticks(x + width)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Accuracy (%)", fontsize=10)
    ax.set_title("Per-Subject Breakdown")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "phase14b_mmlu.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


def main():
    print("=" * 60)
    print("Phase 14b: MMLU Benchmark (v3 — GPU-native hooks)")
    print("=" * 60)
    t0 = time.time()

    print("\n📚 Loading MMLU subset...")
    dataset = load_mmlu()

    print("\n📦 Loading model...")
    model, tokenizer = load_model()

    # Hardcoded Base result from first run
    base_result = {
        "condition": "Base (No Noise)",
        "accuracy": 21.19,
        "correct": 339,
        "total": 1600,
        "per_subject": {
            "clinical_knowledge": 19.0,
            "conceptual_physics": 29.0,
            "elementary_mathematics": 22.0,
            "high_school_mathematics": 21.0,
            "moral_scenarios": 23.5,
            "philosophy": 18.0,
            "professional_medicine": 17.0,
            "security_studies": 20.0,
        },
        "time_minutes": 15.7,
    }
    print(f"\n  ♻ Reusing Base result: {base_result['accuracy']:.1f}%")

    all_results = [base_result]

    # SNN Low-Dose (GPU-native)
    snn_low = make_gpu_snn_hook(sigma=SIGMA_LOW, seed=2026)
    hooks_low = [(idx, snn_low) for idx in MID_LAYERS]
    r2 = evaluate_mmlu(model, tokenizer, dataset,
                       f"SNN σ={SIGMA_LOW} L15-20", hooks=hooks_low)
    all_results.append(r2)

    # SNN High-Dose (GPU-native)
    snn_high = make_gpu_snn_hook(sigma=SIGMA_HIGH, seed=2026)
    hooks_high = [(idx, snn_high) for idx in MID_LAYERS]
    r3 = evaluate_mmlu(model, tokenizer, dataset,
                       f"SNN σ={SIGMA_HIGH} L15-20", hooks=hooks_high)
    all_results.append(r3)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n🎨 Creating visualization...")
    visualize(all_results)

    output = {
        "experiment": "Phase 14b: MMLU Benchmark (v3 GPU-native)",
        "config": {
            "sigma_high": SIGMA_HIGH,
            "sigma_low": SIGMA_LOW,
            "mid_layers": MID_LAYERS,
            "subjects": TARGET_SUBJECTS,
            "max_per_subject": MAX_PER_SUBJECT,
            "num_questions": len(dataset),
            "hook_type": "GPU-native structured noise (0.7*gaussian + 0.3*low_freq)",
        },
        "results": all_results,
        "finished": str(datetime.datetime.now()),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "phase14b_mmlu_log.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Results saved: {results_path}")

    print("\n" + "=" * 60)
    print("PHASE 14b COMPLETE — MMLU BENCHMARK")
    print("=" * 60)
    base_acc = all_results[0]["accuracy"]
    for r in all_results:
        delta = r["accuracy"] - base_acc
        sign = "+" if delta >= 0 else ""
        tax = f"(Tax: {sign}{delta:.1f}%)" if r != all_results[0] else "(baseline)"
        print(f"  {r['condition']:30s}: {r['accuracy']:.1f}% {tax}")

    total_min = (time.time() - t0) / 60
    print(f"\n  ⏱ Total time: {total_min:.1f} min")

    try:
        import winsound
        winsound.Beep(800, 200)
    except Exception:
        print("\a")


if __name__ == "__main__":
    main()
