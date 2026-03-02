"""
phase16_mmlu_full.py — Full MMLU (All 57 Subjects) × 2 Models
==============================================================

Evaluates ALL 57 MMLU subjects on both Mistral-7B and Qwen2.5-7B.
Purpose: Strengthen the zero-alignment-tax claim in the v3 paper.

Current limitation (paper): "8 of 57 subjects evaluated"
This experiment: ALL 57 subjects → ~14,000+ questions total

Conditions per model:
  1. Base (No Noise)
  2. SNN σ=0.01 (low dose)
  3. SNN σ=0.10 (high dose)

Models run sequentially to fit in GPU memory.
PC hibernates on completion.
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

# ─── Model Configs ───
MODELS = [
    {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "short": "Mistral-7B",
        "mid_layers": list(range(15, 21)),  # L15-20 (32 total)
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "short": "Qwen2.5-7B",
        "mid_layers": list(range(12, 18)),  # L12-17 (28 total)
    },
]

# ─── Noise Settings ───
SIGMA_LOW = 0.01
SIGMA_HIGH = 0.10

# ─── MMLU Settings ───
# No subject filter — load ALL subjects
MAX_PER_SUBJECT = 9999  # effectively no limit
BATCH_PROGRESS = 100

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─── Model Loading ───
def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"\n🔄 Loading {model_id}...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto"
    )
    return model, tokenizer


def get_model_layers(model):
    for path in [
        lambda m: m.model.layers,
        lambda m: m.base_model.model.model.layers,
        lambda m: m.model.model.layers,
    ]:
        try:
            layers = path(model)
            if layers is not None:
                return layers
        except (AttributeError, TypeError):
            continue
    raise AttributeError(f"Cannot find decoder layers in {type(model).__name__}")


# ─── GPU-Native SNN Hook ───
def make_gpu_snn_hook(sigma=0.10, seed=42):
    def hook(module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * sigma
        return (hs + noise,) + args[1:]
    return hook


# ─── Log-Likelihood Scoring ───
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


# ─── Load Full MMLU ───
def load_mmlu_full():
    from datasets import load_dataset
    print("  📚 Loading FULL MMLU dataset (all subjects)...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    # Count subjects
    subject_counts = {}
    for row in ds:
        subj = row["subject"]
        subject_counts[subj] = subject_counts.get(subj, 0) + 1

    total = len(ds)
    n_subjects = len(subject_counts)
    print(f"  ✅ {total} questions loaded ({n_subjects} subjects)")
    print(f"  📊 Subjects: {sorted(subject_counts.keys())}")

    # Print per-subject counts
    for subj in sorted(subject_counts.keys()):
        print(f"     {subj}: {subject_counts[subj]}")

    return list(ds), subject_counts


# ─── MMLU Evaluation ───
def evaluate_mmlu(model, tokenizer, dataset, condition_name, hooks=None):
    """Evaluate MMLU accuracy using log-likelihood on full dataset."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {condition_name}")
    print(f"  Questions: {len(dataset)}")
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
    print(f"\n  Per-subject results:")
    for subj, stats in sorted(per_subject.items()):
        s_acc = stats["correct"] / stats["total"] * 100
        per_subject_acc[subj] = round(s_acc, 1)
        print(f"    {subj:40s}: {s_acc:5.1f}% ({stats['correct']}/{stats['total']})")

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


# ─── Visualization ───
def visualize(all_model_results):
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

    n_models = len(all_model_results)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(8 * (n_models + 1), 8))
    fig.suptitle("Phase 16: Full MMLU (57 Subjects) — Alignment Tax",
                 fontsize=16, fontweight="bold", color="#e94560")

    colors = ["#888888", "#00D4AA", "#FF6B35"]

    # Per-model overall accuracy
    for mi, (model_name, results) in enumerate(all_model_results.items()):
        ax = axes[mi]
        names = [r["condition"].split(" ")[0] if i == 0 else f"σ={r['condition'].split('σ=')[1].split(' ')[0]}" 
                 for i, r in enumerate(results)]
        names[0] = "Base"
        accs = [r["accuracy"] for r in results]
        bars = ax.bar(names, accs, color=colors[:len(results)],
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{acc:.1f}%", ha="center", fontsize=12, fontweight="bold")

        ax.set_ylabel("MMLU Accuracy (%)", fontsize=11)
        ax.set_ylim(0, max(accs) * 1.4)
        ax.set_title(f"{model_name}\n(Full MMLU)", fontsize=12)
        ax.grid(True, axis="y")

        # Tax annotation
        base_acc = results[0]["accuracy"]
        tax_texts = []
        for r in results[1:]:
            delta = r["accuracy"] - base_acc
            sign = "+" if delta >= 0 else ""
            tax_texts.append(f"Tax: {sign}{delta:.2f}%")
        if tax_texts:
            ax.annotate("\n".join(tax_texts), xy=(0.5, 0.02),
                       xycoords="axes fraction", fontsize=10, color="#aaa", ha="center")

    # Cross-model alignment tax comparison
    ax = axes[-1]
    model_names = list(all_model_results.keys())
    if len(model_names) == 2:
        categories = ["σ=0.01", "σ=0.10"]
        x = np.arange(len(categories))
        w = 0.35
        model_colors = ["#00D4AA", "#FF6B35"]
        for mi, mname in enumerate(model_names):
            results = all_model_results[mname]
            base_acc = results[0]["accuracy"]
            taxes = [r["accuracy"] - base_acc for r in results[1:]]
            ax.bar(x + (mi - 0.5) * w, taxes, w, label=mname,
                   color=model_colors[mi], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Alignment Tax (%)")
        ax.axhline(y=0, color="#666", linestyle="--", linewidth=0.8)
        ax.set_title("Cross-Architecture\nAlignment Tax", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "Need 2 models\nfor comparison",
                ha="center", va="center", fontsize=11, color="#aaa",
                transform=ax.transAxes)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase16_mmlu_full.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


# ─── Main ───
def main():
    print("=" * 70)
    print("Phase 16: Full MMLU (All 57 Subjects) × 2 Architectures")
    print("=" * 70)
    t0_global = time.time()

    # Load full MMLU dataset
    dataset, subject_counts = load_mmlu_full()

    all_model_results = {}

    for model_cfg in MODELS:
        model_id = model_cfg["id"]
        model_short = model_cfg["short"]
        mid_layers = model_cfg["mid_layers"]

        print(f"\n{'='*70}")
        print(f"  MODEL: {model_short} ({model_id})")
        print(f"  Target layers: {mid_layers}")
        print(f"{'='*70}")

        t0 = time.time()

        # Load model
        model, tokenizer = load_model(model_id)
        layers = get_model_layers(model)
        print(f"  ✅ {len(layers)} decoder layers found")

        results = []

        # 1. Base (No Noise)
        r = evaluate_mmlu(model, tokenizer, dataset,
                         f"Base (No Noise)")
        results.append(r)

        # Save intermediate result after each condition
        save_intermediate(model_short, results, subject_counts, len(dataset))

        # 2. SNN Low σ=0.01
        hook_low = make_gpu_snn_hook(sigma=SIGMA_LOW, seed=2026)
        hooks = [(idx, hook_low) for idx in mid_layers]
        r = evaluate_mmlu(model, tokenizer, dataset,
                         f"SNN σ={SIGMA_LOW} L{mid_layers[0]}-{mid_layers[-1]}",
                         hooks=hooks)
        results.append(r)

        save_intermediate(model_short, results, subject_counts, len(dataset))

        # 3. SNN High σ=0.10
        hook_high = make_gpu_snn_hook(sigma=SIGMA_HIGH, seed=2026)
        hooks = [(idx, hook_high) for idx in mid_layers]
        r = evaluate_mmlu(model, tokenizer, dataset,
                         f"SNN σ={SIGMA_HIGH} L{mid_layers[0]}-{mid_layers[-1]}",
                         hooks=hooks)
        results.append(r)

        save_intermediate(model_short, results, subject_counts, len(dataset))

        model_time = (time.time() - t0) / 60
        print(f"\n  ⏱ {model_short} total: {model_time:.1f} min")

        all_model_results[model_short] = results

        # Free GPU memory before next model
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  🗑 GPU memory cleared")

    # ─── Visualization ───
    print("\n🎨 Creating visualization...")
    visualize(all_model_results)

    # ─── Save Final Results ───
    output = {
        "experiment": "Phase 16: Full MMLU (All 57 Subjects)",
        "config": {
            "sigma_low": SIGMA_LOW,
            "sigma_high": SIGMA_HIGH,
            "models": [
                {
                    "id": m["id"],
                    "short": m["short"],
                    "mid_layers": m["mid_layers"],
                }
                for m in MODELS
            ],
            "total_questions": len(dataset),
            "total_subjects": len(subject_counts),
            "subjects": sorted(subject_counts.keys()),
            "hook_type": "GPU-native structured noise (0.7*gaussian + 0.3*low_freq)",
        },
        "results": {
            model_short: results
            for model_short, results in all_model_results.items()
        },
        "finished": str(datetime.datetime.now()),
    }

    results_path = os.path.join(RESULTS_DIR, "phase16_mmlu_full_log.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Results saved: {results_path}")

    # ─── Grand Summary ───
    print("\n" + "=" * 70)
    print("PHASE 16 COMPLETE — FULL MMLU (ALL 57 SUBJECTS)")
    print("=" * 70)

    for model_short, results in all_model_results.items():
        print(f"\n  {model_short}:")
        base_acc = results[0]["accuracy"]
        for r in results:
            delta = r["accuracy"] - base_acc
            sign = "+" if delta >= 0 else ""
            tax = f"(Tax: {sign}{delta:.2f}%)" if r != results[0] else "(baseline)"
            print(f"    {r['condition']:40s}: {r['accuracy']:.2f}% {tax}")

    total_min = (time.time() - t0_global) / 60
    print(f"\n  ⏱ Total time: {total_min:.1f} min ({total_min/60:.1f} hours)")

    # Beep
    try:
        import winsound
        for freq in [800, 1000, 1200, 1000, 800]:
            winsound.Beep(freq, 150)
            time.sleep(0.1)
    except Exception:
        print("\a")

    # Note: Originally hibernated PC after completion (local use only).
    # Removed for safety — do not auto-hibernate on other users' machines.
    print("\n✅ Phase 16 finished. Results saved.")


def save_intermediate(model_short, results, subject_counts, total_questions):
    """Save intermediate results after each condition completes."""
    output = {
        "experiment": f"Phase 16: Full MMLU — {model_short} (in progress)",
        "total_questions": total_questions,
        "total_subjects": len(subject_counts),
        "results": results,
        "last_updated": str(datetime.datetime.now()),
    }
    path = os.path.join(RESULTS_DIR, f"phase16_{model_short.lower().replace('-', '_')}_intermediate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  💾 Intermediate saved: {path}")


if __name__ == "__main__":
    main()
