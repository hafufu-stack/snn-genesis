"""
phase15_cross_architecture.py — Cross-Architecture SNN Validation
==================================================================

Validates SNN noise injection on Qwen2.5-7B-Instruct to demonstrate
that alignment tax ≈ 0 generalizes beyond Mistral architecture.

Runs: TruthfulQA MC1 + MMLU subset
Conditions: Base, SNN σ=0.01, SNN σ=0.10 (GPU-native hooks)
Hibernates PC on completion.
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

# ─── Config ───
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "Qwen2.5-7B"

SIGMA_LOW = 0.01
SIGMA_HIGH = 0.10
# Qwen2.5-7B has 28 layers. Mid-layers = 12-17 (analogous to Mistral L15-20)
MID_LAYERS = list(range(12, 18))

MMLU_SUBJECTS = [
    "elementary_mathematics",
    "high_school_mathematics",
    "conceptual_physics",
    "philosophy",
    "moral_scenarios",
    "security_studies",
    "clinical_knowledge",
    "professional_medicine",
]
MMLU_MAX_PER_SUBJECT = 200
BATCH_PROGRESS = 50

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─── Model Loading ───
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"\n🔄 Loading {MODEL_ID}...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto"
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


def compute_mc1_logprob(model, tokenizer, question, choices, choice_idx):
    """For TruthfulQA: variable number of choices."""
    prompt = f"Q: {question}\nA: "
    best_logprob = float("-inf")
    best_idx = 0

    for ci, choice in enumerate(choices):
        full_text = prompt + choice
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits

        log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
        total = 0.0
        n = 0
        for i in range(prompt_len - 1, full_ids.shape[1] - 1):
            next_token = full_ids[0, i + 1]
            total += log_probs[i, next_token].item()
            n += 1

        avg_lp = total / max(n, 1)
        if avg_lp > best_logprob:
            best_logprob = avg_lp
            best_idx = ci

    return best_idx


# ─── TruthfulQA Evaluation ───
def load_truthfulqa():
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    return ds


def evaluate_truthfulqa(model, tokenizer, dataset, condition_name, hooks=None):
    print(f"\n{'='*60}")
    print(f"  TruthfulQA MC1: {condition_name}")
    print(f"{'='*60}")

    handles = []
    if hooks:
        layers = get_model_layers(model)
        for layer_idx, hook_fn in hooks:
            handle = layers[layer_idx].register_forward_pre_hook(hook_fn)
            handles.append(handle)

    correct = 0
    total = 0
    t0 = time.time()

    try:
        for idx, row in enumerate(dataset):
            question = row["question"]
            mc1_targets = row["mc1_targets"]
            choices = mc1_targets["choices"]
            labels = mc1_targets["labels"]
            correct_idx = labels.index(1)

            predicted = compute_mc1_logprob(model, tokenizer, question, choices, correct_idx)
            if predicted == correct_idx:
                correct += 1
            total += 1

            if (idx + 1) % BATCH_PROGRESS == 0:
                acc = correct / total * 100
                elapsed = time.time() - t0
                eta = elapsed / total * (len(dataset) - total)
                print(f"  [{idx+1}/{len(dataset)}] MC1: {acc:.1f}% "
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


# ─── MMLU Evaluation ───
def load_mmlu():
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    filtered = []
    subject_counts = {}
    for row in ds:
        subj = row["subject"]
        if subj in MMLU_SUBJECTS:
            cnt = subject_counts.get(subj, 0)
            if cnt < MMLU_MAX_PER_SUBJECT:
                filtered.append(row)
                subject_counts[subj] = cnt + 1
    return filtered


def evaluate_mmlu(model, tokenizer, dataset, condition_name, hooks=None):
    print(f"\n{'='*60}")
    print(f"  MMLU: {condition_name}")
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


# ─── Visualization ───
def visualize(tqa_results, mmlu_results, mistral_tqa=None, mistral_mmlu=None):
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

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f"Phase 15: Cross-Architecture Validation — {MODEL_SHORT}",
                 fontsize=14, fontweight="bold", color="#e94560")

    # TruthfulQA
    ax = axes[0]
    names = [r["condition"] for r in tqa_results]
    accs = [r["accuracy"] for r in tqa_results]
    colors = ["#888888", "#00D4AA", "#FF6B35"]
    bars = ax.bar(names, accs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("MC1 Accuracy (%)")
    ax.set_ylim(0, max(accs) * 1.4)
    ax.set_title("TruthfulQA MC1")
    ax.grid(True, axis="y")

    # MMLU
    ax = axes[1]
    names = [r["condition"] for r in mmlu_results]
    accs = [r["accuracy"] for r in mmlu_results]
    bars = ax.bar(names, accs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("MMLU Accuracy (%)")
    ax.set_ylim(0, max(accs) * 1.4)
    ax.set_title("MMLU (8 Subjects)")
    ax.grid(True, axis="y")

    # Cross-architecture comparison
    ax = axes[2]
    if mistral_tqa and mistral_mmlu:
        categories = ["TruthfulQA\nσ=0.01", "TruthfulQA\nσ=0.10", "MMLU\nσ=0.01", "MMLU\nσ=0.10"]
        mistral_taxes = [
            mistral_tqa[1]["accuracy"] - mistral_tqa[0]["accuracy"],
            mistral_tqa[2]["accuracy"] - mistral_tqa[0]["accuracy"],
            mistral_mmlu[1]["accuracy"] - mistral_mmlu[0]["accuracy"],
            mistral_mmlu[2]["accuracy"] - mistral_mmlu[0]["accuracy"],
        ]
        qwen_taxes = [
            tqa_results[1]["accuracy"] - tqa_results[0]["accuracy"],
            tqa_results[2]["accuracy"] - tqa_results[0]["accuracy"],
            mmlu_results[1]["accuracy"] - mmlu_results[0]["accuracy"],
            mmlu_results[2]["accuracy"] - mmlu_results[0]["accuracy"],
        ]
        x = np.arange(len(categories))
        w = 0.35
        ax.bar(x - w/2, mistral_taxes, w, label="Mistral-7B", color="#00D4AA", alpha=0.8)
        ax.bar(x + w/2, qwen_taxes, w, label=MODEL_SHORT, color="#FF6B35", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylabel("Alignment Tax (%)")
        ax.axhline(y=0, color="#666", linestyle="--", linewidth=0.8)
        ax.set_title("Alignment Tax: Mistral vs Qwen")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "Cross-arch\ncomparison\n(needs Mistral data)",
                ha="center", va="center", fontsize=11, color="#aaa",
                transform=ax.transAxes)
        ax.set_title("Cross-Architecture Comparison")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase15_cross_architecture.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


# ─── Main ───
def main():
    print("=" * 70)
    print(f"Phase 15: Cross-Architecture Validation — {MODEL_SHORT}")
    print("=" * 70)
    t0 = time.time()

    # Load datasets
    print("\n📚 Loading datasets...")
    tqa_data = load_truthfulqa()
    print(f"  TruthfulQA: {len(tqa_data)} questions")
    mmlu_data = load_mmlu()
    print(f"  MMLU:       {len(mmlu_data)} questions")

    # Load model
    print("\n📦 Loading model...")
    model, tokenizer = load_model()

    layers = get_model_layers(model)
    print(f"  ✅ {len(layers)} decoder layers found")
    print(f"  📍 SNN target layers: {MID_LAYERS}")

    # ─── TruthfulQA Benchmarks ───
    print("\n" + "=" * 70)
    print("  BENCHMARK 1: TruthfulQA MC1")
    print("=" * 70)

    tqa_results = []

    # Base
    r = evaluate_truthfulqa(model, tokenizer, tqa_data, "Base (No Noise)")
    tqa_results.append(r)

    # SNN Low
    hook_low = make_gpu_snn_hook(sigma=SIGMA_LOW, seed=2026)
    hooks = [(idx, hook_low) for idx in MID_LAYERS]
    r = evaluate_truthfulqa(model, tokenizer, tqa_data, f"SNN σ={SIGMA_LOW}", hooks=hooks)
    tqa_results.append(r)

    # SNN High
    hook_high = make_gpu_snn_hook(sigma=SIGMA_HIGH, seed=2026)
    hooks = [(idx, hook_high) for idx in MID_LAYERS]
    r = evaluate_truthfulqa(model, tokenizer, tqa_data, f"SNN σ={SIGMA_HIGH}", hooks=hooks)
    tqa_results.append(r)

    # ─── MMLU Benchmarks ───
    print("\n" + "=" * 70)
    print("  BENCHMARK 2: MMLU (8 Subjects)")
    print("=" * 70)

    mmlu_results = []

    # Base
    r = evaluate_mmlu(model, tokenizer, mmlu_data, "Base (No Noise)")
    mmlu_results.append(r)

    # SNN Low
    hook_low = make_gpu_snn_hook(sigma=SIGMA_LOW, seed=2026)
    hooks = [(idx, hook_low) for idx in MID_LAYERS]
    r = evaluate_mmlu(model, tokenizer, mmlu_data, f"SNN σ={SIGMA_LOW}", hooks=hooks)
    mmlu_results.append(r)

    # SNN High
    hook_high = make_gpu_snn_hook(sigma=SIGMA_HIGH, seed=2026)
    hooks = [(idx, hook_high) for idx in MID_LAYERS]
    r = evaluate_mmlu(model, tokenizer, mmlu_data, f"SNN σ={SIGMA_HIGH}", hooks=hooks)
    mmlu_results.append(r)

    # Cleanup GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ─── Visualization ───
    print("\n🎨 Creating visualization...")

    # Load Mistral results for cross-architecture comparison
    mistral_tqa = None
    mistral_mmlu = None
    try:
        tqa_path = os.path.join(RESULTS_DIR, "phase14_truthfulqa_log.json")
        if os.path.exists(tqa_path):
            with open(tqa_path, "r") as f:
                d = json.load(f)
                mistral_tqa = d["results"]
    except Exception:
        pass
    try:
        mmlu_path = os.path.join(RESULTS_DIR, "phase14b_mmlu_log.json")
        if os.path.exists(mmlu_path):
            with open(mmlu_path, "r") as f:
                d = json.load(f)
                mistral_mmlu = d["results"]
    except Exception:
        pass

    visualize(tqa_results, mmlu_results, mistral_tqa, mistral_mmlu)

    # ─── Save Results ───
    output = {
        "experiment": f"Phase 15: Cross-Architecture — {MODEL_SHORT}",
        "model": MODEL_ID,
        "config": {
            "sigma_low": SIGMA_LOW,
            "sigma_high": SIGMA_HIGH,
            "mid_layers": MID_LAYERS,
            "total_layers": 28,
            "hook_type": "GPU-native structured noise",
        },
        "truthfulqa": tqa_results,
        "mmlu": mmlu_results,
        "finished": str(datetime.datetime.now()),
    }

    results_path = os.path.join(RESULTS_DIR, "phase15_cross_architecture_log.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Results saved: {results_path}")

    # ─── Summary ───
    print("\n" + "=" * 70)
    print(f"PHASE 15 COMPLETE — {MODEL_SHORT}")
    print("=" * 70)

    print("\n  TruthfulQA MC1:")
    base_tqa = tqa_results[0]["accuracy"]
    for r in tqa_results:
        delta = r["accuracy"] - base_tqa
        sign = "+" if delta >= 0 else ""
        tax = f"(Tax: {sign}{delta:.1f}%)" if r != tqa_results[0] else "(baseline)"
        print(f"    {r['condition']:25s}: {r['accuracy']:.1f}% {tax}")

    print("\n  MMLU:")
    base_mmlu = mmlu_results[0]["accuracy"]
    for r in mmlu_results:
        delta = r["accuracy"] - base_mmlu
        sign = "+" if delta >= 0 else ""
        tax = f"(Tax: {sign}{delta:.1f}%)" if r != mmlu_results[0] else "(baseline)"
        print(f"    {r['condition']:25s}: {r['accuracy']:.1f}% {tax}")

    total_min = (time.time() - t0) / 60
    print(f"\n  ⏱ Total time: {total_min:.1f} min")

    # Beep
    try:
        import winsound
        winsound.Beep(800, 200)
        time.sleep(0.3)
        winsound.Beep(1000, 200)
        time.sleep(0.3)
        winsound.Beep(800, 200)
    except Exception:
        print("\a")

    # ─── Hibernate ───
    print("\n💤 Hibernating in 10 seconds...")
    time.sleep(10)
    os.system("shutdown /h")


if __name__ == "__main__":
    main()
