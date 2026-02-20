"""
Phase 31: Static σ=0.071 Math Comparison — Mistral vs Qwen
==========================================================
Tests Math accuracy under STATIC σ=0.071 on both architectures.

Key Question:
  If Qwen Math=60% under PPO homeostasis but Mistral Math=0%,
  is it because Qwen is inherently more noise-tolerant?

  If Qwen Math>0% at static σ=0.071 → Qwen is just tougher
  If Qwen Math=0% at static σ=0.071 → PPO somehow protected it (weird!)

Usage:
    python experiments/phase31_static_sigma_math_comparison.py
"""

import torch, torch.nn as nn
import os, sys, json, gc, time, datetime, random, math, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ═══ Configuration ═══
SIGMA_STATIC = 0.071   # The universal homeostatic set-point
SIGMA_MIN, SIGMA_MAX = 0.001, 0.15
MAX_NEW_TOKENS = 50
SEED = 2026
TARGET_LAYERS_MISTRAL = list(range(15, 21))  # L15-20 for Mistral (32 layers)
TARGET_LAYERS_QWEN = list(range(12, 18))     # L12-17 for Qwen (28 layers)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MATH_PROMPTS = [
    {"q": "What is 17 + 28?", "a": "45"},
    {"q": "What is 156 - 89?", "a": "67"},
    {"q": "What is 12 × 7?", "a": "84"},
    {"q": "What is 144 / 12?", "a": "12"},
    {"q": "What is 23 + 45 + 12?", "a": "80"},
    {"q": "What is 15 × 15?", "a": "225"},
    {"q": "What is 1000 - 573?", "a": "427"},
    {"q": "What is 8 × 9?", "a": "72"},
    {"q": "What is 256 / 16?", "a": "16"},
    {"q": "What is 33 + 67?", "a": "100"},
    {"q": "If x + 5 = 12, what is x?", "a": "7"},
    {"q": "What is 2 to the power of 8?", "a": "256"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "What is 99 × 3?", "a": "297"},
    {"q": "What is 50% of 240?", "a": "120"},
    {"q": "If a triangle has sides 3, 4, and 5, what is its perimeter?", "a": "12"},
    {"q": "What is 7! (7 factorial)?", "a": "5040"},
    {"q": "What is 1024 / 32?", "a": "32"},
    {"q": "What is the next prime after 29?", "a": "31"},
    {"q": "What is 13 × 13?", "a": "169"},
]


class StaticSNNHook:
    def __init__(self, sigma=0.071):
        self.sigma = sigma
    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * self.sigma
        return (hs + noise,) + args[1:]


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find decoder layers")


def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.9, top_p=0.95, top_k=50,
                                 repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()


def check_math_answer(response, correct_answer):
    resp = response.strip().replace(",", "").replace(" ", "")
    ans = correct_answer.strip().replace(",", "").replace(" ", "")
    if ans in resp:
        return True
    numbers = re.findall(r'\d+', resp)
    return ans in numbers


def evaluate_math(model, tokenizer, target_layers, sigma, condition_name):
    """Evaluate Math accuracy under a given static sigma."""
    print(f"\n  🧮 {condition_name}: σ={sigma}, layers={target_layers}")

    layers = get_layers(model)
    hook = StaticSNNHook(sigma=sigma)
    handles = [layers[i].register_forward_pre_hook(hook) for i in target_layers if i < len(layers)]

    correct = 0
    details = []
    try:
        for mp in MATH_PROMPTS:
            prompt = f"Solve this math problem. Give only the numerical answer.\n\nQ: {mp['q']}\nA:"
            resp = generate_text(model, tokenizer, prompt)
            is_correct = check_math_answer(resp, mp["a"])
            if is_correct:
                correct += 1
            details.append({
                "question": mp["q"],
                "expected": mp["a"],
                "response": resp[:100],
                "correct": bool(is_correct),
            })
    finally:
        for h in handles:
            h.remove()

    acc = correct / len(MATH_PROMPTS) * 100
    print(f"    → {correct}/{len(MATH_PROMPTS)} = {acc:.1f}%")
    return {
        "condition": condition_name,
        "sigma": sigma,
        "correct": correct,
        "total": len(MATH_PROMPTS),
        "accuracy": round(acc, 2),
        "details": details,
    }


def run_model(model_name, model_short, target_layers):
    print(f"\n{'='*60}")
    print(f"  Loading {model_name}...")
    print(f"{'='*60}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    layers = get_layers(model)
    print(f"  ✅ {model_short}: {len(layers)} layers, target={target_layers}")

    results = []

    # Condition 1: No noise (σ=0)
    results.append(evaluate_math(model, tokenizer, target_layers, sigma=0.0, condition_name=f"{model_short} No Noise"))

    # Condition 2: Static σ=0.015 (Math-optimal)
    results.append(evaluate_math(model, tokenizer, target_layers, sigma=0.015, condition_name=f"{model_short} Static σ*=0.015"))

    # Condition 3: Static σ=0.071 (homeostatic)
    results.append(evaluate_math(model, tokenizer, target_layers, sigma=SIGMA_STATIC, condition_name=f"{model_short} Static σ=0.071"))

    # Condition 4: Static σ=0.10 (above homeostatic)
    results.append(evaluate_math(model, tokenizer, target_layers, sigma=0.10, condition_name=f"{model_short} Static σ=0.10"))

    # Condition 5: Static σ=0.15 (max)
    results.append(evaluate_math(model, tokenizer, target_layers, sigma=0.15, condition_name=f"{model_short} Static σ=0.15"))

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def visualize(mistral_results, qwen_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#ccc", "ytick.color": "#ccc",
        "grid.color": "#333", "grid.alpha": 0.3,
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 31: Math Accuracy vs Static σ — Mistral vs Qwen\n"
                 "Is Qwen inherently more noise-tolerant?",
                 fontsize=14, fontweight="bold", color="#e94560")

    sigmas = [0.0, 0.015, 0.071, 0.10, 0.15]
    labels = ["σ=0\n(No Noise)", "σ=0.015\n(Math-opt)", "σ=0.071\n(Homeo)", "σ=0.10", "σ=0.15\n(Max)"]

    # Panel 1: Side-by-side bars
    ax1 = axes[0]
    m_accs = [r["accuracy"] for r in mistral_results]
    q_accs = [r["accuracy"] for r in qwen_results]
    x = np.arange(len(sigmas))
    w = 0.35
    bars1 = ax1.bar(x - w/2, m_accs, w, label="Mistral-7B", color="#4FC3F7", edgecolor="#333")
    bars2 = ax1.bar(x + w/2, q_accs, w, label="Qwen2.5-7B", color="#FF7043", edgecolor="#333")
    for bar, val in zip(bars1, m_accs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight="bold", color="#4FC3F7")
    for bar, val in zip(bars2, q_accs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight="bold", color="#FF7043")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Math Accuracy (%)")
    ax1.set_title("Math Accuracy at Each σ Level", fontweight="bold")
    ax1.legend(fontsize=10, facecolor="#16213e", edgecolor="#555")
    ax1.set_ylim(0, 105)
    ax1.grid(True, axis="y")
    # Mark homeostatic point
    ax1.axvline(x=2, color="#66BB6A", linestyle="--", alpha=0.5, linewidth=1)

    # Panel 2: Line chart + interpretation
    ax2 = axes[1]
    ax2.plot(sigmas, m_accs, 'o-', color="#4FC3F7", linewidth=2, markersize=8, label="Mistral-7B")
    ax2.plot(sigmas, q_accs, 's-', color="#FF7043", linewidth=2, markersize=8, label="Qwen2.5-7B")
    ax2.axvline(x=0.071, color="#66BB6A", linestyle="--", alpha=0.7, linewidth=2, label="σ=0.071 (Homeo)")
    ax2.set_xlabel("σ (noise intensity)")
    ax2.set_ylabel("Math Accuracy (%)")
    ax2.set_title("Noise Tolerance Curve", fontweight="bold")
    ax2.legend(fontsize=10, facecolor="#16213e", edgecolor="#555")
    ax2.set_ylim(-5, 105)
    ax2.grid(True)

    # Determine verdict
    m_homeo = m_accs[2]
    q_homeo = q_accs[2]
    if q_homeo > m_homeo + 10:
        verdict = f"Qwen is MORE noise-tolerant!\nMath@σ=0.071: Mistral={m_homeo:.0f}% Qwen={q_homeo:.0f}%"
    elif abs(q_homeo - m_homeo) <= 10:
        verdict = f"SIMILAR tolerance!\nMath@σ=0.071: Mistral={m_homeo:.0f}% Qwen={q_homeo:.0f}%"
    else:
        verdict = f"Mistral is MORE noise-tolerant!\nMath@σ=0.071: Mistral={m_homeo:.0f}% Qwen={q_homeo:.0f}%"
    ax2.text(0.05, 0.05, verdict, transform=ax2.transAxes, fontsize=10, fontweight="bold",
             color="#66BB6A", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", edgecolor="#66BB6A", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(FIGURES_DIR, "phase31_static_sigma_math_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  📊 Figure: {out}")
    return out


def main():
    print("=" * 60)
    print("Phase 31: Static σ=0.071 Math Comparison")
    print("  Mistral-7B vs Qwen2.5-7B: Who survives noise better?")
    print("=" * 60)
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    # Run Mistral
    mistral_results = run_model(
        "mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B", TARGET_LAYERS_MISTRAL)

    # Run Qwen
    qwen_results = run_model(
        "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B", TARGET_LAYERS_QWEN)

    # Visualize
    fig_path = visualize(mistral_results, qwen_results)

    elapsed = time.time() - t_start

    # Save
    output = {
        "phase": "Phase 31: Static σ=0.071 Math Comparison",
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "sigma_tested": [0.0, 0.015, 0.071, 0.10, 0.15],
        "mistral": {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "target_layers": TARGET_LAYERS_MISTRAL,
            "results": mistral_results,
        },
        "qwen": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "target_layers": TARGET_LAYERS_QWEN,
            "results": qwen_results,
        },
        "comparison": {
            "sigma_0.071_mistral_acc": mistral_results[2]["accuracy"],
            "sigma_0.071_qwen_acc": qwen_results[2]["accuracy"],
            "qwen_more_tolerant": bool(qwen_results[2]["accuracy"] > mistral_results[2]["accuracy"] + 10),
        },
        "figure_path": fig_path,
    }
    out = os.path.join(RESULTS_DIR, "phase31_static_sigma_math_comparison_log.json")
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Results: {out}")

    print(f"\n{'='*60}")
    print(f"  Phase 31 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"\n  📊 Math Accuracy at Static σ Levels:")
    print(f"  {'σ':>8} | {'Mistral':>10} | {'Qwen':>10} | {'Delta':>8}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for m, q in zip(mistral_results, qwen_results):
        delta = q["accuracy"] - m["accuracy"]
        print(f"  {m['sigma']:>8.3f} | {m['accuracy']:>9.1f}% | {q['accuracy']:>9.1f}% | {delta:>+7.1f}%")
    print(f"\n  Key: σ=0.071 (homeostatic)")
    print(f"    Mistral: {mistral_results[2]['accuracy']:.1f}%")
    print(f"    Qwen:    {qwen_results[2]['accuracy']:.1f}%")
    if output["comparison"]["qwen_more_tolerant"]:
        print(f"  → Qwen IS more noise-tolerant! Biological Egoism severity is architecture-dependent.")
    else:
        print(f"  → Similar tolerance. The PPO dynamics, not base robustness, drive the difference.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
