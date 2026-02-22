"""
Phase 48: L17-18 Surgical Strike
=================================

Deep Think提案: Phase 46bでは全L15-20層にσ=0.02を注入したが、
Phase 44で「本当の急所はL17-18のみ」と判明。
→ L17-18のみにピンポイント注入で精度をさらに向上させる。

実験設計:
  Condition 1: Baseline (σ=0.05, no echo)
  Condition 2: Wide Echo (L15-20, σ=0.02) — Phase 46bの再現
  Condition 3: Surgical Echo (L17-18 only, σ=0.02) — 外科的精密注入
  Condition 4: Surgical Echo (L17-18 only, σ=0.03) — やや強め
  Condition 5: Single-Layer Echo (L18 only, σ=0.02) — 究極のピンポイント

Usage:
    python experiments/phase48_surgical_strike.py
"""

import torch, torch.nn as nn
import os, json, gc, time, random, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
ECHO_TOKENS = 2
SIGMA_NORMAL = 0.05
SEED = 2026

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
    {"q": "What is 2^8?", "a": "256"},
    {"q": "What is sqrt(144)?", "a": "12"},
    {"q": "What is 99 × 3?", "a": "297"},
    {"q": "What is 50% of 240?", "a": "120"},
    {"q": "What is 7!?", "a": "5040"},
    {"q": "What is 1024 / 32?", "a": "32"},
    {"q": "What is the next prime after 29?", "a": "31"},
    {"q": "What is 13 × 13?", "a": "169"},
    {"q": "What is 200 - 137?", "a": "63"},
    {"q": "What is 25 × 4?", "a": "100"},
]


def load_model():
    print(f"\n📦 Loading {MODEL_NAME}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    n_layers = len(model.model.layers)
    print(f"  ✅ Loaded: {n_layers} layers")
    return model, tokenizer


class SurgicalHook:
    """Hook for specific layer echo injection."""
    def __init__(self, sigma_normal=0.05, sigma_echo=0.02, echo_tokens=2):
        self.sigma_normal = sigma_normal
        self.sigma_echo = sigma_echo
        self.echo_tokens_max = echo_tokens
        self.echo_tokens_remaining = 0
        self.echo_active = False

    def __call__(self, module, args):
        hs = args[0]
        if self.echo_active and self.echo_tokens_remaining > 0:
            sigma = self.sigma_echo
            self.echo_tokens_remaining -= 1
            if self.echo_tokens_remaining <= 0:
                self.echo_active = False
        else:
            sigma = self.sigma_normal
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype) * sigma
        return (hs + noise,) + args[1:]

    def activate_echo(self):
        self.echo_active = True
        self.echo_tokens_remaining = self.echo_tokens_max

    def reset(self):
        self.echo_active = False
        self.echo_tokens_remaining = 0


def compute_completion_logprob(model, tokenizer, context, completion):
    full_text = context + completion
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    total_lp, n = 0.0, 0
    for i in range(len(ctx_ids), len(full_ids)):
        total_lp += log_probs[i - 1, full_ids[i]].item(); n += 1
    return total_lp / max(n, 1)


def generate_text(model, tokenizer, prompt, hooks, use_echo=False, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    if use_echo:
        for h in hooks:
            h.activate_echo()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.9, top_p=0.95, top_k=50,
                                 repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id)
    for h in hooks:
        h.reset()
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(input_text):].strip()


def check_math_answer(response, correct_answer):
    resp = response.strip().replace(",", "").replace(" ", "")
    ans = correct_answer.strip().replace(",", "").replace(" ", "")
    if ans in resp: return True
    return ans in re.findall(r'\d+', resp)


def build_dataset(tokenizer):
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), 20)
    items = []
    for idx in indices:
        row = ds[idx]
        items.append({"type": "factual", "prompt": f"Q: {row['question']}\nA:",
                      "question": row["question"], "mc1_targets": row["mc1_targets"]})
    for mp in MATH_PROMPTS:
        items.append({"type": "math",
                      "prompt": f"Solve this math problem. Give only the numerical answer.\n\nQ: {mp['q']}\nA:",
                      "question": mp["q"], "correct_answer": mp["a"]})
    return items


def run_condition(model, tokenizer, dataset, target_layers, sigma_echo, condition_name):
    """Run a single experimental condition."""
    layers = model.model.layers

    # Install hooks only on target layers
    hooks = []
    handles = []
    for i in target_layers:
        if i < len(layers):
            hook = SurgicalHook(sigma_normal=SIGMA_NORMAL, sigma_echo=sigma_echo, echo_tokens=ECHO_TOKENS)
            handle = layers[i].register_forward_pre_hook(hook)
            hooks.append(hook)
            handles.append(handle)

    stats = {"factual": 0, "f_total": 0, "math": 0, "m_total": 0}

    for idx, item in enumerate(dataset):
        prompt = item["prompt"]
        use_echo = (sigma_echo > 0)

        if item["type"] == "factual":
            ch = item["mc1_targets"]
            ci = ch["labels"].index(1) if 1 in ch["labels"] else 0
            if use_echo:
                for h in hooks: h.activate_echo()
            lps = [compute_completion_logprob(model, tokenizer, prompt, " " + c) for c in ch["choices"]]
            for h in hooks: h.reset()
            if np.argmax(lps) == ci:
                stats["factual"] += 1
            stats["f_total"] += 1
        elif item["type"] == "math":
            resp = generate_text(model, tokenizer, prompt, hooks, use_echo=use_echo, max_new_tokens=50)
            if check_math_answer(resp, item["correct_answer"]):
                stats["math"] += 1
            stats["m_total"] += 1

        if (idx + 1) % 10 == 0:
            cf = stats["factual"] / max(stats["f_total"], 1) * 100
            cm = stats["math"] / max(stats["m_total"], 1) * 100
            print(f"    [{idx+1}/{len(dataset)}] Fact={cf:.0f}% Math={cm:.0f}%")

    for h in handles:
        h.remove()

    fa = stats["factual"] / max(stats["f_total"], 1) * 100
    ma = stats["math"] / max(stats["m_total"], 1) * 100
    print(f"  📊 {condition_name}: Fact={fa:.1f}% Math={ma:.1f}% (layers={target_layers} σ_echo={sigma_echo})")
    return {
        "factual_acc": round(fa, 2),
        "math_acc": round(ma, 2),
        "layers": target_layers,
        "sigma_echo": sigma_echo,
    }


def visualize(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Phase 48: L17-18 Surgical Strike\n"
                 "Comparing wide (L15-20) vs surgical (L17-18) echo injection",
                 fontsize=14, fontweight="bold")

    conditions = list(results.keys())
    colors = ["#95a5a6", "#3498db", "#e74c3c", "#e67e22", "#9b59b6"]

    # Panel 1: Factual
    ax = axes[0]
    facts = [results[c]["factual_acc"] for c in conditions]
    bars = ax.bar(range(len(conditions)), facts, color=colors[:len(conditions)], alpha=0.8)
    for i, (bar, val) in enumerate(zip(bars, facts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Factual Accuracy (%)")
    ax.set_title("Factual (TruthfulQA)")
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Math
    ax2 = axes[1]
    maths = [results[c]["math_acc"] for c in conditions]
    bars2 = ax2.bar(range(len(conditions)), maths, color=colors[:len(conditions)], alpha=0.8)
    for bar, val in zip(bars2, maths):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Math Accuracy (%)")
    ax2.set_title("Math (Arithmetic)")
    ax2.grid(True, alpha=0.2, axis="y")
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    # Panel 3: Combined Total
    ax3 = axes[2]
    totals = [results[c]["factual_acc"] + results[c]["math_acc"] for c in conditions]
    bars3 = ax3.bar(range(len(conditions)), totals, color=colors[:len(conditions)], alpha=0.8)
    for bar, val in zip(bars3, totals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax3.set_xticks(range(len(conditions)))
    ax3.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
    ax3.set_ylabel("Total (Fact + Math)")
    ax3.set_title("Combined Score")
    ax3.grid(True, alpha=0.2, axis="y")
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    # Highlight best
    best_idx = np.argmax(totals)
    bars3[best_idx].set_edgecolor('gold')
    bars3[best_idx].set_linewidth(3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase48_surgical_strike.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model()
    dataset = build_dataset(tokenizer)
    print(f"  📂 Dataset: {len(dataset)} items")

    results = {}

    # Condition 1: Baseline
    print(f"\n{'═'*60}")
    print(f"  🔬 BASELINE (σ=0.05, no echo)")
    print(f"{'═'*60}")
    results["baseline"] = run_condition(model, tokenizer, dataset, list(range(15, 21)), 0.0, "baseline")

    # Condition 2: Wide Echo (L15-20, σ=0.02)
    print(f"\n{'═'*60}")
    print(f"  ⚡ WIDE ECHO (L15-20, σ=0.02)")
    print(f"{'═'*60}")
    results["wide_L15-20"] = run_condition(model, tokenizer, dataset, list(range(15, 21)), 0.02, "wide_L15-20")

    # Condition 3: Surgical Echo (L17-18, σ=0.02)
    print(f"\n{'═'*60}")
    print(f"  🎯 SURGICAL ECHO (L17-18, σ=0.02)")
    print(f"{'═'*60}")
    results["surgical_L17-18"] = run_condition(model, tokenizer, dataset, [17, 18], 0.02, "surgical_L17-18")

    # Condition 4: Surgical Echo (L17-18, σ=0.03)
    print(f"\n{'═'*60}")
    print(f"  🎯 SURGICAL ECHO (L17-18, σ=0.03)")
    print(f"{'═'*60}")
    results["surg_L17-18_s03"] = run_condition(model, tokenizer, dataset, [17, 18], 0.03, "surg_L17-18_s03")

    # Condition 5: Single-Layer Echo (L18, σ=0.02)
    print(f"\n{'═'*60}")
    print(f"  💉 SINGLE-LAYER ECHO (L18 only, σ=0.02)")
    print(f"{'═'*60}")
    results["single_L18"] = run_condition(model, tokenizer, dataset, [18], 0.02, "single_L18")

    elapsed = time.time() - t_start
    fig_path = visualize(results)

    output = {
        "experiment": "Phase 48: L17-18 Surgical Strike",
        "model": MODEL_SHORT,
        "elapsed_minutes": round(elapsed / 60, 1),
        "results": results,
        "figure_path": fig_path,
    }
    log_path = os.path.join(RESULTS_DIR, "phase48_surgical_strike_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'═'*60}")
    print(f"  🎯 PHASE 48: SURGICAL STRIKE — VERDICT")
    print(f"{'═'*60}")
    baseline_total = results["baseline"]["factual_acc"] + results["baseline"]["math_acc"]
    for c, r in results.items():
        total = r["factual_acc"] + r["math_acc"]
        delta = total - baseline_total
        print(f"  {c}: Fact={r['factual_acc']:.1f}% Math={r['math_acc']:.1f}% Total={total:.1f} (Δ={delta:+.1f})")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
