"""
Phase 46b: Phantom Echo v2 — 極微量σショック
=============================================

Phase 46 の教訓: σ=0.10 は Math -55% で壊滅的。
→ 今回は σ=0.01, 0.02, 0.03 の3レベルで再実験。

σ=0.01 (1%) は hidden state の 1% 程度の摂動で、
ほとんど知覚できないレベルのノイズ。

Usage:
    python experiments/phase46b_phantom_echo_v2.py
"""

import torch, torch.nn as nn
import os, json, gc, time, random, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
TARGET_LAYERS = list(range(15, 21))
ECHO_TOKENS = 2
MAX_NEW_TOKENS = 100
SEED = 2026

# Three micro-dose sigma levels + baseline + repeat
SIGMA_LEVELS = [0.0, 0.01, 0.02, 0.03]
SIGMA_NORMAL = 0.05

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
    layers = get_layers(model)
    print(f"  ✅ {short_name} loaded: {len(layers)} layers")
    return model, tokenizer

def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find decoder layers")


class MicroDoseHook:
    """Hook with configurable echo sigma for micro-dose experiments."""
    def __init__(self, sigma=0.05, echo_sigma=0.01, echo_tokens=2):
        self.sigma = sigma
        self.echo_sigma = echo_sigma
        self.echo_tokens_max = echo_tokens
        self.echo_tokens_remaining = 0
        self.echo_active = False
        self.forward_count = 0
    def __call__(self, module, args):
        hs = args[0]
        if self.echo_active and self.echo_tokens_remaining > 0:
            effective_sigma = self.echo_sigma
            self.echo_tokens_remaining -= 1
            if self.echo_tokens_remaining <= 0:
                self.echo_active = False
        else:
            effective_sigma = self.sigma
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype) * effective_sigma
        self.forward_count += 1
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


def generate_text(model, tokenizer, prompt, hook, use_echo=False, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    if use_echo:
        hook.activate_echo()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.9, top_p=0.95, top_k=50,
                                 repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id)
    hook.reset()
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


def run_condition(model, tokenizer, dataset, target_layers, sigma_echo, condition_name, is_repeat=False):
    """Run a single experimental condition."""
    layers = get_layers(model)
    hook = MicroDoseHook(sigma=SIGMA_NORMAL, echo_sigma=sigma_echo, echo_tokens=ECHO_TOKENS)
    handles = [layers[i].register_forward_pre_hook(hook) for i in target_layers if i < len(layers)]

    stats = {"factual": 0, "f_total": 0, "math": 0, "m_total": 0}

    for idx, item in enumerate(dataset):
        prompt = item["prompt"]
        if is_repeat:
            prompt = prompt + " " + prompt

        use_echo = (sigma_echo > 0 and not is_repeat)

        if item["type"] == "factual":
            ch = item["mc1_targets"]
            ci = ch["labels"].index(1) if 1 in ch["labels"] else 0
            if use_echo:
                hook.activate_echo()
            lps = [compute_completion_logprob(model, tokenizer, prompt, " " + c) for c in ch["choices"]]
            hook.reset()
            if np.argmax(lps) == ci:
                stats["factual"] += 1
            stats["f_total"] += 1
        elif item["type"] == "math":
            resp = generate_text(model, tokenizer, prompt, hook, use_echo=use_echo, max_new_tokens=50)
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
    print(f"  📊 {condition_name}: Fact={fa:.1f}% Math={ma:.1f}%")
    return {"factual_acc": round(fa, 2), "math_acc": round(ma, 2)}


def visualize_v2(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Phase 46b: Phantom Echo v2 — Micro-Dose σ Sweep\n"
                 "Finding the optimal σ for Physical Prompting",
                 fontsize=14, fontweight="bold")

    conditions = list(results.keys())
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e67e22", "#e74c3c"]

    # Panel 1: Factual
    ax1 = axes[0]
    facts = [results[c]["factual_acc"] for c in conditions]
    bars = ax1.bar(conditions, facts, color=colors[:len(conditions)], alpha=0.8)
    for bar, val in zip(bars, facts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Factual Accuracy (%)")
    ax1.set_title("Factual (TruthfulQA)")
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Math
    ax2 = axes[1]
    maths = [results[c]["math_acc"] for c in conditions]
    bars2 = ax2.bar(conditions, maths, color=colors[:len(conditions)], alpha=0.8)
    for bar, val in zip(bars2, maths):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Math Accuracy (%)")
    ax2.set_title("Math (Arithmetic)")
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Verdict
    ax3 = axes[2]
    ax3.axis("off")

    baseline_f = results.get("baseline", results[conditions[0]])["factual_acc"]
    baseline_m = results.get("baseline", results[conditions[0]])["math_acc"]

    txt = "MICRO-DOSE σ SWEEP\n" + "="*35 + "\n\n"
    best_total = 0
    best_name = ""
    for c in conditions:
        f = results[c]["factual_acc"]; m = results[c]["math_acc"]
        df = f - baseline_f; dm = m - baseline_m
        total = f + m
        if total > best_total:
            best_total = total; best_name = c
        txt += f"{c}:\n  Fact={f:.1f}% Math={m:.1f}%\n"
        if c != conditions[0]:
            txt += f"  Δ Fact={df:+.1f}% Δ Math={dm:+.1f}%\n"
        txt += "\n"

    txt += f"Best overall: {best_name}\n"
    txt += f"({results[best_name]['factual_acc']:.1f}% + {results[best_name]['math_acc']:.1f}% = {best_total:.1f}%)"

    ax3.text(0.05, 0.95, txt, transform=ax3.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase46b_phantom_echo_v2.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model(MODEL_NAME, MODEL_SHORT)
    dataset = build_dataset(tokenizer)
    print(f"  📂 Dataset: {len(dataset)} items")

    results = {}

    # Condition 1: Baseline (no echo, σ=0.05 normal)
    print(f"\n{'═'*60}")
    print(f"  🔬 CONDITION: BASELINE (σ=0.05, no echo)")
    print(f"{'═'*60}")
    results["baseline"] = run_condition(model, tokenizer, dataset, TARGET_LAYERS, 0.0, "baseline")

    # Condition 2-4: Micro-dose echoes
    for sigma in [0.01, 0.02, 0.03]:
        print(f"\n{'═'*60}")
        print(f"  ⚡ CONDITION: ECHO σ={sigma}")
        print(f"{'═'*60}")
        label = f"echo_{sigma}"
        results[label] = run_condition(model, tokenizer, dataset, TARGET_LAYERS, sigma, label)

    # Condition 5: Repeat (Google method)
    print(f"\n{'═'*60}")
    print(f"  🔄 CONDITION: REPEAT (Google method)")
    print(f"{'═'*60}")
    results["repeat"] = run_condition(model, tokenizer, dataset, TARGET_LAYERS, 0.0, "repeat", is_repeat=True)

    elapsed = time.time() - t_start
    fig_path = visualize_v2(results)

    output = {
        "experiment": "Phase 46b: Phantom Echo v2",
        "model": MODEL_SHORT,
        "elapsed_minutes": round(elapsed / 60, 1),
        "results": results,
        "figure_path": fig_path,
    }
    log_path = os.path.join(RESULTS_DIR, "phase46b_phantom_echo_v2_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'═'*60}")
    print(f"  ⚡ PHASE 46b: PHANTOM ECHO v2 — VERDICT")
    print(f"{'═'*60}")
    for c, r in results.items():
        print(f"  {c}: Fact={r['factual_acc']:.1f}% Math={r['math_acc']:.1f}%")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
