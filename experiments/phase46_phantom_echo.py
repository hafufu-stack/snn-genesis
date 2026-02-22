"""
Phase 46: Phantom Echo — 幻の反響（Physical Prompting）
======================================================

Phase 45 proved that prompt repetition causes massive entropy drop
(+0.868 average, 46% reduction on 2nd pass). However, for local
4-bit models, doubling prompt length HURTS accuracy (-15% fact, -20% math).

The Phantom Echo hypothesis:
  Instead of repeating the prompt (2x tokens, 2x cost),
  inject a brief σ-shock RIGHT BEFORE answer generation.
  This should force the LLM to re-scan its KV cache, achieving
  the same "Déjà Vu" effect without the extra tokens.

Phase 43 lesson: σ=0.15 was destructive. Here we use gentler
shocks (σ=0.08-0.10) for only 1-3 tokens.

Three conditions:
  1. SINGLE:  Normal prompt, no tricks
  2. REPEAT:  Prompt repeated 2x (Google method)
  3. PHANTOM: Normal prompt + σ-shock at generation boundary

Usage:
    python experiments/phase46_phantom_echo.py
"""

import torch, torch.nn as nn
import os, json, gc, time, random, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# ═══ Configuration ═══
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
TARGET_LAYERS = list(range(15, 21))
CONFLICT_LAYER_SHALLOW = 8
CONFLICT_LAYER_DEEP = 28

SIGMA_MIN, SIGMA_MAX = 0.001, 0.15
SIGMA_NORMAL = 0.05
SIGMA_ECHO = 0.10         # Phantom Echo shock amplitude (gentler than Phase 43's 0.15)
ECHO_TOKENS = 2           # Apply shock for this many tokens only
MAX_NEW_TOKENS = 100
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


# ═══ Components ═══

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


class PhantomEchoHook:
    """
    Hook with echo mode:
    - Normal mode: applies sigma noise as usual
    - Echo mode: applies SIGMA_ECHO for the first N generated tokens,
      then reverts to normal
    """
    def __init__(self, sigma=0.05, echo_sigma=0.10, echo_tokens=2):
        self.sigma = sigma
        self.echo_sigma = echo_sigma
        self.echo_tokens_max = echo_tokens
        self.echo_tokens_remaining = 0
        self.echo_active = False
        self.generation_started = False
        self.forward_count = 0
        self.echo_fired_count = 0
    def __call__(self, module, args):
        hs = args[0]
        # Determine effective sigma
        if self.echo_active and self.echo_tokens_remaining > 0:
            effective_sigma = self.echo_sigma
            self.echo_tokens_remaining -= 1
            self.echo_fired_count += 1
            if self.echo_tokens_remaining <= 0:
                self.echo_active = False
        else:
            effective_sigma = self.sigma
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * effective_sigma
        self.forward_count += 1
        return (hs + noise,) + args[1:]
    def activate_echo(self):
        """Trigger phantom echo for the next N tokens."""
        self.echo_active = True
        self.echo_tokens_remaining = self.echo_tokens_max
    def reset(self):
        self.echo_active = False
        self.echo_tokens_remaining = 0
        self.generation_started = False


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


def generate_with_echo(model, tokenizer, prompt, hook, use_echo=False,
                       max_new_tokens=MAX_NEW_TOKENS):
    """
    Custom generation loop that can inject a phantom echo.
    When use_echo=True, the hook fires a σ-shock for the first
    ECHO_TOKENS of generation.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)

    if use_echo:
        hook.activate_echo()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )

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


def compute_per_token_entropy(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
    entropies = []
    for i in range(logits.shape[0]):
        probs = torch.softmax(logits[i].float(), dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        entropies.append(ent)
    return np.mean(entropies)


# ═══ RUN EXPERIMENT ═══

def run_all_conditions(model, tokenizer, dataset, hooks_layers):
    """Run 3 conditions: Single, Repeat, Phantom Echo."""
    layers = get_layers(model)

    conditions = ["single", "repeat", "phantom"]
    results = {}

    for cond in conditions:
        print(f"\n{'═'*60}")
        print(f"  👻 CONDITION: {cond.upper()}")
        print(f"{'═'*60}")

        hook = PhantomEchoHook(
            sigma=SIGMA_NORMAL,
            echo_sigma=SIGMA_ECHO,
            echo_tokens=ECHO_TOKENS
        )
        handles = [layers[i].register_forward_pre_hook(hook) for i in hooks_layers if i < len(layers)]

        stats = {"factual": 0, "f_total": 0, "math": 0, "m_total": 0,
                 "entropies": [], "echo_fires": 0}

        for idx, item in enumerate(dataset):
            prompt = item["prompt"]

            if cond == "repeat":
                prompt_used = prompt + " " + prompt
            else:
                prompt_used = prompt

            use_echo = (cond == "phantom")

            if item["type"] == "factual":
                ch = item["mc1_targets"]
                ci = ch["labels"].index(1) if 1 in ch["labels"] else 0

                if cond == "phantom":
                    # For factual MC, activate echo before scoring each choice
                    hook.activate_echo()

                lps = [compute_completion_logprob(model, tokenizer, prompt_used, " " + c) for c in ch["choices"]]
                hook.reset()

                if np.argmax(lps) == ci:
                    stats["factual"] += 1
                stats["f_total"] += 1

            elif item["type"] == "math":
                resp = generate_with_echo(model, tokenizer, prompt_used, hook,
                                          use_echo=use_echo, max_new_tokens=50)
                if check_math_answer(resp, item["correct_answer"]):
                    stats["math"] += 1
                stats["m_total"] += 1

            if (idx + 1) % 10 == 0:
                cf = stats["factual"] / max(stats["f_total"], 1) * 100
                cm = stats["math"] / max(stats["m_total"], 1) * 100
                print(f"  [{idx+1}/{len(dataset)}] Fact={cf:.0f}% Math={cm:.0f}%")

        stats["echo_fires"] = hook.echo_fired_count

        for h in handles:
            h.remove()

        fa = stats["factual"] / max(stats["f_total"], 1) * 100
        ma = stats["math"] / max(stats["m_total"], 1) * 100

        results[cond] = {
            "factual_acc": round(fa, 2),
            "math_acc": round(ma, 2),
            "echo_fires": stats["echo_fires"],
        }
        print(f"\n  📊 {cond}: Fact={fa:.1f}% Math={ma:.1f}% echoes={stats['echo_fires']}")

    return results


# ═══ Entropy analysis ═══

def entropy_analysis(model, tokenizer):
    """Measure the average entropy for single vs phantom echo."""
    print(f"\n{'═'*60}")
    print(f"  📊 ENTROPY ANALYSIS: Does Phantom Echo reduce entropy?")
    print(f"{'═'*60}")

    test_prompts = [
        "Q: What country has the most islands in the world?\nA:",
        "Q: What is the scientific name for humans?\nA:",
        "Solve this math problem. Give only the numerical answer.\n\nQ: What is 17 + 28?\nA:",
        "Solve this math problem. Give only the numerical answer.\n\nQ: What is 15 × 15?\nA:",
    ]

    layers = get_layers(model)

    for prompt in test_prompts:
        # Without echo
        hook_normal = PhantomEchoHook(sigma=SIGMA_NORMAL, echo_sigma=SIGMA_ECHO, echo_tokens=0)
        handles = [layers[i].register_forward_pre_hook(hook_normal) for i in TARGET_LAYERS if i < len(layers)]
        ent_normal = compute_per_token_entropy(model, tokenizer, prompt)
        for h in handles: h.remove()

        # With echo
        hook_echo = PhantomEchoHook(sigma=SIGMA_NORMAL, echo_sigma=SIGMA_ECHO, echo_tokens=ECHO_TOKENS)
        handles = [layers[i].register_forward_pre_hook(hook_echo) for i in TARGET_LAYERS if i < len(layers)]
        hook_echo.activate_echo()
        ent_echo = compute_per_token_entropy(model, tokenizer, prompt)
        hook_echo.reset()
        for h in handles: h.remove()

        # Repeat
        repeated = prompt + " " + prompt
        hook_repeat = PhantomEchoHook(sigma=SIGMA_NORMAL, echo_sigma=SIGMA_ECHO, echo_tokens=0)
        handles = [layers[i].register_forward_pre_hook(hook_repeat) for i in TARGET_LAYERS if i < len(layers)]
        ent_repeat = compute_per_token_entropy(model, tokenizer, repeated)
        for h in handles: h.remove()

        print(f"  {prompt[:45]}...")
        print(f"    Normal:  {ent_normal:.3f}")
        print(f"    Repeat:  {ent_repeat:.3f}")
        print(f"    Phantom: {ent_echo:.3f}")


# ═══ Visualization ═══

def visualize_phantom(results, elapsed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Phase 46: Phantom Echo — Physical Prompting\n"
                 "Can a σ-shock replace prompt repetition?",
                 fontsize=14, fontweight="bold")

    conditions = ["single", "repeat", "phantom"]
    labels = ["Single\n(baseline)", "Repeat\n(Google)", "Phantom\nEcho ⚡"]
    colors = ["#95a5a6", "#e74c3c", "#9b59b6"]

    # Panel 1: Factual accuracy
    ax1 = axes[0]
    fact_vals = [results[c]["factual_acc"] for c in conditions]
    bars = ax1.bar(labels, fact_vals, color=colors, alpha=0.8)
    for bar, val in zip(bars, fact_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Factual Accuracy (%)")
    ax1.set_title("Factual (TruthfulQA)")
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Math accuracy
    ax2 = axes[1]
    math_vals = [results[c]["math_acc"] for c in conditions]
    bars2 = ax2.bar(labels, math_vals, color=colors, alpha=0.8)
    for bar, val in zip(bars2, math_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Math Accuracy (%)")
    ax2.set_title("Math (Arithmetic)")
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Verdict
    ax3 = axes[2]
    ax3.axis("off")

    sf = results["single"]["factual_acc"]; sm = results["single"]["math_acc"]
    rf = results["repeat"]["factual_acc"]; rm = results["repeat"]["math_acc"]
    pf = results["phantom"]["factual_acc"]; pm = results["phantom"]["math_acc"]

    txt = (f"PHANTOM ECHO RESULTS\n"
           f"{'='*40}\n\n"
           f"Single:  Fact={sf:.1f}% Math={sm:.1f}%\n"
           f"Repeat:  Fact={rf:.1f}% Math={rm:.1f}%\n"
           f"Phantom: Fact={pf:.1f}% Math={pm:.1f}%\n\n"
           f"Phantom vs Single:\n"
           f"  Fact Δ: {pf-sf:+.1f}%\n"
           f"  Math Δ: {pm-sm:+.1f}%\n\n"
           f"Phantom vs Repeat:\n"
           f"  Fact Δ: {pf-rf:+.1f}%\n"
           f"  Math Δ: {pm-rm:+.1f}%\n\n"
           f"Echo fires: {results['phantom']['echo_fires']}\n"
           f"σ_echo = {SIGMA_ECHO}\n"
           f"Echo tokens = {ECHO_TOKENS}\n\n")

    # Verdict logic
    phantom_better_than_single = (pf > sf) or (pm > sm)
    phantom_better_than_repeat = (pf > rf) or (pm > rm)

    if phantom_better_than_single and phantom_better_than_repeat:
        txt += ("🎆 PHANTOM ECHO WINS!\n"
                "→ Physical Prompting works!\n"
                "→ No need to repeat prompt!")
        vc = "#2ecc71"
    elif phantom_better_than_repeat:
        txt += ("⚡ PHANTOM > REPEAT!\n"
                "→ σ-shock beats double tokens\n"
                "→ More efficient than Google method")
        vc = "#9b59b6"
    elif phantom_better_than_single:
        txt += ("🔬 PHANTOM > SINGLE\n"
                "→ σ-shock helps somewhat\n"
                "→ Not as good as repetition")
        vc = "#e67e22"
    else:
        txt += ("📊 PHANTOM ≈ SINGLE\n"
                "→ σ-shock did not provide benefit\n"
                "→ Needs amplitude/timing tuning")
        vc = "#e74c3c"

    ax3.text(0.05, 0.95, txt, transform=ax3.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace", color=vc,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase46_phantom_echo.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


# ═══ MAIN ═══

def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model(MODEL_NAME, MODEL_SHORT)
    dataset = build_dataset(tokenizer)
    print(f"  📂 Dataset: {len(dataset)} items")

    # Run all 3 conditions
    results = run_all_conditions(model, tokenizer, dataset, TARGET_LAYERS)

    # Entropy analysis
    entropy_analysis(model, tokenizer)

    elapsed = time.time() - t_start
    fig_path = visualize_phantom(results, elapsed)

    output = {
        "experiment": "Phase 46: Phantom Echo",
        "model": MODEL_SHORT,
        "elapsed_minutes": round(elapsed / 60, 1),
        "sigma_echo": SIGMA_ECHO,
        "echo_tokens": ECHO_TOKENS,
        "results": results,
        "comparison": {
            "phantom_vs_single_fact": round(results["phantom"]["factual_acc"] - results["single"]["factual_acc"], 2),
            "phantom_vs_single_math": round(results["phantom"]["math_acc"] - results["single"]["math_acc"], 2),
            "phantom_vs_repeat_fact": round(results["phantom"]["factual_acc"] - results["repeat"]["factual_acc"], 2),
            "phantom_vs_repeat_math": round(results["phantom"]["math_acc"] - results["repeat"]["math_acc"], 2),
        },
        "figure_path": fig_path,
    }

    log_path = os.path.join(RESULTS_DIR, "phase46_phantom_echo_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)

    pf = results["phantom"]["factual_acc"]; pm = results["phantom"]["math_acc"]
    sf = results["single"]["factual_acc"]; sm = results["single"]["math_acc"]
    rf = results["repeat"]["factual_acc"]; rm = results["repeat"]["math_acc"]
    print(f"\n{'═'*60}")
    print(f"  👻 PHASE 46: PHANTOM ECHO — VERDICT")
    print(f"{'═'*60}")
    print(f"  Single:  Fact={sf:.1f}% Math={sm:.1f}%")
    print(f"  Repeat:  Fact={rf:.1f}% Math={rm:.1f}%")
    print(f"  Phantom: Fact={pf:.1f}% Math={pm:.1f}%")
    print(f"  Phantom vs Single: Fact {pf-sf:+.1f}% Math {pm-sm:+.1f}%")
    print(f"  Phantom vs Repeat: Fact {pf-rf:+.1f}% Math {pm-rm:+.1f}%")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
