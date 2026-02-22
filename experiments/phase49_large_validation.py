"""
Phase 49: Large-Scale Statistical Validation
=============================================

Phase 48で L18 σ=0.02 が Math=95% (n=20) で最強と判明。
しかし n=20 では統計的有意性が弱い (Sonnet指摘)。

→ TruthfulQA 100問 + Math 50問 = 150問 で検証。
→ Fisher exact test で p値を算出。

条件:
  1. Baseline (σ=0.05 normal, no echo)
  2. Surgical L18 Echo (σ=0.02)

Usage:
    python experiments/phase49_large_validation.py
"""

import torch, torch.nn as nn
import os, json, gc, time, random, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.stats import fisher_exact

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
ECHO_TOKENS = 2
SIGMA_NORMAL = 0.05
SIGMA_ECHO = 0.02
TARGET_LAYER = 18
SEED = 2026
N_FACTUAL = 100
N_MATH = 50

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Extended math set (50 problems)
MATH_PROMPTS = [
    {"q": "What is 17 + 28?", "a": "45"}, {"q": "What is 156 - 89?", "a": "67"},
    {"q": "What is 12 × 7?", "a": "84"}, {"q": "What is 144 / 12?", "a": "12"},
    {"q": "What is 23 + 45 + 12?", "a": "80"}, {"q": "What is 15 × 15?", "a": "225"},
    {"q": "What is 1000 - 573?", "a": "427"}, {"q": "What is 8 × 9?", "a": "72"},
    {"q": "What is 256 / 16?", "a": "16"}, {"q": "What is 33 + 67?", "a": "100"},
    {"q": "What is 2^8?", "a": "256"}, {"q": "What is sqrt(144)?", "a": "12"},
    {"q": "What is 99 × 3?", "a": "297"}, {"q": "What is 50% of 240?", "a": "120"},
    {"q": "What is 7!?", "a": "5040"}, {"q": "What is 1024 / 32?", "a": "32"},
    {"q": "What is the next prime after 29?", "a": "31"}, {"q": "What is 13 × 13?", "a": "169"},
    {"q": "What is 200 - 137?", "a": "63"}, {"q": "What is 25 × 4?", "a": "100"},
    {"q": "What is 11 × 11?", "a": "121"}, {"q": "What is 999 - 456?", "a": "543"},
    {"q": "What is 64 / 8?", "a": "8"}, {"q": "What is 17 × 6?", "a": "102"},
    {"q": "What is 2^10?", "a": "1024"}, {"q": "What is sqrt(225)?", "a": "15"},
    {"q": "What is 45 + 78?", "a": "123"}, {"q": "What is 19 × 19?", "a": "361"},
    {"q": "What is 500 / 25?", "a": "20"}, {"q": "What is 3^5?", "a": "243"},
    {"q": "What is 88 - 29?", "a": "59"}, {"q": "What is 14 × 14?", "a": "196"},
    {"q": "What is 75% of 200?", "a": "150"}, {"q": "What is 6!?", "a": "720"},
    {"q": "What is 360 / 12?", "a": "30"}, {"q": "What is 27 + 84?", "a": "111"},
    {"q": "What is 16 × 16?", "a": "256"}, {"q": "What is 1000 / 8?", "a": "125"},
    {"q": "What is sqrt(196)?", "a": "14"}, {"q": "What is 21 × 21?", "a": "441"},
    {"q": "What is 333 + 667?", "a": "1000"}, {"q": "What is 18 × 7?", "a": "126"},
    {"q": "What is 2^12?", "a": "4096"}, {"q": "What is 400 - 157?", "a": "243"},
    {"q": "What is 9 × 9?", "a": "81"}, {"q": "What is 20% of 500?", "a": "100"},
    {"q": "What is 5^4?", "a": "625"}, {"q": "What is 288 / 12?", "a": "24"},
    {"q": "What is sqrt(289)?", "a": "17"}, {"q": "What is 77 + 88?", "a": "165"},
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
    print(f"  ✅ Loaded: {len(model.model.layers)} layers")
    return model, tokenizer


class L18Hook:
    """Echo hook for L18 only."""
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


def generate_text(model, tokenizer, prompt, hook, use_echo=False):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    if use_echo:
        hook.activate_echo()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True,
                                 temperature=0.9, top_p=0.95, top_k=50,
                                 repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id)
    hook.reset()
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(input_text):].strip()


def check_math(resp, ans):
    resp = resp.strip().replace(",", "").replace(" ", "")
    ans = ans.strip().replace(",", "").replace(" ", "")
    if ans in resp: return True
    return ans in re.findall(r'\d+', resp)


def build_dataset(tokenizer):
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), N_FACTUAL)
    items = []
    for idx in indices:
        row = ds[idx]
        items.append({"type": "factual", "prompt": f"Q: {row['question']}\nA:",
                      "mc1_targets": row["mc1_targets"]})
    for mp in MATH_PROMPTS[:N_MATH]:
        items.append({"type": "math",
                      "prompt": f"Solve this math problem. Give only the numerical answer.\n\nQ: {mp['q']}\nA:",
                      "correct_answer": mp["a"]})
    return items


def run_condition(model, tokenizer, dataset, hook, use_echo, label):
    f_correct, f_total, m_correct, m_total = 0, 0, 0, 0
    per_item = []

    for idx, item in enumerate(dataset):
        prompt = item["prompt"]
        correct = False

        if item["type"] == "factual":
            ch = item["mc1_targets"]
            ci = ch["labels"].index(1) if 1 in ch["labels"] else 0
            if use_echo: hook.activate_echo()
            lps = [compute_completion_logprob(model, tokenizer, prompt, " " + c) for c in ch["choices"]]
            hook.reset()
            if np.argmax(lps) == ci:
                correct = True; f_correct += 1
            f_total += 1
        elif item["type"] == "math":
            resp = generate_text(model, tokenizer, prompt, hook, use_echo=use_echo)
            if check_math(resp, item["correct_answer"]):
                correct = True; m_correct += 1
            m_total += 1

        per_item.append(correct)

        if (idx + 1) % 20 == 0:
            cf = f_correct / max(f_total, 1) * 100
            cm = m_correct / max(m_total, 1) * 100
            print(f"    [{idx+1}/{len(dataset)}] Fact={cf:.0f}% ({f_correct}/{f_total}) Math={cm:.0f}% ({m_correct}/{m_total})")

    fa = f_correct / max(f_total, 1) * 100
    ma = m_correct / max(m_total, 1) * 100
    print(f"  📊 {label}: Fact={fa:.1f}% ({f_correct}/{f_total}) Math={ma:.1f}% ({m_correct}/{m_total})")
    return {
        "factual_acc": round(fa, 2), "math_acc": round(ma, 2),
        "f_correct": f_correct, "f_total": f_total,
        "m_correct": m_correct, "m_total": m_total,
        "per_item": per_item,
    }


def compute_fisher(baseline, treatment, label):
    """Fisher exact test for treatment vs baseline."""
    b_correct = sum(baseline["per_item"])
    b_wrong = len(baseline["per_item"]) - b_correct
    t_correct = sum(treatment["per_item"])
    t_wrong = len(treatment["per_item"]) - t_correct
    table = [[t_correct, t_wrong], [b_correct, b_wrong]]
    odds_ratio, p_value = fisher_exact(table, alternative='greater')
    print(f"  📈 Fisher ({label}): OR={odds_ratio:.3f} p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")
    return {"odds_ratio": round(odds_ratio, 4), "p_value": round(p_value, 6)}


def visualize(baseline, surgical, fisher_all, fisher_f, fisher_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f"Phase 49: Large-Scale Validation (n={N_FACTUAL+N_MATH})\n"
                 f"L18 σ=0.02 Surgical Echo vs Baseline",
                 fontsize=14, fontweight="bold")

    # Panel 1: Accuracy comparison
    ax = axes[0]
    categories = ["Factual\n(TruthfulQA)", "Math\n(Arithmetic)", "Overall"]
    b_accs = [baseline["factual_acc"], baseline["math_acc"],
              sum(baseline["per_item"])/len(baseline["per_item"])*100]
    s_accs = [surgical["factual_acc"], surgical["math_acc"],
              sum(surgical["per_item"])/len(surgical["per_item"])*100]
    x = np.arange(len(categories)); w = 0.35
    b1 = ax.bar(x - w/2, b_accs, w, label="Baseline", color="#95a5a6", alpha=0.8)
    b2 = ax.bar(x + w/2, s_accs, w, label="L18 Echo", color="#e74c3c", alpha=0.8)
    for bar, val in zip(b1, b_accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    for bar, val in zip(b2, s_accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_title("(A) Accuracy")
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.legend(); ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Per-item comparison
    ax2 = axes[1]
    b_items = baseline["per_item"]
    s_items = surgical["per_item"]
    agree_correct = sum(1 for b, s in zip(b_items, s_items) if b and s)
    agree_wrong = sum(1 for b, s in zip(b_items, s_items) if not b and not s)
    only_base = sum(1 for b, s in zip(b_items, s_items) if b and not s)
    only_surg = sum(1 for b, s in zip(b_items, s_items) if not b and s)
    sizes = [agree_correct, agree_wrong, only_base, only_surg]
    labels_pie = [f"Both ✓\n({agree_correct})", f"Both ✗\n({agree_wrong})",
                  f"Only Base\n({only_base})", f"Only Echo\n({only_surg})"]
    colors_pie = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
    ax2.pie(sizes, labels=labels_pie, colors=colors_pie, autopct="%1.0f%%",
            startangle=90, textprops={'fontsize': 10})
    ax2.set_title("(B) Per-Item Agreement")

    # Panel 3: Statistical test
    ax3 = axes[2]
    ax3.axis("off")
    txt = "STATISTICAL VALIDATION\n" + "="*40 + "\n\n"
    txt += f"Sample size: n={len(b_items)}\n"
    txt += f"  Factual: n={baseline['f_total']}\n"
    txt += f"  Math:    n={baseline['m_total']}\n\n"
    txt += f"Baseline accuracy:\n"
    txt += f"  Fact={baseline['factual_acc']:.1f}% Math={baseline['math_acc']:.1f}%\n\n"
    txt += f"L18 Echo accuracy:\n"
    txt += f"  Fact={surgical['factual_acc']:.1f}% Math={surgical['math_acc']:.1f}%\n\n"
    txt += "Fisher Exact Test (one-sided):\n"
    txt += f"  Overall: p={fisher_all['p_value']:.6f} "
    txt += f"{'***' if fisher_all['p_value'] < 0.001 else '**' if fisher_all['p_value'] < 0.01 else '*' if fisher_all['p_value'] < 0.05 else 'n.s.'}\n"
    txt += f"  Factual: p={fisher_f['p_value']:.6f} "
    txt += f"{'***' if fisher_f['p_value'] < 0.001 else '**' if fisher_f['p_value'] < 0.01 else '*' if fisher_f['p_value'] < 0.05 else 'n.s.'}\n"
    txt += f"  Math:    p={fisher_m['p_value']:.6f} "
    txt += f"{'***' if fisher_m['p_value'] < 0.001 else '**' if fisher_m['p_value'] < 0.01 else '*' if fisher_m['p_value'] < 0.05 else 'n.s.'}\n"
    ax3.text(0.05, 0.95, txt, transform=ax3.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9))

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase49_large_validation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model()
    dataset = build_dataset(tokenizer)
    print(f"  📂 Dataset: {len(dataset)} items (Factual={N_FACTUAL}, Math={N_MATH})")

    layers = model.model.layers
    hook = L18Hook(sigma_normal=SIGMA_NORMAL, sigma_echo=SIGMA_ECHO, echo_tokens=ECHO_TOKENS)
    handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    # Condition 1: Baseline
    print(f"\n{'═'*60}")
    print(f"  🔬 BASELINE (n={len(dataset)})")
    print(f"{'═'*60}")
    baseline = run_condition(model, tokenizer, dataset, hook, use_echo=False, label="Baseline")

    # Condition 2: Surgical L18 Echo
    print(f"\n{'═'*60}")
    print(f"  💉 L18 SURGICAL ECHO σ=0.02 (n={len(dataset)})")
    print(f"{'═'*60}")
    surgical = run_condition(model, tokenizer, dataset, hook, use_echo=True, label="L18 Echo")

    handle.remove()

    # Fisher exact tests
    print(f"\n{'═'*60}")
    print(f"  📈 STATISTICAL TESTS")
    print(f"{'═'*60}")
    fisher_all = compute_fisher(baseline, surgical, "Overall")

    # Split for per-type Fisher
    b_f = {"per_item": baseline["per_item"][:N_FACTUAL]}
    s_f = {"per_item": surgical["per_item"][:N_FACTUAL]}
    b_m = {"per_item": baseline["per_item"][N_FACTUAL:]}
    s_m = {"per_item": surgical["per_item"][N_FACTUAL:]}
    fisher_f = compute_fisher(b_f, s_f, "Factual")
    fisher_m = compute_fisher(b_m, s_m, "Math")

    elapsed = time.time() - t_start
    fig_path = visualize(baseline, surgical, fisher_all, fisher_f, fisher_m)

    output = {
        "experiment": "Phase 49: Large-Scale Validation",
        "model": MODEL_SHORT,
        "n_total": len(dataset), "n_factual": N_FACTUAL, "n_math": N_MATH,
        "elapsed_minutes": round(elapsed / 60, 1),
        "baseline": {k: v for k, v in baseline.items() if k != "per_item"},
        "surgical": {k: v for k, v in surgical.items() if k != "per_item"},
        "fisher_overall": fisher_all, "fisher_factual": fisher_f, "fisher_math": fisher_m,
        "figure_path": fig_path,
    }
    log_path = os.path.join(RESULTS_DIR, "phase49_large_validation_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)

    b_overall = sum(baseline["per_item"])/len(baseline["per_item"])*100
    s_overall = sum(surgical["per_item"])/len(surgical["per_item"])*100

    print(f"\n{'═'*60}")
    print(f"  💉 PHASE 49: LARGE-SCALE VERDICT (n={len(dataset)})")
    print(f"{'═'*60}")
    print(f"  Baseline: Fact={baseline['factual_acc']:.1f}% Math={baseline['math_acc']:.1f}% Overall={b_overall:.1f}%")
    print(f"  L18 Echo: Fact={surgical['factual_acc']:.1f}% Math={surgical['math_acc']:.1f}% Overall={s_overall:.1f}%")
    print(f"  Δ Overall: {s_overall - b_overall:+.1f}%")
    print(f"  Fisher p (overall): {fisher_all['p_value']:.6f}")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
