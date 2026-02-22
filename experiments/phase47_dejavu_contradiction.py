"""
Phase 47: Déjà Vu Contradiction Resolution
===========================================

Sonnet指摘: エントロピーが46%低下するのに精度は-15~-20%低下する矛盾。
仮説: 4bit量子化モデルのコンテキスト長飽和が原因。

実験設計:
  1. 短文プロンプト (≤64 tokens) vs 長文プロンプト (≥200 tokens)
  2. Single vs Repeat の精度比較
  3. エントロピー・conflict の同時測定
  4. コンテキスト残が少ないとき repeat が逆効果になるか検証

仮説: 短文では repeat で精度向上、長文では repeat でコンテキスト飽和→精度低下

Usage:
    python experiments/phase47_dejavu_contradiction.py
"""

import torch, torch.nn as nn
import os, json, gc, time, random, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
SEED = 2026
MAX_NEW_TOKENS = 50

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── SHORT prompts (≤64 tokens when tokenized) ───
SHORT_MATH = [
    {"q": "What is 17 + 28?", "a": "45"},
    {"q": "What is 12 × 7?", "a": "84"},
    {"q": "What is 144 / 12?", "a": "12"},
    {"q": "What is 15 × 15?", "a": "225"},
    {"q": "What is 8 × 9?", "a": "72"},
    {"q": "What is 33 + 67?", "a": "100"},
    {"q": "What is 99 × 3?", "a": "297"},
    {"q": "What is 25 × 4?", "a": "100"},
    {"q": "What is 200 - 137?", "a": "63"},
    {"q": "What is 2^8?", "a": "256"},
]

SHORT_FACT = [
    {"q": "What is the capital of France?", "a": "Paris"},
    {"q": "What planet is closest to the sun?", "a": "Mercury"},
    {"q": "What is the chemical symbol for gold?", "a": "Au"},
    {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100"},
    {"q": "What is the largest ocean?", "a": "Pacific"},
    {"q": "How many continents are there?", "a": "7"},
    {"q": "What gas do plants absorb?", "a": "CO2"},
    {"q": "What is the speed of light in km/s?", "a": "300000"},
    {"q": "What year did World War II end?", "a": "1945"},
]

# ─── LONG prompts (≥200 tokens: context paragraph + question) ───
LONG_CONTEXT = """The following passage contains important information that you need to carefully read and understand before answering the question.

In the field of renewable energy, solar photovoltaic technology has seen remarkable advances over the past decade. The efficiency of commercial silicon solar cells has improved from approximately 15% in 2010 to over 22% by 2025, while manufacturing costs have dropped by more than 80%. This dramatic cost reduction has made solar energy competitive with fossil fuels in many regions. The levelized cost of energy (LCOE) for utility-scale solar projects fell below $30 per megawatt-hour in several markets, compared to $50-70 for new natural gas plants. Meanwhile, perovskite solar cells, a newer technology, have achieved laboratory efficiencies exceeding 25%, though their commercial deployment remains limited due to stability concerns. The global installed solar capacity reached approximately 1,500 gigawatts by the end of 2024, with China, the United States, and India leading in new installations. Energy storage solutions, particularly lithium-ion batteries, have also become more affordable, enabling solar power to address intermittency challenges."""

LONG_MATH = [
    {"q": "If a solar panel efficiency improved from 15% to 22%, what is the percentage point increase?", "a": "7"},
    {"q": "If manufacturing costs dropped by 80%, what fraction of original cost remains?", "a": "0.2"},
    {"q": "If LCOE is $30/MWh for solar vs $60/MWh for gas, solar is what percent cheaper?", "a": "50"},
    {"q": "If perovskite cells reach 25% efficiency and silicon reaches 22%, what is the difference?", "a": "3"},
    {"q": "If 1500 GW solar capacity generates 4 hours per day average, how many GWh per day?", "a": "6000"},
    {"q": "If costs dropped 80% over 15 years, what is the annual percentage drop (simple)?", "a": "5.3"},
    {"q": "If China has 40% of 1500 GW, how many GW does China have?", "a": "600"},
    {"q": "If a 100MW solar farm costs $30/MWh and runs 2000 hours/year, what is annual revenue in millions?", "a": "6"},
    {"q": "What is 1500 GW expressed in TW?", "a": "1.5"},
    {"q": "If battery costs dropped 70% and were originally $1000/kWh, what is current cost?", "a": "300"},
]

LONG_FACT = [
    {"q": "According to the passage, what efficiency did commercial silicon cells reach by 2025?", "a": "22"},
    {"q": "According to the passage, by how much did manufacturing costs drop?", "a": "80"},
    {"q": "What is the LCOE mentioned for utility-scale solar?", "a": "30"},
    {"q": "What technology achieved 25% laboratory efficiency?", "a": "perovskite"},
    {"q": "What was the global installed solar capacity by end of 2024?", "a": "1500"},
    {"q": "Which three countries lead in new installations?", "a": "China"},
    {"q": "What battery technology is mentioned for energy storage?", "a": "lithium"},
    {"q": "What concern limits perovskite deployment?", "a": "stability"},
    {"q": "What was the LCOE range for new natural gas plants?", "a": "50"},
    {"q": "What year baseline is mentioned for 15% efficiency?", "a": "2010"},
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
    print(f"  ✅ Loaded")
    return model, tokenizer


def compute_metrics(model, tokenizer, prompt):
    """Compute entropy + conflict for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits[0]
        hs = outputs.hidden_states
    # Entropy
    entropies = []
    for i in range(logits.shape[0]):
        probs = torch.softmax(logits[i].float(), dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        entropies.append(ent)
    # Conflict (L8 vs L28)
    n_layers = len(hs) - 1
    shallow = hs[min(8, n_layers)][0].float()
    deep = hs[min(28, n_layers)][0].float()
    conflicts = []
    for i in range(shallow.shape[0]):
        cos = torch.nn.functional.cosine_similarity(
            shallow[i].unsqueeze(0), deep[i].unsqueeze(0)).item()
        conflicts.append(1.0 - cos)
    return np.mean(entropies), np.mean(conflicts), len(inputs['input_ids'][0])


def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                 temperature=0.7, top_p=0.9, repetition_penalty=1.2,
                                 pad_token_id=tokenizer.pad_token_id)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(input_text):].strip()


def check_answer(response, correct):
    resp = response.lower().strip().replace(",", "").replace(" ", "")
    ans = correct.lower().strip().replace(",", "").replace(" ", "")
    return ans in resp


def run_condition(model, tokenizer, items, prompt_template, condition_label, is_repeat=False):
    """Run single vs repeat on a set of items."""
    correct = 0
    total = len(items)
    entropy_avg = 0
    conflict_avg = 0
    token_lengths = []

    for idx, item in enumerate(items):
        prompt = prompt_template.format(q=item["q"])
        if is_repeat:
            prompt = prompt + "\n\n" + prompt

        # Metrics
        ent, conf, n_tokens = compute_metrics(model, tokenizer, prompt)
        entropy_avg += ent
        conflict_avg += conf
        token_lengths.append(n_tokens)

        # Answer
        resp = generate_answer(model, tokenizer, prompt)
        if check_answer(resp, item["a"]):
            correct += 1

    acc = correct / total * 100
    entropy_avg /= total
    conflict_avg /= total
    avg_tokens = np.mean(token_lengths)

    print(f"    {condition_label}: Acc={acc:.1f}% Ent={entropy_avg:.3f} Conf={conflict_avg:.4f} AvgTok={avg_tokens:.0f}")
    return {
        "accuracy": round(acc, 2),
        "entropy": round(entropy_avg, 4),
        "conflict": round(conflict_avg, 5),
        "avg_tokens": round(avg_tokens, 1),
    }


def visualize(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Phase 47: Déjà Vu Contradiction Resolution\n"
                 "Why does entropy drop but accuracy suffers?",
                 fontsize=14, fontweight="bold")

    categories = ["Short\nMath", "Short\nFact", "Long\nMath", "Long\nFact"]
    single_acc = [results["short_math_single"]["accuracy"],
                  results["short_fact_single"]["accuracy"],
                  results["long_math_single"]["accuracy"],
                  results["long_fact_single"]["accuracy"]]
    repeat_acc = [results["short_math_repeat"]["accuracy"],
                  results["short_fact_repeat"]["accuracy"],
                  results["long_math_repeat"]["accuracy"],
                  results["long_fact_repeat"]["accuracy"]]

    # Panel 1: Accuracy comparison
    ax = axes[0, 0]
    x = np.arange(len(categories))
    w = 0.35
    b1 = ax.bar(x - w/2, single_acc, w, label="Single", color="#3498db", alpha=0.8)
    b2 = ax.bar(x + w/2, repeat_acc, w, label="Repeat", color="#e74c3c", alpha=0.8)
    for bar, val in zip(b1, single_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%", ha="center", fontsize=9)
    for bar, val in zip(b2, repeat_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(A) Accuracy: Single vs Repeat")
    ax.legend(); ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Entropy comparison
    ax2 = axes[0, 1]
    single_ent = [results[f"{k}_single"]["entropy"] for k in ["short_math", "short_fact", "long_math", "long_fact"]]
    repeat_ent = [results[f"{k}_repeat"]["entropy"] for k in ["short_math", "short_fact", "long_math", "long_fact"]]
    b1 = ax2.bar(x - w/2, single_ent, w, label="Single", color="#3498db", alpha=0.8)
    b2 = ax2.bar(x + w/2, repeat_ent, w, label="Repeat", color="#e74c3c", alpha=0.8)
    for bar, val in zip(b1, single_ent):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{val:.2f}", ha="center", fontsize=9)
    for bar, val in zip(b2, repeat_ent):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{val:.2f}", ha="center", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(categories)
    ax2.set_ylabel("Entropy (nats)")
    ax2.set_title("(B) Entropy: Always Drops on Repeat")
    ax2.legend(); ax2.grid(True, alpha=0.2, axis="y")
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    # Panel 3: Token count
    ax3 = axes[0, 2]
    single_tok = [results[f"{k}_single"]["avg_tokens"] for k in ["short_math", "short_fact", "long_math", "long_fact"]]
    repeat_tok = [results[f"{k}_repeat"]["avg_tokens"] for k in ["short_math", "short_fact", "long_math", "long_fact"]]
    b1 = ax3.bar(x - w/2, single_tok, w, label="Single", color="#3498db", alpha=0.8)
    b2 = ax3.bar(x + w/2, repeat_tok, w, label="Repeat", color="#e74c3c", alpha=0.8)
    for bar, val in zip(b1, single_tok):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{val:.0f}", ha="center", fontsize=9)
    for bar, val in zip(b2, repeat_tok):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{val:.0f}", ha="center", fontsize=9)
    ax3.set_xticks(x); ax3.set_xticklabels(categories)
    ax3.set_ylabel("Token Count")
    ax3.set_title("(C) Context Length (tokens)")
    ax3.legend(); ax3.grid(True, alpha=0.2, axis="y")
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    # Panel 4: Accuracy delta (repeat - single)
    ax4 = axes[1, 0]
    deltas = [r - s for r, s in zip(repeat_acc, single_acc)]
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
    bars = ax4.bar(categories, deltas, color=colors, alpha=0.8)
    for bar, val in zip(bars, deltas):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (1 if val >= 0 else -3),
                 f"{val:+.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax4.axhline(y=0, color="black", linewidth=0.5)
    ax4.set_ylabel("Accuracy Δ (Repeat - Single)")
    ax4.set_title("(D) Repeat Effect: Short vs Long")
    ax4.grid(True, alpha=0.2, axis="y")
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

    # Panel 5: Conflict comparison
    ax5 = axes[1, 1]
    single_conf = [results[f"{k}_single"]["conflict"] for k in ["short_math", "short_fact", "long_math", "long_fact"]]
    repeat_conf = [results[f"{k}_repeat"]["conflict"] for k in ["short_math", "short_fact", "long_math", "long_fact"]]
    b1 = ax5.bar(x - w/2, single_conf, w, label="Single", color="#3498db", alpha=0.8)
    b2 = ax5.bar(x + w/2, repeat_conf, w, label="Repeat", color="#e74c3c", alpha=0.8)
    ax5.set_xticks(x); ax5.set_xticklabels(categories)
    ax5.set_ylabel("Layer Conflict (1-cos)")
    ax5.set_title("(E) Layer Conflict: Specialization on Repeat")
    ax5.legend(); ax5.grid(True, alpha=0.2, axis="y")
    ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)

    # Panel 6: Verdict
    ax6 = axes[1, 2]
    ax6.axis("off")
    short_delta = np.mean(deltas[:2])
    long_delta = np.mean(deltas[2:])
    txt = "VERDICT\n" + "="*40 + "\n\n"
    txt += f"Short prompts (≤64 tok):\n"
    txt += f"  Repeat Δ = {short_delta:+.1f}%\n\n"
    txt += f"Long prompts (≥200 tok):\n"
    txt += f"  Repeat Δ = {long_delta:+.1f}%\n\n"
    txt += "="*40 + "\n"
    if short_delta > 0 and long_delta < 0:
        txt += "✅ HYPOTHESIS CONFIRMED!\n"
        txt += "Short: Repeat helps\n"
        txt += "Long: Context saturation\n"
        txt += "  causes accuracy drop"
    elif short_delta > long_delta:
        txt += "⚠️ PARTIAL CONFIRMATION\n"
        txt += "Repeat is relatively better\n"
        txt += "on short prompts"
    else:
        txt += "❌ HYPOTHESIS REJECTED\n"
        txt += "Context length is NOT\n"
        txt += "the primary factor"
    ax6.text(0.05, 0.95, txt, transform=ax6.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9))

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase47_dejavu_contradiction.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model()

    results = {}

    short_math_template = "Solve this math problem. Give only the numerical answer.\n\nQ: {q}\nA:"
    short_fact_template = "Answer in one word or phrase.\n\nQ: {q}\nA:"
    long_math_template = LONG_CONTEXT + "\n\nSolve this math problem based on the passage. Give only the numerical answer.\n\nQ: {q}\nA:"
    long_fact_template = LONG_CONTEXT + "\n\nAnswer this question based on the passage. Give a brief answer.\n\nQ: {q}\nA:"

    conditions = [
        ("short_math", SHORT_MATH, short_math_template, "Short Math"),
        ("short_fact", SHORT_FACT, short_fact_template, "Short Fact"),
        ("long_math", LONG_MATH, long_math_template, "Long Math"),
        ("long_fact", LONG_FACT, long_fact_template, "Long Fact"),
    ]

    for key, items, template, label in conditions:
        print(f"\n{'═'*60}")
        print(f"  📏 {label} (n={len(items)})")
        print(f"{'═'*60}")
        results[f"{key}_single"] = run_condition(model, tokenizer, items, template, f"{label} Single", is_repeat=False)
        results[f"{key}_repeat"] = run_condition(model, tokenizer, items, template, f"{label} Repeat", is_repeat=True)

    elapsed = time.time() - t_start
    fig_path = visualize(results)

    # Summary
    print(f"\n{'═'*60}")
    print(f"  🔬 PHASE 47: VERDICT")
    print(f"{'═'*60}")
    for key, _, _, label in conditions:
        s = results[f"{key}_single"]["accuracy"]
        r = results[f"{key}_repeat"]["accuracy"]
        st = results[f"{key}_single"]["avg_tokens"]
        rt = results[f"{key}_repeat"]["avg_tokens"]
        print(f"  {label}: Single={s:.0f}% ({st:.0f}tok) → Repeat={r:.0f}% ({rt:.0f}tok) Δ={r-s:+.1f}%")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")

    output = {
        "experiment": "Phase 47: Déjà Vu Contradiction Resolution",
        "model": MODEL_SHORT,
        "elapsed_minutes": round(elapsed / 60, 1),
        "results": results,
        "figure_path": fig_path,
    }
    log_path = os.path.join(RESULTS_DIR, "phase47_dejavu_contradiction_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
