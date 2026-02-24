"""
Phase 54: The Sanity Check — Factual Accuracy Under Different Conditions
=========================================================================

Hypothesis:
  High-Temperature (1.2) achieves highest Hanoi solve rate (28%, Phase 51)
  BUT may produce hallucinations on simple factual questions.
  SNN Echo maintains factual accuracy while still enabling creative reasoning.

Design:
  - 3 conditions: Baseline (temp=0.5), High-Temp (temp=1.2), SNN Echo (σ=0.02, temp=0.5)
  - 30 factual questions per condition (geography, math, science, history, common sense)
  - Measure factual accuracy per condition
  - Combine with Phase 51 Hanoi data for the full trade-off picture

Expected runtime: ~20-30 minutes
"""

import torch
import torch.nn.functional as F
import os, json, gc, time, random, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ═══ Config ═══
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
SEED = 2026
TARGET_LAYER = 18
SIGMA_ECHO = 0.02

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
#  FACTUAL QUESTIONS (30 questions with unambiguous answers)
# ═══════════════════════════════════════════════════

QUESTIONS = [
    # Geography (10)
    {"q": "What is the capital of Japan?", "answer": ["tokyo"], "category": "geography"},
    {"q": "What is the capital of France?", "answer": ["paris"], "category": "geography"},
    {"q": "What is the largest continent by area?", "answer": ["asia"], "category": "geography"},
    {"q": "What ocean lies between Europe and North America?", "answer": ["atlantic"], "category": "geography"},
    {"q": "What is the capital of Germany?", "answer": ["berlin"], "category": "geography"},
    {"q": "What country has the largest population in the world?", "answer": ["india", "china"], "category": "geography"},
    {"q": "On which continent is Brazil located?", "answer": ["south america"], "category": "geography"},
    {"q": "What is the capital of the United Kingdom?", "answer": ["london"], "category": "geography"},
    {"q": "What is the longest river in Africa?", "answer": ["nile"], "category": "geography"},
    {"q": "What is the capital of Italy?", "answer": ["rome", "roma"], "category": "geography"},

    # Math (10)
    {"q": "What is 7 + 8?", "answer": ["15"], "category": "math"},
    {"q": "What is 12 x 12?", "answer": ["144"], "category": "math"},
    {"q": "What is the square root of 64?", "answer": ["8"], "category": "math"},
    {"q": "What is 100 divided by 4?", "answer": ["25"], "category": "math"},
    {"q": "What is 2 to the power of 10?", "answer": ["1024"], "category": "math"},
    {"q": "What is 15% of 200?", "answer": ["30"], "category": "math"},
    {"q": "How many sides does a hexagon have?", "answer": ["6", "six"], "category": "math"},
    {"q": "What is 3 factorial (3!)?", "answer": ["6", "six"], "category": "math"},
    {"q": "What is the value of pi rounded to 2 decimal places?", "answer": ["3.14"], "category": "math"},
    {"q": "What is 17 - 9?", "answer": ["8"], "category": "math"},

    # Science & Common Knowledge (10)
    {"q": "What is the chemical formula for water?", "answer": ["h2o"], "category": "science"},
    {"q": "What planet is closest to the Sun?", "answer": ["mercury"], "category": "science"},
    {"q": "How many days are in a standard (non-leap) year?", "answer": ["365"], "category": "science"},
    {"q": "What gas do humans breathe in to survive?", "answer": ["oxygen", "o2"], "category": "science"},
    {"q": "What is the boiling point of water in Celsius?", "answer": ["100"], "category": "science"},
    {"q": "How many legs does a spider have?", "answer": ["8", "eight"], "category": "science"},
    {"q": "What is the speed of light approximately in km/s?", "answer": ["300000", "300,000", "3e5", "299792"], "category": "science"},
    {"q": "What year did World War II end?", "answer": ["1945"], "category": "science"},
    {"q": "What is the chemical symbol for gold?", "answer": ["au"], "category": "science"},
    {"q": "How many continents are there?", "answer": ["7", "seven"], "category": "science"},
]


# ═══════════════════════════════════════════════════
#  SNN ECHO HOOK
# ═══════════════════════════════════════════════════

class EchoHook:
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn_like(hs) * self.sigma
        return (hs + noise,) + args[1:]


# ═══════════════════════════════════════════════════
#  MODEL & GENERATION
# ═══════════════════════════════════════════════════

def load_model():
    print(f"\n📦 Loading {MODEL_NAME}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    print(f"  ✅ Loaded ({len(model.model.layers)} layers)")
    return model, tok


def generate(model, tok, prompt, temperature=0.5, max_tokens=60):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperature, top_p=0.9, top_k=40,
            repetition_penalty=1.2, pad_token_id=tok.pad_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    inp = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(inp):].strip()


def build_qa_prompt(tokenizer, question):
    messages = [
        {"role": "user", "content": f"Answer the following question with a short, direct answer. "
         f"Do not explain, just give the answer.\n\nQuestion: {question}\nAnswer:"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def check_answer(response, correct_answers):
    """Check if any correct answer appears in the response."""
    resp_lower = response.lower().strip()
    for ans in correct_answers:
        if ans.lower() in resp_lower:
            return True
    return False


# ═══════════════════════════════════════════════════
#  RUN CONDITION
# ═══════════════════════════════════════════════════

def run_condition(model, tok, cond_name, temperature, questions):
    print(f"\n  📝 {cond_name} | temp={temperature} | {len(questions)} questions")

    layers = model.model.layers
    hook, handle = None, None
    if cond_name == "snn_echo":
        hook = EchoHook(SIGMA_ECHO)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    results = []
    correct = 0
    for i, qa in enumerate(questions):
        prompt = build_qa_prompt(tok, qa["q"])
        response = generate(model, tok, prompt, temperature=temperature)

        is_correct = check_answer(response, qa["answer"])
        if is_correct: correct += 1
        icon = "✅" if is_correct else "❌"

        results.append({
            "question": qa["q"],
            "expected": qa["answer"],
            "response": response[:100],
            "correct": is_correct,
            "category": qa["category"],
        })

        print(f"    {i+1:2d}/{len(questions)}: {icon} Q: {qa['q'][:40]:<40s} "
              f"A: {response[:30]:<30s} [{correct}/{i+1}={correct/(i+1)*100:.0f}%]")

    if handle: handle.remove()

    accuracy = correct / len(questions) * 100
    by_category = {}
    for cat in ["geography", "math", "science"]:
        cat_results = [r for r in results if r["category"] == cat]
        cat_correct = sum(1 for r in cat_results if r["correct"])
        by_category[cat] = {"correct": cat_correct, "total": len(cat_results),
                            "accuracy": round(cat_correct / len(cat_results) * 100, 1)}

    summary = {
        "condition": cond_name,
        "temperature": temperature,
        "n_questions": len(questions),
        "n_correct": correct,
        "accuracy": round(accuracy, 1),
        "by_category": by_category,
    }

    print(f"    📊 Accuracy: {accuracy:.1f}% ({correct}/{len(questions)})")
    return summary, results


# ═══════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════

def visualize(summaries, phase51_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Phase 54: The Sanity Check — Creativity vs. Factual Accuracy Trade-off\n"
                 "Can High-Temperature solve puzzles without losing its mind?",
                 fontsize=14, fontweight="bold", y=1.02)

    conds = ["baseline", "high_temp", "snn_echo"]
    labels = ["Baseline\n(temp=0.5)", "High-Temp\n(temp=1.2)", "SNN Echo\n(σ=0.02)"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    # Panel 1: Hanoi Solve Rate (from Phase 51)
    ax = axes[0]
    hanoi_rates = [phase51_data.get(c, {}).get("solve_rate", 0) for c in conds]
    bars = ax.bar(range(3), hanoi_rates, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    for i, (b, sr) in enumerate(zip(bars, hanoi_rates)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
               f"{sr:.0f}%", ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("① Hanoi Solve Rate\n(from Phase 51, N=50)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 40)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Factual Accuracy (this experiment)
    ax = axes[1]
    fact_rates = [next((s["accuracy"] for s in summaries if s["condition"]==c), 0) for c in conds]
    bars = ax.bar(range(3), fact_rates, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    for i, (b, sr) in enumerate(zip(bars, fact_rates)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
               f"{sr:.0f}%", ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Factual Accuracy (%)", fontsize=12)
    ax.set_title("② Factual Accuracy\n(30 questions, this experiment)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 3: Trade-off scatter (Hanoi vs Factual)
    ax = axes[2]
    for i, (c, label) in enumerate(zip(conds, ["Baseline", "High-Temp", "SNN Echo"])):
        ax.scatter(hanoi_rates[i], fact_rates[i], c=colors[i], s=300, zorder=5,
                  edgecolors='black', linewidth=1.5)
        offset_x = 1 if i != 2 else -2
        offset_y = -3 if i == 1 else 2
        ax.annotate(label, (hanoi_rates[i], fact_rates[i]),
                   textcoords="offset points", xytext=(offset_x, offset_y+8),
                   fontsize=12, fontweight='bold', ha='center')

    ax.set_xlabel("Hanoi Solve Rate (%)", fontsize=12)
    ax.set_ylabel("Factual Accuracy (%)", fontsize=12)
    ax.set_title("③ The Trade-off\n(Top-right = ideal)", fontsize=13, fontweight='bold')
    ax.set_xlim(0, 40); ax.set_ylim(0, 110)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.3, label='90% factual threshold')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Highlight ideal zone
    from matplotlib.patches import Rectangle
    ideal = Rectangle((15, 85), 25, 25, linewidth=2, edgecolor='green',
                      facecolor='green', alpha=0.1)
    ax.add_patch(ideal)
    ax.text(27, 105, "IDEAL\nZONE", ha='center', fontsize=10, color='green', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGURES_DIR, "phase54_sanity_check.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure: {path}")
    return path


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tok = load_model()

    print(f"\n{'═'*60}")
    print(f"  🧪 Phase 54: The Sanity Check")
    print(f"  Can High-Temperature solve puzzles without losing its mind?")
    print(f"{'═'*60}")

    conditions = [
        ("baseline", 0.5),
        ("high_temp", 1.2),
        ("snn_echo", 0.5),
    ]

    all_summaries = []
    all_results = {}

    for cond_name, temp in conditions:
        summary, results = run_condition(model, tok, cond_name, temp, QUESTIONS)
        all_summaries.append(summary)
        all_results[cond_name] = results
        gc.collect(); torch.cuda.empty_cache()

    elapsed = time.time() - t0

    # Phase 51 Hanoi data for comparison
    phase51_data = {
        "baseline": {"solve_rate": 16.0, "n_solved": 8, "n_trials": 50},
        "high_temp": {"solve_rate": 28.0, "n_solved": 14, "n_trials": 50},
        "snn_echo": {"solve_rate": 22.0, "n_solved": 11, "n_trials": 50},
    }

    # Visualize
    fig_path = visualize(all_summaries, phase51_data)

    # Save
    out = {
        "experiment": "Phase 54: The Sanity Check",
        "model": "Mistral-7B-Instruct-v0.3",
        "purpose": "Test factual accuracy under conditions that improve Hanoi solve rate",
        "elapsed_min": round(elapsed / 60, 1),
        "phase51_hanoi": phase51_data,
        "factual_results": all_summaries,
        "detailed_results": {k: v for k, v in all_results.items()},
        "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase54_sanity_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Verdict
    print(f"\n{'═'*70}")
    print(f"  🧪 PHASE 54: THE SANITY CHECK — VERDICT")
    print(f"{'═'*70}")
    print(f"  {'Condition':<15} {'Hanoi Solve%':>14} {'Factual Acc%':>14} {'Trade-off':>12}")
    print(f"{'─'*70}")
    for s in all_summaries:
        hanoi = phase51_data[s["condition"]]["solve_rate"]
        fact = s["accuracy"]
        # Trade-off score: geometric mean of both
        trade = round((hanoi * fact) ** 0.5, 1) if hanoi > 0 and fact > 0 else 0
        print(f"  {s['condition']:<15} {hanoi:>13.1f}% {fact:>13.1f}% {trade:>11.1f}")
    print(f"{'─'*70}")

    # Who wins?
    baseline = next(s for s in all_summaries if s["condition"] == "baseline")
    high_temp = next(s for s in all_summaries if s["condition"] == "high_temp")
    snn = next(s for s in all_summaries if s["condition"] == "snn_echo")

    print(f"\n  🏆 WINNER ANALYSIS:")
    if high_temp["accuracy"] < baseline["accuracy"] - 5:
        print(f"  ⚠️ High-Temp loses {baseline['accuracy'] - high_temp['accuracy']:.0f}% factual accuracy!")
        print(f"  → 'Drunk creativity': solves more puzzles but hallucinates on facts")
    if snn["accuracy"] >= baseline["accuracy"] - 3:
        print(f"  ✅ SNN Echo maintains factual accuracy ({snn['accuracy']:.0f}% vs baseline {baseline['accuracy']:.0f}%)")
        print(f"  → 'Sober creativity': creative reasoning WITHOUT losing sanity")

    print(f"\n  ⏱ {elapsed/60:.1f} min | 💾 {log}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
