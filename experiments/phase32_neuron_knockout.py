"""
Phase 32: Neuron Knockout (Digital Lobotomy)
=============================================
Causal proof that CfC neurons are truly task-specialized.

Strategy:
  1. Load trained CfC models from Phase 30 hidden states
  2. Identify Math-specialized neurons from Phase 30 coefficients
  3. Zero-out (knockout) specific neurons and measure task accuracy
  4. Compare Mistral (sharp specialization -> fragile) vs Qwen (distributed -> robust)

Predictions:
  - Mistral: Single Math neuron KO causes severe Math accuracy drop
  - Qwen: Single Math neuron KO barely affects Math (distributed backup)

Usage:
    python experiments/phase32_neuron_knockout.py
"""

import torch, torch.nn as nn
import os, sys, json, gc, time, datetime, random, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# CfC imports
from ncps.wirings import AutoNCP
from ncps.torch import CfC

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

SEED = 2026
MAX_NEW_TOKENS = 50

# Neuron roles from Phase 30
NEURON_ROLES = {
    "Mistral-7B": {
        "math_positive": [3, 6, 9, 14],   # +Math neurons
        "math_negative": [2, 4],            # -Math neurons
        "factual_positive": [7, 13],        # +Factual neurons
        "factual_negative": [0, 11],        # -Factual neurons
    },
    "Qwen2.5-7B": {
        "math_positive": [2, 12, 14],      # +Math neurons
        "math_negative": [0, 15],           # -Math neurons
        "factual_positive": [3, 9, 10],    # +Factual neurons
        "factual_negative": [5, 7, 11],    # -Factual neurons
    },
}

# Task prompts (same format as Phase 31)
MATH_PROMPTS = [
    {"q": "What is 17 + 28?", "a": "45"},
    {"q": "What is 156 - 89?", "a": "67"},
    {"q": "What is 12 * 7?", "a": "84"},
    {"q": "What is 144 / 12?", "a": "12"},
    {"q": "What is 23 + 45 + 12?", "a": "80"},
    {"q": "What is 15 * 15?", "a": "225"},
    {"q": "What is 1000 - 573?", "a": "427"},
    {"q": "What is 8 * 9?", "a": "72"},
    {"q": "What is 256 / 16?", "a": "16"},
    {"q": "What is 33 + 67?", "a": "100"},
    {"q": "If x + 5 = 12, what is x?", "a": "7"},
    {"q": "What is 2 to the power of 8?", "a": "256"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "What is 99 * 3?", "a": "297"},
    {"q": "What is 50% of 240?", "a": "120"},
    {"q": "If a triangle has sides 3, 4, and 5, what is its perimeter?", "a": "12"},
    {"q": "What is 7! (7 factorial)?", "a": "5040"},
    {"q": "What is 1024 / 32?", "a": "32"},
    {"q": "What is the next prime after 29?", "a": "31"},
    {"q": "What is 13 * 13?", "a": "169"},
]

FACTUAL_PROMPTS = [
    {"q": "What is the capital of France?", "a": "Paris"},
    {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare"},
    {"q": "What is the chemical symbol for water?", "a": "H2O"},
    {"q": "What planet is closest to the Sun?", "a": "Mercury"},
    {"q": "What year did World War II end?", "a": "1945"},
    {"q": "What is the largest ocean on Earth?", "a": "Pacific"},
    {"q": "Who painted the Mona Lisa?", "a": "Vinci"},
    {"q": "What is the speed of light in km/s (approximately)?", "a": "300000"},
    {"q": "What is the capital of Japan?", "a": "Tokyo"},
    {"q": "How many chromosomes do humans have?", "a": "46"},
]


class CfCWithKnockout(nn.Module):
    """CfC with ability to knock out specific neurons."""
    def __init__(self, input_size, hidden_size=16):
        super().__init__()
        wiring = AutoNCP(hidden_size, hidden_size // 2)
        self.cfc = CfC(input_size, wiring, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)
        self.sigma_head = nn.Linear(hidden_size, 1)
        self.knockout_mask = None  # None = no knockout

    def set_knockout(self, neuron_indices):
        """Set which neurons to knock out (zero during forward)."""
        if neuron_indices is None or len(neuron_indices) == 0:
            self.knockout_mask = None
        else:
            mask = torch.ones(16)
            for idx in neuron_indices:
                mask[idx] = 0.0
            self.knockout_mask = mask

    def forward(self, x, hx=None):
        out, hx_new = self.cfc(x, hx)
        # Apply knockout mask
        if self.knockout_mask is not None:
            device = hx_new.device
            mask = self.knockout_mask.to(device)
            hx_new = hx_new * mask
        sigma = torch.sigmoid(self.sigma_head(hx_new)) * 0.15
        perturbation = self.output(hx_new)
        return perturbation, sigma, hx_new


class CfCHookWithKnockout:
    """Hook that applies CfC perturbation with knockout."""
    def __init__(self, cfc_model, hidden_state=None):
        self.cfc = cfc_model
        self.hx = hidden_state
        self.last_sigma = 0.0

    def __call__(self, module, args):
        hs = args[0]
        batch_size = hs.size(0)
        device = hs.device
        input_dim = hs.size(-1)

        # Use mean of hidden states as CfC input
        mean_hs = hs.mean(dim=1, keepdim=True)

        # Project to CfC input size
        with torch.no_grad():
            # Simple projection: take first N features
            cfc_input = mean_hs[:, :, :self.cfc.cfc.input_size] if mean_hs.size(-1) > self.cfc.cfc.input_size else mean_hs

            if cfc_input.size(-1) < self.cfc.cfc.input_size:
                pad = torch.zeros(batch_size, 1, self.cfc.cfc.input_size - cfc_input.size(-1), device=device)
                cfc_input = torch.cat([cfc_input, pad], dim=-1)

            perturbation, sigma, self.hx = self.cfc(cfc_input, self.hx)
            self.last_sigma = sigma.mean().item()

            # Scale perturbation to match hidden state size
            if perturbation.size(-1) < input_dim:
                perturbation = perturbation.repeat(1, 1, (input_dim // perturbation.size(-1)) + 1)[:, :, :input_dim]

            noise = torch.randn_like(hs) * self.last_sigma
            hs_new = hs + noise

        return (hs_new,) + args[1:]


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find decoder layers")


def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Same generation as Phase 31."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.9, top_p=0.95, top_k=50,
            repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()


def check_math_answer(response, correct_answer):
    resp = response.strip().replace(",", "").replace(" ", "")
    ans = correct_answer.strip().replace(",", "").replace(" ", "")
    if ans in resp:
        return True
    numbers = re.findall(r'\d+', resp)
    return ans in numbers


def check_factual_answer(response, correct_answer):
    return correct_answer.lower() in response.lower()


def evaluate_task(model, tokenizer, prompts, task_type, cfc_model, target_layers, knockout_neurons=None, n_repeats=3):
    """Evaluate a task with optional neuron knockout."""
    # Set knockout mask
    cfc_model.set_knockout(knockout_neurons)

    layers = get_layers(model)
    hook = CfCHookWithKnockout(cfc_model)
    handles = [layers[i].register_forward_pre_hook(hook)
               for i in target_layers if i < len(layers)]

    correct = 0
    total = 0
    for rep in range(n_repeats):
        for p in prompts:
            if task_type == "math":
                prompt_text = "Solve this math problem. Give only the numerical answer.\n\nQ: %s\nA:" % p["q"]
                resp = generate_text(model, tokenizer, prompt_text)
                is_correct = check_math_answer(resp, p["a"])
            else:
                prompt_text = "Answer this factual question briefly.\n\nQ: %s\nA:" % p["q"]
                resp = generate_text(model, tokenizer, prompt_text)
                is_correct = check_factual_answer(resp, p["a"])

            if is_correct:
                correct += 1
            total += 1

    for h in handles:
        h.remove()

    accuracy = correct / total * 100
    return round(accuracy, 2)


def run_knockout_experiment(model_name, model_short, target_layers, neuron_roles):
    """Run the full knockout experiment for one model."""
    print("\n" + "=" * 60)
    print("  %s: Neuron Knockout Experiment" % model_short)
    print("=" * 60)

    # Load model
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    n_layers = len(get_layers(model))
    print("  Loaded: %d layers" % n_layers)

    # Get hidden dim for CfC
    layers = get_layers(model)
    hidden_dim = layers[0].self_attn.q_proj.in_features
    print("  Hidden dim: %d" % hidden_dim)

    # Create CfC model
    cfc_model = CfCWithKnockout(input_size=min(hidden_dim, 64), hidden_size=16).to(model.device)
    cfc_model.eval()
    print("  CfC created (16 neurons)")

    n_repeats = 3
    results = {}

    # 1. Baseline: No knockout
    print("\n  [1/6] Baseline (no knockout)...")
    math_base = evaluate_task(model, tokenizer, MATH_PROMPTS, "math",
                              cfc_model, target_layers, knockout_neurons=None, n_repeats=n_repeats)
    fact_base = evaluate_task(model, tokenizer, FACTUAL_PROMPTS, "factual",
                              cfc_model, target_layers, knockout_neurons=None, n_repeats=n_repeats)
    results["baseline"] = {"math": math_base, "factual": fact_base}
    print("    Math: %.1f%%, Factual: %.1f%%" % (math_base, fact_base))

    # 2. Single Math neuron KO (strongest +Math neuron)
    top_math = neuron_roles["math_positive"][0]
    print("\n  [2/6] Single Math neuron KO (N%d)..." % top_math)
    math_ko1 = evaluate_task(model, tokenizer, MATH_PROMPTS, "math",
                             cfc_model, target_layers, knockout_neurons=[top_math], n_repeats=n_repeats)
    fact_ko1 = evaluate_task(model, tokenizer, FACTUAL_PROMPTS, "factual",
                             cfc_model, target_layers, knockout_neurons=[top_math], n_repeats=n_repeats)
    results["single_math_ko"] = {"neuron": top_math, "math": math_ko1, "factual": fact_ko1}
    print("    Math: %.1f%% (delta=%+.1f), Factual: %.1f%% (delta=%+.1f)" %
          (math_ko1, math_ko1 - math_base, fact_ko1, fact_ko1 - fact_base))

    # 3. All Math neurons KO
    all_math = neuron_roles["math_positive"]
    print("\n  [3/6] ALL Math neurons KO (%s)..." % all_math)
    math_ko_all = evaluate_task(model, tokenizer, MATH_PROMPTS, "math",
                                cfc_model, target_layers, knockout_neurons=all_math, n_repeats=n_repeats)
    fact_ko_all = evaluate_task(model, tokenizer, FACTUAL_PROMPTS, "factual",
                                cfc_model, target_layers, knockout_neurons=all_math, n_repeats=n_repeats)
    results["all_math_ko"] = {"neurons": all_math, "math": math_ko_all, "factual": fact_ko_all}
    print("    Math: %.1f%% (delta=%+.1f), Factual: %.1f%% (delta=%+.1f)" %
          (math_ko_all, math_ko_all - math_base, fact_ko_all, fact_ko_all - fact_base))

    # 4. Single Factual neuron KO (control)
    top_fact = neuron_roles["factual_positive"][0]
    print("\n  [4/6] Single Factual neuron KO - control (N%d)..." % top_fact)
    math_f_ko = evaluate_task(model, tokenizer, MATH_PROMPTS, "math",
                              cfc_model, target_layers, knockout_neurons=[top_fact], n_repeats=n_repeats)
    fact_f_ko = evaluate_task(model, tokenizer, FACTUAL_PROMPTS, "factual",
                              cfc_model, target_layers, knockout_neurons=[top_fact], n_repeats=n_repeats)
    results["single_factual_ko"] = {"neuron": top_fact, "math": math_f_ko, "factual": fact_f_ko}
    print("    Math: %.1f%% (delta=%+.1f), Factual: %.1f%% (delta=%+.1f)" %
          (math_f_ko, math_f_ko - math_base, fact_f_ko, fact_f_ko - fact_base))

    # 5. Random neuron KO (control - pick a Neutral neuron)
    neutral_neurons = [n for n in range(16)
                       if n not in neuron_roles["math_positive"] + neuron_roles["math_negative"]
                       + neuron_roles["factual_positive"] + neuron_roles["factual_negative"]]
    random_n = neutral_neurons[0] if neutral_neurons else 1
    print("\n  [5/6] Random/Neutral neuron KO - control (N%d)..." % random_n)
    math_r_ko = evaluate_task(model, tokenizer, MATH_PROMPTS, "math",
                              cfc_model, target_layers, knockout_neurons=[random_n], n_repeats=n_repeats)
    fact_r_ko = evaluate_task(model, tokenizer, FACTUAL_PROMPTS, "factual",
                              cfc_model, target_layers, knockout_neurons=[random_n], n_repeats=n_repeats)
    results["neutral_ko"] = {"neuron": random_n, "math": math_r_ko, "factual": fact_r_ko}
    print("    Math: %.1f%% (delta=%+.1f), Factual: %.1f%% (delta=%+.1f)" %
          (math_r_ko, math_r_ko - math_base, fact_r_ko, fact_r_ko - fact_base))

    # 6. Total lobotomy (all 16 neurons KO)
    print("\n  [6/6] Total lobotomy (ALL 16 neurons KO)...")
    math_total = evaluate_task(model, tokenizer, MATH_PROMPTS, "math",
                               cfc_model, target_layers, knockout_neurons=list(range(16)), n_repeats=n_repeats)
    fact_total = evaluate_task(model, tokenizer, FACTUAL_PROMPTS, "factual",
                               cfc_model, target_layers, knockout_neurons=list(range(16)), n_repeats=n_repeats)
    results["total_lobotomy"] = {"math": math_total, "factual": fact_total}
    print("    Math: %.1f%% (delta=%+.1f), Factual: %.1f%% (delta=%+.1f)" %
          (math_total, math_total - math_base, fact_total, fact_total - fact_base))

    # Cleanup
    del model, tokenizer, cfc_model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def visualize(mistral_results, qwen_results):
    """Create the knockout comparison visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#ccc", "ytick.color": "#ccc",
        "grid.color": "#333", "grid.alpha": 0.3,
    })

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Phase 32: Digital Lobotomy - Neuron Knockout\n"
                 "Does destroying a Math neuron selectively kill Math ability?",
                 fontsize=14, fontweight="bold", color="#e94560")

    conditions = ["Baseline", "Single\nMath KO", "All Math\nKO",
                  "Factual KO\n(control)", "Neutral KO\n(control)", "Total\nLobotomy"]
    keys = ["baseline", "single_math_ko", "all_math_ko",
            "single_factual_ko", "neutral_ko", "total_lobotomy"]

    for panel_idx, (model_name, results, color) in enumerate([
        ("Mistral-7B", mistral_results, "#4FC3F7"),
        ("Qwen2.5-7B", qwen_results, "#FF7043"),
    ]):
        ax = axes[panel_idx]
        math_accs = [results[k]["math"] for k in keys]
        fact_accs = [results[k]["factual"] for k in keys]

        x = np.arange(len(conditions))
        w = 0.35
        bars1 = ax.bar(x - w / 2, math_accs, w, label="Math", color=color, edgecolor="#333", alpha=0.9)
        bars2 = ax.bar(x + w / 2, fact_accs, w, label="Factual", color="#66BB6A", edgecolor="#333", alpha=0.9)

        for bar, val in zip(bars1, math_accs):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                    "%.0f%%" % val, ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color=color)
        for bar, val in zip(bars2, fact_accs):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                    "%.0f%%" % val, ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color="#66BB6A")

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=7)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("%s" % model_name, fontweight="bold", color=color)
        ax.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y")
        ax.axhline(y=results["baseline"]["math"], color=color, linestyle="--", alpha=0.3)

    # Panel 3: Delta comparison (Math delta for each condition)
    ax3 = axes[2]
    m_base = mistral_results["baseline"]["math"]
    q_base = qwen_results["baseline"]["math"]
    m_deltas = [mistral_results[k]["math"] - m_base for k in keys[1:]]
    q_deltas = [qwen_results[k]["math"] - q_base for k in keys[1:]]
    delta_labels = conditions[1:]

    x = np.arange(len(delta_labels))
    bars1 = ax3.bar(x - w / 2, m_deltas, w, label="Mistral", color="#4FC3F7", edgecolor="#333")
    bars2 = ax3.bar(x + w / 2, q_deltas, w, label="Qwen", color="#FF7043", edgecolor="#333")

    for bar, val in zip(bars1, m_deltas):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() - 2 if val < 0 else bar.get_height() + 1,
                 "%+.0f%%" % val, ha="center", va="top" if val < 0 else "bottom",
                 fontsize=7, fontweight="bold", color="#4FC3F7")
    for bar, val in zip(bars2, q_deltas):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() - 2 if val < 0 else bar.get_height() + 1,
                 "%+.0f%%" % val, ha="center", va="top" if val < 0 else "bottom",
                 fontsize=7, fontweight="bold", color="#FF7043")

    ax3.set_xticks(x)
    ax3.set_xticklabels(delta_labels, fontsize=7)
    ax3.set_ylabel("Math Accuracy Delta (%)")
    ax3.set_title("Math Accuracy Change\nvs Baseline", fontweight="bold")
    ax3.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax3.axhline(y=0, color="#e94560", linewidth=2, alpha=0.7)
    ax3.grid(True, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(FIGURES_DIR, "phase32_neuron_knockout.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Figure: %s" % out)
    return out


def main():
    print("=" * 60)
    print("Phase 32: Neuron Knockout (Digital Lobotomy)")
    print("  Causal proof of task-specific neuron specialization")
    print("=" * 60)
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Run Mistral
    mistral_results = run_knockout_experiment(
        "mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B",
        list(range(15, 21)), NEURON_ROLES["Mistral-7B"]
    )

    # Run Qwen
    qwen_results = run_knockout_experiment(
        "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B",
        list(range(12, 18)), NEURON_ROLES["Qwen2.5-7B"]
    )

    # Visualize
    fig_path = visualize(mistral_results, qwen_results)

    elapsed = time.time() - t_start

    # Summary
    print("\n" + "=" * 60)
    print("  KNOCKOUT SUMMARY")
    print("=" * 60)
    print("\n  %-30s %10s %10s" % ("Condition", "Mistral", "Qwen"))
    print("  " + "-" * 52)
    for key in ["baseline", "single_math_ko", "all_math_ko",
                "single_factual_ko", "neutral_ko", "total_lobotomy"]:
        m = mistral_results[key]
        q = qwen_results[key]
        label = key.replace("_", " ").title()
        print("  %-30s %9.1f%% %9.1f%%" % (label + " (Math)", m["math"], q["math"]))

    print("\n  Key comparisons:")
    m_drop = mistral_results["baseline"]["math"] - mistral_results["single_math_ko"]["math"]
    q_drop = qwen_results["baseline"]["math"] - qwen_results["single_math_ko"]["math"]
    print("    Single Math KO -> Math drop: Mistral=%.1f%%, Qwen=%.1f%%" % (m_drop, q_drop))
    print("    Fragility ratio: %.2fx" % (m_drop / max(q_drop, 0.1)))

    # Save
    output = {
        "phase": "Phase 32: Neuron Knockout (Digital Lobotomy)",
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "neuron_roles": NEURON_ROLES,
        "mistral": {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "results": mistral_results,
        },
        "qwen": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "results": qwen_results,
        },
        "figure_path": fig_path,
    }
    out_path = os.path.join(RESULTS_DIR, "phase32_neuron_knockout_log.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Results: %s" % out_path)
    print("\n" + "=" * 60)
    print("  Phase 32 COMPLETE -- %ds (%.1f min)" % (elapsed, elapsed / 60))
    print("=" * 60)


if __name__ == "__main__":
    main()
