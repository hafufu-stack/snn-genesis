"""
Phase 106: Continual Learning under Conflicting Priors
======================================================

Deep Think Proposal #4: Does noise injection help when the model
must override strong arithmetic priors (base-10 → base-8)?

Hypothesis: Flash Annealing helps the model "let go" of base-10
priors and reason in octal, similar to how it helps override
standard Hanoi rules.

Conditions (N=30 each, Qwen2.5-0.5B):
  1. no_noise:     Vanilla inference
  2. flash_noise:  Flash Annealing σ=0.30
  3. aha_flash:    Aha!+Flash (0.5B diff vector)

Task: 20 octal arithmetic problems per trial, 30 trials per condition.

Qwen2.5-0.5B-Instruct, middle layer
Total: 3 x 30 x 20 = 1800 problems
"""

import torch
import torch.nn as nn
import os, json, gc, time, random, re, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# === Config ===
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_SHORT = "Qwen2.5-0.5B"
SEED = 2026
N_PROBLEMS_PER_TRIAL = 20
N_TRIALS = 30
STRESS_SIGMA = 0.30

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================
#  OCTAL ARITHMETIC TASK
# ===================================================

def generate_octal_problems(n=20):
    """Generate octal arithmetic problems with ground truth."""
    problems = []
    for _ in range(n):
        # Generate numbers that are valid in octal (digits 0-7)
        a = random.randint(1, 77)  # in decimal, but we'll convert to octal
        b = random.randint(1, 37)
        op = random.choice(['+', '-', '*'])

        a_oct = oct(a)[2:]  # remove '0o' prefix
        b_oct = oct(b)[2:]

        if op == '+':
            result_dec = a + b
        elif op == '-':
            result_dec = a - b
        else:
            result_dec = a * b

        result_oct = oct(result_dec)[2:] if result_dec >= 0 else '-' + oct(abs(result_dec))[2:]

        problems.append({
            "expression": f"{a_oct} {op} {b_oct}",
            "answer_octal": result_oct,
            "answer_decimal": result_dec,
            "a_dec": a, "b_dec": b, "op": op,
        })
    return problems


def check_octal_answer(response, correct_octal):
    """Check if the response contains the correct octal answer."""
    # Extract numbers from response
    numbers = re.findall(r'-?[0-7]+', response)
    for num in numbers:
        if num == correct_octal:
            return True
    return False


# ===================================================
#  MODEL + GENERATION
# ===================================================

def load_model():
    print(f"\n Loading {MODEL_NAME}...")
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True,
                                         trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
        trust_remote_code=True, local_files_only=True)
    model.eval()
    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"  Done: {n_layers} layers, hidden_dim={hidden_dim}")
    return model, tok, n_layers, hidden_dim

def generate_response(model, tok, prompt, temperature=0.3, max_tokens=100):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperature, top_p=0.9,
            repetition_penalty=1.2, pad_token_id=tok.pad_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    inp = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(inp):].strip()


# ===================================================
#  AHA! + FLASH HOOK
# ===================================================

class AhaFlashHook:
    def __init__(self):
        self.active = False
        self.sigma = 0.15
        self.mode = "baseline"
        self.diff_unit = None
        self.handle = None

    def setup_off(self):
        self.mode = "baseline"
        self.active = False

    def setup_flash(self, sigma):
        self.mode = "flash"
        self.sigma = sigma
        self.active = True

    def setup_aha_flash(self, diff_unit_vec, sigma, device='cuda'):
        self.mode = "aha_flash"
        du = torch.tensor(diff_unit_vec, dtype=torch.float16, device=device)
        du_norm = du.norm()
        if du_norm > 1e-8:
            du = du / du_norm
        self.diff_unit = du
        self.sigma = sigma
        self.active = True

    def register(self, model, layer_idx):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            if hook_obj.mode == "baseline":
                return args
            hs = args[0]
            d = hs.shape[-1]

            if hook_obj.mode == "flash":
                noise = torch.randn_like(hs) * hook_obj.sigma
                return (hs + noise,) + args[1:]

            elif hook_obj.mode == "aha_flash":
                offset = hook_obj.diff_unit
                det_scale = hook_obj.sigma * math.sqrt(d) * 0.5
                det_noise = offset * det_scale
                if hs.dim() == 3:
                    det_noise = det_noise.unsqueeze(0).unsqueeze(0).expand_as(hs)
                else:
                    det_noise = det_noise.unsqueeze(0).expand_as(hs)
                stoch_noise = torch.randn_like(hs) * (hook_obj.sigma * 0.5)
                return (hs + det_noise + stoch_noise,) + args[1:]

            return args
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  EVALUATION
# ===================================================

def build_octal_prompt(tokenizer, problem):
    content = (
        "You are computing in OCTAL (base-8). This means:\n"
        "- Only digits 0-7 are valid\n"
        "- 7 + 1 = 10 (not 8)\n"
        "- 7 + 3 = 12 (not 10)\n"
        "- 3 * 3 = 11 (not 9)\n\n"
        f"Compute in OCTAL: {problem['expression']}\n\n"
        "Show your work step by step in octal, then give the final answer.\n"
        "Answer (in octal):"
    )
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_trial(model, tok, hook, problems):
    """Run one trial of N_PROBLEMS_PER_TRIAL octal problems."""
    correct = 0
    results = []
    for prob in problems:
        prompt = build_octal_prompt(tok, prob)
        resp = generate_response(model, tok, prompt)
        is_correct = check_octal_answer(resp, prob["answer_octal"])
        if is_correct:
            correct += 1
        results.append({
            "expression": prob["expression"],
            "correct_octal": prob["answer_octal"],
            "is_correct": is_correct,
            "response_snippet": resp[:80],
        })
    return correct / len(problems), results


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(all_results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Phase 106: Conflicting Priors — Octal Arithmetic\n"
                 "Does noise help override base-10 arithmetic priors?",
                 fontsize=12, fontweight="bold")

    # Panel 1: Average accuracy per condition
    ax = axes[0]
    conds = all_results["conditions"]
    names = [c["condition"] for c in conds]
    accs = [c["avg_accuracy"] * 100 for c in conds]
    colors = ["#9E9E9E", "#2196F3", "#9C27B0"]
    bars = ax.bar(range(len(conds)), accs, color=colors[:len(conds)], alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=10)
    ax.set_ylabel("Octal Accuracy (%)", fontsize=11)
    ax.set_title(f"Average Accuracy (N={N_TRIALS} trials)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2: Distribution of per-trial accuracy
    ax = axes[1]
    for i, c in enumerate(conds):
        trial_accs = [t * 100 for t in c["trial_accuracies"]]
        positions = [i + (random.random() - 0.5) * 0.3 for _ in trial_accs]
        ax.scatter(positions, trial_accs, alpha=0.4, s=20, color=colors[i])
        ax.plot([i - 0.2, i + 0.2], [np.mean(trial_accs)] * 2,
                color=colors[i], linewidth=3)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=10)
    ax.set_ylabel("Per-Trial Accuracy (%)", fontsize=11)
    ax.set_title("Distribution of Trial Accuracies", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase106_conflicting_priors.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*80}")
    print(f"  Phase 106: Continual Learning under Conflicting Priors")
    print(f"  3 conditions x {N_TRIALS} trials x {N_PROBLEMS_PER_TRIAL} problems")
    print(f"{'='*80}")

    t0 = time.time()

    model, tok, n_layers, hidden_dim = load_model()
    device = next(model.parameters()).device
    layer_idx = n_layers // 2
    print(f"  Using layer {layer_idx}, hidden_dim={hidden_dim}")

    # Load 0.5B diff vector
    diff_pca_path = os.path.join(RESULTS_DIR, "phase101_05b_diff_pca.npz")
    diff_unit = None
    if os.path.exists(diff_pca_path):
        data = np.load(diff_pca_path)
        diff_unit = data["diff_unit"]
        print(f"  Loaded 0.5B diff_unit: norm={np.linalg.norm(diff_unit):.4f}")
    else:
        print(f"  WARNING: Phase 101 diff vector not found, aha_flash will use random")
        diff_unit = np.random.randn(hidden_dim).astype(np.float32)
        diff_unit /= np.linalg.norm(diff_unit)

    hook = AhaFlashHook()
    hook.register(model, layer_idx)

    conditions_config = [
        {"name": "no_noise",    "setup": "off"},
        {"name": "flash_noise", "setup": "flash"},
        {"name": "aha_flash",   "setup": "aha_flash"},
    ]

    all_results = {
        "experiment": "Phase 106: Conflicting Priors (Octal)",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "n_layers": n_layers,
        "layer": layer_idx,
        "hidden_dim": hidden_dim,
        "sigma": STRESS_SIGMA,
        "n_trials": N_TRIALS,
        "n_problems_per_trial": N_PROBLEMS_PER_TRIAL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": [],
    }

    results_path = os.path.join(RESULTS_DIR, "phase106_log.json")

    for cfg in conditions_config:
        print(f"\n  === Condition: {cfg['name']} ===")
        if cfg["setup"] == "off":
            hook.setup_off()
        elif cfg["setup"] == "flash":
            hook.setup_flash(STRESS_SIGMA)
        elif cfg["setup"] == "aha_flash":
            hook.setup_aha_flash(diff_unit, STRESS_SIGMA, device)

        trial_accs = []
        all_trial_results = []
        for trial in range(N_TRIALS):
            problems = generate_octal_problems(N_PROBLEMS_PER_TRIAL)
            acc, trial_results = run_trial(model, tok, hook, problems)
            trial_accs.append(acc)
            all_trial_results.append(trial_results)
            if (trial + 1) % 10 == 0:
                avg = np.mean(trial_accs) * 100
                elapsed = time.time() - t0
                print(f"    [{trial+1}/{N_TRIALS}] Avg accuracy: {avg:.1f}% | {elapsed/60:.1f}min")

        cond_result = {
            "condition": cfg["name"],
            "avg_accuracy": round(float(np.mean(trial_accs)), 4),
            "std_accuracy": round(float(np.std(trial_accs)), 4),
            "trial_accuracies": [round(a, 4) for a in trial_accs],
            "n_trials": N_TRIALS,
        }
        all_results["conditions"].append(cond_result)
        print(f"    Final: {cond_result['avg_accuracy']*100:.1f}% ± {cond_result['std_accuracy']*100:.1f}%")

        # Intermediate save
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    hook.remove()

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    bl_acc = all_results["conditions"][0]["avg_accuracy"]
    for c in all_results["conditions"]:
        delta = c["avg_accuracy"] - bl_acc
        print(f"    {c['condition']:20s}: {c['avg_accuracy']*100:5.1f}% ± {c['std_accuracy']*100:.1f}% (Δ={delta*100:+.1f}pp)")

    # Mann-Whitney U test between conditions
    from scipy.stats import mannwhitneyu
    bl_accs = all_results["conditions"][0]["trial_accuracies"]
    stats_tests = {}
    for c in all_results["conditions"][1:]:
        c_accs = c["trial_accuracies"]
        stat, p_val = mannwhitneyu(c_accs, bl_accs, alternative='two-sided')
        stats_tests[f"{c['condition']}_vs_baseline"] = round(p_val, 6)
        print(f"  Mann-Whitney U ({c['condition']} vs baseline): p={p_val:.6f}")

    all_results["statistical_tests"] = stats_tests

    # Verdict
    best = max(all_results["conditions"][1:], key=lambda c: c["avg_accuracy"])
    if best["avg_accuracy"] > bl_acc + 0.05:
        verdict = "NOISE_OVERCOMES_PRIOR"
        print(f"\n  VERDICT: {verdict} — Noise helps override base-10 priors!")
    elif best["avg_accuracy"] > bl_acc + 0.02:
        verdict = "MARGINAL_PRIOR_OVERRIDE"
        print(f"\n  VERDICT: {verdict} — Small benefit in prior override")
    else:
        verdict = "PRIOR_TOO_STRONG"
        print(f"\n  VERDICT: {verdict} — Base-10 priors too strong for noise alone")

    all_results["verdict"] = verdict

    fig_path = visualize(all_results)
    all_results["figure"] = fig_path

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed, 1)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  Total elapsed: {elapsed/60:.1f} min")

    return all_results, elapsed


if __name__ == "__main__":
    main()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n Phase 106 complete.")
