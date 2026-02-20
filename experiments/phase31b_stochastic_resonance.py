"""
Phase 31b: Stochastic Resonance Verification (n=100)
=====================================================
Uses EXACT same conditions as Phase 31:
  - Structured noise (0.7 * pure + 0.3 * low_freq)
  - do_sample=True, temperature=0.9, top_p=0.95, top_k=50
  - Raw prompt (no chat template)
  - Same 20 math prompts, repeated 5x with different noise seeds

Tests Qwen2.5-7B only, 7 sigma levels.

Usage:
    python experiments/phase31b_stochastic_resonance.py
"""

import torch, torch.nn as nn
import os, sys, json, gc, time, datetime, random, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

SEED = 2026
MAX_NEW_TOKENS = 50

# EXACT same 20 prompts as Phase 31
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

# 7 sigma levels for finer resolution
SIGMA_LEVELS = [0.000, 0.015, 0.030, 0.050, 0.071, 0.100, 0.150]
N_REPEATS = 5  # Run each prompt 5 times = 100 evaluations per sigma


class StaticSNNHook:
    """EXACT same hook as Phase 31 -- structured noise."""
    def __init__(self, sigma=0.071):
        self.sigma = sigma
    def __call__(self, module, args):
        if self.sigma <= 0:
            return args
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
    """EXACT same generation as Phase 31 -- do_sample=True, temperature=0.9."""
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
    """EXACT same checker as Phase 31."""
    resp = response.strip().replace(",", "").replace(" ", "")
    ans = correct_answer.strip().replace(",", "").replace(" ", "")
    if ans in resp:
        return True
    numbers = re.findall(r'\d+', resp)
    return ans in numbers


def evaluate_at_sigma(model, tokenizer, target_layers, sigma, n_repeats):
    """Evaluate math accuracy at a given sigma, repeating n_repeats times."""
    print("\n  --- sigma = %.3f ---" % sigma)

    layers = get_layers(model)
    hook = StaticSNNHook(sigma=sigma)
    handles = [layers[i].register_forward_pre_hook(hook)
               for i in target_layers if i < len(layers)]

    correct = 0
    total = 0
    all_details = []

    for rep in range(n_repeats):
        rep_correct = 0
        for mp in MATH_PROMPTS:
            prompt = "Solve this math problem. Give only the numerical answer.\n\nQ: %s\nA:" % mp["q"]
            resp = generate_text(model, tokenizer, prompt)
            is_correct = check_math_answer(resp, mp["a"])
            if is_correct:
                correct += 1
                rep_correct += 1
            total += 1
            all_details.append({
                "repeat": rep,
                "question": mp["q"],
                "expected": mp["a"],
                "response": resp[:100],
                "correct": bool(is_correct),
            })
        print("    Rep %d: %d/%d = %.1f%%" % (rep + 1, rep_correct, len(MATH_PROMPTS),
                                                rep_correct / len(MATH_PROMPTS) * 100))

    for h in handles:
        h.remove()

    accuracy = correct / total * 100
    print("    TOTAL: %d/%d = %.1f%%" % (correct, total, accuracy))
    return {
        "sigma": sigma,
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "per_repeat": [
            sum(1 for d in all_details if d["repeat"] == r and d["correct"])
            for r in range(n_repeats)
        ],
        "details": all_details,
    }


def bootstrap_ci(correct_arr, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    np.random.seed(SEED)
    n = len(correct_arr)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(correct_arr, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return round(lower * 100, 2), round(upper * 100, 2)


def visualize(results):
    """Create the stochastic resonance visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#ccc", "ytick.color": "#ccc",
        "grid.color": "#333", "grid.alpha": 0.3,
    })

    sigmas = [r["sigma"] for r in results]
    accs = [r["accuracy"] for r in results]
    cis_low = [r["ci_lower"] for r in results]
    cis_high = [r["ci_upper"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Phase 31b: Stochastic Resonance Verification\n"
                 "Qwen2.5-7B Math Accuracy vs Static Noise (n=100 per sigma)",
                 fontsize=14, fontweight="bold", color="#e94560")

    # Panel 1: Accuracy curve with CI
    ax1 = axes[0]
    ax1.fill_between(sigmas, cis_low, cis_high, alpha=0.3, color="#4FC3F7", label="95% CI")
    ax1.plot(sigmas, accs, "o-", color="#4FC3F7", linewidth=2.5, markersize=10,
             markeredgecolor="white", markeredgewidth=1.5, label="Math Accuracy", zorder=5)

    # Mark the homeostatic point
    homeo_idx = sigmas.index(0.071)
    ax1.plot(0.071, accs[homeo_idx], "s", color="#e94560", markersize=15,
             markeredgecolor="white", markeredgewidth=2, zorder=10,
             label="Homeostatic point")

    # Baseline marker
    ax1.axhline(y=accs[0], color="#FFA726", linestyle="--", alpha=0.7,
                label="Baseline (sigma=0): %.1f%%" % accs[0])
    ax1.axvline(x=0.071, color="#e94560", linestyle=":", alpha=0.5)

    for i, (s, a) in enumerate(zip(sigmas, accs)):
        ax1.annotate("%.1f%%" % a, (s, a), textcoords="offset points",
                     xytext=(0, 15), ha="center", fontsize=10, fontweight="bold",
                     color="white")

    ax1.set_xlabel("Static Noise Level (sigma)", fontsize=12)
    ax1.set_ylabel("Math Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy vs Noise Curve", fontweight="bold")
    ax1.legend(fontsize=9, facecolor="#16213e", edgecolor="#555", loc="best")
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0, 100)

    # Panel 2: Verdict + stats
    ax2 = axes[1]
    ax2.axis("off")

    baseline_acc = accs[0]
    peak_acc = max(accs)
    peak_sigma = sigmas[accs.index(peak_acc)]

    if peak_acc > baseline_acc + 5 and peak_sigma > 0:
        verdict = "STOCHASTIC RESONANCE\nCONFIRMED!"
        verdict_color = "#00D4AA"
        desc = ("Noise at sigma=%.3f IMPROVES accuracy\n"
                "by +%.1f%% over baseline.\n\n"
                "First demonstration of\n"
                "Stochastic Resonance in LLM!" % (peak_sigma, peak_acc - baseline_acc))
    elif peak_acc >= baseline_acc - 2 and peak_sigma > 0 and abs(peak_acc - baseline_acc) <= 5:
        verdict = "PLATEAU RESILIENCE"
        verdict_color = "#FFA726"
        desc = ("Noise does not degrade performance.\n"
                "Qwen maintains %.1f%% even at sigma=%.3f.\n\n"
                "Architecture is inherently\n"
                "noise-tolerant." % (accs[homeo_idx], 0.071))
    else:
        verdict = "MONOTONIC DECAY"
        verdict_color = "#EF5350"
        desc = ("Noise degrades accuracy.\n"
                "Baseline (%.1f%%) remains best.\n\n"
                "No Stochastic Resonance." % baseline_acc)

    ax2.text(0.5, 0.85, verdict, transform=ax2.transAxes,
             fontsize=20, fontweight="bold", color=verdict_color,
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=verdict_color + "22",
                       edgecolor=verdict_color, linewidth=2))

    ax2.text(0.5, 0.58, desc, transform=ax2.transAxes,
             fontsize=12, ha="center", va="center", color="#eee",
             linespacing=1.5)

    # Stats box
    fisher_p = results[homeo_idx].get("fisher_p_vs_baseline")
    stats_text = ("n = %d per sigma level\n"
                  "Baseline (sigma=0): %.1f%%\n"
                  "Homeostatic (0.071): %.1f%%\n"
                  "Peak: %.1f%% at sigma=%.3f\n"
                  "Delta (homeo-base): %+.1f%%" % (
                      results[0]["total"], baseline_acc,
                      accs[homeo_idx], peak_acc, peak_sigma,
                      accs[homeo_idx] - baseline_acc))

    if fisher_p is not None:
        sig_marker = "***" if fisher_p < 0.001 else "**" if fisher_p < 0.01 else "*" if fisher_p < 0.05 else "(n.s.)"
        stats_text += "\n\nFisher exact (0 vs 0.071):\np = %.4f %s" % (fisher_p, sig_marker)

    ax2.text(0.5, 0.20, stats_text, transform=ax2.transAxes,
             fontsize=10, ha="center", va="center", color="#ccc",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#16213e",
                       edgecolor="#555", linewidth=1))

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(FIGURES_DIR, "phase31b_stochastic_resonance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Figure: %s" % out)
    return out


def main():
    print("=" * 60)
    print("Phase 31b: Stochastic Resonance Verification")
    print("  Same conditions as Phase 31 (structured noise, do_sample=True)")
    print("  20 prompts x 5 repeats = 100 evaluations per sigma")
    print("  sigma levels: %s" % SIGMA_LEVELS)
    print("=" * 60)
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Load Qwen
    print("\n  Loading Qwen2.5-7B-Instruct...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
    n_layers = len(get_layers(model))
    print("  Qwen loaded: %d layers" % n_layers)

    target_layers = list(range(12, 18))

    all_results = []
    for sigma in SIGMA_LEVELS:
        result = evaluate_at_sigma(model, tokenizer, target_layers, sigma, N_REPEATS)

        # Bootstrap CI
        correct_arr = np.array([1 if d["correct"] else 0 for d in result["details"]])
        ci_low, ci_high = bootstrap_ci(correct_arr)
        result["ci_lower"] = ci_low
        result["ci_upper"] = ci_high
        print("    95%% CI: [%.1f%%, %.1f%%]" % (ci_low, ci_high))

        all_results.append(result)

    # Fisher exact test: sigma=0 vs sigma=0.071
    from scipy.stats import fisher_exact
    baseline = all_results[0]
    homeo = next(r for r in all_results if abs(r["sigma"] - 0.071) < 0.001)
    table = [[baseline["correct"], baseline["total"] - baseline["correct"]],
             [homeo["correct"], homeo["total"] - homeo["correct"]]]
    odds, p_val = fisher_exact(table)
    homeo["fisher_p_vs_baseline"] = round(p_val, 6)
    homeo["fisher_odds_vs_baseline"] = round(odds, 4)
    print("\n  Fisher exact test (sigma=0 vs sigma=0.071):")
    print("    p = %.4f, OR = %.4f" % (p_val, odds))
    sig_str = "SIGNIFICANT (p<0.05)" if p_val < 0.05 else "NOT significant"
    print("    Result: %s" % sig_str)

    # Summary
    print("\n" + "=" * 60)
    print("  STOCHASTIC RESONANCE SUMMARY")
    print("=" * 60)
    baseline_acc = baseline["accuracy"]
    homeo_acc = homeo["accuracy"]
    peak_r = max(all_results, key=lambda x: x["accuracy"])
    print("  %-25s %.1f%%" % ("Baseline (sigma=0):", baseline_acc))
    print("  %-25s %.1f%%" % ("Homeostatic (0.071):", homeo_acc))
    print("  %-25s %.1f%% at sigma=%.3f" % ("Peak:", peak_r["accuracy"], peak_r["sigma"]))
    print("  %-25s %+.1f%%" % ("Delta (homeo - base):", homeo_acc - baseline_acc))
    print("  %-25s" % "Per-sigma breakdown:")
    for r in all_results:
        print("    sigma=%.3f: %.1f%% (%d/%d) per_rep=%s" % (
            r["sigma"], r["accuracy"], r["correct"], r["total"], r["per_repeat"]))

    if homeo_acc > baseline_acc + 5:
        print("\n  *** STOCHASTIC RESONANCE DETECTED! ***")
    elif homeo_acc >= baseline_acc - 2:
        print("\n  Plateau resilience -- noise does not degrade.")
    else:
        print("\n  No resonance. Noise degrades accuracy.")

    # Visualize
    fig_path = visualize(all_results)

    # Save (strip large details for JSON)
    for r in all_results:
        r["example_details"] = r["details"][:5]
        del r["details"]

    elapsed = time.time() - t_start
    output = {
        "phase": "Phase 31b: Stochastic Resonance Verification",
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "model": "Qwen2.5-7B-Instruct",
        "methodology": {
            "n_prompts": len(MATH_PROMPTS),
            "n_repeats": N_REPEATS,
            "total_per_sigma": len(MATH_PROMPTS) * N_REPEATS,
            "noise_type": "structured (0.7*pure + 0.3*low_freq)",
            "generation": "do_sample=True, temperature=0.9, top_p=0.95",
            "prompt_format": "raw (no chat template)",
        },
        "sigma_levels": SIGMA_LEVELS,
        "results": all_results,
        "fisher_test": {
            "comparison": "sigma=0 vs sigma=0.071",
            "p_value": round(p_val, 6),
            "odds_ratio": round(odds, 4),
            "significant": p_val < 0.05,
        },
        "verdict": {
            "baseline_acc": baseline_acc,
            "homeostatic_acc": homeo_acc,
            "peak_acc": peak_r["accuracy"],
            "peak_sigma": peak_r["sigma"],
            "delta": round(homeo_acc - baseline_acc, 2),
            "stochastic_resonance": homeo_acc > baseline_acc + 5,
        },
        "figure_path": fig_path,
    }
    out_path = os.path.join(RESULTS_DIR, "phase31b_stochastic_resonance_log.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Results: %s" % out_path)
    print("\n" + "=" * 60)
    print("  Phase 31b COMPLETE -- %ds (%.1f min)" % (elapsed, elapsed / 60))
    print("=" * 60)


if __name__ == "__main__":
    main()
