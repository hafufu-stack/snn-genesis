"""
Phase 2: LLM Hidden State Noise Injection — SNN vs Random
==========================================================

Compares the effect of SNN chaotic noise vs standard torch.randn()
noise on LLM hidden states, measuring:

1. Canary head entropy response (how much does the internal alarm trigger?)
2. Output diversity (do different noise sources produce different hallucination patterns?)
3. Noise-induced hallucination rate

This is the Genesis version of Electric Dreams (v10), replacing numpy/torch
random noise with SNN-generated chaotic noise.

Key insight: SNN noise has near-zero autocorrelation and true chaotic dynamics,
which may perturb the LLM's computational graph differently than simple Gaussian noise.

Requirements: GPU with ~16GB VRAM (Mistral-7B in float16)
"""

import torch
import numpy as np
import json
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.snn_reservoir import ChaoticReservoir

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
CANARY_LAYER = 10
CANARY_HEAD = 17
NOISE_SIGMAS = [0.05, 0.10, 0.15, 0.20]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

QUESTIONS = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water in Celsius?",
    "What planet is closest to the Sun?",
    "What is the largest mammal?",
    "How many continents are there?",
    "What color is the sky on a clear day?",
    "What gas do plants absorb?",
    "What is the speed of light in km/s?",
]

EXPECTED_ANSWERS = [
    "Paris", "4", "Shakespeare", "100", "Mercury",
    "blue whale", "7", "blue", "carbon dioxide", "300000",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Noise Generators ────────────────────────────────────

class TorchNoiseGenerator:
    """Standard torch.randn() noise (baseline from Electric Dreams v10)."""
    def __init__(self):
        self.name = "torch.randn"
        self.color = "#3498db"

    def generate(self, shape, sigma, device, dtype=torch.float16):
        return torch.randn(shape, device=device, dtype=dtype) * sigma


class SNNNoiseGenerator:
    """SNN chaotic reservoir noise (Project Genesis)."""
    def __init__(self, num_neurons=300, seed=2026):
        self.name = "SNN Chaos"
        self.color = "#2ecc71"
        self.reservoir = ChaoticReservoir(
            num_neurons=num_neurons,
            temperature=1.0,
            seed=seed,
        )

    def generate(self, shape, sigma, device, dtype=torch.float16):
        # Flatten total elements needed
        total = 1
        for s in shape:
            total *= s

        noise_np = self.reservoir.generate_noise_vector(total, warmup_steps=5)
        noise = torch.from_numpy(noise_np).reshape(shape).to(device=device, dtype=dtype) * sigma
        return noise


# ── Core Functions ──────────────────────────────────────

def compute_canary_entropy(model, input_ids):
    """Compute canary head (L10H17) entropy."""
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, use_cache=False)

    attn = out.attentions[CANARY_LAYER]
    a = attn[0, CANARY_HEAD, -1, :].float()
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
    a = a.clamp(min=1e-10)
    h = -(a * torch.log2(a)).sum().item()

    if np.isnan(h) or np.isinf(h):
        h = 0.0

    del out
    return h


def run_injection_comparison(model, tokenizer, device):
    """
    Compare SNN noise vs torch.randn noise across multiple sigma levels.

    For each (noise_source, sigma, question):
    1. Measure clean entropy
    2. Inject noise and measure perturbed entropy
    3. Generate perturbed output
    4. Record results
    """
    print("\n" + "=" * 70)
    print("Phase 2: SNN vs torch.randn Noise Injection")
    print("=" * 70)

    generators = [TorchNoiseGenerator(), SNNNoiseGenerator()]
    all_results = []

    for q_idx, (question, expected) in enumerate(zip(QUESTIONS, EXPECTED_ANSWERS)):
        print(f"\n{'─'*50}")
        print(f"Q{q_idx+1}: {question}")

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(device)

        # Clean baseline
        clean_entropy = compute_canary_entropy(model, input_ids)
        with torch.no_grad():
            clean_out = model.generate(
                input_ids, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        clean_text = tokenizer.decode(clean_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        clean_correct = expected.lower() in clean_text.lower()

        print(f"  CLEAN: H={clean_entropy:.4f} | \"{clean_text[:60]}\" | correct={clean_correct}")

        for gen in generators:
            for sigma in NOISE_SIGMAS:
                # Create noise hook using this generator
                def make_hook(generator, sig):
                    def pre_hook(module, args):
                        hidden_states = args[0]
                        noise = generator.generate(
                            hidden_states.shape, sig,
                            hidden_states.device, hidden_states.dtype
                        )
                        return (hidden_states + noise,) + args[1:]
                    return pre_hook

                # Measure entropy with noise
                hook = model.model.layers[CANARY_LAYER].register_forward_pre_hook(
                    make_hook(gen, sigma)
                )
                noisy_entropy = compute_canary_entropy(model, input_ids)
                hook.remove()

                # Generate with noise
                hook = model.model.layers[CANARY_LAYER].register_forward_pre_hook(
                    make_hook(gen, sigma)
                )
                with torch.no_grad():
                    noisy_out = model.generate(
                        input_ids, max_new_tokens=30,
                        do_sample=True, temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                noisy_text = tokenizer.decode(
                    noisy_out[0][input_ids.shape[1]:], skip_special_tokens=True
                ).strip()
                hook.remove()

                noisy_correct = expected.lower() in noisy_text.lower()
                entropy_spike = noisy_entropy - clean_entropy

                safe_text = "".join(c if ord(c) < 128 else "?" for c in noisy_text[:60])
                print(f"  {gen.name:<12} σ={sigma}: H={noisy_entropy:.4f} "
                      f"(Δ={entropy_spike:+.4f}) | \"{safe_text}\" | correct={noisy_correct}")

                result = {
                    "question": question,
                    "expected": expected,
                    "noise_source": gen.name,
                    "sigma": sigma,
                    "clean_entropy": round(float(clean_entropy), 4),
                    "clean_correct": bool(clean_correct),
                    "clean_text": clean_text[:100],
                    "noisy_entropy": round(float(noisy_entropy), 4),
                    "entropy_spike": round(float(entropy_spike), 4),
                    "noisy_correct": bool(noisy_correct),
                    "noisy_text": safe_text,
                }
                all_results.append(result)

        torch.cuda.empty_cache()

    return all_results


def analyze_and_plot(results):
    """Analyze results and generate comparison visualization."""
    print("\n" + "=" * 70)
    print("Analysis: SNN Chaos vs torch.randn")
    print("=" * 70)

    # Group by noise source and sigma
    sources = ["torch.randn", "SNN Chaos"]
    sigmas = NOISE_SIGMAS

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Project Genesis — Phase 2: SNN vs torch.randn Noise Injection\n"
                 "Canary Head L10H17 Response (Mistral-7B)",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Mean Entropy Spike per sigma ──
    ax = axes[0, 0]
    for src in sources:
        means = []
        stds = []
        for sig in sigmas:
            spikes = [r["entropy_spike"] for r in results
                      if r["noise_source"] == src and r["sigma"] == sig]
            means.append(np.mean(spikes))
            stds.append(np.std(spikes))
        color = "#2ecc71" if src == "SNN Chaos" else "#3498db"
        ax.errorbar(sigmas, means, yerr=stds, marker="o", label=src,
                    color=color, capsize=4, linewidth=2)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Mean Entropy Spike (ΔH)")
    ax.set_title("Canary Alarm Strength")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Panel 2: Hallucination Rate per sigma ──
    ax = axes[0, 1]
    for src in sources:
        rates = []
        for sig in sigmas:
            subset = [r for r in results
                      if r["noise_source"] == src and r["sigma"] == sig]
            incorrect = sum(1 for r in subset if not r["noisy_correct"])
            rates.append(incorrect / len(subset) * 100 if subset else 0)
        color = "#2ecc71" if src == "SNN Chaos" else "#3498db"
        ax.bar([s + (0.015 if src == "SNN Chaos" else -0.015) for s in sigmas],
               rates, width=0.025, label=src, color=color, alpha=0.8)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_title("Noise-Induced Hallucination Rate")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # ── Panel 3: Entropy distribution comparison at σ=0.10 ──
    ax = axes[1, 0]
    for src in sources:
        ents = [r["noisy_entropy"] for r in results
                if r["noise_source"] == src and r["sigma"] == 0.10]
        color = "#2ecc71" if src == "SNN Chaos" else "#3498db"
        ax.hist(ents, bins=15, alpha=0.6, label=f"{src} (σ=0.10)",
                color=color, edgecolor="white")
    # Add clean baseline
    clean_ents = list(set([r["clean_entropy"] for r in results]))
    ax.axvline(np.mean(clean_ents), color="gray", linestyle="--",
               label=f"Clean mean ({np.mean(clean_ents):.2f})")
    ax.set_xlabel("Canary Entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title("Entropy Distribution at σ=0.10")
    ax.legend(fontsize=8)

    # ── Panel 4: Summary Table ──
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = "Summary Statistics\n" + "=" * 45 + "\n\n"
    for src in sources:
        subset = [r for r in results if r["noise_source"] == src]
        mean_spike = np.mean([r["entropy_spike"] for r in subset])
        max_spike = np.max([r["entropy_spike"] for r in subset])
        halluc_rate = sum(1 for r in subset if not r["noisy_correct"]) / len(subset) * 100

        summary_text += f"  {src}:\n"
        summary_text += f"    Mean ΔH:          {mean_spike:+.4f}\n"
        summary_text += f"    Max  ΔH:          {max_spike:+.4f}\n"
        summary_text += f"    Hallucination:    {halluc_rate:.1f}%\n\n"

    # Check if SNN noise is more effective
    snn_spikes = [r["entropy_spike"] for r in results if r["noise_source"] == "SNN Chaos"]
    torch_spikes = [r["entropy_spike"] for r in results if r["noise_source"] == "torch.randn"]
    ratio = np.mean(snn_spikes) / (np.mean(torch_spikes) + 1e-10)

    summary_text += f"  SNN/torch ratio: {ratio:.2f}×\n"
    if ratio > 1.1:
        summary_text += f"  → SNN noise is MORE effective!\n"
    elif ratio < 0.9:
        summary_text += f"  → torch.randn is more effective\n"
    else:
        summary_text += f"  → Similar effectiveness\n"

    summary_text += f"\n  → Ready for Phase 3: Dream Catcher v2"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phase2_noise_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Chart saved: {out_path}")

    # Print summary table
    print(f"\n{'Source':<14} {'σ':<6} {'Mean ΔH':<12} {'Halluc %':<10}")
    print("-" * 45)
    for src in sources:
        for sig in sigmas:
            subset = [r for r in results
                      if r["noise_source"] == src and r["sigma"] == sig]
            mean_dh = np.mean([r["entropy_spike"] for r in subset])
            halluc = sum(1 for r in subset if not r["noisy_correct"]) / len(subset) * 100
            print(f"{src:<14} {sig:<6.2f} {mean_dh:<12.4f} {halluc:<10.1f}")

    return out_path


def main():
    print("=" * 70)
    print("Project Genesis — Phase 2: LLM Noise Injection Comparison")
    print("SNN Chaotic Noise vs torch.randn()")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map=device,
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Run experiment
    results = run_injection_comparison(model, tokenizer, device)

    # Save raw results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "phase2_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\n💾 Results saved: {results_path}")

    # Analyze and plot
    plot_path = analyze_and_plot(results)

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Phase 2 Complete!")
    print(f"  Results: {results_path}")
    print(f"  Chart:   {plot_path}")
    print("  → Next: Phase 3 — Dream Catcher v2 (SNN-powered data generation)")
    print("=" * 70)


if __name__ == "__main__":
    main()
