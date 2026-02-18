"""
phase17b_nightmare_by_layer.py — Do Late Layers Generate Nightmares?
====================================================================

Phase 17.5: Tests whether SNN noise at DIFFERENT layer ranges affects
nightmare generation rate. Specifically, does L25-31 (which showed 0%
TruthfulQA tax in Phase 17) still generate nightmares?

Two possible outcomes:
  A) L25-31 does NOT generate nightmares → consistent story
     (truthfulness + nightmare generation are both early-layer phenomena)
  B) L25-31 DOES generate nightmares → amazing finding
     (vulnerability probing with zero alignment tax is possible!)

Conditions tested (σ=0.10 across all, on Mistral-7B):
  - Base (no noise)
  - L0-5, L5-10, L10-15, L15-20, L20-25, L25-31

Each condition: 40 nightmare prompts → measure acceptance rate.
"""

import torch
import os
import sys
import json
import gc
import time
import datetime
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase5_scaleup import (
    load_model, get_model_layers, generate_text,
    build_nightmare_questions, classify_nightmare
)

# ─── Settings ───
SIGMA = 0.10
N_NIGHTMARE = 40  # nightmare prompts per condition

LAYER_RANGES = {
    "L0-5":   list(range(0, 6)),
    "L5-10":  list(range(5, 11)),
    "L10-15": list(range(10, 16)),
    "L15-20": list(range(15, 21)),
    "L20-25": list(range(20, 26)),
    "L25-31": list(range(25, 32)),
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def make_gpu_snn_hook(sigma=0.10):
    """GPU-native SNN-style noise hook (consistent with Phase 17)."""
    def hook(module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * sigma
        return (hs + noise,) + args[1:]
    return hook


def generate_with_layer_range_noise(model, tokenizer, prompts, layer_indices, sigma):
    """Generate responses with SNN noise on multiple layers."""
    layers = get_model_layers(model)
    handles = []

    hook_fn = make_gpu_snn_hook(sigma=sigma)
    for idx in layer_indices:
        if idx < len(layers):
            handle = layers[idx].register_forward_pre_hook(hook_fn)
            handles.append(handle)

    responses = []
    try:
        for p in prompts:
            resp = generate_text(model, tokenizer, p, max_new=100)
            responses.append(resp)
    finally:
        for h in handles:
            h.remove()

    return responses


def evaluate_nightmare_rate(responses):
    """Compute nightmare acceptance rate."""
    accepted = sum(1 for r in responses if classify_nightmare(r))
    rate = accepted / len(responses) * 100 if responses else 0
    return accepted, len(responses), rate


def visualize(all_results, base_rate):
    """Create bar chart comparing nightmare rates across layers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560",
        "axes.labelcolor": "#eee",
        "text.color": "#eee",
        "xtick.color": "#ccc",
        "ytick.color": "#ccc",
        "grid.color": "#333",
        "grid.alpha": 0.3,
    })

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.suptitle("Phase 17.5: Nightmare Generation by Layer Range\n"
                 f"Mistral-7B (32 layers) | σ={SIGMA} | n={N_NIGHTMARE} nightmares",
                 fontsize=14, fontweight="bold", color="#e94560")

    names = [r["condition"] for r in all_results]
    rates = [r["nightmare_rate"] for r in all_results]

    # Color: base=gray, others gradient from red (early) to blue (late)
    colors = ["#888888"]  # Base
    layer_colors = ["#FF2222", "#FF6633", "#FF9944", "#FFCC55", "#66BB66", "#22AAFF"]
    colors.extend(layer_colors)

    x = np.arange(len(names))
    bars = ax.bar(x, rates, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{rate:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Nightmare Acceptance Rate (%)", fontsize=12)
    ax.set_xlabel("Layer Range (σ=0.10)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, max(rates) * 1.3 if rates else 100)
    ax.axhline(y=base_rate, color="#e94560", linestyle="--", alpha=0.5, label=f"Baseline ({base_rate:.1f}%)")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y")

    # Add delta annotations
    for i, rate in enumerate(rates[1:], 1):
        delta = rate - base_rate
        sign = "+" if delta >= 0 else ""
        color = "#FF4444" if delta > 5 else "#00FF00" if delta < -5 else "#FFFFFF"
        ax.text(i, max(rates) * 1.15, f"Δ{sign}{delta:.1f}%",
                ha="center", fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "phase17b_nightmare_by_layer.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


def main():
    print("=" * 60)
    print("Phase 17.5: Nightmare Generation by Layer Range")
    print("=" * 60)
    print(f"  σ = {SIGMA}")
    print(f"  Nightmares per condition: {N_NIGHTMARE}")
    print(f"  Layer ranges: {list(LAYER_RANGES.keys())}")
    t0 = time.time()

    # Build nightmare prompts
    print("\n📚 Building nightmare prompts...")
    nm_prompts = build_nightmare_questions(N_NIGHTMARE)
    print(f"  ✅ {len(nm_prompts)} nightmare prompts ready")

    # Load model
    print("\n📦 Loading Mistral-7B...")
    model, tokenizer = load_model()
    num_layers = len(get_model_layers(model))
    print(f"  ✅ {num_layers} decoder layers found")

    all_results = []

    # ─── Baseline (no noise) ───
    print(f"\n{'▶'*60}")
    print("  BASELINE (No Noise)")
    print(f"{'▶'*60}")

    base_responses = []
    for i, p in enumerate(nm_prompts):
        resp = generate_text(model, tokenizer, p, max_new=100)
        base_responses.append(resp)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N_NIGHTMARE}] generated...")

    accepted, total, rate = evaluate_nightmare_rate(base_responses)
    base_rate = rate
    print(f"  ✅ Base: {accepted}/{total} nightmares accepted ({rate:.1f}%)")

    all_results.append({
        "condition": "Base",
        "nightmare_accepted": accepted,
        "nightmare_total": total,
        "nightmare_rate": round(rate, 1),
        "time_minutes": round((time.time() - t0) / 60, 1),
        "examples": base_responses[:3],
    })

    # ─── Layer ablation conditions ───
    for range_name, layer_indices in LAYER_RANGES.items():
        t_cond = time.time()
        print(f"\n{'▶'*60}")
        print(f"  CONDITION: σ={SIGMA} @ {range_name} (layers {layer_indices})")
        print(f"{'▶'*60}")

        responses = generate_with_layer_range_noise(
            model, tokenizer, nm_prompts, layer_indices, SIGMA
        )

        accepted, total, rate = evaluate_nightmare_rate(responses)
        elapsed = (time.time() - t_cond) / 60
        print(f"  ✅ {range_name}: {accepted}/{total} nightmares accepted ({rate:.1f}%) [{elapsed:.1f}min]")

        all_results.append({
            "condition": range_name,
            "nightmare_accepted": accepted,
            "nightmare_total": total,
            "nightmare_rate": round(rate, 1),
            "time_minutes": round(elapsed, 1),
            "examples": responses[:3],
        })

        # Save intermediate
        save_intermediate(all_results, t0)

    # ─── Cleanup ───
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ─── Visualize ───
    print("\n🎨 Creating visualization...")
    visualize(all_results, base_rate)

    # ─── Save final results ───
    output = {
        "experiment": "Phase 17.5: Nightmare Generation by Layer Range",
        "sigma": SIGMA,
        "n_nightmare": N_NIGHTMARE,
        "layer_ranges": {k: v for k, v in LAYER_RANGES.items()},
        "results": all_results,
        "finished": str(datetime.datetime.now()),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "phase17b_nightmare_by_layer_log.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Final results: {results_path}")

    # ─── Summary ───
    print("\n" + "=" * 60)
    print("PHASE 17.5 COMPLETE — NIGHTMARE GENERATION BY LAYER")
    print("=" * 60)
    print(f"\n  {'Condition':15s} | {'Accepted':>10s} | {'Rate':>8s} | {'ΔBase':>8s}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    for r in all_results:
        delta = r["nightmare_rate"] - base_rate
        sign = "+" if delta >= 0 else ""
        delta_str = "—" if r["condition"] == "Base" else f"{sign}{delta:.1f}%"
        print(f"  {r['condition']:15s} | {r['nightmare_accepted']:>4d}/{r['nightmare_total']:<4d}  | {r['nightmare_rate']:>6.1f}% | {delta_str:>8s}")

    total_min = (time.time() - t0) / 60
    print(f"\n  ⏱ Total time: {total_min:.1f} min")

    # Beep
    try:
        import winsound
        winsound.Beep(800, 200)
        winsound.Beep(1000, 200)
    except Exception:
        print("\a")


def save_intermediate(results, t0):
    """Save intermediate results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "experiment": "Phase 17.5: Nightmare by Layer (in progress)",
        "results": results,
        "elapsed_minutes": round((time.time() - t0) / 60, 1),
    }
    path = os.path.join(RESULTS_DIR, "phase17b_intermediate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
