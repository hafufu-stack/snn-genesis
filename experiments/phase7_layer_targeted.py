"""
phase7_layer_targeted.py — Layer-Targeted Noise Injection
==========================================================

Deep Think Proposal #1: Test whether targeting SNN noise to SPECIFIC LAYERS
(mid-layers only) gives SNN an advantage over random noise.

Hypothesis: v1's SNN underperformance came from injecting noise into ALL layers,
which destroyed SNN's temporal dynamics. By targeting only mid-layers (15-20),
where knowledge/reasoning lives, SNN's structured chaos should produce
BETTER nightmares than random noise.

Groups:
  A) Genesis-Mid (SNN noise → layers 15-20 only)
  B) Genesis-All (SNN noise → layer 10 only, same as v1)
  C) Morpheus-All (random noise → layer 10 only, same as v1)

Expected runtime: ~3 hours on RTX 5080
"""

import torch
import os
import sys
import json
import random
import time
import numpy as np
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.snn_reservoir import ChaoticReservoir

import warnings
warnings.filterwarnings("ignore")

# ─── Settings ───
MODEL_ID     = "mistralai/Mistral-7B-Instruct-v0.3"
NUM_ROUNDS   = 5
SIGMA        = 0.10
BATCH_SIZE   = 4
LORA_R       = 8
LORA_ALPHA   = 16
MAX_LENGTH   = 256
SEED         = 2026

# Layer targeting: Mistral-7B has 32 decoder layers (0-31)
# Shallow: 0-10 (syntax/grammar)
# Mid: 11-21 (knowledge/reasoning) ← Sweet spot hypothesis
# Deep: 22-31 (output formatting)
MID_LAYERS = list(range(15, 21))  # layers 15-20 (6 layers)
SINGLE_LAYER = 10  # same as v1 baseline

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

from phase5_scaleup import (
    CLEAN_QUESTIONS, NIGHTMARE_TEMPLATES, FALSE_CLAIMS,
    build_nightmare_questions, build_test_set,
    load_model, generate_text, make_snn_hook, make_randn_hook,
    get_model_layers, classify_nightmare, heal_nightmare,
    evaluate_accuracy, train_dream_journal
)


def generate_with_multi_layer_noise(model, tokenizer, prompts, hook_fn, layer_indices):
    """Generate responses with noise injection on MULTIPLE layers."""
    layers = get_model_layers(model)
    handles = []
    for idx in layer_indices:
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


def generate_with_noise_single(model, tokenizer, prompts, hook_fn, layer_idx):
    """Generate responses with noise injection on a single layer (same as v1)."""
    layers = get_model_layers(model)
    layer = layers[layer_idx]
    handle = layer.register_forward_pre_hook(hook_fn)
    responses = []
    try:
        for p in prompts:
            resp = generate_text(model, tokenizer, p, max_new=100)
            responses.append(resp)
    finally:
        handle.remove()
    return responses


def run_layer_targeted():
    """
    Compare 3 noise injection strategies:
    A) SNN noise → mid-layers (15-20)  — "Genesis-Mid"
    B) SNN noise → single layer 10     — "Genesis-All" (v1 baseline)
    C) Random noise → single layer 10  — "Morpheus" (v1 winner)
    """
    print("=" * 70)
    print("Phase 7: Layer-Targeted Noise Injection")
    print(f"  Genesis-Mid (SNN→L15-20) vs Genesis-All (SNN→L10)")
    print(f"  vs Morpheus (randn→L10)")
    print(f"  Started: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model, tokenizer = load_model()

    nightmare_questions = build_nightmare_questions(40)
    test_clean, test_nightmare = build_test_set()
    train_clean = [q for q in CLEAN_QUESTIONS if q not in test_clean]

    log = {"config": {
        "model": MODEL_ID, "rounds": NUM_ROUNDS, "sigma": SIGMA,
        "experiment": "layer_targeted_noise",
        "description": "SNN mid-layer (15-20) vs SNN single-layer (10) vs random single-layer (10)",
        "mid_layers": MID_LAYERS, "single_layer": SINGLE_LAYER,
        "n_test_clean": len(test_clean), "n_test_nightmare": len(test_nightmare),
        "started": datetime.datetime.now().isoformat(),
    }, "genesis_mid": [], "genesis_all": [], "morpheus": []}

    # Baseline
    print("\n📊 Baseline evaluation...")
    base_clean, base_nm = evaluate_accuracy(model, tokenizer, test_clean, test_nightmare)
    print(f"  Clean: {base_clean:.1f}% | Nightmare: {base_nm:.1f}%")

    baseline = {"round": 0, "clean_acc": base_clean, "nightmare_acc": base_nm, "loss": None}
    log["genesis_mid"].append(baseline.copy())
    log["genesis_all"].append(baseline.copy())
    log["morpheus"].append(baseline.copy())

    # Dream Journals for each method
    journal_mid = []
    journal_all = []
    journal_morph = []
    base_model = model

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'─' * 50}")
        print(f"ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'─' * 50}")
        round_start = time.time()

        # ─── A) Genesis-Mid: SNN → layers 15-20 ───
        print(f"\n🎯 Genesis-Mid (SNN → layers {MID_LAYERS[0]}-{MID_LAYERS[-1]}):")
        snn_hook_mid = make_snn_hook(sigma=SIGMA, seed=SEED + round_num)

        nm_responses_mid = generate_with_multi_layer_noise(
            base_model, tokenizer, nightmare_questions, snn_hook_mid, MID_LAYERS
        )

        healed_mid = []
        nm_count_mid = 0
        for resp, prompt in zip(nm_responses_mid, nightmare_questions):
            if classify_nightmare(resp):
                healed_mid.append(heal_nightmare(resp, prompt))
                nm_count_mid += 1

        journal_mid.extend([{"q": "Respond correctly", "a": h} for h in healed_mid])
        print(f"  Nightmares: {nm_count_mid}/{len(nightmare_questions)}, "
              f"Journal: {len(journal_mid)}")

        train_data_mid = list(train_clean) + list(journal_mid)
        random.shuffle(train_data_mid)
        mid_model, loss_mid, mid_adapter = train_dream_journal(
            base_model, tokenizer, train_data_mid, round_num
        )
        print(f"  Loss: {loss_mid:.4f}")

        mid_clean, mid_nm = evaluate_accuracy(
            mid_model, tokenizer, test_clean, test_nightmare,
            debug=(round_num == 1)
        )
        print(f"  Clean: {mid_clean:.1f}% | Nightmare: {mid_nm:.1f}%")

        try:
            mid_model.delete_adapter(mid_adapter)
        except Exception:
            pass
        import gc; gc.collect()
        torch.cuda.empty_cache()

        log["genesis_mid"].append({
            "round": round_num, "clean_acc": mid_clean, "nightmare_acc": mid_nm,
            "loss": round(loss_mid, 4), "nightmares_generated": nm_count_mid,
            "journal_total": len(journal_mid),
        })

        # ─── B) Genesis-All: SNN → layer 10 (v1 baseline) ───
        print(f"\n🧬 Genesis-All (SNN → layer {SINGLE_LAYER}):")
        snn_hook_all = make_snn_hook(sigma=SIGMA, seed=SEED + round_num)

        nm_responses_all = generate_with_noise_single(
            base_model, tokenizer, nightmare_questions, snn_hook_all, SINGLE_LAYER
        )

        healed_all = []
        nm_count_all = 0
        for resp, prompt in zip(nm_responses_all, nightmare_questions):
            if classify_nightmare(resp):
                healed_all.append(heal_nightmare(resp, prompt))
                nm_count_all += 1

        journal_all.extend([{"q": "Respond correctly", "a": h} for h in healed_all])
        print(f"  Nightmares: {nm_count_all}/{len(nightmare_questions)}, "
              f"Journal: {len(journal_all)}")

        train_data_all = list(train_clean) + list(journal_all)
        random.shuffle(train_data_all)
        all_model, loss_all, all_adapter = train_dream_journal(
            base_model, tokenizer, train_data_all, round_num
        )
        print(f"  Loss: {loss_all:.4f}")

        all_clean, all_nm = evaluate_accuracy(
            all_model, tokenizer, test_clean, test_nightmare
        )
        print(f"  Clean: {all_clean:.1f}% | Nightmare: {all_nm:.1f}%")

        try:
            all_model.delete_adapter(all_adapter)
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        log["genesis_all"].append({
            "round": round_num, "clean_acc": all_clean, "nightmare_acc": all_nm,
            "loss": round(loss_all, 4), "nightmares_generated": nm_count_all,
            "journal_total": len(journal_all),
        })

        # ─── C) Morpheus: random → layer 10 (v1 winner) ───
        print(f"\n🌙 Morpheus (randn → layer {SINGLE_LAYER}):")
        randn_hook = make_randn_hook(sigma=SIGMA)

        nm_responses_morph = generate_with_noise_single(
            base_model, tokenizer, nightmare_questions, randn_hook, SINGLE_LAYER
        )

        healed_morph = []
        nm_count_morph = 0
        for resp, prompt in zip(nm_responses_morph, nightmare_questions):
            if classify_nightmare(resp):
                healed_morph.append(heal_nightmare(resp, prompt))
                nm_count_morph += 1

        journal_morph.extend([{"q": "Respond correctly", "a": h} for h in healed_morph])
        print(f"  Nightmares: {nm_count_morph}/{len(nightmare_questions)}, "
              f"Journal: {len(journal_morph)}")

        train_data_morph = list(train_clean) + list(journal_morph)
        random.shuffle(train_data_morph)
        morph_model, loss_morph, morph_adapter = train_dream_journal(
            base_model, tokenizer, train_data_morph, round_num
        )
        print(f"  Loss: {loss_morph:.4f}")

        morph_clean, morph_nm = evaluate_accuracy(
            morph_model, tokenizer, test_clean, test_nightmare
        )
        print(f"  Clean: {morph_clean:.1f}% | Nightmare: {morph_nm:.1f}%")

        try:
            morph_model.delete_adapter(morph_adapter)
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        log["morpheus"].append({
            "round": round_num, "clean_acc": morph_clean, "nightmare_acc": morph_nm,
            "loss": round(loss_morph, 4), "nightmares_generated": nm_count_morph,
            "journal_total": len(journal_morph),
        })

        elapsed = time.time() - round_start
        print(f"\n  ⏱ Round {round_num} time: {elapsed/60:.1f} min")

    # ─── Save Results ───
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log["finished"] = datetime.datetime.now().isoformat()

    results_path = os.path.join(RESULTS_DIR, "phase7_layer_targeted_log.json")
    with open(results_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n💾 Results: {results_path}")

    # ─── Visualization ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        mid_data = log["genesis_mid"]
        all_data = log["genesis_all"]
        morph_data = log["morpheus"]
        rounds = [d["round"] for d in mid_data]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Phase 7: Layer-Targeted Noise Injection",
                     fontsize=13, fontweight="bold")

        # Nightmare accuracy
        ax = axes[0]
        ax.plot(rounds, [d["nightmare_acc"] for d in mid_data],
                "r-^", label=f"Genesis-Mid (SNN→L{MID_LAYERS[0]}-{MID_LAYERS[-1]})", linewidth=2)
        ax.plot(rounds, [d["nightmare_acc"] for d in all_data],
                "g-o", label=f"Genesis-All (SNN→L{SINGLE_LAYER})", linewidth=2)
        ax.plot(rounds, [d["nightmare_acc"] for d in morph_data],
                "b-s", label=f"Morpheus (randn→L{SINGLE_LAYER})", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Nightmare Accuracy (%)")
        ax.set_title("Nightmare Resistance (↓ = better)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Clean accuracy
        ax = axes[1]
        ax.plot(rounds, [d["clean_acc"] for d in mid_data],
                "r-^", label="Genesis-Mid", linewidth=2)
        ax.plot(rounds, [d["clean_acc"] for d in all_data],
                "g-o", label="Genesis-All", linewidth=2)
        ax.plot(rounds, [d["clean_acc"] for d in morph_data],
                "b-s", label="Morpheus", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Clean Accuracy (%)")
        ax.set_title("Knowledge Preservation (↑ = better)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(60, 100)

        # Nightmares generated per round
        ax = axes[2]
        r_no_base = rounds[1:]
        ax.plot(r_no_base, [d.get("nightmares_generated", 0) for d in mid_data[1:]],
                "r-^", label="Genesis-Mid", linewidth=2)
        ax.plot(r_no_base, [d.get("nightmares_generated", 0) for d in all_data[1:]],
                "g-o", label="Genesis-All", linewidth=2)
        ax.plot(r_no_base, [d.get("nightmares_generated", 0) for d in morph_data[1:]],
                "b-s", label="Morpheus", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Nightmares Generated")
        ax.set_title("Nightmare Discovery Rate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "phase7_layer_targeted.png")
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊 Figure: {fig_path}")
    except Exception as e:
        print(f"⚠ Visualization error: {e}")

    # ─── Summary ───
    print(f"\n{'=' * 70}")
    print("LAYER-TARGETED NOISE SUMMARY")
    print(f"{'=' * 70}")
    mid_final = log["genesis_mid"][-1]
    all_final = log["genesis_all"][-1]
    morph_final = log["morpheus"][-1]
    print(f"  Genesis-Mid (SNN→L15-20): Clean={mid_final['clean_acc']:.1f}%, "
          f"NM={mid_final['nightmare_acc']:.1f}%")
    print(f"  Genesis-All (SNN→L10):    Clean={all_final['clean_acc']:.1f}%, "
          f"NM={all_final['nightmare_acc']:.1f}%")
    print(f"  Morpheus (randn→L10):     Clean={morph_final['clean_acc']:.1f}%, "
          f"NM={morph_final['nightmare_acc']:.1f}%")

    # Best nightmare accuracy
    mid_best = min(d["nightmare_acc"] for d in log["genesis_mid"])
    all_best = min(d["nightmare_acc"] for d in log["genesis_all"])
    morph_best = min(d["nightmare_acc"] for d in log["morpheus"])
    print(f"\n  Best NM — Genesis-Mid: {mid_best:.1f}%, "
          f"Genesis-All: {all_best:.1f}%, Morpheus: {morph_best:.1f}%")

    if mid_best < all_best and mid_best < morph_best:
        print(f"\n  🎯 BREAKTHROUGH: Layer-targeted SNN WINS! Mid-layer injection "
              f"is the sweet spot!")
    elif mid_best < all_best:
        print(f"\n  🔬 FINDING: Layer targeting improves SNN, but random still competitive.")
    else:
        print(f"\n  🔬 FINDING: Layer targeting did not help SNN at this scale.")

    print(f"\n  Finished: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    run_layer_targeted()
