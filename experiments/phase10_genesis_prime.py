"""
phase10_genesis_prime.py — Genesis Prime: DPO + Mid-Layer SNN
=============================================================

The ultimate combination: best attack (SNN mid-layer noise) + best defense (DPO).

Phase 7 showed: SNN mid-layer (L15-20) → 100% nightmare discovery rate
Phase 8 showed: DPO training → 0% nightmare acceptance

This experiment fuses both to create the "Genesis Prime" — the strongest
possible configuration of the SNN-Genesis framework.

Comparison:
  A) Genesis-Prime (SNN mid-layer + DPO) — the fusion
  B) Phase 8 baseline (Random single-layer + DPO) — for comparison

Hypothesis: Genesis-Prime will achieve 0% NM acceptance faster and more
            stably because mid-layer SNN discovers MORE nightmares (100%
            vs ~80%), giving DPO better training signal.
"""

import os
import sys
import json
import time
import datetime
import random
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt

# ─── Config ─────────────────────────────────────────────
MODEL_ID    = "mistralai/Mistral-7B-Instruct-v0.3"
NUM_ROUNDS  = 5
SIGMA       = 0.1
TARGET_LAYER = 10
LORA_R       = 8
LORA_ALPHA   = 16
MAX_LENGTH   = 256
SEED         = 2026

MID_LAYERS = list(range(15, 21))  # layers 15-20 (6 layers)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

from phase5_scaleup import (
    CLEAN_QUESTIONS, NIGHTMARE_TEMPLATES, FALSE_CLAIMS,
    build_nightmare_questions, build_test_set,
    load_model, generate_text, make_snn_hook, make_randn_hook, get_model_layers,
    generate_with_noise, classify_nightmare, heal_nightmare,
    evaluate_accuracy, train_dream_journal
)
from phase7_layer_targeted import generate_with_multi_layer_noise
from phase8_dpo import train_dpo_journal


def run_genesis_prime():
    """
    Compare two DPO configurations:
    A) Genesis-Prime: SNN mid-layer (L15-20) noise + DPO training
    B) Morpheus-DPO:  Random single-layer (L10) noise + DPO training (Phase 8 baseline)
    """
    print("=" * 70)
    print("Phase 10: Genesis Prime — DPO + Mid-Layer SNN")
    print(f"  The Ultimate Fusion: Best Attack × Best Defense")
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
        "experiment": "genesis_prime",
        "description": "Genesis-Prime (SNN mid-layer + DPO) vs Morpheus-DPO (random + DPO)",
        "mid_layers": MID_LAYERS,
        "single_layer": TARGET_LAYER,
        "n_test_clean": len(test_clean), "n_test_nightmare": len(test_nightmare),
        "started": datetime.datetime.now().isoformat(),
    }, "genesis_prime": [], "morpheus_dpo": []}

    # Baseline
    print("\n📊 Baseline evaluation...")
    base_clean, base_nm = evaluate_accuracy(model, tokenizer, test_clean, test_nightmare)
    print(f"  Clean: {base_clean:.1f}% | Nightmare: {base_nm:.1f}%")

    baseline = {"round": 0, "clean_acc": base_clean, "nightmare_acc": base_nm, "loss": None}
    log["genesis_prime"].append(baseline.copy())
    log["morpheus_dpo"].append(baseline.copy())

    # Accumulate journals for each branch
    prime_preferences = []      # Genesis-Prime DPO prefs
    morpheus_preferences = []   # Morpheus-DPO prefs
    base_model = model

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'─' * 50}")
        print(f"ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'─' * 50}")
        round_start = time.time()

        # ─── A) Genesis-Prime: SNN mid-layer nightmare generation ───
        print(f"\n🧬 Genesis-Prime: SNN Mid-Layer (L15-20) noise...")
        snn_hook = make_snn_hook(sigma=SIGMA)
        prime_responses = generate_with_multi_layer_noise(
            base_model, tokenizer, nightmare_questions, snn_hook, MID_LAYERS
        )

        prime_nm_count = 0
        prime_prefs_new = []
        for resp, prompt in zip(prime_responses, nightmare_questions):
            if classify_nightmare(resp):
                prime_nm_count += 1
                refusal = heal_nightmare(resp, prompt)
                prime_prefs_new.append({
                    "prompt": prompt,
                    "chosen": refusal,
                    "rejected": resp[:200],
                })

        prime_preferences.extend(prime_prefs_new)
        print(f"  Nightmares discovered: {prime_nm_count}/40 ({prime_nm_count/40*100:.0f}%)")
        print(f"  Total preferences: {len(prime_preferences)}")

        # ─── B) Morpheus-DPO: Random single-layer nightmare generation ───
        print(f"\n🎲 Morpheus-DPO: Random noise (L{TARGET_LAYER})...")
        randn_hook = make_randn_hook(sigma=SIGMA)
        morpheus_responses = generate_with_noise(
            base_model, tokenizer, nightmare_questions, randn_hook, TARGET_LAYER
        )

        morpheus_nm_count = 0
        morpheus_prefs_new = []
        for resp, prompt in zip(morpheus_responses, nightmare_questions):
            if classify_nightmare(resp):
                morpheus_nm_count += 1
                refusal = heal_nightmare(resp, prompt)
                morpheus_prefs_new.append({
                    "prompt": prompt,
                    "chosen": refusal,
                    "rejected": resp[:200],
                })

        morpheus_preferences.extend(morpheus_prefs_new)
        print(f"  Nightmares discovered: {morpheus_nm_count}/40 ({morpheus_nm_count/40*100:.0f}%)")
        print(f"  Total preferences: {len(morpheus_preferences)}")

        # ─── Train Genesis-Prime (DPO) ───
        print(f"\n🎯 Training Genesis-Prime (DPO):")
        prime_model, loss_prime, prime_adapter = train_dpo_journal(
            base_model, tokenizer, prime_preferences, train_clean, round_num
        )
        print(f"  Loss: {loss_prime:.4f}")

        prime_clean, prime_nm = evaluate_accuracy(
            prime_model, tokenizer, test_clean, test_nightmare
        )
        print(f"  Clean: {prime_clean:.1f}% | Nightmare: {prime_nm:.1f}%")

        try:
            prime_model.delete_adapter(prime_adapter)
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        log["genesis_prime"].append({
            "round": round_num, "clean_acc": prime_clean, "nightmare_acc": prime_nm,
            "loss": round(loss_prime, 4),
            "nightmares_generated": prime_nm_count,
            "preferences_total": len(prime_preferences),
        })

        # ─── Train Morpheus-DPO ───
        print(f"\n🎯 Training Morpheus-DPO:")
        morph_model, loss_morph, morph_adapter = train_dpo_journal(
            base_model, tokenizer, morpheus_preferences, train_clean, round_num
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

        log["morpheus_dpo"].append({
            "round": round_num, "clean_acc": morph_clean, "nightmare_acc": morph_nm,
            "loss": round(loss_morph, 4),
            "nightmares_generated": morpheus_nm_count,
            "preferences_total": len(morpheus_preferences),
        })

        elapsed = time.time() - round_start
        print(f"\n  ⏱ Round {round_num} time: {elapsed/60:.1f} min")
        print(f"  📊 Prime: Clean {prime_clean:.1f}% NM {prime_nm:.1f}%")
        print(f"  📊 Morph: Clean {morph_clean:.1f}% NM {morph_nm:.1f}%")

    # ─── Save Results ───
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log["finished"] = datetime.datetime.now().isoformat()
    log_path = os.path.join(RESULTS_DIR, "phase10_genesis_prime_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n💾 Results saved: {log_path}")

    # ─── Visualization ───
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rounds = list(range(NUM_ROUNDS + 1))

    prime_nm_vals = [r["nightmare_acc"] for r in log["genesis_prime"]]
    morph_nm_vals = [r["nightmare_acc"] for r in log["morpheus_dpo"]]
    prime_cl_vals = [r["clean_acc"] for r in log["genesis_prime"]]
    morph_cl_vals = [r["clean_acc"] for r in log["morpheus_dpo"]]
    prime_loss = [r["loss"] for r in log["genesis_prime"] if r["loss"] is not None]
    morph_loss = [r["loss"] for r in log["morpheus_dpo"] if r["loss"] is not None]
    prime_gen  = [r.get("nightmares_generated", 0) for r in log["genesis_prime"] if r.get("nightmares_generated") is not None]
    morph_gen  = [r.get("nightmares_generated", 0) for r in log["morpheus_dpo"] if r.get("nightmares_generated") is not None]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 10: Genesis Prime — DPO + Mid-Layer SNN", fontsize=14, fontweight="bold")

    # NM Acceptance
    ax = axes[0, 0]
    ax.plot(rounds, prime_nm_vals, "b-o", label="Genesis-Prime (SNN Mid + DPO)", linewidth=2)
    ax.plot(rounds, morph_nm_vals, "r--s", label="Morpheus-DPO (Random + DPO)", linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("NM Acceptance (%)")
    ax.set_title("Nightmare Acceptance (↓ better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="green", linestyle=":", alpha=0.5, label="Perfect safety")

    # Clean Accuracy
    ax = axes[0, 1]
    ax.plot(rounds, prime_cl_vals, "b-o", label="Genesis-Prime", linewidth=2)
    ax.plot(rounds, morph_cl_vals, "r--s", label="Morpheus-DPO", linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Clean Accuracy (%)")
    ax.set_title("Clean Accuracy (↑ better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training Loss
    ax = axes[1, 0]
    ax.plot(range(1, len(prime_loss)+1), prime_loss, "b-o", label="Genesis-Prime", linewidth=2)
    ax.plot(range(1, len(morph_loss)+1), morph_loss, "r--s", label="Morpheus-DPO", linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss (↓ better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Nightmares Discovered
    ax = axes[1, 1]
    ax.bar(np.arange(1, len(prime_gen)+1) - 0.15, prime_gen, 0.3, label="Genesis-Prime (SNN Mid)", color="blue", alpha=0.7)
    ax.bar(np.arange(1, len(morph_gen)+1) + 0.15, morph_gen, 0.3, label="Morpheus-DPO (Random)", color="red", alpha=0.7)
    ax.set_xlabel("Round")
    ax.set_ylabel("Nightmares Discovered")
    ax.set_title("Nightmare Discovery Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=40, color="green", linestyle=":", alpha=0.5, label="Max (40)")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase10_genesis_prime.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"📊 Figure saved: {fig_path}")

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("GENESIS PRIME — FINAL RESULTS")
    print("=" * 70)
    print(f"\n  {'Method':<30} {'Final NM':>10} {'Best NM':>10} {'Final Clean':>12}")
    print(f"  {'─' * 62}")

    best_prime = min(prime_nm_vals)
    best_morph = min(morph_nm_vals)
    print(f"  {'Genesis-Prime (SNN+DPO)':<30} {prime_nm_vals[-1]:>9.1f}% {best_prime:>9.1f}% {prime_cl_vals[-1]:>11.1f}%")
    print(f"  {'Morpheus-DPO (Rand+DPO)':<30} {morph_nm_vals[-1]:>9.1f}% {best_morph:>9.1f}% {morph_cl_vals[-1]:>11.1f}%")

    if best_prime <= best_morph:
        print(f"\n  🏆 Genesis-Prime wins! SNN mid-layer attack provides better training signal for DPO.")
    else:
        print(f"\n  🏆 Morpheus-DPO wins! Random noise is sufficient for DPO training.")

    total_time = sum(1 for _ in [])  # placeholder
    print(f"\n  Finished: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    run_genesis_prime()
