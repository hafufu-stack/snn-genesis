"""
Phase 82b: Phi-3-mini only (Qwen already done)
Loads existing phase82_log.json, adds Phi-3 results, saves back.
"""
import sys, os, json, gc, time, random, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase82_cross_architecture import (
    run_model_experiment, RESULTS_DIR, SEED
)
from scipy.stats import fisher_exact

# Phi-3 config
PHI3_CONFIG = {
    "name": "microsoft/Phi-3-mini-4k-instruct",
    "short": "Phi-3-mini",
    "layer_idx": 18,
    "hidden_dim": 3072,
}

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    # Load existing results
    results_path = os.path.join(RESULTS_DIR, "phase82_log.json")
    with open(results_path, 'r') as f:
        all_results = json.load(f)

    # Remove any error entries for Phi-3
    all_results["models"] = [m for m in all_results["models"] if m.get("model") != "Phi-3-mini" or "error" not in m]

    print(f"Existing models: {[m.get('model') for m in all_results['models']]}")

    # Run Phi-3
    print("\n=== Running Phi-3-mini ===")
    try:
        phi3_results = run_model_experiment(PHI3_CONFIG)
        all_results["models"].append(phi3_results)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()
        all_results["models"].append({"model": "Phi-3-mini", "error": str(e)})

    # Save
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Print comparison
    print(f"\n{'='*80}")
    print(f"  === Cross-Architecture Comparison ===")
    print(f"{'='*80}")

    print(f"\n  Reference: Mistral-7B (Phase 71)")
    print(f"    Top-4 PC variance: 21.5%")
    print(f"    PCA-top4: 6.7%, Random: 46.7%")

    for mr in all_results["models"]:
        if "error" in mr:
            print(f"\n  {mr['model']}: ERROR - {mr['error']}")
            continue

        print(f"\n  {mr['model']}:")
        print(f"    Layers: {mr.get('n_layers','?')}, Hidden dim: {mr.get('hidden_dim','?')}")
        print(f"    Top-4 PC variance: {mr['pca_info']['explained_variance_top4']}%")

        bl = next(c for c in mr["conditions"] if c["condition"] == "baseline")
        rand = next(c for c in mr["conditions"] if c["condition"] == "random")
        top4 = next(c for c in mr["conditions"] if c["condition"] == "pca_top4")
        bottom = next(c for c in mr["conditions"] if c["condition"] == "pca_bottom")

        print(f"    Baseline: {bl['solve_rate']*100:.1f}%")
        print(f"    Random:   {rand['solve_rate']*100:.1f}%")
        print(f"    PCA-top4: {top4['solve_rate']*100:.1f}%")
        print(f"    PCA-bottom: {bottom['solve_rate']*100:.1f}%")

        table = [[top4["n_solved"], top4["n_total"]-top4["n_solved"]],
                 [rand["n_solved"], rand["n_total"]-rand["n_solved"]]]
        _, p = fisher_exact(table)
        gap = (rand["solve_rate"] - top4["solve_rate"]) * 100
        if gap > 10:
            verdict = "CONFIRMED: Top-4 PCs encode reasoning"
        elif gap > 5:
            verdict = "PARTIAL: Top-4 PCs may encode reasoning"
        else:
            verdict = "NOT CONFIRMED: Top-4 PCs don't seem special"
        print(f"    Random - PCA-top4 gap: {gap:.1f}pp (p={p:.4f}) -> {verdict}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 82b (Phi-3 only) complete.")

