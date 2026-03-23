"""
Phase 94: Cross-Architecture Discriminant Axis Analysis
=======================================================

Pure computational analysis (no GPU games needed).
Compares the Differential PCA axes found by Phase 87 (Qwen) and
Phase 91 (Mistral) to answer the grand question:

  "Is there a UNIVERSAL reasoning direction across architectures?"

Analyses:
  1. Qwen Diff-PCA vs Qwen Standard PCA
     - Where in standard-PCA space does Qwen's discriminant axis live?
     - Does it overlap with top-64 (reasoning), mid-band, or PC 257+ (safe)?

  2. Mistral Diff-PCA vs Mistral Standard PCA
     - Same analysis for Mistral
     - Does Mistral's discriminant axis align with PC 257+?

  3. Structural comparison (Cross-Architecture)
     - Both models have different hidden_dim (3584 vs 4096)
     - Cannot directly compute cosine sim between raw vectors
     - Instead compare: WHERE in standard-PCA space each discriminant lives
     - If both discriminant axes live in PC 257+ → universal safe-zone for reasoning
     - If Qwen's is in top-64 and Mistral's in PC 257+ → architecture-specific

Depends on:
  - phase87_diff_pca.npz (Qwen — already exists)
  - phase84_qwen_pca.npz (already exists)
  - phase91_diff_pca.npz (Mistral — created by Phase 91)
  - phase86_mistral_pca.npz (already exists)
"""

import os, json, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def analyze_subspace_overlap(diff_unit, std_Vt, model_name):
    """Analyze where the discriminant axis lives in standard PCA space.

    Returns a dict with overlap fractions for each PCA band.
    """
    print(f"\n  === {model_name}: Discriminant vs Standard PCA ===")

    # Cosine similarities between diff_unit and each standard PC
    cos_sims = np.abs(std_Vt @ diff_unit)  # (n_pcs,)
    n_pcs = len(cos_sims)

    # Band boundaries
    bands = {
        "top_10": (0, min(10, n_pcs)),
        "top_64": (0, min(64, n_pcs)),
        "mid_band_65_256": (64, min(256, n_pcs)),
        "pc_257_plus": (256, n_pcs),
    }

    results = {}
    for band_name, (start, end) in bands.items():
        if start >= n_pcs:
            results[band_name] = {"overlap": 0.0, "top_pc": None}
            continue
        band_cos = cos_sims[start:end]
        overlap = float(np.sum(band_cos**2))  # Fraction of discriminant variance in this band
        top_idx_in_band = np.argmax(band_cos)
        top_pc = start + top_idx_in_band + 1
        top_cos = float(band_cos[top_idx_in_band])
        results[band_name] = {
            "overlap": round(overlap, 4),
            "overlap_pct": round(overlap * 100, 1),
            "strongest_pc": int(top_pc),
            "strongest_cos": round(top_cos, 4),
        }
        print(f"    {band_name:20s}: {overlap*100:5.1f}% overlap "
              f"(strongest: PC{top_pc} cos={top_cos:.4f})")

    # Top-10 most aligned PCs
    top_k_idx = np.argsort(cos_sims)[::-1][:10]
    top_k = [{"pc": int(idx+1), "cos_sim": round(float(cos_sims[idx]), 4)} for idx in top_k_idx]
    results["top_10_aligned_pcs"] = top_k
    top_str = ', '.join(f'PC{p["pc"]}({p["cos_sim"]:.3f})' for p in top_k[:5])
    print(f"    Top aligned PCs: {top_str}")

    # Cumulative overlap curve
    sorted_cos2 = np.sort(cos_sims**2)[::-1]
    cumulative = np.cumsum(sorted_cos2)
    # How many PCs needed to capture 90% of discriminant?
    n_for_90 = int(np.searchsorted(cumulative, 0.9) + 1)
    n_for_50 = int(np.searchsorted(cumulative, 0.5) + 1)
    results["n_pcs_for_50pct"] = int(n_for_50)
    results["n_pcs_for_90pct"] = int(n_for_90)
    print(f"    PCs needed: {n_for_50} for 50%, {n_for_90} for 90% of discriminant")

    return results, cos_sims


def structural_comparison(qwen_overlap, mistral_overlap):
    """Compare WHERE in PCA space each architecture's discriminant lives."""
    print(f"\n  === Cross-Architecture Structural Comparison ===")

    comparison = {}
    for band in ["top_10", "top_64", "mid_band_65_256", "pc_257_plus"]:
        q_pct = qwen_overlap.get(band, {}).get("overlap_pct", 0)
        m_pct = mistral_overlap.get(band, {}).get("overlap_pct", 0)
        diff = q_pct - m_pct
        comparison[band] = {
            "qwen_pct": q_pct,
            "mistral_pct": m_pct,
            "difference": round(diff, 1),
        }
        marker = "★" if abs(diff) < 5 else ("↑Q" if diff > 0 else "↑M")
        print(f"    {band:20s}: Qwen={q_pct:5.1f}%  Mistral={m_pct:5.1f}%  "
              f"(diff={diff:+5.1f}pp) {marker}")

    # Interpretation
    q_safe = qwen_overlap.get("pc_257_plus", {}).get("overlap_pct", 0)
    m_safe = mistral_overlap.get("pc_257_plus", {}).get("overlap_pct", 0)
    q_top = qwen_overlap.get("top_64", {}).get("overlap_pct", 0)
    m_top = mistral_overlap.get("top_64", {}).get("overlap_pct", 0)

    print(f"\n  === Interpretation ===")
    if abs(q_safe - m_safe) < 10 and q_safe > 30:
        print(f"    UNIVERSAL: Both discriminant axes live primarily in PC 257+")
        print(f"    → The safe subspace is universally where reasoning direction hides")
        comparison["verdict"] = "universal_safe_zone"
    elif abs(q_top - m_top) < 10 and q_top > 30:
        print(f"    UNIVERSAL: Both discriminant axes overlap with top-64 variance PCs")
        comparison["verdict"] = "universal_top_pcs"
    else:
        print(f"    ARCHITECTURE-SPECIFIC: Discriminant axes live in different PCA bands")
        print(f"    → Each model invents its own reasoning geometry")
        comparison["verdict"] = "architecture_specific"

    return comparison


def visualize(qwen_cos, mistral_cos, qwen_overlap, mistral_overlap, comparison):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 94: Cross-Architecture Discriminant Axis Analysis\n"
                 "Where does the reasoning direction live in standard PCA space?",
                 fontsize=13, fontweight="bold")

    # Panel 1: Qwen cosine similarity spectrum
    ax = axes[0, 0]
    ax.plot(range(1, len(qwen_cos)+1), qwen_cos, color="#2196F3", alpha=0.7, linewidth=0.8)
    ax.axvline(x=64, color="red", linestyle="--", alpha=0.5, label="PC 64")
    ax.axvline(x=256, color="orange", linestyle="--", alpha=0.5, label="PC 256")
    ax.set_xlabel("Standard PC Index", fontsize=10)
    ax.set_ylabel("|cos(diff_unit, PC_i)|", fontsize=10)
    ax.set_title("Qwen: Discriminant vs Standard PCA", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, len(qwen_cos))
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2: Mistral cosine similarity spectrum
    ax = axes[0, 1]
    ax.plot(range(1, len(mistral_cos)+1), mistral_cos, color="#4CAF50", alpha=0.7, linewidth=0.8)
    ax.axvline(x=64, color="red", linestyle="--", alpha=0.5, label="PC 64")
    ax.axvline(x=256, color="orange", linestyle="--", alpha=0.5, label="PC 256")
    ax.set_xlabel("Standard PC Index", fontsize=10)
    ax.set_ylabel("|cos(diff_unit, PC_i)|", fontsize=10)
    ax.set_title("Mistral: Discriminant vs Standard PCA", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, len(mistral_cos))
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 3: Band overlap comparison
    ax = axes[1, 0]
    bands = ["top_10", "top_64", "mid_band_65_256", "pc_257_plus"]
    band_labels = ["Top 10", "Top 64", "Mid\n(65-256)", "PC 257+"]
    q_vals = [qwen_overlap[b]["overlap_pct"] for b in bands]
    m_vals = [mistral_overlap[b]["overlap_pct"] for b in bands]

    x = np.arange(len(bands))
    width = 0.35
    bars1 = ax.bar(x - width/2, q_vals, width, label="Qwen", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width/2, m_vals, width, label="Mistral", color="#4CAF50", alpha=0.8)

    for bar, val in zip(bars1, q_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, m_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=9)
    ax.set_ylabel("Discriminant Overlap (%)", fontsize=10)
    ax.set_title("Band Overlap Comparison", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis("off")
    verdict = comparison.get("verdict", "unknown")
    verdict_text = {
        "universal_safe_zone": "UNIVERSAL: Both discriminant axes\nlive in PC 257+ (safe subspace)\n→ Reasoning hides in low-variance dims",
        "universal_top_pcs": "UNIVERSAL: Both discriminant axes\nalign with top variance PCs\n→ Reasoning direction = variance direction",
        "architecture_specific": "ARCHITECTURE-SPECIFIC:\nDifferent PCA regions across models\n→ Each model has unique reasoning geometry",
    }.get(verdict, f"Verdict: {verdict}")

    ax.text(0.5, 0.7, verdict_text, transform=ax.transAxes,
            fontsize=13, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3E5F5", alpha=0.8))

    q_n50 = qwen_overlap.get("n_pcs_for_50pct", "?")
    m_n50 = mistral_overlap.get("n_pcs_for_50pct", "?")
    q_n90 = qwen_overlap.get("n_pcs_for_90pct", "?")
    m_n90 = mistral_overlap.get("n_pcs_for_90pct", "?")
    stats_text = (f"PCs for 50% discriminant:\n"
                  f"  Qwen: {q_n50}  |  Mistral: {m_n50}\n\n"
                  f"PCs for 90% discriminant:\n"
                  f"  Qwen: {q_n90}  |  Mistral: {m_n90}")
    ax.text(0.5, 0.25, stats_text, transform=ax.transAxes,
            fontsize=11, ha="center", va="center", family="monospace")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase94_cross_axis_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


def main():
    print(f"\n{'='*80}")
    print(f"  Phase 94: Cross-Architecture Discriminant Axis Analysis")
    print(f"{'='*80}")

    t0 = time.time()

    # === Load all NPZ files ===
    # Qwen
    qwen_diff_path = os.path.join(RESULTS_DIR, "phase87_diff_pca.npz")
    qwen_std_path = os.path.join(RESULTS_DIR, "phase84_qwen_pca.npz")

    if not os.path.exists(qwen_diff_path):
        print(f"  ERROR: {qwen_diff_path} not found. Run Phase 87 first.")
        return
    if not os.path.exists(qwen_std_path):
        print(f"  ERROR: {qwen_std_path} not found. Run Phase 84 first.")
        return

    # Mistral
    mistral_diff_path = os.path.join(RESULTS_DIR, "phase91_diff_pca.npz")
    mistral_std_path = os.path.join(RESULTS_DIR, "phase86_mistral_pca.npz")

    if not os.path.exists(mistral_diff_path):
        print(f"  ERROR: {mistral_diff_path} not found. Run Phase 91 first.")
        return
    if not os.path.exists(mistral_std_path):
        print(f"  ERROR: {mistral_std_path} not found. Run Phase 86 first.")
        return

    print(f"  Loading Qwen Diff-PCA: {qwen_diff_path}")
    qwen_diff = np.load(qwen_diff_path)
    qwen_diff_unit = qwen_diff["diff_unit"]

    print(f"  Loading Qwen Standard PCA: {qwen_std_path}")
    qwen_std = np.load(qwen_std_path)
    qwen_std_Vt = qwen_std["Vt"]

    print(f"  Loading Mistral Diff-PCA: {mistral_diff_path}")
    mistral_diff = np.load(mistral_diff_path)
    mistral_diff_unit = mistral_diff["diff_unit"]

    print(f"  Loading Mistral Standard PCA: {mistral_std_path}")
    mistral_std = np.load(mistral_std_path)
    mistral_std_Vt = mistral_std["Vt"]

    print(f"\n  Dimensions: Qwen={len(qwen_diff_unit)}, Mistral={len(mistral_diff_unit)}")

    # === Analysis 1: Qwen ===
    qwen_overlap, qwen_cos = analyze_subspace_overlap(qwen_diff_unit, qwen_std_Vt, "Qwen")

    # === Analysis 2: Mistral ===
    mistral_overlap, mistral_cos = analyze_subspace_overlap(mistral_diff_unit, mistral_std_Vt, "Mistral")

    # === Analysis 3: Cross-architecture ===
    comparison = structural_comparison(qwen_overlap, mistral_overlap)

    # Visualization
    fig_path = visualize(qwen_cos, mistral_cos, qwen_overlap, mistral_overlap, comparison)

    # Save results
    elapsed = time.time() - t0
    all_results = {
        "experiment": "Phase 94: Cross-Architecture Discriminant Axis Analysis",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "qwen": {
            "diff_pca_source": "phase87_diff_pca.npz",
            "std_pca_source": "phase84_qwen_pca.npz",
            "hidden_dim": int(len(qwen_diff_unit)),
            "subspace_overlap": qwen_overlap,
        },
        "mistral": {
            "diff_pca_source": "phase91_diff_pca.npz",
            "std_pca_source": "phase86_mistral_pca.npz",
            "hidden_dim": int(len(mistral_diff_unit)),
            "subspace_overlap": mistral_overlap,
        },
        "cross_architecture_comparison": comparison,
        "figure": fig_path,
    }

    results_path = os.path.join(RESULTS_DIR, "phase94_log.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  Total elapsed: {elapsed:.1f} seconds")

    return all_results


if __name__ == "__main__":
    main()
    print(f"\n Phase 94 complete.")
