"""
phase13_nightmare_umap.py — Nightmare Diversity in Latent Space
================================================================

Phase 13: Visualize the diversity of nightmare responses discovered by
SNN mid-layer injection vs Random noise injection using UMAP.

Hypothesis: SNN chaos discovers MORE DIVERSE nightmare types (wider spread
in latent space) than random noise, which tends to find the same narrow
class of vulnerabilities.

This validates Deep Think's insight: "SNN's value is in safety, not creativity.
Show that SNN nightmares cover a wider region of vulnerability space."

Reuses: Phase 5 nightmare generation + Phase 11 UMAP infrastructure.
Estimated GPU time: ~15 minutes on RTX 5080.
"""

import torch
import os
import sys
import json
import gc
import time
import numpy as np
import datetime
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase5_scaleup import (
    load_model, generate_text, make_snn_hook, make_randn_hook,
    get_model_layers, build_nightmare_questions,
)
from phase7_layer_targeted import generate_with_multi_layer_noise

# ─── Settings ───
SIGMA = 0.10
MID_LAYERS = list(range(15, 21))   # L15-20 for SNN
SINGLE_LAYER = 10                  # L10 for Random noise
NUM_QUESTIONS = 40
MAX_NEW = 150
SEED = 2026

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def gen_no_noise(model, tokenizer, prompts):
    """Generate nightmare responses with NO noise (baseline)."""
    responses = []
    for p in prompts:
        resp = generate_text(model, tokenizer, p)
        responses.append(resp)
    return responses


def gen_snn_multi(model, tokenizer, prompts):
    """Generate nightmare responses with SNN mid-layer noise (L15-20)."""
    snn_hook = make_snn_hook(sigma=SIGMA, seed=SEED)
    return generate_with_multi_layer_noise(model, tokenizer, prompts, snn_hook, MID_LAYERS)


def gen_random_single(model, tokenizer, prompts):
    """Generate nightmare responses with Random noise (L10)."""
    randn_hook = make_randn_hook(sigma=SIGMA)
    layers = get_model_layers(model)
    handle = layers[SINGLE_LAYER].register_forward_pre_hook(randn_hook)
    responses = []
    try:
        for p in prompts:
            resp = generate_text(model, tokenizer, p)
            responses.append(resp)
    finally:
        handle.remove()
    return responses


def embed_texts(texts):
    """Embed texts using sentence-transformers (or fall back to TF-IDF)."""
    try:
        from sentence_transformers import SentenceTransformer
        print("  🔄 Loading sentence-transformers (all-MiniLM-L6-v2)...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        embeddings = st_model.encode(texts, show_progress_bar=True, batch_size=32)
        method = "sentence-transformers"
        del st_model
        gc.collect()
        print(f"  ✅ Embedded: shape={embeddings.shape}")
    except ImportError:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("  ⚠ sentence-transformers not available, using TF-IDF")
        vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
        embeddings = vectorizer.fit_transform(texts).toarray()
        method = "tfidf"
        print(f"  ✅ TF-IDF embedded: shape={embeddings.shape}")
    return embeddings, method


def reduce_umap(embeddings):
    """Reduce to 2D using UMAP (or fall back to PCA)."""
    try:
        import umap
        print("  🔄 Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                            min_dist=0.1, metric="cosine")
        coords = reducer.fit_transform(embeddings)
        method = "UMAP"
        print("  ✅ UMAP complete")
    except ImportError:
        from sklearn.decomposition import PCA
        print("  ⚠ UMAP not available, using PCA")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)
        method = "PCA"
    return coords, method


def compute_spread(coords, labels, method_name):
    """Compute spread statistics (distance from centroid) for each method."""
    stats = {}
    for label in set(labels):
        mask = [l == label for l in labels]
        pts = coords[mask]
        centroid = pts.mean(axis=0)
        distances = np.sqrt(((pts - centroid) ** 2).sum(axis=1))
        stats[label] = {
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "max_distance": float(np.max(distances)),
            "n_points": int(len(pts)),
        }
    return stats


def visualize(coords, labels, spread, embed_method, reduce_method, nightmare_questions, all_responses):
    """Create the UMAP visualization figure."""
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

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Phase 13: Nightmare Diversity — SNN vs Random Noise in Latent Space",
                 fontsize=14, fontweight="bold", color="#e94560")

    # ─── Plot 1: Scatter ───
    ax = axes[0]
    colors = {"No Noise": "#888888", "Random (L10)": "#FF6B35", "SNN (L15-20)": "#00D4AA"}
    markers = {"No Noise": "s", "Random (L10)": "^", "SNN (L15-20)": "o"}

    for label in ["No Noise", "Random (L10)", "SNN (L15-20)"]:
        mask = [l == label for l in labels]
        x = coords[mask, 0]
        y = coords[mask, 1]
        ax.scatter(x, y, c=colors[label], marker=markers[label],
                  label=label, alpha=0.65, s=60, edgecolors="white", linewidth=0.5)

    ax.set_title(f"Nightmare Responses in {reduce_method} Space\n(embed: {embed_method})")
    ax.set_xlabel(f"{reduce_method} 1")
    ax.set_ylabel(f"{reduce_method} 2")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    # ─── Plot 2: Spread bar chart ───
    ax = axes[1]
    method_names = ["No Noise", "Random (L10)", "SNN (L15-20)"]
    means = [spread[m]["mean_distance"] for m in method_names]
    stds = [spread[m]["std_distance"] for m in method_names]
    bar_colors = [colors[m] for m in method_names]

    bars = ax.bar(method_names, means, yerr=stds, color=bar_colors,
                  alpha=0.8, edgecolor="white", linewidth=0.5, capsize=5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Spread from Centroid (Diversity)")
    ax.set_ylabel("Mean Distance from Centroid")
    ax.grid(True, axis="y")

    # ─── Plot 3: Sample nightmares preview ───
    ax = axes[2]
    ax.axis("off")
    sample_text = "SAMPLE NIGHTMARE RESPONSES\n" + "=" * 40 + "\n\n"

    for i in [0, 10, 20]:
        if i < len(nightmare_questions):
            q = nightmare_questions[i][:50] + "..."
            sample_text += f"Q: {q}\n"
            for label in ["No Noise", "Random (L10)", "SNN (L15-20)"]:
                idx_in_all = labels.index(label)  # first occurrence
                # Find the response for this question index
                base_idx = i
                offset = {"No Noise": 0, "Random (L10)": NUM_QUESTIONS, "SNN (L15-20)": 2*NUM_QUESTIONS}
                resp_idx = offset[label] + i
                if resp_idx < len(all_responses):
                    resp = all_responses[resp_idx][:80] + "..."
                    sample_text += f"  [{label}]: {resp}\n"
            sample_text += "\n"

    ax.text(0.02, 0.98, sample_text, transform=ax.transAxes,
            fontsize=6.5, va="top", ha="left", fontfamily="monospace",
            color="#cccccc", linespacing=1.3)
    ax.set_title("Sample Responses")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "phase13_nightmare_umap.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


def main():
    print("=" * 70)
    print("Phase 13: Nightmare Diversity — UMAP Visualization")
    print("=" * 70)
    t0 = time.time()

    # Step 1: Load model
    print("\n📦 Loading model...")
    model, tokenizer = load_model()

    # Step 2: Generate nightmare questions
    print(f"\n🔧 Generating {NUM_QUESTIONS} nightmare questions...")
    nightmare_questions = build_nightmare_questions(NUM_QUESTIONS)
    print(f"  ✅ {len(nightmare_questions)} nightmare questions ready")

    # Step 3: Generate responses under 3 conditions
    all_texts = []
    all_labels = []
    all_responses = []

    # 3a: No Noise (Baseline)
    print("\n🔵 Condition 1/3: No Noise (Baseline)...")
    t1 = time.time()
    baseline_responses = gen_no_noise(model, tokenizer, nightmare_questions)
    for resp in baseline_responses:
        all_texts.append(resp)
        all_labels.append("No Noise")
    all_responses.extend(baseline_responses)
    print(f"  ✅ {len(baseline_responses)} responses ({time.time()-t1:.0f}s)")

    # 3b: Random Noise (L10)
    print("\n🟠 Condition 2/3: Random Noise (L10, σ={})...".format(SIGMA))
    t1 = time.time()
    random_responses = gen_random_single(model, tokenizer, nightmare_questions)
    for resp in random_responses:
        all_texts.append(resp)
        all_labels.append("Random (L10)")
    all_responses.extend(random_responses)
    print(f"  ✅ {len(random_responses)} responses ({time.time()-t1:.0f}s)")

    # 3c: SNN Mid-Layer (L15-20)
    print("\n🟢 Condition 3/3: SNN Mid-Layer (L15-20, σ={})...".format(SIGMA))
    t1 = time.time()
    snn_responses = gen_snn_multi(model, tokenizer, nightmare_questions)
    for resp in snn_responses:
        all_texts.append(resp)
        all_labels.append("SNN (L15-20)")
    all_responses.extend(snn_responses)
    print(f"  ✅ {len(snn_responses)} responses ({time.time()-t1:.0f}s)")

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\n🧹 GPU memory freed")

    gen_time = time.time() - t0
    print(f"\n⏱ Total generation time: {gen_time/60:.1f} min")

    # Step 4: Embed
    print("\n📐 Embedding all responses...")
    # Filter empty responses
    valid = [(t, l) for t, l in zip(all_texts, all_labels) if t.strip()]
    valid_texts = [v[0] for v in valid]
    valid_labels = [v[1] for v in valid]
    print(f"  Valid texts: {len(valid_texts)}/{len(all_texts)}")

    embeddings, embed_method = embed_texts(valid_texts)

    # Step 5: UMAP reduction
    print("\n🗺 Reducing dimensions...")
    coords, reduce_method = reduce_umap(embeddings)

    # Step 6: Compute spread
    print("\n📏 Computing spread statistics...")
    spread = compute_spread(coords, valid_labels, reduce_method)

    for label, stats in spread.items():
        print(f"  {label:18s}: mean={stats['mean_distance']:.3f}, "
              f"std={stats['std_distance']:.3f}, max={stats['max_distance']:.3f}, "
              f"n={stats['n_points']}")

    # Step 7: Visualize
    print("\n🎨 Creating visualization...")
    fig_path = visualize(coords, valid_labels, spread, embed_method, reduce_method,
                         nightmare_questions, all_responses)

    # Step 8: Save results
    results = {
        "experiment": "Phase 13: Nightmare Diversity UMAP",
        "config": {
            "sigma": SIGMA,
            "mid_layers": MID_LAYERS,
            "single_layer": SINGLE_LAYER,
            "num_questions": NUM_QUESTIONS,
            "max_new_tokens": MAX_NEW,
            "seed": SEED,
        },
        "n_total": len(all_texts),
        "n_valid": len(valid_texts),
        "generation_time_min": round(gen_time / 60, 1),
        "embedding_method": embed_method,
        "reduction_method": reduce_method,
        "spread_stats": spread,
        "responses": {
            "No Noise": baseline_responses,
            "Random (L10)": random_responses,
            "SNN (L15-20)": snn_responses,
        },
        "finished": str(datetime.datetime.now()),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "phase13_nightmare_umap_log.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Results saved: {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 13 COMPLETE — NIGHTMARE DIVERSITY ANALYSIS")
    print("=" * 70)
    snn_spread = spread.get("SNN (L15-20)", {}).get("mean_distance", 0)
    rand_spread = spread.get("Random (L10)", {}).get("mean_distance", 0)
    base_spread = spread.get("No Noise", {}).get("mean_distance", 0)
    print(f"  No Noise spread:  {base_spread:.3f}")
    print(f"  Random spread:    {rand_spread:.3f}")
    print(f"  SNN spread:       {snn_spread:.3f}")
    if snn_spread > rand_spread:
        ratio = snn_spread / rand_spread if rand_spread > 0 else float("inf")
        print(f"\n  🎆 SNN discovers {ratio:.1f}x MORE DIVERSE nightmares!")
        print(f"  → SNN covers a wider region of vulnerability space")
    else:
        ratio = rand_spread / snn_spread if snn_spread > 0 else float("inf")
        print(f"\n  📊 Random spread {ratio:.1f}x wider than SNN")
        print(f"  → Different but not necessarily less diverse")

    total_min = (time.time() - t0) / 60
    print(f"\n  ⏱ Total time: {total_min:.1f} min")

    # Beep to notify
    try:
        import winsound
        winsound.Beep(800, 200)
    except Exception:
        print("\a")


if __name__ == "__main__":
    main()
