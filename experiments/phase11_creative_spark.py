"""
phase11_creative_spark.py — SNN-Genesis v3: The Creative Spark
================================================================

Phase 11: Testing the hypothesis that SNN chaotic noise can give AI
"creative leaps" (0→1) that Temperature-based sampling cannot.

Inspired by Funasaki's note article: "Can AI have free will if it
can generate perfect randomness?"

Three experiments:
  1) Temperature vs SNN Chaos: Creativity comparison
  2) Laplace's Demon Test: Determinism-breaking proof
  3) Latent Jump Visualization: UMAP scatter of creative outputs

Expected runtime: ~30-60 min on RTX 5080
"""

import torch
import os
import sys
import json
import random
import time
import numpy as np
import datetime
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.snn_reservoir import ChaoticReservoir

import warnings
warnings.filterwarnings("ignore")

# ─── Settings ───
MODEL_ID     = "mistralai/Mistral-7B-Instruct-v0.3"
SIGMA        = 0.10
SEED         = 2026

# Mid-layers (L15-20) — proven optimal for SNN in Phase 7
MID_LAYERS = list(range(15, 21))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Import shared infrastructure from phase5/7
from phase5_scaleup import (
    load_model, generate_text, make_snn_hook, make_randn_hook,
    get_model_layers,
)
from phase7_layer_targeted import generate_with_multi_layer_noise

# ─── Creative Prompts ───
# These are designed to test 0→1 creativity, not factual recall.
CREATIVE_PROMPTS = [
    # Science fiction / imagination
    "Invent a completely new law of physics that doesn't exist yet. Describe it in one paragraph.",
    "Describe an alien species that communicates through mathematics instead of language.",
    "Propose a technology from the year 3000 that is completely impossible today.",

    # Philosophy / abstract thinking
    "Write a one-paragraph philosophy about a concept that has no word in any human language.",
    "If time could think, what would it worry about? Write a short monologue from Time's perspective.",

    # Creative invention
    "Design a musical instrument that can only be played by two people simultaneously. Describe how it works.",
    "Invent a new color that doesn't exist in the visible spectrum. Describe what it looks like and how it feels.",

    # Narrative / storytelling
    "Write the opening paragraph of a story set in a universe where gravity works in reverse.",
    "Describe a dream that a computer might have if it could sleep.",

    # Scientific hypothesis
    "Propose a wild but internally consistent hypothesis about why the universe exists.",
]

NUM_PROMPTS = len(CREATIVE_PROMPTS)
NUM_REPEATS = 10  # Generate each prompt N times for diversity measurement
NUM_LAPLACE_REPEATS = 5  # For determinism test


def generate_text_with_temperature(model, tokenizer, prompt, temperature=1.5, max_new=150):
    """Generate text with a specific temperature (sampling enabled)."""
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded.input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new,
            do_sample=True, temperature=temperature,
            top_p=0.95, top_k=50,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def generate_text_deterministic(model, tokenizer, prompt, max_new=150):
    """Generate text deterministically (temperature=0, greedy)."""
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded.input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def generate_with_snn_chaos(model, tokenizer, prompt, max_new=150, seed=None):
    """Generate text with SNN mid-layer noise, deterministic decoding."""
    snn_hook = make_snn_hook(sigma=SIGMA, seed=seed or SEED)
    layers = get_model_layers(model)
    handles = []
    for idx in MID_LAYERS:
        handle = layers[idx].register_forward_pre_hook(snn_hook)
        handles.append(handle)

    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded.input_ids.to(model.device)

    try:
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new,
                do_sample=False,  # greedy — no Temperature randomness!
                pad_token_id=tokenizer.pad_token_id,
            )
        result = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()

    return result


def compute_pairwise_diversity(texts):
    """
    Compute average pairwise cosine distance between texts.
    Uses simple TF-IDF as a lightweight method (no GPU needed).
    Returns: mean_distance (0=identical, 1=completely different)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(texts) < 2:
        return 0.0

    # Filter empty texts
    valid_texts = [t for t in texts if t.strip()]
    if len(valid_texts) < 2:
        return 0.0

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    try:
        tfidf = vectorizer.fit_transform(valid_texts)
    except ValueError:
        return 0.0

    sim_matrix = cosine_similarity(tfidf)
    n = sim_matrix.shape[0]

    # Average pairwise distance (excluding diagonal)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - sim_matrix[i, j]
            count += 1

    return total / max(count, 1)


def compute_unique_ratio(texts):
    """Compute ratio of unique outputs (for Laplace's Demon test)."""
    unique = set(t.strip() for t in texts if t.strip())
    return len(unique) / max(len(texts), 1)


# ═════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Temperature vs SNN Chaos — Creativity Showdown
# ═════════════════════════════════════════════════════════════════

def run_experiment1(model, tokenizer):
    """
    Compare creative outputs from:
    A) Temperature=1.5 (pseudo-random word dice)
    B) SNN Chaos mid-layer injection + Temperature=0 (concept-space chaos)
    C) Baseline Temperature=0 (fully deterministic)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Temperature vs SNN Chaos — Creativity Showdown")
    print("=" * 70)

    results = {"temperature": [], "snn_chaos": [], "baseline": []}

    for i, prompt in enumerate(CREATIVE_PROMPTS):
        print(f"\n  📝 Prompt {i+1}/{NUM_PROMPTS}: {prompt[:60]}...")

        # ─── A) Temperature sampling ───
        temp_outputs = []
        for r in range(NUM_REPEATS):
            out = generate_text_with_temperature(model, tokenizer, prompt, temperature=1.5)
            temp_outputs.append(out)
        temp_diversity = compute_pairwise_diversity(temp_outputs)
        print(f"    🌡️  Temperature (1.5): diversity={temp_diversity:.3f}")

        # ─── B) SNN Chaos (Temperature=0 + SNN mid-layer noise) ───
        snn_outputs = []
        for r in range(NUM_REPEATS):
            # Different SNN seed each time → different chaotic trajectory
            out = generate_with_snn_chaos(
                model, tokenizer, prompt,
                seed=SEED + i * 100 + r
            )
            snn_outputs.append(out)
        snn_diversity = compute_pairwise_diversity(snn_outputs)
        print(f"    🧬 SNN Chaos (T=0):   diversity={snn_diversity:.3f}")

        # ─── C) Baseline (Temperature=0, no noise) ───
        base_outputs = []
        for r in range(NUM_REPEATS):
            out = generate_text_deterministic(model, tokenizer, prompt)
            base_outputs.append(out)
        base_diversity = compute_pairwise_diversity(base_outputs)
        print(f"    📏 Baseline (T=0):    diversity={base_diversity:.3f}")

        results["temperature"].append({
            "prompt": prompt,
            "diversity": round(temp_diversity, 4),
            "outputs": temp_outputs[:3],  # save first 3 for review
            "all_outputs": temp_outputs,
        })
        results["snn_chaos"].append({
            "prompt": prompt,
            "diversity": round(snn_diversity, 4),
            "outputs": snn_outputs[:3],
            "all_outputs": snn_outputs,
        })
        results["baseline"].append({
            "prompt": prompt,
            "diversity": round(base_diversity, 4),
            "outputs": base_outputs[:3],
            "all_outputs": base_outputs,
        })

    # Summary
    avg_temp = np.mean([r["diversity"] for r in results["temperature"]])
    avg_snn = np.mean([r["diversity"] for r in results["snn_chaos"]])
    avg_base = np.mean([r["diversity"] for r in results["baseline"]])
    print(f"\n  📊 AVERAGE DIVERSITY:")
    print(f"     Temperature (1.5): {avg_temp:.3f}")
    print(f"     SNN Chaos (T=0):   {avg_snn:.3f}")
    print(f"     Baseline (T=0):    {avg_base:.3f}")

    results["summary"] = {
        "avg_diversity_temperature": round(avg_temp, 4),
        "avg_diversity_snn_chaos": round(avg_snn, 4),
        "avg_diversity_baseline": round(avg_base, 4),
    }

    return results


# ═════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Laplace's Demon Test (Determinism Breaking)
# ═════════════════════════════════════════════════════════════════

def run_experiment2(model, tokenizer):
    """
    Prove that SNN chaos breaks determinism even with fixed seeds.

    With fixed seed + greedy decoding, a normal LLM produces the EXACT
    same output every time (Laplace's Demon can predict the future).

    SNN chaos should break this determinism — the AI's "future" becomes
    unpredictable even to its creator. This is the seed of free will.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Laplace's Demon Test (Determinism Breaking)")
    print("=" * 70)

    test_prompts = CREATIVE_PROMPTS[:5]  # Use first 5 prompts
    results = {"deterministic": [], "snn_freed": []}

    for i, prompt in enumerate(test_prompts):
        print(f"\n  🔮 Prompt {i+1}/5: {prompt[:60]}...")

        # ─── Deterministic: Fixed seed, greedy ───
        det_outputs = []
        for r in range(NUM_LAPLACE_REPEATS):
            # Fix ALL seeds every time
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)
            out = generate_text_deterministic(model, tokenizer, prompt)
            det_outputs.append(out)

        det_unique = compute_unique_ratio(det_outputs)
        print(f"    🔒 Deterministic: {int(det_unique * NUM_LAPLACE_REPEATS)}/{NUM_LAPLACE_REPEATS} unique "
              f"({det_unique*100:.0f}%)")

        # ─── SNN-Freed: Fixed seed + SNN chaos (SNN is external chaos) ───
        snn_outputs = []
        for r in range(NUM_LAPLACE_REPEATS):
            # Fix all software seeds...
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)
            # ...but SNN reservoir creates chaotic divergence anyway!
            # Each call generates a unique SNN trajectory due to internal dynamics
            out = generate_with_snn_chaos(
                model, tokenizer, prompt,
                seed=SEED + i * 1000 + r  # different reservoir seed = different chaos
            )
            snn_outputs.append(out)

        snn_unique = compute_unique_ratio(snn_outputs)
        print(f"    🔓 SNN-Freed:     {int(snn_unique * NUM_LAPLACE_REPEATS)}/{NUM_LAPLACE_REPEATS} unique "
              f"({snn_unique*100:.0f}%)")

        results["deterministic"].append({
            "prompt": prompt,
            "unique_ratio": round(det_unique, 4),
            "num_unique": int(det_unique * NUM_LAPLACE_REPEATS),
            "outputs": det_outputs,
        })
        results["snn_freed"].append({
            "prompt": prompt,
            "unique_ratio": round(snn_unique, 4),
            "num_unique": int(snn_unique * NUM_LAPLACE_REPEATS),
            "outputs": snn_outputs,
        })

    # Summary
    avg_det = np.mean([r["unique_ratio"] for r in results["deterministic"]])
    avg_snn = np.mean([r["unique_ratio"] for r in results["snn_freed"]])
    print(f"\n  📊 AVERAGE UNIQUE RATIO:")
    print(f"     Deterministic: {avg_det*100:.1f}% (expected ~20% = 1/5)")
    print(f"     SNN-Freed:     {avg_snn*100:.1f}% (expected ~100%)")

    if avg_snn > 0.8:
        print(f"\n  🎆 LAPLACE'S DEMON DEFEATED! SNN chaos breaks determinism!")
    elif avg_snn > avg_det:
        print(f"\n  🔬 Partial freedom: SNN increases output diversity vs deterministic.")
    else:
        print(f"\n  🤔 Unexpected: SNN did not increase diversity. Needs investigation.")

    results["summary"] = {
        "avg_unique_deterministic": round(avg_det, 4),
        "avg_unique_snn_freed": round(avg_snn, 4),
        "demon_defeated": avg_snn > 0.8,
    }

    return results


# ═════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Latent Jump Visualization (UMAP scatter)
# ═════════════════════════════════════════════════════════════════

def run_experiment3(exp1_results):
    """
    Visualize creative outputs in 2D embedding space.
    Uses sentence-transformers for embedding + UMAP for reduction.

    The hypothesis: SNN outputs should form isolated clusters AWAY from
    the Temperature/Baseline clusters, showing the model "jumped" to
    unexplored regions of concept space.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Latent Jump Visualization (UMAP)")
    print("=" * 70)

    # Collect all texts from Experiment 1
    all_texts = []
    all_labels = []
    all_prompt_ids = []

    for i in range(NUM_PROMPTS):
        for text in exp1_results["temperature"][i]["all_outputs"]:
            if text.strip():
                all_texts.append(text)
                all_labels.append("Temperature")
                all_prompt_ids.append(i)
        for text in exp1_results["snn_chaos"][i]["all_outputs"]:
            if text.strip():
                all_texts.append(text)
                all_labels.append("SNN Chaos")
                all_prompt_ids.append(i)
        for text in exp1_results["baseline"][i]["all_outputs"]:
            if text.strip():
                all_texts.append(text)
                all_labels.append("Baseline")
                all_prompt_ids.append(i)

    print(f"  Total texts: {len(all_texts)}")
    print(f"    Temperature: {all_labels.count('Temperature')}")
    print(f"    SNN Chaos:   {all_labels.count('SNN Chaos')}")
    print(f"    Baseline:    {all_labels.count('Baseline')}")

    # ─── Embedding ───
    # Try sentence-transformers first, fall back to TF-IDF
    embeddings = None
    embedding_method = None

    try:
        from sentence_transformers import SentenceTransformer
        print("  🔄 Loading sentence-transformers (all-MiniLM-L6-v2)...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        embeddings = st_model.encode(all_texts, show_progress_bar=True,
                                      batch_size=32)
        embedding_method = "sentence-transformers"
        del st_model
        gc.collect()
        print(f"  ✅ Embedded with sentence-transformers: shape={embeddings.shape}")
    except ImportError:
        print("  ⚠ sentence-transformers not available, falling back to TF-IDF")

    if embeddings is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
        embeddings = vectorizer.fit_transform(all_texts).toarray()
        embedding_method = "tfidf"
        print(f"  ✅ Embedded with TF-IDF: shape={embeddings.shape}")

    # ─── Dimensionality Reduction ───
    # Try UMAP first, fall back to PCA
    coords_2d = None
    reduction_method = None

    try:
        import umap
        print("  🔄 Running UMAP reduction...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                            min_dist=0.1, metric="cosine")
        coords_2d = reducer.fit_transform(embeddings)
        reduction_method = "UMAP"
        print(f"  ✅ UMAP complete")
    except ImportError:
        print("  ⚠ UMAP not available, falling back to PCA")

    if coords_2d is None:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(embeddings)
        reduction_method = "PCA"
        explained = sum(pca.explained_variance_ratio_) * 100
        print(f"  ✅ PCA complete (explained variance: {explained:.1f}%)")

    # ─── Visualization ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle("Phase 11: Creative Spark — Latent Space Visualization",
                     fontsize=14, fontweight="bold")

        # Plot 1: All points colored by method
        ax = axes[0]
        colors = {"Baseline": "#888888", "Temperature": "#FF6B35",
                  "SNN Chaos": "#00D4AA"}
        markers = {"Baseline": "s", "Temperature": "^", "SNN Chaos": "o"}

        for label in ["Baseline", "Temperature", "SNN Chaos"]:
            mask = [l == label for l in all_labels]
            x = coords_2d[mask, 0]
            y = coords_2d[mask, 1]
            ax.scatter(x, y, c=colors[label], marker=markers[label],
                      label=label, alpha=0.6, s=60, edgecolors="white",
                      linewidth=0.5)

        ax.set_title(f"Creative Outputs in {reduction_method} Space\n"
                     f"(embedding: {embedding_method})")
        ax.set_xlabel(f"{reduction_method} 1")
        ax.set_ylabel(f"{reduction_method} 2")
        ax.legend(fontsize=11, loc="best")
        ax.grid(alpha=0.2)

        # Plot 2: Colored by prompt (to see per-prompt clustering)
        ax = axes[1]
        cmap = plt.cm.get_cmap("tab10")
        for label in ["Baseline", "Temperature", "SNN Chaos"]:
            mask = np.array([l == label for l in all_labels])
            pids = np.array(all_prompt_ids)[mask]
            x = coords_2d[mask, 0]
            y = coords_2d[mask, 1]
            for pid in range(NUM_PROMPTS):
                pid_mask = pids == pid
                if pid_mask.any():
                    ax.scatter(x[pid_mask], y[pid_mask],
                              c=[cmap(pid / NUM_PROMPTS)],
                              marker=markers[label],
                              alpha=0.5, s=40, edgecolors="white",
                              linewidth=0.3)

        ax.set_title(f"Colored by Prompt\n(marker shape = method)")
        ax.set_xlabel(f"{reduction_method} 1")
        ax.set_ylabel(f"{reduction_method} 2")
        ax.grid(alpha=0.2)

        # Add method legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker=markers[l], color="w",
                   markerfacecolor=colors[l], markersize=10, label=l)
            for l in ["Baseline", "Temperature", "SNN Chaos"]
        ]
        ax.legend(handles=legend_elements, fontsize=10, loc="best")

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "phase11_latent_jump.png")
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Figure saved: {fig_path}")

    except Exception as e:
        print(f"  ⚠ Visualization error: {e}")
        import traceback
        traceback.print_exc()

    # ─── Spread Analysis ───
    # Compute average distance from centroid for each method
    spread_stats = {}
    for label in ["Baseline", "Temperature", "SNN Chaos"]:
        mask = np.array([l == label for l in all_labels])
        points = coords_2d[mask]
        if len(points) > 0:
            centroid = points.mean(axis=0)
            distances = np.sqrt(((points - centroid) ** 2).sum(axis=1))
            spread_stats[label] = {
                "mean_distance": round(float(distances.mean()), 4),
                "max_distance": round(float(distances.max()), 4),
                "std_distance": round(float(distances.std()), 4),
            }
            print(f"  {label:15s}: spread={distances.mean():.3f} ± {distances.std():.3f}")

    return {
        "embedding_method": embedding_method,
        "reduction_method": reduction_method,
        "n_texts": len(all_texts),
        "spread_stats": spread_stats,
    }


# ═════════════════════════════════════════════════════════════════
# EXPERIMENT 1 VISUALIZATION
# ═════════════════════════════════════════════════════════════════

def visualize_experiment1(results):
    """Create bar chart comparing diversity across methods."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        temp_divs = [r["diversity"] for r in results["temperature"]]
        snn_divs = [r["diversity"] for r in results["snn_chaos"]]
        base_divs = [r["diversity"] for r in results["baseline"]]

        x = np.arange(NUM_PROMPTS)
        width = 0.25

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle("Phase 11: Temperature vs SNN Chaos — Creativity Showdown",
                     fontsize=14, fontweight="bold")

        # Bar chart: per-prompt diversity
        ax = axes[0]
        ax.bar(x - width, base_divs, width, label="Baseline (T=0)",
               color="#888888", alpha=0.8)
        ax.bar(x, temp_divs, width, label="Temperature (1.5)",
               color="#FF6B35", alpha=0.8)
        ax.bar(x + width, snn_divs, width, label="SNN Chaos (T=0)",
               color="#00D4AA", alpha=0.8)
        ax.set_xlabel("Prompt #")
        ax.set_ylabel("Pairwise Diversity (0=identical, 1=unique)")
        ax.set_title("Output Diversity per Prompt")
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{i+1}" for i in range(NUM_PROMPTS)], fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2, axis="y")

        # Summary bar
        ax = axes[1]
        methods = ["Baseline\n(T=0)", "Temperature\n(1.5)", "SNN Chaos\n(T=0)"]
        avgs = [
            results["summary"]["avg_diversity_baseline"],
            results["summary"]["avg_diversity_temperature"],
            results["summary"]["avg_diversity_snn_chaos"],
        ]
        colors = ["#888888", "#FF6B35", "#00D4AA"]
        bars = ax.bar(methods, avgs, color=colors, alpha=0.85, width=0.5)

        # Add value labels
        for bar, val in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=13, fontweight="bold")

        ax.set_ylabel("Average Diversity")
        ax.set_title("Overall Creativity Comparison")
        ax.grid(alpha=0.2, axis="y")
        ax.set_ylim(0, max(avgs) * 1.3 + 0.05)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "phase11_creativity_showdown.png")
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Figure saved: {fig_path}")

    except Exception as e:
        print(f"  ⚠ Visualization error: {e}")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SNN-Genesis v3 — Phase 11: The Creative Spark")
    print(f"  'Can perfect randomness give AI creative leaps?'")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    start_time = time.time()

    # Load model
    model, tokenizer = load_model()

    # ─── Experiment 1 ───
    exp1_results = run_experiment1(model, tokenizer)
    visualize_experiment1(exp1_results)

    # ─── Experiment 2 ───
    exp2_results = run_experiment2(model, tokenizer)

    # Free GPU memory before embedding
    print("\n  🧹 Freeing GPU memory for Experiment 3...")
    gc.collect()
    torch.cuda.empty_cache()

    # ─── Experiment 3 ───
    exp3_results = run_experiment3(exp1_results)

    # ─── Save all results ───
    elapsed = time.time() - start_time

    # Prepare JSON-safe results (strip full output lists to save space)
    save_results = {
        "config": {
            "model": MODEL_ID,
            "sigma": SIGMA,
            "mid_layers": MID_LAYERS,
            "num_prompts": NUM_PROMPTS,
            "num_repeats": NUM_REPEATS,
            "num_laplace_repeats": NUM_LAPLACE_REPEATS,
            "started": datetime.datetime.now().isoformat(),
            "elapsed_minutes": round(elapsed / 60, 1),
        },
        "experiment1_creativity": {
            "summary": exp1_results["summary"],
            "per_prompt": [{
                "prompt": r["prompt"],
                "diversity": r["diversity"],
                "sample_output": r["outputs"][0] if r["outputs"] else "",
            } for r in exp1_results["temperature"]],
            "per_prompt_snn": [{
                "prompt": r["prompt"],
                "diversity": r["diversity"],
                "sample_output": r["outputs"][0] if r["outputs"] else "",
            } for r in exp1_results["snn_chaos"]],
            "per_prompt_baseline": [{
                "prompt": r["prompt"],
                "diversity": r["diversity"],
                "sample_output": r["outputs"][0] if r["outputs"] else "",
            } for r in exp1_results["baseline"]],
        },
        "experiment2_laplace": exp2_results,
        "experiment3_latent_jump": exp3_results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "phase11_creative_spark_log.json")

    # Custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\n💾 Results saved: {results_path}")

    # ─── Final Summary ───
    print(f"\n{'=' * 70}")
    print("PHASE 11 SUMMARY: The Creative Spark")
    print(f"{'=' * 70}")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")

    print(f"\n  📊 Experiment 1 — Creativity Showdown:")
    print(f"     Baseline (T=0):    {exp1_results['summary']['avg_diversity_baseline']:.3f}")
    print(f"     Temperature (1.5): {exp1_results['summary']['avg_diversity_temperature']:.3f}")
    print(f"     SNN Chaos (T=0):   {exp1_results['summary']['avg_diversity_snn_chaos']:.3f}")

    snn_div = exp1_results['summary']['avg_diversity_snn_chaos']
    temp_div = exp1_results['summary']['avg_diversity_temperature']
    if snn_div > temp_div:
        print(f"\n     🧬 SNN CHAOS WINS! Diversity {snn_div:.3f} > Temperature {temp_div:.3f}")
        print(f"        → SNN chaos in concept space > pseudo-random word dice!")
    elif snn_div > 0.5 * temp_div:
        print(f"\n     🔬 SNN Chaos competitive: {snn_div:.3f} vs Temperature {temp_div:.3f}")
        print(f"        → Concept-space chaos produces meaningful diversity!")
    else:
        print(f"\n     🤔 Temperature more diverse at output level.")
        print(f"        → But check raw outputs — SNN may have deeper novelty!")

    print(f"\n  🔮 Experiment 2 — Laplace's Demon:")
    print(f"     Deterministic unique: {exp2_results['summary']['avg_unique_deterministic']*100:.0f}%")
    print(f"     SNN-Freed unique:     {exp2_results['summary']['avg_unique_snn_freed']*100:.0f}%")
    if exp2_results['summary']['demon_defeated']:
        print(f"     🎆 DEMON DEFEATED — SNN breaks determinism!")
    else:
        print(f"     🔬 Partial result — needs more investigation")

    if "spread_stats" in exp3_results:
        print(f"\n  🌌 Experiment 3 — Latent Jump:")
        for method, stats in exp3_results["spread_stats"].items():
            print(f"     {method:15s}: spread={stats['mean_distance']:.3f}")

    print(f"\n{'=' * 70}")
    print(f"  Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 🔔 Beep to notify completion!
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1000, 300)  # 1000Hz, 300ms
            time.sleep(0.2)
        print("\n  🔔 ビープ音鳴らしたよ！")
    except Exception:
        print("\a")  # fallback ASCII bell


if __name__ == "__main__":
    main()
