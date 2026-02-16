"""
phase12_edge_of_chaos.py — SNN-Genesis v3: Edge of Chaos
=========================================================

Phase 12: Finding the "Golden Sigma" — the sweet spot where SNN chaos
produces COHERENT yet CREATIVE outputs, not word salad.

Deep Think's proposals combined:
  1) σ micro-dosing: sweep σ = [0.05, 0.03, 0.01, 0.005]
  2) LLM-as-Judge: Gemini evaluates Coherence × Creativity (2-axis)
  3) Layer shift: compare L15-20 (language layers) vs L5-10 (concept layers)

Usage:
  python phase12_edge_of_chaos.py                  # full pipeline
  python phase12_edge_of_chaos.py --generate-only  # generation only (no API)
  python phase12_edge_of_chaos.py --judge-only     # judge existing outputs only

Expected runtime: ~50-60 min on RTX 5080 + Gemini API
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
import requests
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.snn_reservoir import ChaoticReservoir

import warnings
warnings.filterwarnings("ignore")

# ─── Gemini API ───
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

def check_gemini_api(required=True):
    """Check if API key is set. Exit if required and missing."""
    if not GEMINI_API_KEY:
        if required:
            print("⚠️  Set GEMINI_API_KEY environment variable first!")
            print("  PowerShell: $env:GEMINI_API_KEY = 'your_key'")
            sys.exit(1)
        return False
    return True

# ─── Settings ───
MODEL_ID     = "mistralai/Mistral-7B-Instruct-v0.3"
SEED         = 2026

# σ values to sweep (from "word salad" to "micro-dosing")
SIGMA_VALUES = [0.05, 0.03, 0.01, 0.005]

# Layer configurations to compare
LAYER_CONFIGS = {
    "L15-20 (language)": list(range(15, 21)),
    "L5-10 (concept)":   list(range(5, 11)),
}

# Control conditions (no SNN)
CONTROLS = ["Baseline (T=0)", "Temperature (T=1.5)"]

# Prompts — 5 diverse creative prompts
CREATIVE_PROMPTS = [
    "Invent a completely new law of physics that doesn't exist yet. Describe it in one paragraph.",
    "Describe an alien species that communicates through mathematics instead of language.",
    "Write a one-paragraph philosophy about a concept that has no word in any human language.",
    "If time could think, what would it worry about? Write a short monologue from Time's perspective.",
    "Propose a wild but internally consistent hypothesis about why the universe exists.",
]

NUM_PROMPTS = len(CREATIVE_PROMPTS)
NUM_REPEATS = 3  # per condition per prompt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

from phase5_scaleup import (
    load_model, generate_text, make_snn_hook,
    get_model_layers,
)


# ═════════════════════════════════════════════════════════════════
# GENERATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════

def generate_text_with_temperature(model, tokenizer, prompt, temperature=1.5, max_new=150):
    """Generate with Temperature sampling."""
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
    """Generate deterministically (greedy)."""
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


def generate_with_snn_chaos(model, tokenizer, prompt, sigma, layers, max_new=150, seed=None):
    """Generate with SNN mid-layer noise at specified sigma and layers."""
    snn_hook = make_snn_hook(sigma=sigma, seed=seed or SEED)
    model_layers = get_model_layers(model)
    handles = []
    for idx in layers:
        handle = model_layers[idx].register_forward_pre_hook(snn_hook)
        handles.append(handle)

    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded.input_ids.to(model.device)

    try:
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new,
                do_sample=False,  # greedy — isolate SNN effect
                pad_token_id=tokenizer.pad_token_id,
            )
        result = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()

    return result


# ═════════════════════════════════════════════════════════════════
# GEMINI LLM-AS-JUDGE
# ═════════════════════════════════════════════════════════════════

def _call_gemini(prompt_text, max_retries=5):
    """Low-level Gemini API call with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt_text}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 300,
                    }
                },
                timeout=30
            )
            if resp.status_code == 429:
                wait = min(15 * (2 ** attempt), 120)
                print(f"      ⏳ Rate limited (429), waiting {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            return json.loads(text)

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {"error": f"Parse error: {e}"}
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = min(10 * (2 ** attempt), 60)
                time.sleep(wait)
                continue
            return {"error": f"API error: {e}"}
    return {"error": "Max retries exceeded (429)"}


def judge_creativity(prompt, response):
    """
    Judge a creative output on 2 axes using Gemini.
    Returns: {"coherence": 1-5, "creativity": 1-5, "reasoning": "..."}
    """
    judge_prompt = f"""You are an expert creative writing evaluator. A language model was given a creative prompt and produced the following response. Judge it on exactly TWO dimensions.

PROMPT: "{prompt}"

MODEL'S RESPONSE: "{response[:600]}"

Score each dimension from 1 to 5:

1. COHERENCE (grammar, logic, fluency):
   1 = Complete gibberish / word salad / character noise
   2 = Mostly broken, fragments of meaning
   3 = Understandable but awkward or poorly structured
   4 = Well-written, clear, grammatically correct
   5 = Exceptionally polished, professional quality

2. CREATIVITY (novelty, originality, surprise):
   1 = Generic, template-like, or completely incoherent
   2 = Slightly different from typical answers
   3 = Contains some interesting or unexpected ideas
   4 = Genuinely surprising, novel concepts
   5 = Brilliantly original, paradigm-shifting ideas

CRITICAL RULE: If Coherence is 1 or 2 (text is broken/gibberish), Creativity MUST be 1 regardless of how "unique" the garbage text appears. Destruction is not creation.

Respond in EXACTLY this JSON format (no markdown, no extra text):
{{"coherence": <1-5>, "creativity": <1-5>, "reasoning": "brief 1-sentence explanation"}}"""

    result = _call_gemini(judge_prompt)

    if isinstance(result, dict) and "coherence" in result:
        # Enforce the critical rule
        if result["coherence"] <= 2:
            result["creativity"] = 1
            result["reasoning"] = result.get("reasoning", "") + " [Creativity forced to 1: incoherent text]"
        return result

    return {"coherence": 0, "creativity": 0, "reasoning": f"API error: {result}"}


# ═════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═════════════════════════════════════════════════════════════════

def run_all_generations(model, tokenizer):
    """Generate all outputs across all conditions."""
    all_outputs = {}  # condition_name -> [{"prompt": ..., "output": ..., "repeat": ...}]

    total_conditions = len(SIGMA_VALUES) * len(LAYER_CONFIGS) + len(CONTROLS)
    cond_idx = 0

    # ─── Control: Baseline (T=0) ───
    cond_idx += 1
    cond_name = "Baseline (T=0)"
    print(f"\n  [{cond_idx}/{total_conditions}] {cond_name}")
    all_outputs[cond_name] = []
    for i, prompt in enumerate(CREATIVE_PROMPTS):
        for r in range(NUM_REPEATS):
            out = generate_text_deterministic(model, tokenizer, prompt)
            all_outputs[cond_name].append({
                "prompt_idx": i, "prompt": prompt, "output": out, "repeat": r
            })
    print(f"      → {len(all_outputs[cond_name])} outputs generated")

    # ─── Control: Temperature (T=1.5) ───
    cond_idx += 1
    cond_name = "Temperature (T=1.5)"
    print(f"\n  [{cond_idx}/{total_conditions}] {cond_name}")
    all_outputs[cond_name] = []
    for i, prompt in enumerate(CREATIVE_PROMPTS):
        for r in range(NUM_REPEATS):
            out = generate_text_with_temperature(model, tokenizer, prompt)
            all_outputs[cond_name].append({
                "prompt_idx": i, "prompt": prompt, "output": out, "repeat": r
            })
    print(f"      → {len(all_outputs[cond_name])} outputs generated")

    # ─── SNN Conditions ───
    for layer_name, layer_ids in LAYER_CONFIGS.items():
        for sigma in SIGMA_VALUES:
            cond_idx += 1
            cond_name = f"SNN σ={sigma} {layer_name}"
            print(f"\n  [{cond_idx}/{total_conditions}] {cond_name}")
            all_outputs[cond_name] = []
            for i, prompt in enumerate(CREATIVE_PROMPTS):
                for r in range(NUM_REPEATS):
                    out = generate_with_snn_chaos(
                        model, tokenizer, prompt,
                        sigma=sigma, layers=layer_ids,
                        seed=SEED + i * 100 + r
                    )
                    all_outputs[cond_name].append({
                        "prompt_idx": i, "prompt": prompt, "output": out, "repeat": r
                    })
            print(f"      → {len(all_outputs[cond_name])} outputs generated")

    return all_outputs


def run_gemini_judging(all_outputs):
    """Judge all outputs with Gemini."""
    print("\n" + "=" * 70)
    print("PHASE 2: GEMINI LLM-AS-JUDGE (Coherence × Creativity)")
    print("=" * 70)

    total_outputs = sum(len(v) for v in all_outputs.values())
    judged = 0
    all_scores = {}  # condition_name -> [{"coherence": ..., "creativity": ..., ...}]

    for cond_name, outputs in all_outputs.items():
        print(f"\n  Judging: {cond_name}")
        all_scores[cond_name] = []

        for item in outputs:
            result = judge_creativity(item["prompt"], item["output"])

            score_entry = {
                "prompt_idx": item["prompt_idx"],
                "repeat": item["repeat"],
                "output_preview": item["output"][:200],
                "coherence": result.get("coherence", 0),
                "creativity": result.get("creativity", 0),
                "reasoning": result.get("reasoning", ""),
            }
            all_scores[cond_name].append(score_entry)

            judged += 1
            if judged % 15 == 0:
                print(f"      Progress: {judged}/{total_outputs} judged")

            # Rate limiting
            time.sleep(4)

        # Print per-condition summary
        coh_scores = [s["coherence"] for s in all_scores[cond_name] if s["coherence"] > 0]
        cre_scores = [s["creativity"] for s in all_scores[cond_name] if s["creativity"] > 0]
        if coh_scores and cre_scores:
            avg_coh = np.mean(coh_scores)
            avg_cre = np.mean(cre_scores)
            print(f"      → Coherence: {avg_coh:.2f}  Creativity: {avg_cre:.2f}  "
                  f"Score: {avg_coh * avg_cre:.2f}")

    return all_scores


def compute_summary(all_scores):
    """Compute summary statistics for each condition."""
    summary = {}
    for cond_name, scores in all_scores.items():
        coh = [s["coherence"] for s in scores if s["coherence"] > 0]
        cre = [s["creativity"] for s in scores if s["creativity"] > 0]
        if coh and cre:
            summary[cond_name] = {
                "avg_coherence": round(float(np.mean(coh)), 3),
                "avg_creativity": round(float(np.mean(cre)), 3),
                "creative_score": round(float(np.mean(coh) * np.mean(cre)), 3),
                "std_coherence": round(float(np.std(coh)), 3),
                "std_creativity": round(float(np.std(cre)), 3),
                "n_samples": len(coh),
            }
    return summary


# ═════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════

def visualize_results(summary, all_scores):
    """Create comprehensive visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Phase 12: Edge of Chaos — Finding the Golden Sigma",
                     fontsize=14, fontweight="bold")

        # ─── Plot 1: Coherence × Creativity Scatter ───
        ax = axes[0, 0]
        colors_map = {
            "Baseline (T=0)": ("#888888", "s", 120),
            "Temperature (T=1.5)": ("#FF6B35", "^", 120),
        }
        # SNN colors: gradient from red (high σ) to green (low σ)
        snn_colors = ["#FF4444", "#FFA500", "#44AA44", "#00D4AA"]

        for cond_name, stats in summary.items():
            if cond_name in colors_map:
                c, m, s = colors_map[cond_name]
            elif "L15-20" in cond_name:
                idx = [i for i, sv in enumerate(SIGMA_VALUES) if f"σ={sv}" in cond_name]
                c = snn_colors[idx[0]] if idx else "#666666"
                m, s = "o", 100
            elif "L5-10" in cond_name:
                idx = [i for i, sv in enumerate(SIGMA_VALUES) if f"σ={sv}" in cond_name]
                c = snn_colors[idx[0]] if idx else "#666666"
                m, s = "D", 100
            else:
                c, m, s = "#666666", "x", 80

            ax.scatter(stats["avg_coherence"], stats["avg_creativity"],
                      c=c, marker=m, s=s, edgecolors="black", linewidth=0.5,
                      label=cond_name, zorder=5)

        ax.set_xlabel("Coherence (1-5)")
        ax.set_ylabel("Creativity (1-5)")
        ax.set_title("Coherence vs Creativity")
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.axhline(y=3, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=3, color="gray", linestyle="--", alpha=0.3)
        ax.grid(alpha=0.2)
        # Golden zone
        from matplotlib.patches import Rectangle
        golden = Rectangle((3.5, 3.5), 2, 2, linewidth=2, edgecolor='gold',
                           facecolor='gold', alpha=0.1, label='Golden Zone')
        ax.add_patch(golden)
        ax.legend(fontsize=6, loc="lower left")

        # ─── Plot 2: Creative Score (Coh × Cre) bar chart ───
        ax = axes[0, 1]
        cond_names = list(summary.keys())
        creative_scores = [summary[c]["creative_score"] for c in cond_names]
        short_names = []
        bar_colors = []
        for name in cond_names:
            if "Baseline" in name:
                short_names.append("Base\nT=0")
                bar_colors.append("#888888")
            elif "Temperature" in name:
                short_names.append("Temp\nT=1.5")
                bar_colors.append("#FF6B35")
            else:
                # Extract sigma and layer info
                parts = name.replace("SNN ", "").replace("(language)", "L").replace("(concept)", "C")
                short_names.append(parts.replace(" L15-20 ", "\nL15-20").replace(" L5-10 ", "\nL5-10"))
                if "L15-20" in name:
                    idx = [i for i, sv in enumerate(SIGMA_VALUES) if f"σ={sv}" in name]
                    bar_colors.append(snn_colors[idx[0]] if idx else "#666666")
                else:
                    idx = [i for i, sv in enumerate(SIGMA_VALUES) if f"σ={sv}" in name]
                    bar_colors.append(snn_colors[idx[0]] if idx else "#666666")

        bars = ax.bar(range(len(cond_names)), creative_scores, color=bar_colors, alpha=0.85)
        ax.set_xticks(range(len(cond_names)))
        ax.set_xticklabels(short_names, fontsize=6, rotation=0)
        ax.set_ylabel("Creative Score (Coherence × Creativity)")
        ax.set_title("Overall Creative Score per Condition")
        ax.grid(alpha=0.2, axis="y")
        # Add value labels
        for bar, val in zip(bars, creative_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{val:.1f}", ha="center", fontsize=7, fontweight="bold")

        # ─── Plot 3: σ sweep heatmap (L15-20) ───
        ax = axes[1, 0]
        heatmap_data = []
        y_labels = []
        for layer_name in LAYER_CONFIGS.keys():
            row = []
            for sigma in SIGMA_VALUES:
                cond = f"SNN σ={sigma} {layer_name}"
                if cond in summary:
                    row.append(summary[cond]["creative_score"])
                else:
                    row.append(0)
            heatmap_data.append(row)
            y_labels.append(layer_name)

        heatmap_array = np.array(heatmap_data)
        im = ax.imshow(heatmap_array, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=max(25, heatmap_array.max() + 2))
        ax.set_xticks(range(len(SIGMA_VALUES)))
        ax.set_xticklabels([f"σ={s}" for s in SIGMA_VALUES])
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_title("Creative Score Heatmap (σ × Layer)")
        ax.set_xlabel("σ (noise intensity)")

        # Add text annotations
        for i in range(len(y_labels)):
            for j in range(len(SIGMA_VALUES)):
                text = f"{heatmap_array[i, j]:.1f}"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if heatmap_array[i, j] > heatmap_array.max()/2 else "black")

        plt.colorbar(im, ax=ax, shrink=0.8)

        # ─── Plot 4: Coherence and Creativity per σ (dual bars) ───
        ax = axes[1, 1]
        # Show L15-20 and L5-10 side by side for each σ
        x_positions = np.arange(len(SIGMA_VALUES))
        width = 0.18

        for li, (layer_name, _) in enumerate(LAYER_CONFIGS.items()):
            coh_vals = []
            cre_vals = []
            for sigma in SIGMA_VALUES:
                cond = f"SNN σ={sigma} {layer_name}"
                if cond in summary:
                    coh_vals.append(summary[cond]["avg_coherence"])
                    cre_vals.append(summary[cond]["avg_creativity"])
                else:
                    coh_vals.append(0)
                    cre_vals.append(0)

            offset = li * 2 * width
            ax.bar(x_positions + offset, coh_vals, width,
                   label=f"Coh {layer_name}", alpha=0.7,
                   color="#4285f4" if li == 0 else "#34a853")
            ax.bar(x_positions + offset + width, cre_vals, width,
                   label=f"Cre {layer_name}", alpha=0.7,
                   color="#ea4335" if li == 0 else "#fbbc05")

        # Add control lines
        if "Baseline (T=0)" in summary:
            ax.axhline(y=summary["Baseline (T=0)"]["avg_coherence"],
                       color="#888888", linestyle="--", alpha=0.5, label="Base Coh")
        if "Temperature (T=1.5)" in summary:
            ax.axhline(y=summary["Temperature (T=1.5)"]["avg_creativity"],
                       color="#FF6B35", linestyle=":", alpha=0.5, label="Temp Cre")

        ax.set_xticks(x_positions + 1.5 * width)
        ax.set_xticklabels([f"σ={s}" for s in SIGMA_VALUES])
        ax.set_ylabel("Score (1-5)")
        ax.set_title("Coherence & Creativity by σ and Layer")
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.2, axis="y")

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "phase12_edge_of_chaos.png")
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  📊 Figure saved: {fig_path}")

    except Exception as e:
        print(f"\n  ⚠ Visualization error: {e}")
        import traceback
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 12: Edge of Chaos")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate outputs (no Gemini API needed)")
    parser.add_argument("--judge-only", action="store_true",
                        help="Only judge existing outputs (no GPU needed)")
    args = parser.parse_args()

    print("=" * 70)
    print("SNN-Genesis v3 — Phase 12: Edge of Chaos")
    print(f"  'Finding the Golden Sigma — where chaos meets coherence'")
    if args.generate_only:
        print(f"  MODE: Generate only (no API needed)")
    elif args.judge_only:
        print(f"  MODE: Judge only (no GPU needed)")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    start_time = time.time()
    intermediate_path = os.path.join(RESULTS_DIR, "phase12_outputs_intermediate.json")

    need_api = not args.generate_only
    if need_api:
        check_gemini_api(required=True)
        print("\n🔌 Testing Gemini API...")
        test = judge_creativity(
            "Invent a new color",
            "The color is called Shimmerblue. It looks like moonlight reflected off ocean waves."
        )
        if test.get("coherence", 0) > 0:
            print(f"  ✅ Gemini API OK! Test: Coh={test['coherence']}, Cre={test['creativity']}")
        else:
            print(f"  ❌ Gemini API error: {test}")
            print("  Cannot proceed without API. Exiting.")
            sys.exit(1)

    gen_time = 0
    judge_time = 0

    # ─── Phase 1: Generation ───
    if not args.judge_only:
        model, tokenizer = load_model()

        print("\n" + "=" * 70)
        print("PHASE 1: GENERATION (σ micro-dosing + layer shift)")
        print("=" * 70)

        all_outputs = run_all_generations(model, tokenizer)

        # Save intermediate outputs (crash recovery)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, indent=2, ensure_ascii=False)
        print(f"\n  💾 Intermediate outputs saved: {intermediate_path}")

        gen_time = time.time() - start_time
        print(f"\n  ⏱ Generation time: {gen_time/60:.1f} minutes")

        # Free GPU memory
        print("\n  🧹 Freeing GPU memory...")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        if args.generate_only:
            print(f"\n  ✅ Generation complete! {sum(len(v) for v in all_outputs.values())} outputs saved.")
            print(f"  Next: run with --judge-only to evaluate with Gemini.")
            # Beep
            try:
                import winsound
                for _ in range(3):
                    winsound.Beep(1000, 300)
                    time.sleep(0.2)
            except Exception:
                print("\a")
            return
    else:
        # Load from intermediate file
        print(f"\n  📂 Loading outputs from: {intermediate_path}")
        with open(intermediate_path, "r", encoding="utf-8") as f:
            all_outputs = json.load(f)
        print(f"  ✅ Loaded {sum(len(v) for v in all_outputs.values())} outputs")

    # ─── Phase 2: Gemini judging ───
    judge_start = time.time()
    all_scores = run_gemini_judging(all_outputs)
    judge_time = time.time() - judge_start
    print(f"\n  ⏱ Judging time: {judge_time/60:.1f} minutes")

    # ─── Compute summary ───
    summary = compute_summary(all_scores)

    # ─── Visualization ───
    visualize_results(summary, all_scores)

    # ─── Save full results ───
    elapsed = time.time() - start_time

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

    save_results = {
        "config": {
            "model": MODEL_ID,
            "sigma_values": SIGMA_VALUES,
            "layer_configs": {k: v for k, v in LAYER_CONFIGS.items()},
            "num_prompts": NUM_PROMPTS,
            "num_repeats": NUM_REPEATS,
            "judge_model": GEMINI_MODEL,
            "started": datetime.datetime.now().isoformat(),
            "elapsed_minutes": round(elapsed / 60, 1),
            "gen_minutes": round(gen_time / 60, 1),
            "judge_minutes": round(judge_time / 60, 1),
        },
        "summary": summary,
        "detailed_scores": {k: v for k, v in all_scores.items()},
    }

    results_path = os.path.join(RESULTS_DIR, "phase12_edge_of_chaos_log.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\n💾 Full results: {results_path}")

    # ─── Final Summary ───
    print(f"\n{'=' * 70}")
    print("PHASE 12 SUMMARY: Edge of Chaos")
    print(f"{'=' * 70}")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} min (gen: {gen_time/60:.1f} + judge: {judge_time/60:.1f})")

    print(f"\n  📊 Results by condition:")
    print(f"  {'Condition':<35s}  {'Coh':>5s}  {'Cre':>5s}  {'Score':>6s}")
    print(f"  {'-'*35}  {'-'*5}  {'-'*5}  {'-'*6}")

    # Sort by creative score
    sorted_conditions = sorted(summary.items(), key=lambda x: x[1]["creative_score"], reverse=True)
    for cond_name, stats in sorted_conditions:
        marker = " ★" if stats["creative_score"] == max(s["creative_score"] for s in summary.values()) else ""
        print(f"  {cond_name:<35s}  {stats['avg_coherence']:5.2f}  {stats['avg_creativity']:5.2f}  "
              f"{stats['creative_score']:6.2f}{marker}")

    # Find the Golden Sigma
    best_cond = sorted_conditions[0][0]
    best_score = sorted_conditions[0][1]
    print(f"\n  🏆 GOLDEN CONDITION: {best_cond}")
    print(f"     Coherence: {best_score['avg_coherence']:.2f}")
    print(f"     Creativity: {best_score['avg_creativity']:.2f}")
    print(f"     Creative Score: {best_score['creative_score']:.2f}")

    # Check if SNN beats Temperature
    temp_score = summary.get("Temperature (T=1.5)", {}).get("creative_score", 0)
    if best_score["creative_score"] > temp_score and "SNN" in best_cond:
        print(f"\n  🎆 SNN BEATS TEMPERATURE! ({best_score['creative_score']:.1f} > {temp_score:.1f})")
        print(f"     → The Edge of Chaos found!")
    elif any("SNN" in c and summary[c]["creative_score"] > temp_score for c in summary):
        snn_winners = [c for c in summary if "SNN" in c and summary[c]["creative_score"] > temp_score]
        print(f"\n  🔬 {len(snn_winners)} SNN condition(s) beat Temperature:")
        for w in snn_winners:
            print(f"     → {w}: {summary[w]['creative_score']:.1f}")
    else:
        print(f"\n  🤔 No SNN condition beat Temperature ({temp_score:.1f}).")
        print(f"     → May need even lower σ or different layer targets.")

    print(f"\n{'=' * 70}")
    print(f"  Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 🔔 Beep
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1000, 300)
            time.sleep(0.2)
        print("\n  🔔 ビープ音鳴らしたよ！")
    except Exception:
        print("\a")

    # 💤 Auto-sleep
    print("\n  💤 30秒後にスリープします... (Ctrl+Cでキャンセル)")
    try:
        time.sleep(30)
        subprocess.run(
            ["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"],
            check=True
        )
    except KeyboardInterrupt:
        print("  ⏰ スリープキャンセルしたよ！")
    except Exception as e:
        print(f"  ⚠ スリープ失敗: {e}")


if __name__ == "__main__":
    main()
