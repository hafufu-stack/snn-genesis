"""
aggregate_judge_responses_12b.py — Process Phase 12b manual Gemini judge responses.

Usage:
  1. Save Gemini responses as response_01.json through response_09.json
     in results/judge_responses_12b/
  2. Run: python experiments/aggregate_judge_responses_12b.py
"""

import json
import os
import sys
import re
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
RESPONSES_DIR = os.path.join(RESULTS_DIR, "judge_responses_12b")

CONDITIONS = [
    "Baseline (T=0)",
    "Temperature (T=1.5)",
    "SNN σ=0.05 L15-20",
    "SNN σ=0.03 L15-20",
    "SNN σ=0.01 L5-10",
    "SNN σ=0.005 L5-10",
    "Hybrid T=1.5 + σ=0.05 L15-20",
    "Hybrid T=1.5 + σ=0.03 L15-20",
    "Hybrid T=1.5 + σ=0.01 L15-20",
]


def clean_gemini_json(text):
    """Strip Gemini-specific artifacts from JSON text."""
    # Remove [cite_start], [cite_end], [cite: N] tags
    text = re.sub(r'\[cite_start\]', '', text)
    text = re.sub(r'\[cite_end\]', '', text)
    text = re.sub(r'\[cite:\s*\d+\]', '', text)
    # Remove markdown code fences
    if text.strip().startswith("```"):
        lines = text.strip().split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text.strip()


def load_responses():
    all_scores = {}
    for idx, condition in enumerate(CONDITIONS, 1):
        possible_files = [f"response_{idx:02d}.json", f"response_{idx}.json"]
        found = False
        for fname in possible_files:
            fpath = os.path.join(RESPONSES_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if not content:
                    print(f"  ⏳ [{idx:2d}] {condition}: EMPTY (waiting for data)")
                    found = True
                    break
                content = clean_gemini_json(content)
                try:
                    scores = json.loads(content)
                    all_scores[condition] = scores
                    print(f"  ✅ [{idx:2d}] {condition}: {len(scores)} scores")
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  [{idx:2d}] {condition}: JSON parse error: {e}")
                found = True
                break
        if not found:
            print(f"  ❌ [{idx:2d}] {condition}: NOT FOUND")
    return all_scores


def compute_summary(all_scores):
    summary = {}
    for condition, scores in all_scores.items():
        coh = [s["coherence"] for s in scores if isinstance(s.get("coherence"), (int, float))]
        cre = [s["creativity"] for s in scores if isinstance(s.get("creativity"), (int, float))]
        if coh and cre:
            coh_mean = np.mean(coh)
            cre_mean = np.mean(cre)
            summary[condition] = {
                "coherence_mean": round(float(coh_mean), 2),
                "coherence_std": round(float(np.std(coh)), 2),
                "creativity_mean": round(float(cre_mean), 2),
                "creativity_std": round(float(np.std(cre)), 2),
                "creative_score": round(float(np.sqrt(coh_mean * cre_mean)), 2),
                "n": len(coh),
            }
    return summary


def print_results(summary):
    print("\n" + "=" * 80)
    print("PHASE 12b RESULTS: Extended Edge of Chaos (max_tokens=200)")
    print("=" * 80)
    print(f"{'Condition':<40} {'Coh':>5} {'Cre':>5} {'Score':>7} {'N':>3}")
    print("-" * 65)

    sorted_conds = sorted(summary.items(), key=lambda x: x[1]["creative_score"], reverse=True)
    for condition, stats in sorted_conds:
        marker = " ★" if condition == sorted_conds[0][0] else ""
        print(f"  {condition:<38} {stats['coherence_mean']:>5.2f} {stats['creativity_mean']:>5.2f} "
              f"{stats['creative_score']:>7.2f}{marker}")

    print("-" * 65)
    best = sorted_conds[0]
    print(f"\n  🏆 BEST: {best[0]}")
    print(f"     Creative Score: {best[1]['creative_score']:.2f}")
    print(f"     Coherence: {best[1]['coherence_mean']:.2f} ± {best[1]['coherence_std']:.2f}")
    print(f"     Creativity: {best[1]['creativity_mean']:.2f} ± {best[1]['creativity_std']:.2f}")

    # Check hybrid vs pure
    hybrid_conds = {k: v for k, v in summary.items() if "Hybrid" in k}
    pure_temp = summary.get("Temperature (T=1.5)", {})
    if hybrid_conds and pure_temp:
        best_hybrid = max(hybrid_conds.items(), key=lambda x: x[1]["creative_score"])
        print(f"\n  🔬 Hybrid vs Temperature:")
        print(f"     Temperature:  {pure_temp.get('creative_score', 0):.2f}")
        print(f"     Best Hybrid:  {best_hybrid[1]['creative_score']:.2f} ({best_hybrid[0]})")
        if best_hybrid[1]["creative_score"] > pure_temp.get("creative_score", 0):
            print(f"     → 🎆 HYBRID WINS! Synergy confirmed!")
        else:
            print(f"     → Temperature still ahead")


def save_results(all_scores, summary):
    result = {
        "experiment": "Phase 12b: Extended Edge of Chaos",
        "max_tokens": 200,
        "summary": summary,
        "all_scores": all_scores,
    }
    path = os.path.join(RESULTS_DIR, "phase12b_log.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Results saved: {path}")


def main():
    print("=" * 70)
    print("Phase 12b: Aggregating Manual Judge Responses")
    print("=" * 70)

    if not os.path.exists(RESPONSES_DIR):
        os.makedirs(RESPONSES_DIR, exist_ok=True)
        # Create empty files
        for i in range(1, 10):
            fpath = os.path.join(RESPONSES_DIR, f"response_{i:02d}.json")
            if not os.path.exists(fpath):
                open(fpath, "w").close()
        print(f"\n  📁 Created: {RESPONSES_DIR}")
        print(f"  ⚠️  Save Gemini responses as response_01.json through response_09.json")
        return

    print("\n📂 Loading responses...")
    all_scores = load_responses()

    if not all_scores:
        print(f"\n  ❌ No responses found in {RESPONSES_DIR}")
        return

    print("\n📊 Computing summary...")
    summary = compute_summary(all_scores)
    print_results(summary)
    save_results(all_scores, summary)

    print(f"\n  ✅ Done! {len(all_scores)}/{len(CONDITIONS)} conditions processed.")


if __name__ == "__main__":
    main()
