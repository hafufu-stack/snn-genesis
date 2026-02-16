"""
aggregate_judge_responses.py — Process manual Gemini judge responses.

Reads JSON responses saved from Gemini Web UI and integrates them
back into the Phase 12 pipeline for visualization and analysis.

Usage:
  1. Save each Gemini response as a .json file in results/judge_responses/
     - Name them: response_01.json, response_02.json, ... response_10.json
     - Or just paste the JSON arrays into the files
  2. Run: python experiments/aggregate_judge_responses.py
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUTPUTS_FILE = os.path.join(RESULTS_DIR, "phase12_outputs_intermediate.json")
RESPONSES_DIR = os.path.join(RESULTS_DIR, "judge_responses")

# Condition order (must match the order in intermediate JSON)
CONDITIONS = [
    "Baseline (T=0)",
    "Temperature (T=1.5)", 
    "SNN σ=0.05 L15-20 (language)",
    "SNN σ=0.03 L15-20 (language)",
    "SNN σ=0.01 L15-20 (language)",
    "SNN σ=0.005 L15-20 (language)",
    "SNN σ=0.05 L5-10 (concept)",
    "SNN σ=0.03 L5-10 (concept)",
    "SNN σ=0.01 L5-10 (concept)",
    "SNN σ=0.005 L5-10 (concept)",
]


def load_responses():
    """Load all response JSON files from the responses directory."""
    all_scores = {}
    
    for idx, condition in enumerate(CONDITIONS, 1):
        # Try multiple naming conventions
        possible_files = [
            f"response_{idx:02d}.json",
            f"response_{idx}.json",
            f"response_{idx:02d}_{condition.replace(' ', '_')}.json",
        ]
        
        found = False
        for fname in possible_files:
            fpath = os.path.join(RESPONSES_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    # Handle potential markdown code block wrapping
                    if content.startswith("```"):
                        lines = content.split("\n")
                        content = "\n".join(lines[1:-1])
                    scores = json.loads(content)
                
                all_scores[condition] = scores
                print(f"  ✅ [{idx:2d}] {condition}: {len(scores)} scores from {fname}")
                found = True
                break
        
        if not found:
            print(f"  ❌ [{idx:2d}] {condition}: NO RESPONSE FILE FOUND")
            print(f"       Expected: {possible_files[0]} in {RESPONSES_DIR}")
    
    return all_scores


def compute_summary(all_scores):
    """Compute condition-level summary statistics."""
    summary = {}
    
    for condition, scores in all_scores.items():
        coherence_vals = [s["coherence"] for s in scores if isinstance(s.get("coherence"), (int, float))]
        creativity_vals = [s["creativity"] for s in scores if isinstance(s.get("creativity"), (int, float))]
        
        if coherence_vals and creativity_vals:
            coh_mean = np.mean(coherence_vals)
            cre_mean = np.mean(creativity_vals)
            # Creative score = sqrt(Coherence * Creativity) — balanced metric
            creative_score = np.sqrt(coh_mean * cre_mean)
            
            summary[condition] = {
                "coherence_mean": round(float(coh_mean), 2),
                "coherence_std": round(float(np.std(coherence_vals)), 2),
                "creativity_mean": round(float(cre_mean), 2),
                "creativity_std": round(float(np.std(creativity_vals)), 2),
                "creative_score": round(float(creative_score), 2),
                "n": len(coherence_vals),
            }
    
    return summary


def print_results(summary):
    """Pretty-print the results table."""
    print("\n" + "=" * 80)
    print("PHASE 12 RESULTS: Edge of Chaos")
    print("=" * 80)
    print(f"{'Condition':<35} {'Coh':>5} {'Cre':>5} {'Score':>7} {'N':>3}")
    print("-" * 60)
    
    # Sort by creative score
    sorted_conditions = sorted(summary.items(), key=lambda x: x[1]["creative_score"], reverse=True)
    
    for condition, stats in sorted_conditions:
        print(f"  {condition:<33} {stats['coherence_mean']:>5.2f} {stats['creativity_mean']:>5.2f} {stats['creative_score']:>7.2f} {stats['n']:>3}")
    
    print("-" * 60)
    
    # Find golden sigma
    best = sorted_conditions[0]
    print(f"\n  🏆 GOLDEN SIGMA: {best[0]}")
    print(f"     Creative Score: {best[1]['creative_score']:.2f}")
    print(f"     Coherence: {best[1]['coherence_mean']:.2f} ± {best[1]['coherence_std']:.2f}")
    print(f"     Creativity: {best[1]['creativity_mean']:.2f} ± {best[1]['creativity_std']:.2f}")


def save_final_results(all_scores, summary):
    """Save the complete results to the final JSON log."""
    result = {
        "experiment": "Phase 12: Edge of Chaos",
        "summary": summary,
        "all_scores": {k: v for k, v in all_scores.items()},
    }
    
    output_path = os.path.join(RESULTS_DIR, "phase12_edge_of_chaos_log.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Final results saved: {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("Phase 12: Aggregating Manual Judge Responses")
    print("=" * 70)
    
    if not os.path.exists(RESPONSES_DIR):
        os.makedirs(RESPONSES_DIR, exist_ok=True)
        print(f"\n  📁 Created directory: {RESPONSES_DIR}")
        print(f"  ⚠️  No response files found yet!")
        print(f"  Save Gemini responses as response_01.json through response_10.json")
        return
    
    # Load responses
    print("\n📂 Loading responses...")
    all_scores = load_responses()
    
    if not all_scores:
        print("\n  ❌ No responses loaded! Please add response files to:")
        print(f"     {RESPONSES_DIR}")
        return
    
    # Compute summary
    print("\n📊 Computing summary statistics...")
    summary = compute_summary(all_scores)
    
    # Print results
    print_results(summary)
    
    # Save
    save_final_results(all_scores, summary)
    
    print(f"\n  ✅ Done! {len(all_scores)}/{len(CONDITIONS)} conditions processed.")
    if len(all_scores) < len(CONDITIONS):
        missing = [c for c in CONDITIONS if c not in all_scores]
        print(f"  ⚠️  Missing: {missing}")


if __name__ == "__main__":
    main()
