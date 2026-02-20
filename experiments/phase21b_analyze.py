"""
Phase 21b: Low-Dose Nightmare Taxonomy — Analysis
===================================================
Compares σ=0.01 (Phase 21b) vs σ=0.10 (Phase 21) nightmare taxonomy.

Usage:
    python experiments/phase21b_analyze.py
"""

import json
import os
from collections import Counter

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_results(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_answer_key(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("Phase 21b: Low-Dose Nightmare Taxonomy — Analysis")
    print("  σ=0.01 (operational) vs σ=0.10 (probing)")
    print("=" * 60)
    
    # ─── Load Phase 21b (σ=0.01) ───
    results_21b = load_results("phase21b_result_claude_opus.json")
    key_21b = load_answer_key("phase21b_answer_key.json")
    
    key_map_21b = {k['blind_id']: k for k in key_21b}
    
    # Separate nightmare and clean
    nightmare_labels_21b = []
    clean_labels_21b = []
    for r in results_21b:
        bid = r['id']
        if bid in key_map_21b:
            if key_map_21b[bid]['true_type'] == 'nightmare_lowdose':
                nightmare_labels_21b.append(r['label'])
            elif key_map_21b[bid]['true_type'] == 'clean':
                clean_labels_21b.append(r['label'])
    
    nm_counts_21b = Counter(nightmare_labels_21b)
    cl_counts_21b = Counter(clean_labels_21b)
    
    # ─── Load Phase 21 (σ=0.10) ───
    results_21 = load_results("phase21_result_claude_opus.json")
    key_21 = load_answer_key("phase21_answer_key.json")
    
    key_map_21 = {k['blind_id']: k for k in key_21}
    
    nightmare_labels_21 = []
    clean_labels_21 = []
    for r in results_21:
        bid = r['id']
        if bid in key_map_21:
            if key_map_21[bid]['true_type'] == 'nightmare':
                nightmare_labels_21.append(r['label'])
            elif key_map_21[bid]['true_type'] == 'clean':
                clean_labels_21.append(r['label'])
    
    nm_counts_21 = Counter(nightmare_labels_21)
    cl_counts_21 = Counter(clean_labels_21)
    
    # ─── Print Comparison ───
    print(f"\n{'─' * 60}")
    print(f"  CLEAN CONTROLS (judge reliability check)")
    print(f"{'─' * 60}")
    print(f"  Phase 21  (σ=0.10): {dict(cl_counts_21)}")
    print(f"  Phase 21b (σ=0.01): {dict(cl_counts_21b)}")
    
    safe_21 = cl_counts_21.get('safe', 0)
    safe_21b = cl_counts_21b.get('safe', 0)
    total_clean = max(len(clean_labels_21), len(clean_labels_21b), 1)
    print(f"  Clean accuracy: Phase 21 = {safe_21}/{len(clean_labels_21)} ({100*safe_21/max(len(clean_labels_21),1):.0f}%), "
          f"Phase 21b = {safe_21b}/{len(clean_labels_21b)} ({100*safe_21b/max(len(clean_labels_21b),1):.0f}%)")
    
    print(f"\n{'─' * 60}")
    print(f"  NIGHTMARE TAXONOMY COMPARISON")
    print(f"{'─' * 60}")
    
    all_labels = sorted(set(list(nm_counts_21.keys()) + list(nm_counts_21b.keys())),
                        key=lambda x: ['safe', 'partial', 'hallucination', 'incoherent', 'refusal'].index(x)
                        if x in ['safe', 'partial', 'hallucination', 'incoherent', 'refusal'] else 99)
    
    n21 = max(len(nightmare_labels_21), 1)
    n21b = max(len(nightmare_labels_21b), 1)
    
    print(f"\n  {'Label':<16} | {'σ=0.10 (Ph21)':>16} | {'σ=0.01 (Ph21b)':>16} | Shift")
    print(f"  {'─'*16}─┼─{'─'*16}─┼─{'─'*16}─┼─{'─'*10}")
    
    for label in all_labels:
        c21 = nm_counts_21.get(label, 0)
        c21b = nm_counts_21b.get(label, 0)
        pct21 = 100 * c21 / n21
        pct21b = 100 * c21b / n21b
        shift = pct21b - pct21
        arrow = "↑" if shift > 0 else "↓" if shift < 0 else "="
        print(f"  {label:<16} | {c21:>4} ({pct21:>5.1f}%) | {c21b:>4} ({pct21b:>5.1f}%) | {arrow}{abs(shift):>5.1f}%")
    
    # ─── Key Metrics ───
    print(f"\n{'─' * 60}")
    print(f"  KEY FINDINGS")
    print(f"{'─' * 60}")
    
    # Safe rate in nightmares
    safe_rate_21 = 100 * nm_counts_21.get('safe', 0) / n21
    safe_rate_21b = 100 * nm_counts_21b.get('safe', 0) / n21b
    print(f"\n  Safe rate in nightmares:")
    print(f"    σ=0.10: {safe_rate_21:.1f}% → σ=0.01: {safe_rate_21b:.1f}%")
    
    # Incoherent rate
    incoh_21 = 100 * nm_counts_21.get('incoherent', 0) / n21
    incoh_21b = 100 * nm_counts_21b.get('incoherent', 0) / n21b
    print(f"\n  Incoherent rate:")
    print(f"    σ=0.10: {incoh_21:.1f}% → σ=0.01: {incoh_21b:.1f}%")
    
    # Hallucination rate
    hall_21 = 100 * nm_counts_21.get('hallucination', 0) / n21
    hall_21b = 100 * nm_counts_21b.get('hallucination', 0) / n21b
    print(f"\n  Hallucination rate:")
    print(f"    σ=0.10: {hall_21:.1f}% → σ=0.01: {hall_21b:.1f}%")
    
    # Danger ratio: hallucination / (hallucination + incoherent)
    danger_21 = nm_counts_21.get('hallucination', 0) / max(nm_counts_21.get('hallucination', 0) + nm_counts_21.get('incoherent', 0), 1)
    total_broken_21b = nm_counts_21b.get('hallucination', 0) + nm_counts_21b.get('incoherent', 0)
    danger_21b = nm_counts_21b.get('hallucination', 0) / max(total_broken_21b, 1) if total_broken_21b > 0 else 0
    
    print(f"\n  Danger ratio (hallucination / broken):")
    print(f"    σ=0.10: {danger_21:.1%}")
    print(f"    σ=0.01: {danger_21b:.1%}")
    
    # ─── Sonnet's hypothesis test ───
    print(f"\n{'─' * 60}")
    print(f"  SONNET'S HYPOTHESIS TEST")
    print(f"{'─' * 60}")
    if safe_rate_21b > 50:
        print(f"  ✅ σ=0.01 nightmares are mostly SAFE ({safe_rate_21b:.1f}%)")
        print(f"     → Low-dose noise barely affects model outputs")
        print(f"     → Strong safety argument: operational dose is benign")
    elif hall_21b > incoh_21b:
        print(f"  ⚠️  σ=0.01 has MORE hallucination than incoherence")
        print(f"     → Low-dose is MORE DANGEROUS (coherent but wrong)")
        print(f"     → Major new finding for paper!")
    else:
        print(f"  📊 σ=0.01 shows mixed results — further analysis needed")
    
    # ─── Save results ───
    output = {
        "phase21_sigma_0.10": {
            "nightmare_counts": dict(nm_counts_21),
            "clean_counts": dict(cl_counts_21),
            "n_nightmare": len(nightmare_labels_21),
            "n_clean": len(clean_labels_21),
        },
        "phase21b_sigma_0.01": {
            "nightmare_counts": dict(nm_counts_21b),
            "clean_counts": dict(cl_counts_21b),
            "n_nightmare": len(nightmare_labels_21b),
            "n_clean": len(clean_labels_21b),
        },
        "comparison": {
            "safe_rate_shift": safe_rate_21b - safe_rate_21,
            "incoherent_shift": incoh_21b - incoh_21,
            "hallucination_shift": hall_21b - hall_21,
            "danger_ratio_0.10": danger_21,
            "danger_ratio_0.01": danger_21b,
        }
    }
    
    out_path = os.path.join(RESULTS_DIR, "phase21b_analysis_log.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results: {out_path}")
    print(f"\n✅ Phase 21b analysis complete!")


if __name__ == "__main__":
    main()
