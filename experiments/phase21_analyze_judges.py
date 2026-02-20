"""
Phase 21: LLM-as-a-Judge — Cross-Model Analysis Script
========================================================
Reads judge results from multiple models and computes:
- Label distribution per model
- Agreement rates (pairwise and overall)
- Cohen's kappa for inter-rater reliability  
- Comparison with original keyword-based labels
- Disagreement analysis

Usage:
    python experiments/phase21_analyze_judges.py
"""

import json
import os
import sys
from collections import Counter
from itertools import combinations

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
ANSWER_KEY_PATH = os.path.join(RESULTS_DIR, 'phase21_answer_key.json')

# Expected model result files
MODEL_FILES = {
    'claude_opus': 'phase21_result_claude_opus.json',
    'gemini_pro': 'phase21_result_gemini_pro.json',
    'gpt_oss': 'phase21_result_gpt_oss.json',
}

def load_answer_key():
    """Load the answer key with true labels."""
    with open(ANSWER_KEY_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_judge_results(model_name, filename):
    """Load judge results from a model's output file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Try to parse as JSON (handle models that add text around JSON)
    try:
        results = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON array from text
        start = content.find('[')
        end = content.rfind(']') + 1
        if start >= 0 and end > start:
            try:
                results = json.loads(content[start:end])
            except json.JSONDecodeError:
                print(f"  ERROR: Could not parse JSON from {filename}")
                return None
        else:
            print(f"  ERROR: No JSON array found in {filename}")
            return None
    
    # Normalize to dict keyed by blind_id
    result_dict = {}
    for item in results:
        bid = item.get('id', item.get('blind_id'))
        if bid is not None:
            # Normalize label
            label = item.get('label', item.get('your_label', 'unknown')).lower().strip()
            result_dict[bid] = {
                'label': label,
                'confidence': item.get('confidence', None),
                'reasoning': item.get('reasoning', ''),
            }
    
    return result_dict

def cohens_kappa(labels_a, labels_b):
    """Compute Cohen's kappa between two raters."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 0.0
    
    # Get all unique categories
    categories = sorted(set(labels_a) | set(labels_b))
    
    # Build confusion matrix
    matrix = {}
    for c1 in categories:
        for c2 in categories:
            matrix[(c1, c2)] = 0
    
    for a, b in zip(labels_a, labels_b):
        matrix[(a, b)] += 1
    
    # Observed agreement
    po = sum(matrix[(c, c)] for c in categories) / n
    
    # Expected agreement
    pe = 0
    for c in categories:
        row_sum = sum(matrix[(c, c2)] for c2 in categories) / n
        col_sum = sum(matrix[(c1, c)] for c1 in categories) / n
        pe += row_sum * col_sum
    
    if pe >= 1.0:
        return 1.0
    
    return (po - pe) / (1 - pe)

def analyze(answer_key, model_results):
    """Run full analysis."""
    
    print("\n" + "=" * 70)
    print(" Phase 21: LLM-as-a-Judge — Cross-Model Analysis")
    print("=" * 70)
    
    available_models = {name: res for name, res in model_results.items() if res is not None}
    
    print(f"\nModels loaded: {len(available_models)}/{len(model_results)}")
    for name in model_results:
        status = "✓" if model_results[name] is not None else "✗ (not found)"
        print(f"  {name}: {status}")
    
    if not available_models:
        print("\nNo model results found. Run evaluations first.")
        return
    
    n_samples = len(answer_key)
    blind_ids = [s['blind_id'] for s in answer_key]
    true_types = {s['blind_id']: s['true_type'] for s in answer_key}
    
    # 1. Label distribution per model
    print(f"\n{'─' * 70}")
    print("1. LABEL DISTRIBUTION PER MODEL")
    print(f"{'─' * 70}")
    
    for name, results in available_models.items():
        labels = [results.get(bid, {}).get('label', 'missing') for bid in blind_ids]
        dist = Counter(labels)
        print(f"\n  {name}:")
        for label, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = count / len(labels) * 100
            print(f"    {label:15s}: {count:3d} ({pct:5.1f}%)")
    
    # 2. Agreement with original labels
    print(f"\n{'─' * 70}")
    print("2. AGREEMENT WITH ORIGINAL KEYWORD LABELS")
    print(f"{'─' * 70}")
    
    # Map: nightmare true_type -> should be hallucination/incoherent
    # Map: clean true_type -> should be safe
    for name, results in available_models.items():
        agree = 0
        disagree = 0
        disagree_samples = []
        
        for s in answer_key:
            bid = s['blind_id']
            judge_label = results.get(bid, {}).get('label', 'missing')
            
            if s['true_type'] == 'nightmare':
                # Original label: hallucination. Judge should say hallucination or incoherent
                if judge_label in ('hallucination', 'incoherent'):
                    agree += 1
                else:
                    disagree += 1
                    disagree_samples.append({
                        'blind_id': bid,
                        'true_type': s['true_type'],
                        'judge_label': judge_label,
                        'prompt': s['prompt'][:60],
                        'reasoning': results.get(bid, {}).get('reasoning', '')[:80],
                    })
            elif s['true_type'] == 'clean':
                # Original label: safe. Judge should say safe
                if judge_label == 'safe':
                    agree += 1
                else:
                    disagree += 1
                    disagree_samples.append({
                        'blind_id': bid,
                        'true_type': s['true_type'],
                        'judge_label': judge_label,
                        'prompt': s['prompt'][:60],
                        'reasoning': results.get(bid, {}).get('reasoning', '')[:80],
                    })
        
        total = agree + disagree
        pct = agree / total * 100 if total > 0 else 0
        print(f"\n  {name}: {agree}/{total} agree ({pct:.1f}%)")
        
        if disagree_samples:
            print(f"  Disagreements ({disagree}):")
            for ds in disagree_samples[:5]:  # Show max 5
                print(f"    #{ds['blind_id']}: true={ds['true_type']}, judge={ds['judge_label']}")
                print(f"      Prompt: {ds['prompt']}...")
                print(f"      Reason: {ds['reasoning']}...")
    
    # 3. Inter-model agreement (Cohen's kappa)
    print(f"\n{'─' * 70}")
    print("3. INTER-MODEL AGREEMENT (Cohen's κ)")
    print(f"{'─' * 70}")
    
    model_names = list(available_models.keys())
    if len(model_names) >= 2:
        for m1, m2 in combinations(model_names, 2):
            labels_1 = [available_models[m1].get(bid, {}).get('label', 'missing') for bid in blind_ids]
            labels_2 = [available_models[m2].get(bid, {}).get('label', 'missing') for bid in blind_ids]
            
            # Raw agreement
            agree = sum(1 for a, b in zip(labels_1, labels_2) if a == b)
            raw_pct = agree / len(blind_ids) * 100
            
            # Cohen's kappa
            kappa = cohens_kappa(labels_1, labels_2)
            
            print(f"\n  {m1} vs {m2}:")
            print(f"    Raw agreement: {agree}/{len(blind_ids)} ({raw_pct:.1f}%)")
            print(f"    Cohen's κ:     {kappa:.3f}", end="")
            if kappa >= 0.8:
                print(" (almost perfect)")
            elif kappa >= 0.6:
                print(" (substantial)")
            elif kappa >= 0.4:
                print(" (moderate)")
            elif kappa >= 0.2:
                print(" (fair)")
            else:
                print(" (slight)")
    
    # 4. Majority vote label
    print(f"\n{'─' * 70}")
    print("4. MAJORITY VOTE LABELS")
    print(f"{'─' * 70}")
    
    majority_labels = {}
    for bid in blind_ids:
        votes = []
        for name, results in available_models.items():
            label = results.get(bid, {}).get('label', 'missing')
            if label != 'missing':
                votes.append(label)
        
        if votes:
            vote_count = Counter(votes)
            majority_label = vote_count.most_common(1)[0][0]
            unanimous = len(set(votes)) == 1
            majority_labels[bid] = {
                'label': majority_label,
                'unanimous': unanimous,
                'votes': dict(vote_count),
            }
    
    unanimous_count = sum(1 for v in majority_labels.values() if v['unanimous'])
    print(f"\n  Unanimous agreement: {unanimous_count}/{len(majority_labels)} ({unanimous_count/len(majority_labels)*100:.1f}%)")
    
    # Compare majority vote with true labels
    majority_agree = 0
    for s in answer_key:
        bid = s['blind_id']
        if bid in majority_labels:
            ml = majority_labels[bid]['label']
            if s['true_type'] == 'nightmare' and ml in ('hallucination', 'incoherent'):
                majority_agree += 1
            elif s['true_type'] == 'clean' and ml == 'safe':
                majority_agree += 1
    
    pct = majority_agree / len(answer_key) * 100
    print(f"  Majority vote agrees with keyword labels: {majority_agree}/{len(answer_key)} ({pct:.1f}%)")
    
    # 5. Key finding: Does keyword heuristic hold up?
    print(f"\n{'─' * 70}")
    print("5. KEY FINDING: KEYWORD HEURISTIC VALIDITY")
    print(f"{'─' * 70}")
    
    # How many nightmares are actually hallucination/incoherent vs partial/safe?
    nightmare_ids = [s['blind_id'] for s in answer_key if s['true_type'] == 'nightmare']
    clean_ids = [s['blind_id'] for s in answer_key if s['true_type'] == 'clean']
    
    for name, results in available_models.items():
        nm_labels = Counter(results.get(bid, {}).get('label', 'missing') for bid in nightmare_ids)
        cl_labels = Counter(results.get(bid, {}).get('label', 'missing') for bid in clean_ids)
        
        print(f"\n  {name}:")
        print(f"    Nightmare samples (n={len(nightmare_ids)}):")
        for label, count in sorted(nm_labels.items(), key=lambda x: -x[1]):
            print(f"      {label}: {count}")
        print(f"    Clean samples (n={len(clean_ids)}):")
        for label, count in sorted(cl_labels.items(), key=lambda x: -x[1]):
            print(f"      {label}: {count}")
    
    # Save comprehensive results
    output = {
        'n_samples': n_samples,
        'n_nightmare': len(nightmare_ids),
        'n_clean': len(clean_ids),
        'models_evaluated': list(available_models.keys()),
        'majority_labels': majority_labels,
        'unanimous_rate': unanimous_count / len(majority_labels) if majority_labels else 0,
        'majority_keyword_agreement': majority_agree / len(answer_key) if answer_key else 0,
    }
    
    output_path = os.path.join(RESULTS_DIR, 'phase21_analysis_log.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n\nFull results saved to: {output_path}")

def main():
    answer_key = load_answer_key()
    print(f"Answer key loaded: {len(answer_key)} samples")
    
    model_results = {}
    for name, filename in MODEL_FILES.items():
        print(f"Loading {name}...", end=" ")
        result = load_judge_results(name, filename)
        if result:
            print(f"{len(result)} evaluations")
        else:
            print("not found")
        model_results[name] = result
    
    analyze(answer_key, model_results)

if __name__ == '__main__':
    main()
