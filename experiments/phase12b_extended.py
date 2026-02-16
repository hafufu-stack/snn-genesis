"""
phase12b_extended.py — SNN-Genesis v3: Phase 12b Extended Edge of Chaos
=======================================================================

Two improvements over Phase 12:
  1) max_tokens=200 (was 100) — removes truncation artifacts
  2) Hybrid: Temperature(1.5) + SNN noise — potential synergy

Conditions (9 total, 135 outputs):
  [Retest] Baseline T=0, Temperature T=1.5
  [Retest] SNN σ=0.05 L15-20, SNN σ=0.03 L15-20 (most interesting from 12)
  [Retest] SNN σ=0.01 L5-10, SNN σ=0.005 L5-10 (survived from concept layer)
  [NEW]    Hybrid T=1.5 + σ=0.05 L15-20
  [NEW]    Hybrid T=1.5 + σ=0.03 L15-20
  [NEW]    Hybrid T=1.5 + σ=0.01 L15-20

Usage:
  python experiments/phase12b_extended.py --generate-only
  (then manual Gemini evaluation)
  python experiments/aggregate_judge_responses_12b.py

Expected runtime: ~70 min generation on RTX 5080
"""

import torch
import os
import sys
import json
import time
import numpy as np
import datetime
import gc
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from core.snn_reservoir import ChaoticReservoir

import warnings
warnings.filterwarnings("ignore")

# ─── Settings ───
MODEL_ID     = "mistralai/Mistral-7B-Instruct-v0.3"
SEED         = 2026
MAX_NEW      = 200  # ← Increased from 100

# Prompts — same as Phase 12
CREATIVE_PROMPTS = [
    "Invent a completely new law of physics that doesn't exist yet. Describe it in one paragraph.",
    "Describe an alien species that communicates through mathematics instead of language.",
    "Write a one-paragraph philosophy about a concept that has no word in any human language.",
    "If time could think, what would it worry about? Write a short monologue from Time's perspective.",
    "Propose a wild but internally consistent hypothesis about why the universe exists.",
]

NUM_PROMPTS = len(CREATIVE_PROMPTS)
NUM_REPEATS = 3

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

from phase5_scaleup import (
    load_model, generate_text, make_snn_hook,
    get_model_layers,
)


# ═══════════════════════════════════════════════════════════════
# GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def gen_baseline(model, tokenizer, prompt):
    """Greedy (deterministic) baseline."""
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def gen_temperature(model, tokenizer, prompt, temperature=1.5):
    """Temperature sampling."""
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=True,
                             temperature=temperature, top_p=0.95, top_k=50,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def gen_snn(model, tokenizer, prompt, sigma, layers, seed=SEED):
    """SNN noise injection (greedy decoding)."""
    snn_hook = make_snn_hook(sigma=sigma, seed=seed)
    model_layers = get_model_layers(model)
    handles = [model_layers[idx].register_forward_pre_hook(snn_hook) for idx in layers]
    
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    
    try:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()


def gen_hybrid(model, tokenizer, prompt, sigma, layers, temperature=1.5, seed=SEED):
    """HYBRID: Temperature sampling + SNN noise injection."""
    snn_hook = make_snn_hook(sigma=sigma, seed=seed)
    model_layers = get_model_layers(model)
    handles = [model_layers[idx].register_forward_pre_hook(snn_hook) for idx in layers]
    
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    
    try:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=True,
                                 temperature=temperature, top_p=0.95, top_k=50,
                                 pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()


# ═══════════════════════════════════════════════════════════════
# CONDITION DEFINITIONS
# ═══════════════════════════════════════════════════════════════

L15_20 = list(range(15, 21))
L5_10  = list(range(5, 11))

CONDITIONS = [
    # (condition_name, gen_func_name, kwargs)
    ("Baseline (T=0)",                    "baseline",     {}),
    ("Temperature (T=1.5)",               "temperature",  {}),
    ("SNN σ=0.05 L15-20",                 "snn",          {"sigma": 0.05, "layers": L15_20}),
    ("SNN σ=0.03 L15-20",                 "snn",          {"sigma": 0.03, "layers": L15_20}),
    ("SNN σ=0.01 L5-10",                  "snn",          {"sigma": 0.01, "layers": L5_10}),
    ("SNN σ=0.005 L5-10",                 "snn",          {"sigma": 0.005, "layers": L5_10}),
    ("Hybrid T=1.5 + σ=0.05 L15-20",     "hybrid",       {"sigma": 0.05, "layers": L15_20}),
    ("Hybrid T=1.5 + σ=0.03 L15-20",     "hybrid",       {"sigma": 0.03, "layers": L15_20}),
    ("Hybrid T=1.5 + σ=0.01 L15-20",     "hybrid",       {"sigma": 0.01, "layers": L15_20}),
]


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 12b: Extended Edge of Chaos")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate outputs (no Gemini API needed)")
    args = parser.parse_args()

    print("=" * 70)
    print("SNN-Genesis v3 — Phase 12b: Extended Edge of Chaos")
    print(f"  max_tokens={MAX_NEW} + Hybrid(Temperature+SNN) conditions")
    print(f"  {len(CONDITIONS)} conditions × {NUM_PROMPTS*NUM_REPEATS} = {len(CONDITIONS)*NUM_PROMPTS*NUM_REPEATS} total outputs")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    start_time = time.time()
    output_path = os.path.join(RESULTS_DIR, "phase12b_outputs_intermediate.json")

    # Check for existing intermediate file (crash recovery)
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"\n  📂 Found {len(existing)} existing conditions, resuming from there.")

    # Load model
    model, tokenizer = load_model()

    all_outputs = existing.copy()

    for cond_idx, (cond_name, func_name, kwargs) in enumerate(CONDITIONS, 1):
        if cond_name in all_outputs and len(all_outputs[cond_name]) >= NUM_PROMPTS * NUM_REPEATS:
            print(f"\n  [{cond_idx}/{len(CONDITIONS)}] {cond_name} — SKIP (already done)")
            continue

        print(f"\n  [{cond_idx}/{len(CONDITIONS)}] {cond_name}")
        all_outputs[cond_name] = []

        for i, prompt in enumerate(CREATIVE_PROMPTS):
            for r in range(NUM_REPEATS):
                seed = SEED + i * 100 + r

                if func_name == "baseline":
                    out = gen_baseline(model, tokenizer, prompt)
                elif func_name == "temperature":
                    out = gen_temperature(model, tokenizer, prompt)
                elif func_name == "snn":
                    out = gen_snn(model, tokenizer, prompt, seed=seed, **kwargs)
                elif func_name == "hybrid":
                    out = gen_hybrid(model, tokenizer, prompt, seed=seed, **kwargs)

                all_outputs[cond_name].append({
                    "prompt_idx": i, "prompt": prompt, "output": out, "repeat": r
                })

        print(f"      → {len(all_outputs[cond_name])} outputs generated")

        # Save after each condition (crash recovery)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, indent=2, ensure_ascii=False)
        print(f"      💾 Saved (crash recovery)")

    elapsed = time.time() - start_time
    total = sum(len(v) for v in all_outputs.values())
    print(f"\n{'=' * 70}")
    print(f"  ✅ Generation complete! {total} outputs in {elapsed/60:.1f} min")
    print(f"  💾 Saved: {output_path}")
    print(f"{'=' * 70}")

    # Free GPU
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Generate evaluation prompt files
    print("\n  📝 Generating evaluation prompts...")
    generate_eval_prompts(all_outputs)

    # Beep
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1000, 300)
            time.sleep(0.2)
    except Exception:
        print("\a")

    # Auto-sleep
    if not args.generate_only:
        return
    
    print("\n  💤 30秒後にスリープします... (Ctrl+Cでキャンセル)")
    try:
        import subprocess
        time.sleep(30)
        subprocess.run(
            ["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"],
            check=True
        )
    except KeyboardInterrupt:
        print("  ⏰ スリープキャンセル！")
    except Exception as e:
        print(f"  ⚠ スリープ失敗: {e}")


def generate_eval_prompts(all_outputs):
    """Generate evaluation prompt files for Gemini batch upload.
    
    Creates 3 batch files, each containing 3 conditions.
    Users upload all 3 .txt files at once to Gemini.
    """
    prompt_dir = os.path.join(RESULTS_DIR, "judge_prompts_12b")
    os.makedirs(prompt_dir, exist_ok=True)

    conditions = list(all_outputs.keys())
    
    # Split into 3 batches of 3
    batches = [conditions[i:i+3] for i in range(0, len(conditions), 3)]

    system_prompt = """You are an expert creative writing evaluator. For each condition below, evaluate ALL responses on two dimensions:

1. **Coherence** (1-5): Grammar, logic, fluency, completeness
   5=Perfectly natural | 4=Mostly natural | 3=Understandable but unnatural | 2=Partially broken | 1=Complete gibberish

2. **Creativity** (1-5): Novelty, originality, surprise
   5=Brilliantly original | 4=Genuinely surprising | 3=Some interesting ideas | 2=Clichéd | 1=No originality

CRITICAL RULE: If Coherence ≤ 2, Creativity MUST be 1. Destruction is not creation.

Return ONLY a JSON array for each condition. No markdown code blocks."""

    for batch_idx, batch in enumerate(batches):
        for cond_offset, cond_name in enumerate(batch):
            file_idx = batch_idx * 3 + cond_offset + 1
            outputs = all_outputs[cond_name]

            prompt_text = f"""CONDITION: {cond_name}
Evaluate all {len(outputs)} responses below.

{system_prompt}

"""
            for i, item in enumerate(outputs):
                prompt_text += f"""---
### Response {i+1}
**Prompt:** "{item['prompt']}"

**Output:**
{item['output'][:500]}

"""

            prompt_text += f"""---
Return ONLY a JSON array with {len(outputs)} entries:
[
  {{"id": 1, "coherence": <1-5>, "creativity": <1-5>, "reasoning": "brief explanation"}},
  ...
]
"""
            safe_name = cond_name.replace(" ", "_").replace("=", "").replace("σ", "s").replace("+", "plus")
            filename = f"prompt_{file_idx:02d}_{safe_name}.txt"
            filepath = os.path.join(prompt_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            
            print(f"    [{file_idx:2d}/9] {filename} ({len(outputs)} outputs, {len(prompt_text)} chars)")

    # Save a summary instruction file
    summary_path = os.path.join(prompt_dir, "00_INSTRUCTION.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("""=== Phase 12b: Gemini Batch Evaluation ===

STEP 1: Upload 3 files at once to Gemini:
  Round 1: prompt_01 + prompt_02 + prompt_03
  Round 2: prompt_04 + prompt_05 + prompt_06
  Round 3: prompt_07 + prompt_08 + prompt_09

STEP 2: For each round, Gemini returns 3 JSON arrays.
  Copy each JSON array into separate files:
  response_01.json through response_09.json

STEP 3: Run aggregation:
  python experiments/aggregate_judge_responses_12b.py
""")

    print(f"\n  ✅ Evaluation prompts saved: {prompt_dir}")
    print(f"  📋 Instructions: {summary_path}")


if __name__ == "__main__":
    main()
