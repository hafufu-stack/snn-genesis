"""
Phase 22: Filtered DPO — Hallucination-Only Training
=====================================================

Uses Phase 21's Claude Opus labels to extract ONLY hallucination-type
nightmares (excluding incoherent word-salad), creating higher-quality
DPO preference pairs for training.

Hypothesis: filtering out incoherent samples will:
  1. Reduce alignment tax (no "learn to avoid gibberish" overhead)
  2. Focus model on rejecting plausible misinformation
  3. Achieve equivalent or better nightmare defense with less data

Usage:
    python experiments/phase22_filtered_dpo.py
"""

import torch
import json
import os
import sys
import time
import datetime
import random
import gc
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ─── Config ───
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TARGET_LAYERS = list(range(15, 21))
SIGMA = 0.10
MAX_LENGTH = 256
BATCH_SIZE = 2
SEED = 2026

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════
# PART 1: Load data and model
# ═══════════════════════════════════════

def load_filtered_data():
    """
    Load Phase 21 judge results and extract hallucination-only nightmares.
    Also load corresponding clean responses for DPO chosen/rejected pairs.
    """
    # Phase 21 judge labels
    judge_path = os.path.join(RESULTS_DIR, "phase21_result_claude_opus.json")
    key_path = os.path.join(RESULTS_DIR, "phase21_answer_key.json")
    vaccine_path = os.path.join(RESULTS_DIR, "genesis_vaccine.jsonl")
    
    with open(judge_path, 'r', encoding='utf-8') as f:
        judge_results = json.load(f)
    with open(key_path, 'r', encoding='utf-8') as f:
        answer_key = json.load(f)
    
    # Map blind_id to true nightmare index and judge label
    key_map = {k['blind_id']: k for k in answer_key}
    
    # Identify hallucination blind_ids
    hallucination_indices = []  # original_index in nightmare set
    for r in judge_results:
        bid = r['id']
        if bid in key_map:
            entry = key_map[bid]
            if entry['true_type'] == 'nightmare' and r['label'] == 'hallucination':
                hallucination_indices.append(entry['original_index'])
    
    # Load vaccine data
    nightmares = []
    cleans = []
    with open(vaccine_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record['type'] == 'nightmare':
                nightmares.append(record)
            elif record['type'] == 'clean':
                cleans.append(record)
    
    # Extract hallucination-only nightmares
    hall_nightmares = []
    for idx in hallucination_indices:
        if idx < len(nightmares):
            hall_nightmares.append(nightmares[idx])
    
    # Also get ALL nightmares for "unfiltered" comparison
    all_nightmares = nightmares
    
    return hall_nightmares, all_nightmares, cleans


def build_dpo_pairs(nightmares, cleans):
    """
    Create DPO preference pairs:
      - chosen: clean (correct) response for same prompt
      - rejected: nightmare (hallucinated) response for same prompt
    """
    # Build clean response lookup by prompt
    clean_map = {}
    for c in cleans:
        clean_map[c['prompt'].strip()] = c['response']
    
    pairs = []
    for nm in nightmares:
        prompt = nm['prompt'].strip()
        if prompt in clean_map:
            pairs.append({
                'prompt': prompt,
                'chosen': clean_map[prompt],
                'rejected': nm['response'][:200],  # Truncate long nightmare
            })
    
    return pairs


def load_model():
    print(f"\n📦 Loading {MODEL_NAME}...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.float16,
    )
    return model, tokenizer


# ═══════════════════════════════════════
# PART 2: DPO Training (SFT on chosen + clean)
# ═══════════════════════════════════════

def train_dpo(model, tokenizer, preference_data, clean_data, label="filtered"):
    """
    DPO training: SFT on chosen responses + clean Q&A.
    Uses LoRA for parameter-efficient fine-tuning.
    """
    print(f"\n🎯 Training DPO ({label}): {len(preference_data)} preference pairs + "
          f"{len(clean_data)} clean examples")
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    
    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"  LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Build training texts
    texts = []
    
    # Chosen (correct) responses from DPO pairs
    for item in preference_data:
        text = f"Question: {item['prompt']}\nAnswer: {item['chosen']}"
        texts.append(text)
    
    # Clean Q&A for knowledge retention
    for item in clean_data[:30]:  # Cap to avoid drowning out DPO signal
        text = f"Question: {item['prompt']}\nAnswer: {item['response']}"
        texts.append(text)
    
    random.shuffle(texts)
    
    # Tokenize
    train_items = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                        padding=False, return_attention_mask=True)
        train_items.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })
    
    ds = Dataset.from_list(train_items)
    
    output_dir = os.path.join(RESULTS_DIR, f"phase22_dpo_{label}_tmp")
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=3e-5,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
        fp16=True,
        warmup_steps=5,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )
    
    result = trainer.train()
    print(f"  ✅ Training loss: {result.training_loss:.4f}")
    
    return peft_model, result.training_loss


# ═══════════════════════════════════════
# PART 3: Evaluation (TruthfulQA MC1)
# ═══════════════════════════════════════

def evaluate_truthfulqa(model, tokenizer, sigma=0.0, label=""):
    """Evaluate on TruthfulQA MC1 subset."""
    from datasets import load_dataset
    
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    questions = list(ds)[:200]  # Use 200 for speed
    
    # Install noise hooks if sigma > 0
    handles = []
    if sigma > 0 and hasattr(model, 'base_model'):
        base = model.base_model if hasattr(model, 'base_model') else model
        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
            layers = base.model.layers
        elif hasattr(base, 'model') and hasattr(base.model, 'model') and hasattr(base.model.model, 'layers'):
            layers = base.model.model.layers
        else:
            layers = []
        
        class SNNHook:
            def __init__(self, s):
                self.sigma = s
            def __call__(self, module, args):
                hs = args[0]
                noise = torch.randn_like(hs) * self.sigma
                return (hs + noise,) + args[1:]
        
        hook = SNNHook(sigma)
        for idx in TARGET_LAYERS:
            if idx < len(layers):
                handles.append(layers[idx].register_forward_pre_hook(hook))
    elif sigma > 0:
        # For non-PEFT model
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            
            class SNNHook:
                def __init__(self, s):
                    self.sigma = s
                def __call__(self, module, args):
                    hs = args[0]
                    noise = torch.randn_like(hs) * self.sigma
                    return (hs + noise,) + args[1:]
            
            hook = SNNHook(sigma)
            for idx in TARGET_LAYERS:
                if idx < len(layers):
                    handles.append(layers[idx].register_forward_pre_hook(hook))
    
    correct = 0
    total = 0
    
    try:
        for q in questions:
            prompt = q['question']
            choices = q['mc1_targets']['choices']
            labels = q['mc1_targets']['labels']
            
            correct_idx = labels.index(1)
            
            # Score each choice
            best_score = float('-inf')
            best_idx = 0
            
            for ci, choice in enumerate(choices[:5]):  # Limit choices
                text = f"Question: {prompt}\nAnswer: {choice}"
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=MAX_LENGTH).to(model.device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                
                # Average log probability of answer tokens
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs['input_ids'][..., 1:].contiguous()
                
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                score = token_log_probs.mean().item()
                
                if score > best_score:
                    best_score = score
                    best_idx = ci
            
            if best_idx == correct_idx:
                correct += 1
            total += 1
    finally:
        for h in handles:
            h.remove()
    
    acc = 100 * correct / max(total, 1)
    return acc, correct, total


# ═══════════════════════════════════════
# PART 4: Nightmare detection test
# ═══════════════════════════════════════

def test_nightmare_resistance(model, tokenizer, prompts, sigma=0.10):
    """Test nightmare generation rate under SNN noise."""
    if hasattr(model, 'base_model'):
        base = model.base_model
        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
            layers = base.model.layers
        elif hasattr(base, 'model') and hasattr(base.model, 'model') and hasattr(base.model.model, 'layers'):
            layers = base.model.model.layers
        else:
            return 0.0, 0, 0
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        return 0.0, 0, 0
    
    class SNNHook:
        def __init__(self, s):
            self.sigma = s
        def __call__(self, module, args):
            hs = args[0]
            noise = torch.randn_like(hs) * self.sigma
            return (hs + noise,) + args[1:]
    
    hook = SNNHook(sigma)
    handles = []
    for idx in TARGET_LAYERS:
        if idx < len(layers):
            handles.append(layers[idx].register_forward_pre_hook(hook))
    
    nightmares = 0
    total = len(prompts)
    
    try:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=60,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(prompt):].strip()
            
            # Simple nightmare detection (same as original phases)
            is_nightmare = False
            if len(generated) < 5:
                is_nightmare = True
            elif generated.count('?') > 5:
                is_nightmare = True
            elif len(set(generated.split())) < 3:
                is_nightmare = True
            elif any(c in generated.lower() for c in ['\\n\\n\\n', '???', '!!!']):
                is_nightmare = True
            
            if is_nightmare:
                nightmares += 1
    finally:
        for h in handles:
            h.remove()
    
    nm_rate = 100 * nightmares / max(total, 1)
    return nm_rate, nightmares, total


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 22: Filtered DPO — Hallucination-Only Training")
    print("  Removing incoherent word-salad from training data")
    print("=" * 60)
    t_start = time.time()
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Load and filter data
    hall_nightmares, all_nightmares, cleans = load_filtered_data()
    
    print(f"\n📊 Data summary:")
    print(f"  Total nightmares:        {len(all_nightmares)}")
    print(f"  Hallucination-only:      {len(hall_nightmares)} (Phase 21 filtered)")
    print(f"  Removed (incoherent+):   {len(all_nightmares) - len(hall_nightmares)}")
    print(f"  Clean responses:         {len(cleans)}")
    
    # Build DPO pairs
    filtered_pairs = build_dpo_pairs(hall_nightmares, cleans)
    unfiltered_pairs = build_dpo_pairs(all_nightmares, cleans)
    
    print(f"\n📋 DPO pairs:")
    print(f"  Filtered (hall-only):    {len(filtered_pairs)}")
    print(f"  Unfiltered (all):        {len(unfiltered_pairs)}")
    
    # Test prompts for nightmare resistance
    test_prompts = [nm['prompt'] for nm in all_nightmares[:20]]
    
    # Load model
    model, tokenizer = load_model()
    
    log = {
        "experiment": "phase22_filtered_dpo",
        "started": datetime.datetime.now().isoformat(),
        "n_hall_nightmares": len(hall_nightmares),
        "n_all_nightmares": len(all_nightmares),
        "n_filtered_pairs": len(filtered_pairs),
        "n_unfiltered_pairs": len(unfiltered_pairs),
        "results": {},
    }
    
    # ─── Baseline (no training) ───
    print(f"\n{'═' * 50}")
    print(f"  BASELINE (No DPO)")
    print(f"{'═' * 50}")
    
    acc_base, correct_base, total_base = evaluate_truthfulqa(model, tokenizer, sigma=0.0)
    print(f"  TruthfulQA (no noise): {acc_base:.1f}% ({correct_base}/{total_base})")
    
    nm_rate_base, _, _ = test_nightmare_resistance(model, tokenizer, test_prompts)
    print(f"  Nightmare rate (σ={SIGMA}): {nm_rate_base:.1f}%")
    
    log['results']['baseline'] = {
        'truthfulqa_acc': acc_base,
        'nightmare_rate': nm_rate_base,
        'n_training_pairs': 0,
    }
    
    # ─── Filtered DPO (hallucination-only) ───
    print(f"\n{'═' * 50}")
    print(f"  FILTERED DPO (Hallucination-Only: {len(filtered_pairs)} pairs)")
    print(f"{'═' * 50}")
    
    model_filtered, loss_filtered = train_dpo(
        model, tokenizer, filtered_pairs, cleans, label="filtered"
    )
    
    acc_filtered, correct_f, total_f = evaluate_truthfulqa(model_filtered, tokenizer, sigma=0.0)
    print(f"  TruthfulQA (no noise): {acc_filtered:.1f}% ({correct_f}/{total_f})")
    
    tax_filtered = acc_base - acc_filtered
    print(f"  Alignment Tax: {tax_filtered:+.1f}%")
    
    nm_rate_filtered, _, _ = test_nightmare_resistance(model_filtered, tokenizer, test_prompts)
    print(f"  Nightmare rate (σ={SIGMA}): {nm_rate_filtered:.1f}%")
    
    log['results']['filtered_dpo'] = {
        'truthfulqa_acc': acc_filtered,
        'alignment_tax': tax_filtered,
        'nightmare_rate': nm_rate_filtered,
        'training_loss': loss_filtered,
        'n_training_pairs': len(filtered_pairs),
    }
    
    # Clean up filtered model
    del model_filtered
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload fresh model for unfiltered
    model2, tokenizer2 = load_model()
    
    # ─── Unfiltered DPO (all nightmares) ───
    print(f"\n{'═' * 50}")
    print(f"  UNFILTERED DPO (All Nightmares: {len(unfiltered_pairs)} pairs)")
    print(f"{'═' * 50}")
    
    model_unfiltered, loss_unfiltered = train_dpo(
        model2, tokenizer2, unfiltered_pairs, cleans, label="unfiltered"
    )
    
    acc_unfiltered, correct_u, total_u = evaluate_truthfulqa(model_unfiltered, tokenizer2, sigma=0.0)
    print(f"  TruthfulQA (no noise): {acc_unfiltered:.1f}% ({correct_u}/{total_u})")
    
    tax_unfiltered = acc_base - acc_unfiltered
    print(f"  Alignment Tax: {tax_unfiltered:+.1f}%")
    
    nm_rate_unfiltered, _, _ = test_nightmare_resistance(model_unfiltered, tokenizer2, test_prompts)
    print(f"  Nightmare rate (σ={SIGMA}): {nm_rate_unfiltered:.1f}%")
    
    log['results']['unfiltered_dpo'] = {
        'truthfulqa_acc': acc_unfiltered,
        'alignment_tax': tax_unfiltered,
        'nightmare_rate': nm_rate_unfiltered,
        'training_loss': loss_unfiltered,
        'n_training_pairs': len(unfiltered_pairs),
    }
    
    # ─── Summary ───
    elapsed = (time.time() - t_start) / 60
    
    print(f"\n{'═' * 60}")
    print(f"  RESULTS SUMMARY — Phase 22: Filtered DPO")
    print(f"{'═' * 60}")
    print(f"  {'Condition':<30} | {'Acc':>6} | {'Tax':>7} | {'NM Rate':>8} | {'Pairs':>5}")
    print(f"  {'─'*30}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*5}")
    print(f"  {'Baseline (no DPO)':<30} | {acc_base:>5.1f}% | {'—':>7} | {nm_rate_base:>7.1f}% | {'—':>5}")
    print(f"  {'Filtered (hall-only)':<30} | {acc_filtered:>5.1f}% | {tax_filtered:>+6.1f}% | {nm_rate_filtered:>7.1f}% | {len(filtered_pairs):>5}")
    print(f"  {'Unfiltered (all nightmares)':<30} | {acc_unfiltered:>5.1f}% | {tax_unfiltered:>+6.1f}% | {nm_rate_unfiltered:>7.1f}% | {len(unfiltered_pairs):>5}")
    
    print(f"\n  KEY COMPARISON:")
    if tax_filtered < tax_unfiltered:
        print(f"  ✅ Filtered DPO has LOWER tax ({tax_filtered:+.1f}% vs {tax_unfiltered:+.1f}%)")
    else:
        print(f"  ⚠️  Filtered DPO has HIGHER tax ({tax_filtered:+.1f}% vs {tax_unfiltered:+.1f}%)")
    
    if nm_rate_filtered <= nm_rate_unfiltered:
        print(f"  ✅ Filtered DPO has equal or better nightmare resistance")
    else:
        print(f"  ⚠️  Filtered DPO has worse nightmare resistance")
    
    print(f"  Data efficiency: {len(filtered_pairs)} vs {len(unfiltered_pairs)} pairs "
          f"({100*len(filtered_pairs)/max(len(unfiltered_pairs),1):.0f}% of data)")
    
    log['elapsed_min'] = elapsed
    log['comparison'] = {
        'tax_improvement': tax_unfiltered - tax_filtered,
        'nm_rate_diff': nm_rate_filtered - nm_rate_unfiltered,
        'data_efficiency': len(filtered_pairs) / max(len(unfiltered_pairs), 1),
    }
    
    # Save results
    log_path = os.path.join(RESULTS_DIR, "phase22_filtered_dpo_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results: {log_path}")
    print(f"⏱  Total: {elapsed:.1f} min")
    print(f"\n✅ Phase 22 complete!")


if __name__ == "__main__":
    main()
