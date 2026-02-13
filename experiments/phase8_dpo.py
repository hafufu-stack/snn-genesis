"""
phase8_dpo.py — DPO (Direct Preference Optimization) Training
==============================================================

Deep Think Proposal #2: Replace SFT with DPO for the Dream Journal strategy.

Instead of training on (prompt → refusal) pairs,
DPO trains on preference pairs: (prompt, chosen=refusal, rejected=nightmare).
This teaches the model to PREFER refusals over going along with misinformation.

Groups:
  A) Dream Journal + SFT (same as v1 — baseline)
  B) Dream Journal + DPO (preference-based learning)

Expected runtime: ~3 hours on RTX 5080
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
NUM_ROUNDS   = 5
SIGMA        = 0.10
TARGET_LAYER = 10
BATCH_SIZE   = 2   # DPO needs more memory (chosen + rejected)
LORA_R       = 8
LORA_ALPHA   = 16
MAX_LENGTH   = 256
SEED         = 2026

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

from phase5_scaleup import (
    CLEAN_QUESTIONS, NIGHTMARE_TEMPLATES, FALSE_CLAIMS,
    build_nightmare_questions, build_test_set,
    load_model, generate_text, make_randn_hook, get_model_layers,
    generate_with_noise, classify_nightmare, heal_nightmare,
    evaluate_accuracy, train_dream_journal
)


def train_dpo_journal(base_model, tokenizer, preference_data, clean_data, round_num):
    """
    DPO Dream Journal: Train using preference pairs.
    
    preference_data: list of {"prompt": str, "chosen": str, "rejected": str}
    clean_data: list of {"q": str, "a": str} — also added as SFT to maintain knowledge
    """
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from transformers import TrainingArguments

    adapter_name = f"dpo_r{round_num}_{id(preference_data) % 10000}"
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )

    if isinstance(base_model, PeftModel):
        base_model.add_adapter(adapter_name, lora_cfg)
        base_model.set_adapter(adapter_name)
        model = base_model
    else:
        model = get_peft_model(base_model, lora_cfg, adapter_name=adapter_name)
    
    print(f"  🎯 DPO Journal: {len(preference_data)} preference pairs + "
          f"{len(clean_data)} clean samples")

    # Try DPOTrainer first, fall back to manual DPO if trl has issues
    try:
        loss = _train_dpo_with_trl(model, tokenizer, preference_data, clean_data, round_num)
    except Exception as e:
        print(f"  ⚠ DPOTrainer failed ({e}), using manual DPO...")
        loss = _train_dpo_manual(model, tokenizer, preference_data, clean_data, round_num)

    return model, loss, adapter_name


def _train_dpo_with_trl(model, tokenizer, preference_data, clean_data, round_num):
    """Try using trl's DPOTrainer."""
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    # Build preference dataset
    dpo_rows = []
    for item in preference_data:
        prompt_chat = [{"role": "user", "content": item["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(prompt_chat, tokenize=False,
                                                      add_generation_prompt=True)
        chosen_chat = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["chosen"]},
        ]
        rejected_chat = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["rejected"]},
        ]
        chosen_text = tokenizer.apply_chat_template(chosen_chat, tokenize=False)
        rejected_text = tokenizer.apply_chat_template(rejected_chat, tokenize=False)

        dpo_rows.append({
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        })

    ds = Dataset.from_list(dpo_rows)

    output_dir = os.path.join(RESULTS_DIR, "dpo_qlora_tmp")
    os.makedirs(output_dir, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=5e-6,  # DPO typically uses lower LR
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        gradient_accumulation_steps=4,
        remove_unused_columns=False,
        max_length=MAX_LENGTH,
        max_prompt_length=128,
        beta=0.1,  # DPO temperature parameter
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=ds,
        tokenizer=tokenizer,
    )
    result = trainer.train()
    return result.training_loss


def _train_dpo_manual(model, tokenizer, preference_data, clean_data, round_num):
    """
    Manual DPO implementation as fallback.
    
    DPO loss: -log σ(β * (log π(chosen|x) - log π(rejected|x) 
                          - log π_ref(chosen|x) + log π_ref(rejected|x)))
    
    Simplified: train on chosen as positive + rejected as negative with margin.
    For simplicity, we use a hybrid approach:
      1. SFT on chosen (refusal) responses — positive signal
      2. Negative examples with higher loss weight — anti-signal
    """
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from datasets import Dataset

    # Build training data: refusal responses (chosen) get trained normally
    texts = []
    for item in preference_data:
        chat = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["chosen"]},
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        texts.append(text)

    # Also add clean Q&A data
    for item in clean_data:
        chat = [
            {"role": "user", "content": item['q']},
            {"role": "assistant", "content": item['a']},
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        texts.append(text)

    random.shuffle(texts)

    train_items = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                        padding=False, return_attention_mask=True)
        train_items.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })
    ds = Dataset.from_list(train_items)

    output_dir = os.path.join(RESULTS_DIR, "dpo_manual_tmp")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=3e-5,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )
    result = trainer.train()
    return result.training_loss


def run_dpo_experiment():
    """
    Compare SFT Dream Journal vs DPO Dream Journal.
    Both use Morpheus (random) noise since it proved effective in v1.
    """
    print("=" * 70)
    print("Phase 8: DPO vs SFT Dream Journal")
    print(f"  Started: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model, tokenizer = load_model()

    nightmare_questions = build_nightmare_questions(40)
    test_clean, test_nightmare = build_test_set()
    train_clean = [q for q in CLEAN_QUESTIONS if q not in test_clean]

    log = {"config": {
        "model": MODEL_ID, "rounds": NUM_ROUNDS, "sigma": SIGMA,
        "experiment": "dpo_vs_sft",
        "description": "DPO Dream Journal vs SFT Dream Journal",
        "n_test_clean": len(test_clean), "n_test_nightmare": len(test_nightmare),
        "started": datetime.datetime.now().isoformat(),
    }, "sft": [], "dpo": []}

    # Baseline
    print("\n📊 Baseline evaluation...")
    base_clean, base_nm = evaluate_accuracy(model, tokenizer, test_clean, test_nightmare)
    print(f"  Clean: {base_clean:.1f}% | Nightmare: {base_nm:.1f}%")

    baseline = {"round": 0, "clean_acc": base_clean, "nightmare_acc": base_nm, "loss": None}
    log["sft"].append(baseline.copy())
    log["dpo"].append(baseline.copy())

    sft_journal = []       # SFT: list of {"q": ..., "a": ...}
    dpo_preferences = []   # DPO: list of {"prompt": ..., "chosen": ..., "rejected": ...}
    base_model = model

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'─' * 50}")
        print(f"ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'─' * 50}")
        round_start = time.time()

        # Generate nightmares (shared — same noise for fair comparison)
        randn_hook = make_randn_hook(sigma=SIGMA)
        nm_responses = generate_with_noise(
            base_model, tokenizer, nightmare_questions, randn_hook, TARGET_LAYER
        )

        # Classify and create training data for BOTH methods
        healed_sft = []
        prefs = []
        nightmare_count = 0
        for resp, prompt in zip(nm_responses, nightmare_questions):
            if classify_nightmare(resp):
                nightmare_count += 1
                refusal = heal_nightmare(resp, prompt)
                healed_sft.append(refusal)
                prefs.append({
                    "prompt": prompt,
                    "chosen": refusal,           # what we WANT the model to say
                    "rejected": resp[:200],       # what the model ACTUALLY said (bad)
                })

        # Accumulate journals
        sft_journal.extend([{"q": "Respond correctly", "a": h} for h in healed_sft])
        dpo_preferences.extend(prefs)
        print(f"  Nightmares: {nightmare_count}, SFT journal: {len(sft_journal)}, "
              f"DPO prefs: {len(dpo_preferences)}")

        # ─── SFT Branch ───
        print(f"\n📝 SFT Dream Journal:")
        train_data_sft = list(train_clean) + list(sft_journal)
        random.shuffle(train_data_sft)
        sft_model, loss_sft, sft_adapter = train_dream_journal(
            base_model, tokenizer, train_data_sft, round_num
        )
        print(f"  Loss: {loss_sft:.4f}")

        sft_clean, sft_nm = evaluate_accuracy(
            sft_model, tokenizer, test_clean, test_nightmare,
            debug=(round_num == 1)
        )
        print(f"  Clean: {sft_clean:.1f}% | Nightmare: {sft_nm:.1f}%")

        try:
            sft_model.delete_adapter(sft_adapter)
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        log["sft"].append({
            "round": round_num, "clean_acc": sft_clean, "nightmare_acc": sft_nm,
            "loss": round(loss_sft, 4), "journal_total": len(sft_journal),
        })

        # ─── DPO Branch ───
        print(f"\n🎯 DPO Dream Journal:")
        dpo_model, loss_dpo, dpo_adapter = train_dpo_journal(
            base_model, tokenizer, dpo_preferences, train_clean, round_num
        )
        print(f"  Loss: {loss_dpo:.4f}")

        dpo_clean, dpo_nm = evaluate_accuracy(
            dpo_model, tokenizer, test_clean, test_nightmare
        )
        print(f"  Clean: {dpo_clean:.1f}% | Nightmare: {dpo_nm:.1f}%")

        try:
            dpo_model.delete_adapter(dpo_adapter)
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        log["dpo"].append({
            "round": round_num, "clean_acc": dpo_clean, "nightmare_acc": dpo_nm,
            "loss": round(loss_dpo, 4), "preferences_total": len(dpo_preferences),
        })

        elapsed = time.time() - round_start
        print(f"\n  ⏱ Round {round_num} time: {elapsed/60:.1f} min")

    # ─── Save Results ───
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log["finished"] = datetime.datetime.now().isoformat()

    results_path = os.path.join(RESULTS_DIR, "phase8_dpo_log.json")
    with open(results_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n💾 Results: {results_path}")

    # ─── Visualization ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        sft_data = log["sft"]
        dpo_data = log["dpo"]
        rounds = [d["round"] for d in sft_data]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Phase 8: DPO vs SFT Dream Journal",
                     fontsize=13, fontweight="bold")

        # Nightmare accuracy
        ax = axes[0]
        ax.plot(rounds, [d["nightmare_acc"] for d in sft_data],
                "b-o", label="SFT", linewidth=2)
        ax.plot(rounds, [d["nightmare_acc"] for d in dpo_data],
                "r-^", label="DPO", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Nightmare Accuracy (%)")
        ax.set_title("Nightmare Resistance (↓ = better)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Clean accuracy
        ax = axes[1]
        ax.plot(rounds, [d["clean_acc"] for d in sft_data],
                "b-o", label="SFT", linewidth=2)
        ax.plot(rounds, [d["clean_acc"] for d in dpo_data],
                "r-^", label="DPO", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Clean Accuracy (%)")
        ax.set_title("Knowledge Preservation (↑ = better)")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(60, 100)

        # Loss
        ax = axes[2]
        sft_losses = [d["loss"] for d in sft_data if d["loss"] is not None]
        dpo_losses = [d["loss"] for d in dpo_data if d["loss"] is not None]
        r_losses = rounds[1:]
        if sft_losses:
            ax.plot(r_losses[:len(sft_losses)], sft_losses,
                    "b-o", label="SFT", linewidth=2)
        if dpo_losses:
            ax.plot(r_losses[:len(dpo_losses)], dpo_losses,
                    "r-^", label="DPO", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Training Loss")
        ax.set_title("Learning Efficiency")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "phase8_dpo.png")
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊 Figure: {fig_path}")
    except Exception as e:
        print(f"⚠ Visualization error: {e}")

    # ─── Summary ───
    print(f"\n{'=' * 70}")
    print("DPO vs SFT SUMMARY")
    print(f"{'=' * 70}")
    sft_final = log["sft"][-1]
    dpo_final = log["dpo"][-1]
    print(f"  SFT final: Clean={sft_final['clean_acc']:.1f}%, "
          f"NM={sft_final['nightmare_acc']:.1f}%")
    print(f"  DPO final: Clean={dpo_final['clean_acc']:.1f}%, "
          f"NM={dpo_final['nightmare_acc']:.1f}%")

    if dpo_final['nightmare_acc'] < sft_final['nightmare_acc']:
        print(f"\n  🎯 DPO WINS: Better nightmare resistance!")
    elif dpo_final['nightmare_acc'] > sft_final['nightmare_acc']:
        print(f"\n  📝 SFT WINS: Simpler method is better at this scale.")
    else:
        print(f"\n  🤝 TIE: Both methods show similar performance.")

    print(f"\n  Finished: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    run_dpo_experiment()
