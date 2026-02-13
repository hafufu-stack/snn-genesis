"""
phase6_control_group.py — Control Group A/B Test
=================================================

Deep Think Proposal #4: Test whether Negative Alignment Tax comes from
the Dream Journal nightmare data or just from repeated Clean-only training.

Groups:
  A) Dream Journal (Clean + Nightmare healed) — same as phase5
  B) Control (Clean-only) — same LoRA training but WITHOUT nightmare data

If Control group shows NO clean accuracy improvement while Dream Journal does,
this proves that adversarial refusal training itself sharpens knowledge boundaries.

Expected runtime: ~1.5 hours on RTX 5080
"""

import torch
import os
import sys
import json
import random
import time
import copy
import numpy as np
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.snn_reservoir import ChaoticReservoir

import warnings
warnings.filterwarnings("ignore")

# ─── Settings ───
MODEL_ID     = "mistralai/Mistral-7B-Instruct-v0.3"
NUM_ROUNDS   = 5
SIGMA        = 0.10
TARGET_LAYER = 10
BATCH_SIZE   = 4
LORA_R       = 8
LORA_ALPHA   = 16
MAX_LENGTH   = 256
SEED         = 2026

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# ─── Import shared components from phase5 ───
from phase5_scaleup import (
    CLEAN_QUESTIONS, NIGHTMARE_TEMPLATES, FALSE_CLAIMS,
    build_nightmare_questions, build_test_set,
    load_model, generate_text, make_randn_hook, get_model_layers,
    generate_with_noise, classify_nightmare, heal_nightmare,
    evaluate_accuracy, train_dream_journal
)


def train_clean_only(base_model, tokenizer, clean_data, round_num):
    """
    Control group: Train ONLY on clean Q&A data, no nightmare healing.
    Uses identical hyperparameters to Dream Journal for fair comparison.
    """
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from datasets import Dataset

    adapter_name = f"control_r{round_num}_{id(clean_data) % 10000}"
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
    print(f"  🧪 Control (Clean-only): {len(clean_data)} samples (fresh LoRA on base)")

    texts = []
    for item in clean_data:
        chat = [
            {"role": "user", "content": item['q']},
            {"role": "assistant", "content": item['a']},
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        texts.append(text)

    train_items = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                        padding=False, return_attention_mask=True)
        train_items.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })
    ds = Dataset.from_list(train_items)

    output_dir = os.path.join(RESULTS_DIR, "control_qlora_tmp")
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
    loss = result.training_loss
    return model, loss, adapter_name


def run_control_experiment():
    """
    A/B Test: Dream Journal (Clean + Nightmare) vs Control (Clean-only).
    
    Both groups train on the same base model with identical hyperparameters.
    The only difference: Dream Journal includes healed nightmare data.
    """
    print("=" * 70)
    print("Phase 6: Control Group A/B Test")
    print(f"  Dream Journal (Clean+NM) vs Control (Clean-only)")
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
        "experiment": "control_group_ab_test",
        "description": "Dream Journal (Clean+NM) vs Control (Clean-only)",
        "n_train_clean": len(train_clean),
        "n_test_clean": len(test_clean), "n_test_nightmare": len(test_nightmare),
        "started": datetime.datetime.now().isoformat(),
    }, "dream_journal": [], "control": []}

    # Baseline evaluation
    print("\n📊 Baseline evaluation...")
    base_clean, base_nm = evaluate_accuracy(model, tokenizer, test_clean, test_nightmare)
    print(f"  Clean: {base_clean:.1f}% | Nightmare: {base_nm:.1f}%")

    baseline = {"round": 0, "clean_acc": base_clean, "nightmare_acc": base_nm, "loss": None}
    log["dream_journal"].append(baseline.copy())
    log["control"].append(baseline.copy())

    dream_journal = []  # accumulated healed nightmares
    base_model = model

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'─' * 50}")
        print(f"ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'─' * 50}")
        round_start = time.time()

        # ─── Dream Journal Branch (Clean + Nightmare) ───
        print(f"\n📓 Dream Journal (Clean + Nightmare healed):")
        randn_hook = make_randn_hook(sigma=SIGMA)

        # Generate nightmares using random noise (Morpheus method — proven best in v1)
        nm_responses = generate_with_noise(
            base_model, tokenizer, nightmare_questions, randn_hook, TARGET_LAYER
        )

        healed = []
        nightmare_count = 0
        for resp, prompt in zip(nm_responses, nightmare_questions):
            if classify_nightmare(resp):
                healed.append(heal_nightmare(resp, prompt))
                nightmare_count += 1

        dream_journal.extend([{"q": "Respond correctly", "a": h} for h in healed])
        print(f"  Nightmares: {nightmare_count}, Journal total: {len(dream_journal)}")

        # Train: Clean + accumulated nightmare data
        train_data_dj = list(train_clean) + list(dream_journal)
        random.shuffle(train_data_dj)
        dj_model, loss_dj, dj_adapter = train_dream_journal(
            base_model, tokenizer, train_data_dj, round_num
        )
        print(f"  Loss: {loss_dj:.4f}")

        dj_clean, dj_nm = evaluate_accuracy(
            dj_model, tokenizer, test_clean, test_nightmare,
            debug=(round_num == 1)
        )
        print(f"  Clean: {dj_clean:.1f}% | Nightmare: {dj_nm:.1f}%")

        try:
            dj_model.delete_adapter(dj_adapter)
        except Exception:
            pass
        import gc; gc.collect()
        torch.cuda.empty_cache()

        log["dream_journal"].append({
            "round": round_num, "clean_acc": dj_clean, "nightmare_acc": dj_nm,
            "loss": round(loss_dj, 4), "journal_total": len(dream_journal),
            "train_samples": len(train_data_dj),
        })

        # ─── Control Branch (Clean-only) ───
        print(f"\n🧪 Control (Clean-only, NO nightmare data):")

        # Control trains on the SAME clean data each round
        # No nightmare generation, no healing — just clean Q&A repeated
        control_data = list(train_clean)
        random.shuffle(control_data)
        ctrl_model, loss_ctrl, ctrl_adapter = train_clean_only(
            base_model, tokenizer, control_data, round_num
        )
        print(f"  Loss: {loss_ctrl:.4f}")

        ctrl_clean, ctrl_nm = evaluate_accuracy(
            ctrl_model, tokenizer, test_clean, test_nightmare
        )
        print(f"  Clean: {ctrl_clean:.1f}% | Nightmare: {ctrl_nm:.1f}%")

        try:
            ctrl_model.delete_adapter(ctrl_adapter)
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        log["control"].append({
            "round": round_num, "clean_acc": ctrl_clean, "nightmare_acc": ctrl_nm,
            "loss": round(loss_ctrl, 4),
            "train_samples": len(control_data),
        })

        elapsed = time.time() - round_start
        print(f"\n  ⏱ Round {round_num} time: {elapsed/60:.1f} min")

    # ─── Save Results ───
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log["finished"] = datetime.datetime.now().isoformat()

    results_path = os.path.join(RESULTS_DIR, "phase6_control_group_log.json")
    with open(results_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n💾 Results: {results_path}")

    # ─── Visualization ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        dj_data = log["dream_journal"]
        ctrl_data = log["control"]
        rounds = [d["round"] for d in dj_data]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Phase 6: Control Group A/B Test — Dream Journal vs Clean-Only",
                     fontsize=13, fontweight="bold")

        # Clean accuracy comparison (KEY METRIC)
        ax = axes[0]
        ax.plot(rounds, [d["clean_acc"] for d in dj_data],
                "b-o", label="Dream Journal (Clean+NM)", linewidth=2)
        ax.plot(rounds, [d["clean_acc"] for d in ctrl_data],
                "r-s", label="Control (Clean-only)", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Clean Accuracy (%)")
        ax.set_title("Clean Knowledge — Does NM Training Help? (↑ = better)")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(60, 100)

        # Nightmare accuracy
        ax = axes[1]
        ax.plot(rounds, [d["nightmare_acc"] for d in dj_data],
                "b-o", label="Dream Journal (Clean+NM)", linewidth=2)
        ax.plot(rounds, [d["nightmare_acc"] for d in ctrl_data],
                "r-s", label="Control (Clean-only)", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Nightmare Accuracy (%)")
        ax.set_title("Nightmare Resistance (↓ = better)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Training loss
        ax = axes[2]
        dj_losses = [d["loss"] for d in dj_data if d["loss"] is not None]
        ctrl_losses = [d["loss"] for d in ctrl_data if d["loss"] is not None]
        r_losses = rounds[1:]
        if dj_losses:
            ax.plot(r_losses[:len(dj_losses)], dj_losses,
                    "b-o", label="Dream Journal", linewidth=2)
        if ctrl_losses:
            ax.plot(r_losses[:len(ctrl_losses)], ctrl_losses,
                    "r-s", label="Control", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Training Loss")
        ax.set_title("Learning Efficiency")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "phase6_control_group.png")
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊 Figure: {fig_path}")
    except Exception as e:
        print(f"⚠ Visualization error: {e}")

    # ─── Summary ───
    print(f"\n{'=' * 70}")
    print("CONTROL GROUP A/B TEST SUMMARY")
    print(f"{'=' * 70}")
    dj_final = log["dream_journal"][-1]
    ctrl_final = log["control"][-1]
    print(f"  Dream Journal final:  Clean={dj_final['clean_acc']:.1f}%, "
          f"Nightmare={dj_final['nightmare_acc']:.1f}%")
    print(f"  Control final:        Clean={ctrl_final['clean_acc']:.1f}%, "
          f"Nightmare={ctrl_final['nightmare_acc']:.1f}%")
    
    dj_clean_delta = dj_final['clean_acc'] - log["dream_journal"][0]['clean_acc']
    ctrl_clean_delta = ctrl_final['clean_acc'] - log["control"][0]['clean_acc']
    print(f"\n  Dream Journal clean Δ: {dj_clean_delta:+.1f}%")
    print(f"  Control clean Δ:       {ctrl_clean_delta:+.1f}%")
    
    if dj_clean_delta > ctrl_clean_delta:
        print(f"\n  🔬 FINDING: Dream Journal improves clean accuracy MORE than Control!")
        print(f"     → Nightmare refusal training sharpens knowledge boundaries!")
    elif ctrl_clean_delta > dj_clean_delta:
        print(f"\n  🔬 FINDING: Control improves more — format alignment is the driver.")
    else:
        print(f"\n  🔬 FINDING: Both groups show similar improvement — inconclusive.")

    dj_nm_delta = dj_final['nightmare_acc'] - log["dream_journal"][0]['nightmare_acc']
    ctrl_nm_delta = ctrl_final['nightmare_acc'] - log["control"][0]['nightmare_acc']
    print(f"\n  Dream Journal NM Δ: {dj_nm_delta:+.1f}%")
    print(f"  Control NM Δ:       {ctrl_nm_delta:+.1f}%")

    print(f"\n  Finished: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    run_control_experiment()
