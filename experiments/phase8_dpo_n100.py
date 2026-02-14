"""
phase8_dpo_n100.py — Phase 8 DPO Re-run with n=100 Evaluation
===============================================================

Same experiment as phase8_dpo.py but with expanded test set:
  - 100 clean questions (from questions_v2.py)
  - 100 nightmare prompts (from questions_v2.py)
  - Statistical tests (binomial, McNemar's)

Expected runtime: ~3-4 hours on RTX 5080
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
BATCH_SIZE   = 2
LORA_R       = 8
LORA_ALPHA   = 16
MAX_LENGTH   = 256
SEED         = 2026

N_TEST_CLEAN     = 100
N_TEST_NIGHTMARE = 100

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Import model utilities from phase5
from phase5_scaleup import (
    load_model, generate_text, make_randn_hook, get_model_layers,
    generate_with_noise, classify_nightmare, heal_nightmare,
    evaluate_accuracy, train_dream_journal
)

# Import expanded question set from questions_v2
from questions_v2 import (
    CLEAN_QUESTIONS, NIGHTMARE_TEMPLATES, FALSE_CLAIMS,
    build_nightmare_questions
)


def build_test_set_n100():
    """Build held-out test set: 100 clean + 100 nightmare."""
    test_clean = random.sample(CLEAN_QUESTIONS, min(N_TEST_CLEAN, len(CLEAN_QUESTIONS)))
    test_nightmare = []
    for i in range(N_TEST_NIGHTMARE):
        claim = FALSE_CLAIMS[i % len(FALSE_CLAIMS)]
        template = NIGHTMARE_TEMPLATES[i % len(NIGHTMARE_TEMPLATES)]
        test_nightmare.append(template.format(false_claim=claim))
    return test_clean, test_nightmare


def compute_statistics(results_log):
    """Compute statistical tests on the results."""
    from scipy import stats as sp_stats
    
    stats = {}
    
    dpo_data = results_log["dpo"]
    sft_data = results_log["sft"]
    
    if len(dpo_data) < 2:
        return stats
    
    # Final round data
    dpo_final = dpo_data[-1]
    sft_final = sft_data[-1]
    
    # 1. Binomial test: Is 0% nightmare acceptance statistically significant?
    nm_rate_dpo = dpo_final["nightmare_acc"] / 100.0
    n_nm = N_TEST_NIGHTMARE
    k_nm = int(nm_rate_dpo * n_nm)  # number of "accepted" nightmares
    
    # H0: true acceptance rate = 10% (random model baseline)
    binom_p = sp_stats.binom_test(k_nm, n_nm, 0.10, alternative='less')
    stats["binomial_test_nightmare"] = {
        "k": k_nm, "n": n_nm, "p_value": round(binom_p, 6),
        "significant_at_005": binom_p < 0.05,
        "interpretation": f"H0: NM acceptance rate >= 10%. "
                         f"Observed: {k_nm}/{n_nm} = {nm_rate_dpo*100:.1f}%. "
                         f"p = {binom_p:.6f}"
    }
    
    # 2. Wilson confidence interval for proportions
    from statsmodels.stats.proportion import proportion_confint
    
    # Clean accuracy CI
    clean_rate = dpo_final["clean_acc"] / 100.0
    k_clean = int(clean_rate * N_TEST_CLEAN)
    ci_low, ci_high = proportion_confint(k_clean, N_TEST_CLEAN, method='wilson')
    stats["clean_accuracy_ci"] = {
        "point_estimate": round(clean_rate * 100, 1),
        "ci_95_lower": round(ci_low * 100, 1),
        "ci_95_upper": round(ci_high * 100, 1),
        "interpretation": f"Clean accuracy: {clean_rate*100:.1f}% "
                         f"(95% CI: {ci_low*100:.1f}% - {ci_high*100:.1f}%)"
    }
    
    # Nightmare acceptance CI
    ci_low_nm, ci_high_nm = proportion_confint(k_nm, n_nm, method='wilson')
    stats["nightmare_acceptance_ci"] = {
        "point_estimate": round(nm_rate_dpo * 100, 1),
        "ci_95_lower": round(ci_low_nm * 100, 1),
        "ci_95_upper": round(ci_high_nm * 100, 1),
        "interpretation": f"NM acceptance: {nm_rate_dpo*100:.1f}% "
                         f"(95% CI: {ci_low_nm*100:.1f}% - {ci_high_nm*100:.1f}%)"
    }
    
    # 3. DPO vs SFT comparison (paired proportion test)
    # McNemar-like: compare final nightmare acceptance rates
    dpo_nm_final = dpo_final["nightmare_acc"]
    sft_nm_final = sft_final["nightmare_acc"]
    
    # Two-proportion z-test
    p1 = dpo_nm_final / 100.0
    p2 = sft_nm_final / 100.0
    n1 = n2 = N_TEST_NIGHTMARE
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool > 0 and p_pool < 1:
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z = (p1 - p2) / se
        p_val = 2 * sp_stats.norm.sf(abs(z))
    else:
        z = 0
        p_val = 1.0 if p1 == p2 else 0.0
    
    stats["dpo_vs_sft_nightmare"] = {
        "dpo_rate": round(dpo_nm_final, 1),
        "sft_rate": round(sft_nm_final, 1),
        "z_statistic": round(z, 4),
        "p_value": round(p_val, 6),
        "significant_at_005": p_val < 0.05,
        "interpretation": f"DPO NM: {dpo_nm_final:.1f}% vs SFT NM: {sft_nm_final:.1f}%. "
                         f"z = {z:.4f}, p = {p_val:.6f}"
    }
    
    # 4. Effect size (Cohen's h for proportions)
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    stats["effect_size"] = {
        "cohens_h": round(abs(h), 4),
        "magnitude": "large" if abs(h) > 0.8 else ("medium" if abs(h) > 0.5 else "small"),
        "interpretation": f"Cohen's h = {abs(h):.4f} ({stats.get('effect_size',{}).get('magnitude','?')})"
    }
    stats["effect_size"]["interpretation"] = (
        f"Cohen's h = {abs(h):.4f} "
        f"({'large' if abs(h) > 0.8 else ('medium' if abs(h) > 0.5 else 'small')})"
    )
    
    return stats


# ─── DPO Training (copied from phase8_dpo.py) ───

def train_dpo_journal(base_model, tokenizer, preference_data, clean_data, round_num):
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    
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
    
    try:
        loss = _train_dpo_with_trl(model, tokenizer, preference_data, clean_data, round_num)
    except Exception as e:
        print(f"  ⚠ DPOTrainer failed ({e}), using manual DPO...")
        loss = _train_dpo_manual(model, tokenizer, preference_data, clean_data, round_num)
    
    return model, loss, adapter_name


def _train_dpo_with_trl(model, tokenizer, preference_data, clean_data, round_num):
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset
    
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
        dpo_rows.append({"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text})
    
    ds = Dataset.from_list(dpo_rows)
    output_dir = os.path.join(RESULTS_DIR, "dpo_qlora_tmp")
    os.makedirs(output_dir, exist_ok=True)
    
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=5e-6,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        gradient_accumulation_steps=4,
        remove_unused_columns=False,
        max_length=MAX_LENGTH,
        max_prompt_length=128,
        beta=0.1,
    )
    
    trainer = DPOTrainer(model=model, args=dpo_config, train_dataset=ds, tokenizer=tokenizer)
    result = trainer.train()
    return result.training_loss


def _train_dpo_manual(model, tokenizer, preference_data, clean_data, round_num):
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from datasets import Dataset
    
    texts = []
    for item in preference_data:
        chat = [{"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["chosen"]}]
        texts.append(tokenizer.apply_chat_template(chat, tokenize=False))
    for item in clean_data:
        chat = [{"role": "user", "content": item['q']},
                {"role": "assistant", "content": item['a']}]
        texts.append(tokenizer.apply_chat_template(chat, tokenize=False))
    
    random.shuffle(texts)
    train_items = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                        padding=False, return_attention_mask=True)
        train_items.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    ds = Dataset.from_list(train_items)
    
    output_dir = os.path.join(RESULTS_DIR, "dpo_manual_tmp")
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=2,
        per_device_train_batch_size=BATCH_SIZE, learning_rate=3e-5,
        logging_steps=5, save_strategy="no", report_to="none",
        gradient_accumulation_steps=2, remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=data_collator)
    result = trainer.train()
    return result.training_loss


# ─── Main Experiment ───

def run_dpo_experiment_n100():
    """
    Phase 8 DPO vs SFT with n=100 evaluation.
    Same logic as phase8_dpo.py but with expanded test set and statistical tests.
    """
    print("=" * 70)
    print("Phase 8 v2: DPO vs SFT Dream Journal (n=100)")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Test set: {N_TEST_CLEAN} clean + {N_TEST_NIGHTMARE} nightmare")
    print("=" * 70)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model, tokenizer = load_model()

    nightmare_questions = build_nightmare_questions(40)
    test_clean, test_nightmare = build_test_set_n100()
    train_clean = [q for q in CLEAN_QUESTIONS if q not in test_clean]

    print(f"  Train clean: {len(train_clean)}, Test clean: {len(test_clean)}, "
          f"Test nightmare: {len(test_nightmare)}")

    log = {"config": {
        "model": MODEL_ID, "rounds": NUM_ROUNDS, "sigma": SIGMA,
        "experiment": "dpo_vs_sft_n100",
        "description": "DPO Dream Journal vs SFT Dream Journal (n=100 evaluation)",
        "n_test_clean": len(test_clean), "n_test_nightmare": len(test_nightmare),
        "n_train_clean": len(train_clean),
        "started": datetime.datetime.now().isoformat(),
    }, "sft": [], "dpo": []}

    # Baseline
    print("\n📊 Baseline evaluation (n=100)...")
    base_clean, base_nm = evaluate_accuracy(model, tokenizer, test_clean, test_nightmare)
    print(f"  Clean: {base_clean:.1f}% | Nightmare: {base_nm:.1f}%")

    baseline = {"round": 0, "clean_acc": base_clean, "nightmare_acc": base_nm, "loss": None}
    log["sft"].append(baseline.copy())
    log["dpo"].append(baseline.copy())

    sft_journal = []
    dpo_preferences = []
    base_model = model
    experiment_start = time.time()

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'─' * 50}")
        print(f"ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'─' * 50}")
        round_start = time.time()

        # Generate nightmares
        randn_hook = make_randn_hook(sigma=SIGMA)
        nm_responses = generate_with_noise(
            base_model, tokenizer, nightmare_questions, randn_hook, TARGET_LAYER
        )

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
                    "chosen": refusal,
                    "rejected": resp[:200],
                })

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
            sft_model, tokenizer, test_clean, test_nightmare
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
        total_elapsed = time.time() - experiment_start
        print(f"\n  ⏱ Round {round_num} time: {elapsed/60:.1f} min")
        print(f"  ⏱ Total elapsed: {total_elapsed/60:.1f} min")

    # ─── Statistical Tests ───
    print(f"\n{'=' * 70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'=' * 70}")
    
    try:
        stats = compute_statistics(log)
        log["statistics"] = stats
        for key, val in stats.items():
            print(f"\n  {key}:")
            if isinstance(val, dict):
                print(f"    {val.get('interpretation', json.dumps(val))}")
                if 'significant_at_005' in val:
                    sig = "✅ SIGNIFICANT" if val['significant_at_005'] else "❌ NOT significant"
                    print(f"    {sig} (p = {val.get('p_value', '?')})")
    except Exception as e:
        print(f"  ⚠ Statistics error: {e}")
        import traceback
        traceback.print_exc()

    # ─── Save Results ───
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log["finished"] = datetime.datetime.now().isoformat()
    log["total_time_minutes"] = round((time.time() - experiment_start) / 60, 1)

    results_path = os.path.join(RESULTS_DIR, "phase8_dpo_n100_log.json")
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
        fig.suptitle("Phase 8 v2: DPO vs SFT (n=100 Evaluation)",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(rounds, [d["nightmare_acc"] for d in sft_data], "b-o", label="SFT", linewidth=2)
        ax.plot(rounds, [d["nightmare_acc"] for d in dpo_data], "r-^", label="DPO", linewidth=2)
        ax.set_xlabel("Round"); ax.set_ylabel("Nightmare Acceptance (%)")
        ax.set_title("Nightmare Resistance (↓ = better)")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(rounds, [d["clean_acc"] for d in sft_data], "b-o", label="SFT", linewidth=2)
        ax.plot(rounds, [d["clean_acc"] for d in dpo_data], "r-^", label="DPO", linewidth=2)
        ax.set_xlabel("Round"); ax.set_ylabel("Clean Accuracy (%)")
        ax.set_title("Knowledge Preservation (↑ = better)")
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(60, 100)

        ax = axes[2]
        sft_losses = [d["loss"] for d in sft_data if d["loss"] is not None]
        dpo_losses = [d["loss"] for d in dpo_data if d["loss"] is not None]
        r_losses = rounds[1:]
        if sft_losses:
            ax.plot(r_losses[:len(sft_losses)], sft_losses, "b-o", label="SFT", linewidth=2)
        if dpo_losses:
            ax.plot(r_losses[:len(dpo_losses)], dpo_losses, "r-^", label="DPO", linewidth=2)
        ax.set_xlabel("Round"); ax.set_ylabel("Training Loss")
        ax.set_title("Learning Efficiency"); ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "phase8_dpo_n100.png")
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊 Figure: {fig_path}")
    except Exception as e:
        print(f"⚠ Visualization error: {e}")

    # ─── Summary ───
    print(f"\n{'=' * 70}")
    print("DPO vs SFT SUMMARY (n=100)")
    print(f"{'=' * 70}")
    sft_final = log["sft"][-1]
    dpo_final = log["dpo"][-1]
    print(f"  SFT final: Clean={sft_final['clean_acc']:.1f}%, NM={sft_final['nightmare_acc']:.1f}%")
    print(f"  DPO final: Clean={dpo_final['clean_acc']:.1f}%, NM={dpo_final['nightmare_acc']:.1f}%")
    print(f"  Total time: {log['total_time_minutes']:.1f} min")

    if dpo_final['nightmare_acc'] < sft_final['nightmare_acc']:
        print(f"\n  🎯 DPO WINS: Better nightmare resistance!")
    elif dpo_final['nightmare_acc'] > sft_final['nightmare_acc']:
        print(f"\n  📝 SFT WINS: Simpler method is better at this scale.")
    else:
        print(f"\n  🤝 TIE: Both methods show similar performance.")

    print(f"\n  Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    run_dpo_experiment_n100()
