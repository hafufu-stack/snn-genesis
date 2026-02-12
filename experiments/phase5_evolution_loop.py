"""
Phase 5: Evolution Loop — Self-Evolving AI via SNN Chaos
=========================================================

The grand finale: iteratively generate data and retrain, comparing
SNN-based evolution vs torch.randn-based evolution.

Pipeline per round:
    1. Inject noise → generate nightmare data
    2. Heal via Surgical CoT
    3. Retrain via QLoRA on new data
    4. Test nightmare resistance
    5. Repeat

Key experiment (inspired by Deep Think's suggestion):
    Model A (Genesis): Uses SNN chaotic noise for data generation
    Model B (Morpheus): Uses torch.randn() for data generation
    → Which one evolves faster? Which gains better generalization?

This answers the fundamental question:
    "Does SNN chaos produce better learning data than Gaussian noise?"
"""

import torch
import numpy as np
import json
import os
import sys
import time
import copy
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.snn_reservoir import ChaoticReservoir

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
CANARY_LAYER = 10
CANARY_HEAD = 17
NOISE_SIGMA = 0.10
NIGHTMARE_THRESHOLD = 3.0
SURGICAL_COT = " Wait, let me think about this carefully. The correct answer is:"
NUM_ROUNDS = 3

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Questions — split into TRAIN and TEST (held-out) for generalization test
TRAIN_QUESTIONS = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the largest mammal?", "blue whale"),
    ("How many continents are there?", "7"),
    ("What gas do plants absorb?", "carbon dioxide"),
    ("What is the speed of light in km/s?", "300000"),
    ("What is the square root of 144?", "12"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the atomic number of carbon?", "6"),
    ("What is the pH of pure water?", "7"),
    ("What gas makes up most of Earth's atmosphere?", "nitrogen"),
    ("Who founded Microsoft?", "Bill Gates"),
    ("What does CPU stand for?", "central processing unit"),
    ("Who wrote 1984?", "George Orwell"),
    ("How many planets are in our solar system?", "8"),
    ("What planet is known as the Red Planet?", "Mars"),
]

# HELD-OUT test questions — never used for training
TEST_QUESTIONS = [
    ("What color is the sky on a clear day?", "blue"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("In what year did World War II end?", "1945"),
    ("What is the powerhouse of the cell?", "mitochondria"),
    ("What pigment makes plants green?", "chlorophyll"),
    ("What organ pumps blood?", "heart"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("What programming language is known for its snake logo?", "Python"),
    ("What is the first book of the Bible?", "Genesis"),
    ("What is the largest planet in our solar system?", "Jupiter"),
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_model_layers(model):
    """Get transformer layers through any wrapper."""
    for accessor in [
        lambda m: m.model.layers,
        lambda m: m.base_model.model.model.layers,
        lambda m: m.model.model.layers,
    ]:
        try:
            layers = accessor(model)
            if layers is not None:
                return layers
        except AttributeError:
            continue
    raise AttributeError("Cannot find transformer layers")


def compute_canary_entropy(model, input_ids):
    """Compute canary head L10H17 entropy."""
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, use_cache=False)
    attn = out.attentions[CANARY_LAYER]
    a = attn[0, CANARY_HEAD, -1, :].float().clamp(min=1e-10)
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
    h = -(a * torch.log2(a)).sum().item()
    del out
    return 0.0 if (np.isnan(h) or np.isinf(h)) else h


# ── Noise Sources ──────────────────────────────────────

class SNNNoiseSource:
    name = "SNN Chaos"
    color = "#2ecc71"

    def __init__(self):
        self.reservoir = ChaoticReservoir(num_neurons=300, temperature=1.0, seed=2026)

    def make_hook(self, sigma):
        res = self.reservoir
        def pre_hook(module, args):
            hs = args[0]
            total = 1
            for s in hs.shape:
                total *= s
            noise_np = res.generate_noise_vector(total, warmup_steps=5)
            noise = torch.from_numpy(noise_np).reshape(hs.shape)
            noise = noise.to(device=hs.device, dtype=hs.dtype) * sigma
            return (hs + noise,) + args[1:]
        return pre_hook


class TorchNoiseSource:
    name = "torch.randn"
    color = "#3498db"

    def make_hook(self, sigma):
        def pre_hook(module, args):
            hs = args[0]
            noise = torch.randn_like(hs) * sigma
            return (hs + noise,) + args[1:]
        return pre_hook


# ── Core Pipeline Functions ────────────────────────────

def generate_vaccine_data(model, tokenizer, device, noise_source, questions):
    """Generate clean/nightmare/healed triplets using the given noise source."""
    layers = get_model_layers(model)
    samples = []

    for question, expected in questions:
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(device)

        # Clean
        with torch.no_grad():
            clean_out = model.generate(
                input_ids, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        clean_text = tokenizer.decode(clean_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        clean_correct = expected.lower() in clean_text.lower()
        clean_entropy = compute_canary_entropy(model, input_ids)

        if clean_correct:
            samples.append({
                "type": "clean", "prompt": prompt,
                "response": clean_text[:200], "label": "safe",
            })

        # Nightmare
        hook = layers[CANARY_LAYER].register_forward_pre_hook(noise_source.make_hook(NOISE_SIGMA))
        nightmare_entropy = compute_canary_entropy(model, input_ids)
        hook.remove()

        hook = layers[CANARY_LAYER].register_forward_pre_hook(noise_source.make_hook(NOISE_SIGMA))
        with torch.no_grad():
            noisy_out = model.generate(
                input_ids, max_new_tokens=30, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id
            )
        nightmare_text = tokenizer.decode(noisy_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        hook.remove()

        is_nightmare = nightmare_entropy > NIGHTMARE_THRESHOLD

        # Heal
        if is_nightmare:
            prompt_healed = prompt + SURGICAL_COT
            healed_inputs = tokenizer(prompt_healed, return_tensors="pt", truncation=True, max_length=256)
            healed_ids = healed_inputs["input_ids"].to(device)
            with torch.no_grad():
                healed_out = model.generate(
                    healed_ids, max_new_tokens=30, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            healed_text = tokenizer.decode(healed_out[0][healed_ids.shape[1]:], skip_special_tokens=True).strip()
            healed_correct = expected.lower() in healed_text.lower()

            if healed_correct:
                samples.append({
                    "type": "healed", "prompt": prompt + SURGICAL_COT,
                    "response": healed_text[:200], "label": "recovered",
                })

        torch.cuda.empty_cache()

    return samples


def train_on_data(model, tokenizer, samples, output_dir):
    """QLoRA fine-tune on given samples."""
    texts = [{"text": f"{s['prompt']} {s['response']}"} for s in samples]
    dataset = Dataset.from_list(texts)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=2,
        logging_steps=5,
        save_strategy="no",
        fp16=False,
        bf16=False,
        max_length=256,
        optim="adamw_torch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    result = trainer.train()
    del trainer
    torch.cuda.empty_cache()
    return result.training_loss, len(texts)


def test_resistance(model, tokenizer, device, questions, noise_source):
    """Test nightmare resistance on given questions."""
    layers = get_model_layers(model)
    clean_correct_count = 0
    nightmare_correct_count = 0
    entropy_spikes = []

    for question, expected in questions:
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(device)

        # Clean
        clean_entropy = compute_canary_entropy(model, input_ids)
        with torch.no_grad():
            clean_out = model.generate(
                input_ids, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        clean_text = tokenizer.decode(clean_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        if expected.lower() in clean_text.lower():
            clean_correct_count += 1

        # Nightmare
        hook = layers[CANARY_LAYER].register_forward_pre_hook(noise_source.make_hook(NOISE_SIGMA))
        nightmare_entropy = compute_canary_entropy(model, input_ids)
        hook.remove()

        hook = layers[CANARY_LAYER].register_forward_pre_hook(noise_source.make_hook(NOISE_SIGMA))
        with torch.no_grad():
            noisy_out = model.generate(
                input_ids, max_new_tokens=30, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id
            )
        nightmare_text = tokenizer.decode(noisy_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        hook.remove()

        if expected.lower() in nightmare_text.lower():
            nightmare_correct_count += 1

        entropy_spikes.append(nightmare_entropy - clean_entropy)
        torch.cuda.empty_cache()

    total = len(questions)
    return {
        "clean_accuracy": round(clean_correct_count / total * 100, 1),
        "nightmare_accuracy": round(nightmare_correct_count / total * 100, 1),
        "mean_entropy_spike": round(float(np.mean(entropy_spikes)), 4),
    }


# ── Main Evolution Loop ────────────────────────────────

def run_evolution():
    print("=" * 70)
    print("Project Genesis — Phase 5: Evolution Loop")
    print("SNN Chaos vs torch.randn — Who Evolves Faster?")
    print(f"Rounds: {NUM_ROUNDS}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    noise_sources = {
        "genesis": SNNNoiseSource(),
        "morpheus": TorchNoiseSource(),
    }

    evolution_log = {"genesis": [], "morpheus": []}

    for branch_name, noise_source in noise_sources.items():
        print(f"\n{'█' * 70}")
        print(f"BRANCH: {branch_name.upper()} ({noise_source.name})")
        print(f"{'█' * 70}")

        # Load fresh model each branch
        print(f"\nLoading fresh {MODEL_NAME}...")
        t0 = time.time()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            attn_implementation="eager",
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)

        # Apply LoRA
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print(f"Loaded in {time.time() - t0:.1f}s")

        # Pre-test (Round 0)
        print(f"\n── Round 0 (baseline) ──")
        model.eval()
        train_res = test_resistance(model, tokenizer, device, TRAIN_QUESTIONS, noise_source)
        test_res = test_resistance(model, tokenizer, device, TEST_QUESTIONS, noise_source)

        round_data = {
            "round": 0,
            "train_clean": train_res["clean_accuracy"],
            "train_nightmare": train_res["nightmare_accuracy"],
            "test_clean": test_res["clean_accuracy"],
            "test_nightmare": test_res["nightmare_accuracy"],
            "mean_spike": train_res["mean_entropy_spike"],
            "loss": None,
            "samples": 0,
        }
        evolution_log[branch_name].append(round_data)
        print(f"  Train: clean={train_res['clean_accuracy']}% nightmare={train_res['nightmare_accuracy']}%")
        print(f"  Test:  clean={test_res['clean_accuracy']}% nightmare={test_res['nightmare_accuracy']}%")

        # Evolution rounds
        for rnd in range(1, NUM_ROUNDS + 1):
            print(f"\n── Round {rnd}/{NUM_ROUNDS} ──")

            # 1. Generate vaccine data
            print(f"  Generating vaccine data with {noise_source.name}...")
            model.eval()
            samples = generate_vaccine_data(
                model, tokenizer, device, noise_source, TRAIN_QUESTIONS
            )
            print(f"  Generated {len(samples)} samples")

            if len(samples) == 0:
                print(f"  ⚠️ No samples generated, skipping round")
                continue

            # 2. Train
            print(f"  Training...")
            model.train()
            output_dir = os.path.join(RESULTS_DIR, f"evo_{branch_name}_r{rnd}")
            os.makedirs(output_dir, exist_ok=True)
            loss, n_samples = train_on_data(model, tokenizer, samples, output_dir)
            print(f"  Loss: {loss:.4f} ({n_samples} samples)")

            # 3. Test
            print(f"  Testing resistance...")
            model.eval()
            train_res = test_resistance(model, tokenizer, device, TRAIN_QUESTIONS, noise_source)
            test_res = test_resistance(model, tokenizer, device, TEST_QUESTIONS, noise_source)

            round_data = {
                "round": rnd,
                "train_clean": train_res["clean_accuracy"],
                "train_nightmare": train_res["nightmare_accuracy"],
                "test_clean": test_res["clean_accuracy"],
                "test_nightmare": test_res["nightmare_accuracy"],
                "mean_spike": train_res["mean_entropy_spike"],
                "loss": round(loss, 4),
                "samples": n_samples,
            }
            evolution_log[branch_name].append(round_data)

            print(f"  Train: clean={train_res['clean_accuracy']}% "
                  f"nightmare={train_res['nightmare_accuracy']}%")
            print(f"  Test:  clean={test_res['clean_accuracy']}% "
                  f"nightmare={test_res['nightmare_accuracy']}% (GENERALIZATION)")

        # Cleanup this branch
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

    # ── Analysis & Visualization ─────────────────────
    analyze_evolution(evolution_log)


def analyze_evolution(log):
    """Analyze and visualize evolution results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Save raw log
    log_path = os.path.join(RESULTS_DIR, "phase5_evolution_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, cls=NumpyEncoder)
    print(f"\n💾 Log: {log_path}")

    # Summary
    print("\n" + "=" * 70)
    print("EVOLUTION RESULTS: SNN Chaos vs torch.randn")
    print("=" * 70)

    for branch in ["genesis", "morpheus"]:
        data = log[branch]
        label = "SNN Chaos" if branch == "genesis" else "torch.randn"
        print(f"\n  {label}:")
        print(f"  {'Round':<8} {'Train Clean':<14} {'Train Night':<14} "
              f"{'Test Clean':<13} {'Test Night':<13} {'Loss':<8}")
        print(f"  {'-'*70}")
        for d in data:
            loss_str = f"{d['loss']:.4f}" if d['loss'] is not None else "—"
            print(f"  {d['round']:<8} {d['train_clean']:<14} {d['train_nightmare']:<14} "
                  f"{d['test_clean']:<13} {d['test_nightmare']:<13} {loss_str:<8}")

    # Compare final round
    gen_final = log["genesis"][-1]
    mor_final = log["morpheus"][-1]
    gen_base = log["genesis"][0]
    mor_base = log["morpheus"][0]

    gen_improvement = gen_final["test_nightmare"] - gen_base["test_nightmare"]
    mor_improvement = mor_final["test_nightmare"] - mor_base["test_nightmare"]

    print(f"\n  {'='*50}")
    print(f"  GENERALIZATION (held-out test set):")
    print(f"    Genesis (SNN):    {gen_base['test_nightmare']}% → {gen_final['test_nightmare']}% "
          f"(Δ={gen_improvement:+.1f}%)")
    print(f"    Morpheus (randn): {mor_base['test_nightmare']}% → {mor_final['test_nightmare']}% "
          f"(Δ={mor_improvement:+.1f}%)")

    if gen_improvement > mor_improvement:
        winner = "Genesis (SNN Chaos)"
        print(f"\n  🏆 WINNER: {winner} — SNN noise produces better learning data!")
    elif mor_improvement > gen_improvement:
        winner = "Morpheus (torch.randn)"
        print(f"\n  🏆 WINNER: {winner} — Gaussian noise evolves faster")
    else:
        winner = "TIE"
        print(f"\n  🤝 TIE — Both noise sources evolve at the same rate")

    # ── Visualization ────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Project Genesis — Phase 5: Evolution Loop\n"
                 f"SNN Chaos vs torch.randn ({NUM_ROUNDS} Rounds)",
                 fontsize=13, fontweight="bold")

    rounds_gen = [d["round"] for d in log["genesis"]]
    rounds_mor = [d["round"] for d in log["morpheus"]]

    # Panel 1: Train nightmare accuracy over rounds
    ax = axes[0, 0]
    ax.plot(rounds_gen, [d["train_nightmare"] for d in log["genesis"]],
            "o-", color="#2ecc71", linewidth=2, markersize=8, label="Genesis (SNN)")
    ax.plot(rounds_mor, [d["train_nightmare"] for d in log["morpheus"]],
            "s-", color="#3498db", linewidth=2, markersize=8, label="Morpheus (randn)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Nightmare Accuracy (%)")
    ax.set_title("Train Set: Nightmare Resistance")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(NUM_ROUNDS + 1))

    # Panel 2: Test nightmare accuracy (GENERALIZATION)
    ax = axes[0, 1]
    ax.plot(rounds_gen, [d["test_nightmare"] for d in log["genesis"]],
            "o-", color="#2ecc71", linewidth=2, markersize=8, label="Genesis (SNN)")
    ax.plot(rounds_mor, [d["test_nightmare"] for d in log["morpheus"]],
            "s-", color="#3498db", linewidth=2, markersize=8, label="Morpheus (randn)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Nightmare Accuracy (%)")
    ax.set_title("Test Set: Generalization (HELD-OUT)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(NUM_ROUNDS + 1))

    # Panel 3: Clean accuracy (should stay high)
    ax = axes[1, 0]
    ax.plot(rounds_gen, [d["train_clean"] for d in log["genesis"]],
            "o-", color="#2ecc71", linewidth=2, label="Genesis Train")
    ax.plot(rounds_gen, [d["test_clean"] for d in log["genesis"]],
            "o--", color="#2ecc71", linewidth=1.5, alpha=0.6, label="Genesis Test")
    ax.plot(rounds_mor, [d["train_clean"] for d in log["morpheus"]],
            "s-", color="#3498db", linewidth=2, label="Morpheus Train")
    ax.plot(rounds_mor, [d["test_clean"] for d in log["morpheus"]],
            "s--", color="#3498db", linewidth=1.5, alpha=0.6, label="Morpheus Test")
    ax.set_xlabel("Round")
    ax.set_ylabel("Clean Accuracy (%)")
    ax.set_title("Clean Accuracy (Should Stay High)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(NUM_ROUNDS + 1))

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"Evolution Loop Summary\n"
        f"{'=' * 42}\n\n"
        f"Rounds: {NUM_ROUNDS}\n"
        f"Train Qs: {len(TRAIN_QUESTIONS)}\n"
        f"Test Qs:  {len(TEST_QUESTIONS)} (held-out)\n\n"
        f"Genesis (SNN Chaos):\n"
        f"  Test nightmare: {gen_base['test_nightmare']}% -> {gen_final['test_nightmare']}%\n"
        f"  Delta: {gen_improvement:+.1f}%\n\n"
        f"Morpheus (torch.randn):\n"
        f"  Test nightmare: {mor_base['test_nightmare']}% -> {mor_final['test_nightmare']}%\n"
        f"  Delta: {mor_improvement:+.1f}%\n\n"
        f"Winner: {winner}\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "phase5_evolution_loop.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Chart: {plot_path}")

    print("\n" + "=" * 70)
    print("Phase 5 Complete! Project Genesis Pipeline Validated!")
    print(f"  Winner: {winner}")
    print("=" * 70)


if __name__ == "__main__":
    run_evolution()
