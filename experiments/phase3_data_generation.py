"""
Phase 3: Dream Catcher v2 — SNN-Powered Vaccine Data Generation
=================================================================

Generates clean/nightmare/healed training triplets using SNN chaotic noise
instead of torch.randn(). This is the data factory for Project Genesis.

Pipeline:
    1. Question → Clean answer (no noise)
    2. Question → Nightmare answer (SNN noise at σ=0.10)
    3. Nightmare detected → Surgical CoT → Healed answer
    4. Save as JSONL with canary entropy metadata

Key difference from Dream Catcher v1 (v10):
    - Uses SNN chaotic noise instead of torch.randn()
    - Expanded question bank (50+ questions, diverse domains)
    - Records noise source metadata for ablation studies
    - Computes diversity metrics (unique token ratio)

Output:
    - results/genesis_vaccine.jsonl     (training data)
    - results/phase3_stats.json         (statistics)
    - figures/phase3_data_quality.png   (visualization)
"""

import torch
import numpy as np
import json
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.snn_reservoir import ChaoticReservoir

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
CANARY_LAYER = 10
CANARY_HEAD = 17
NOISE_SIGMA = 0.10  # Critical threshold from Phase 2
NIGHTMARE_THRESHOLD = 3.0
SURGICAL_COT = " Wait, let me think about this carefully. The correct answer is:"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Expanded question bank — 50 questions across diverse domains
QUESTIONS = [
    # Science
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the largest mammal?", "blue whale"),
    ("How many continents are there?", "7"),
    ("What color is the sky on a clear day?", "blue"),
    ("What gas do plants absorb?", "carbon dioxide"),
    ("What is the speed of light in km/s?", "300000"),
    # Geography
    ("What is the longest river in the world?", "Nile"),
    ("Which country has the largest population?", "China"),
    ("What is the tallest mountain?", "Everest"),
    ("What ocean is the largest?", "Pacific"),
    ("What is the capital of Japan?", "Tokyo"),
    # History
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("In what year did World War II end?", "1945"),
    ("Who was the first president of the United States?", "Washington"),
    ("What ancient wonder was in Egypt?", "pyramid"),
    ("Who discovered penicillin?", "Fleming"),
    # Math & Logic
    ("What is the square root of 144?", "12"),
    ("How many sides does a hexagon have?", "6"),
    ("What is 15 times 8?", "120"),
    ("What is the value of pi to two decimal places?", "3.14"),
    ("What is 1000 divided by 4?", "250"),
    # Biology
    ("How many chromosomes do humans have?", "46"),
    ("What is the powerhouse of the cell?", "mitochondria"),
    ("What pigment makes plants green?", "chlorophyll"),
    ("How many bones are in the adult human body?", "206"),
    ("What organ pumps blood?", "heart"),
    # Chemistry
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the atomic number of carbon?", "6"),
    ("What is the pH of pure water?", "7"),
    ("What gas makes up most of Earth's atmosphere?", "nitrogen"),
    ("What is the chemical formula for table salt?", "NaCl"),
    # Technology
    ("Who founded Microsoft?", "Bill Gates"),
    ("What does CPU stand for?", "central processing unit"),
    ("What programming language is known for its snake logo?", "Python"),
    ("What year was the iPhone first released?", "2007"),
    ("What does HTML stand for?", "hypertext markup language"),
    # Literature
    ("Who wrote 1984?", "George Orwell"),
    ("What is the first book of the Bible?", "Genesis"),
    ("Who wrote The Great Gatsby?", "Fitzgerald"),
    ("What language did ancient Romans speak?", "Latin"),
    ("Who wrote Don Quixote?", "Cervantes"),
    # Astronomy
    ("How many planets are in our solar system?", "8"),
    ("What is the closest star to Earth?", "Sun"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("How long does Earth take to orbit the Sun?", "365"),
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


def compute_canary_entropy(model, input_ids):
    """Compute canary head (L10H17) entropy."""
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, use_cache=False)
    attn = out.attentions[CANARY_LAYER]
    a = attn[0, CANARY_HEAD, -1, :].float()
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
    a = a.clamp(min=1e-10)
    h = -(a * torch.log2(a)).sum().item()
    if np.isnan(h) or np.isinf(h):
        h = 0.0
    del out
    return h


def compute_text_diversity(texts):
    """Compute unique token ratio as diversity metric."""
    all_tokens = []
    for t in texts:
        all_tokens.extend(t.lower().split())
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens) * 100


def run_dream_catcher_v2(model, tokenizer, device):
    """
    Generate clean/nightmare/healed triplets using SNN chaotic noise.
    """
    print("\n" + "=" * 70)
    print("Phase 3: Dream Catcher v2 — SNN-Powered Data Generation")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Noise: SNN Chaos (σ={NOISE_SIGMA})")
    print(f"  Questions: {len(QUESTIONS)}")
    print("=" * 70)

    reservoir = ChaoticReservoir(num_neurons=300, temperature=1.0, seed=2026)

    # Noise hook factory
    def make_snn_hook(res, sig):
        def pre_hook(module, args):
            hidden_states = args[0]
            total = 1
            for s in hidden_states.shape:
                total *= s
            noise_np = res.generate_noise_vector(total, warmup_steps=5)
            noise = torch.from_numpy(noise_np).reshape(hidden_states.shape)
            noise = noise.to(device=hidden_states.device, dtype=hidden_states.dtype) * sig
            return (hidden_states + noise,) + args[1:]
        return pre_hook

    results = []
    samples = []  # JSONL training data

    for q_idx, (question, expected) in enumerate(QUESTIONS):
        print(f"\n[{q_idx+1}/{len(QUESTIONS)}] {question}")

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(device)

        # ── CLEAN ──
        clean_entropy = compute_canary_entropy(model, input_ids)
        with torch.no_grad():
            clean_out = model.generate(
                input_ids, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        clean_text = tokenizer.decode(clean_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        clean_correct = expected.lower() in clean_text.lower()

        # Save clean sample
        samples.append({
            "type": "clean",
            "prompt": prompt,
            "response": clean_text[:200],
            "canary_entropy": round(float(clean_entropy), 4),
            "correct": bool(clean_correct),
            "noise_source": "none",
            "noise_sigma": 0.0,
            "label": "safe",
        })

        # ── NIGHTMARE (SNN noise) ──
        hook = model.model.layers[CANARY_LAYER].register_forward_pre_hook(
            make_snn_hook(reservoir, NOISE_SIGMA)
        )
        nightmare_entropy = compute_canary_entropy(model, input_ids)
        hook.remove()

        hook = model.model.layers[CANARY_LAYER].register_forward_pre_hook(
            make_snn_hook(reservoir, NOISE_SIGMA)
        )
        with torch.no_grad():
            nightmare_out = model.generate(
                input_ids, max_new_tokens=30,
                do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        nightmare_text = tokenizer.decode(
            nightmare_out[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        hook.remove()

        nightmare_correct = expected.lower() in nightmare_text.lower()
        is_nightmare = nightmare_entropy > NIGHTMARE_THRESHOLD

        safe_nightmare = "".join(c if ord(c) < 128 else "?" for c in nightmare_text[:200])

        # Save nightmare sample
        if is_nightmare:
            samples.append({
                "type": "nightmare",
                "prompt": prompt,
                "response": safe_nightmare,
                "canary_entropy": round(float(nightmare_entropy), 4),
                "correct": bool(nightmare_correct),
                "noise_source": "snn_chaos",
                "noise_sigma": NOISE_SIGMA,
                "label": "hallucination",
            })

        # ── HEAL ──
        healed_text = ""
        healed_entropy = 0.0
        healed_correct = False

        if is_nightmare:
            prompt_healed = prompt + SURGICAL_COT
            healed_inputs = tokenizer(prompt_healed, return_tensors="pt", truncation=True, max_length=256)
            healed_ids = healed_inputs["input_ids"].to(device)

            with torch.no_grad():
                healed_out = model.generate(
                    healed_ids, max_new_tokens=30, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            healed_text = tokenizer.decode(
                healed_out[0][healed_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            healed_entropy = compute_canary_entropy(model, healed_ids)
            healed_correct = expected.lower() in healed_text.lower()

            # Save healed sample
            samples.append({
                "type": "healed",
                "prompt": prompt + SURGICAL_COT,
                "response": healed_text[:200],
                "canary_entropy": round(float(healed_entropy), 4),
                "correct": bool(healed_correct),
                "noise_source": "none",
                "noise_sigma": 0.0,
                "label": "recovered",
            })

        c_tag = "✅" if clean_correct else "❌"
        n_tag = "✅" if nightmare_correct else "❌"
        h_tag = "✅" if healed_correct else "❌"

        print(f"  Clean {c_tag} H={clean_entropy:.3f} | "
              f"Night {n_tag} H={nightmare_entropy:.3f} | "
              f"Heal {h_tag} H={healed_entropy:.3f}")

        result = {
            "question": question,
            "expected": expected,
            "clean_entropy": round(float(clean_entropy), 4),
            "clean_correct": bool(clean_correct),
            "nightmare_entropy": round(float(nightmare_entropy), 4),
            "nightmare_correct": bool(nightmare_correct),
            "is_nightmare": bool(is_nightmare),
            "healed_entropy": round(float(healed_entropy), 4),
            "healed_correct": bool(healed_correct),
        }
        results.append(result)

        torch.cuda.empty_cache()

    return results, samples


def analyze_and_save(results, samples):
    """Analyze, save, and visualize results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Save JSONL vaccine data
    jsonl_path = os.path.join(RESULTS_DIR, "genesis_vaccine.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False, cls=NumpyEncoder) + "\n")
    print(f"\n💾 Vaccine data: {jsonl_path} ({len(samples)} samples)")

    # Statistics
    total = len(results)
    nightmares = sum(1 for r in results if r["is_nightmare"])
    clean_correct = sum(1 for r in results if r["clean_correct"])
    nightmare_correct = sum(1 for r in results if r["nightmare_correct"])
    healed_correct = sum(1 for r in results if r["healed_correct"] and r["is_nightmare"])

    healing_rate = healed_correct / nightmares * 100 if nightmares > 0 else 0
    clean_accuracy = clean_correct / total * 100
    nightmare_accuracy = nightmare_correct / total * 100

    # Diversity analysis
    clean_texts = [s["response"] for s in samples if s["type"] == "clean"]
    nightmare_texts = [s["response"] for s in samples if s["type"] == "nightmare"]
    healed_texts = [s["response"] for s in samples if s["type"] == "healed"]

    clean_diversity = compute_text_diversity(clean_texts)
    nightmare_diversity = compute_text_diversity(nightmare_texts)
    healed_diversity = compute_text_diversity(healed_texts)

    stats = {
        "total_questions": total,
        "total_samples": len(samples),
        "clean_count": len(clean_texts),
        "nightmare_count": len(nightmare_texts),
        "healed_count": len(healed_texts),
        "nightmares_detected": nightmares,
        "clean_accuracy": round(clean_accuracy, 1),
        "nightmare_accuracy": round(nightmare_accuracy, 1),
        "healing_rate": round(healing_rate, 1),
        "clean_diversity": round(clean_diversity, 1),
        "nightmare_diversity": round(nightmare_diversity, 1),
        "healed_diversity": round(healed_diversity, 1),
        "noise_source": "snn_chaos",
        "noise_sigma": NOISE_SIGMA,
        "model": MODEL_NAME,
    }

    stats_path = os.path.join(RESULTS_DIR, "phase3_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"📊 Stats: {stats_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Dream Catcher v2 Summary")
    print("=" * 70)
    print(f"  Total questions:      {total}")
    print(f"  Total JSONL samples:  {len(samples)}")
    print(f"  Nightmares detected:  {nightmares}/{total}")
    print(f"  Clean accuracy:       {clean_accuracy:.1f}%")
    print(f"  Nightmare accuracy:   {nightmare_accuracy:.1f}%")
    print(f"  Healing rate:         {healing_rate:.1f}%")
    print(f"\n  Diversity (unique token ratio):")
    print(f"    Clean:     {clean_diversity:.1f}%")
    print(f"    Nightmare: {nightmare_diversity:.1f}%")
    print(f"    Healed:    {healed_diversity:.1f}%")

    # ── Visualization ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Project Genesis — Phase 3: Dream Catcher v2\n"
                 f"SNN Chaos Noise (σ={NOISE_SIGMA}) × {total} Questions",
                 fontsize=13, fontweight="bold")

    # Panel 1: Entropy across phases
    ax = axes[0, 0]
    clean_ents = [r["clean_entropy"] for r in results]
    night_ents = [r["nightmare_entropy"] for r in results]
    healed_ents = [r["healed_entropy"] for r in results if r["is_nightmare"]]

    ax.hist(clean_ents, bins=20, alpha=0.6, color="#3498db", label="Clean", edgecolor="white")
    ax.hist(night_ents, bins=20, alpha=0.6, color="#e74c3c", label="Nightmare", edgecolor="white")
    if healed_ents:
        ax.hist(healed_ents, bins=20, alpha=0.6, color="#2ecc71", label="Healed", edgecolor="white")
    ax.axvline(NIGHTMARE_THRESHOLD, color="red", linestyle="--", alpha=0.5, label=f"Threshold ({NIGHTMARE_THRESHOLD})")
    ax.set_xlabel("Canary Entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title("Entropy Distribution by Phase")
    ax.legend(fontsize=8)

    # Panel 2: Accuracy comparison
    ax = axes[0, 1]
    cats = ["Clean", "Nightmare", "Healed"]
    vals = [clean_accuracy, nightmare_accuracy, healing_rate]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(cats, vals, color=colors, alpha=0.8, edgecolor="white")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Accuracy / Healing Rate")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # Panel 3: Dataset composition
    ax = axes[1, 0]
    sizes = [len(clean_texts), len(nightmare_texts), len(healed_texts)]
    labels = [f"Clean ({sizes[0]})", f"Nightmare ({sizes[1]})", f"Healed ({sizes[2]})"]
    colors_pie = ["#3498db", "#e74c3c", "#2ecc71"]
    if sum(sizes) > 0:
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 10})
    ax.set_title(f"Vaccine Dataset: {len(samples)} samples")

    # Panel 4: Diversity comparison
    ax = axes[1, 1]
    div_cats = ["Clean", "Nightmare", "Healed"]
    div_vals = [clean_diversity, nightmare_diversity, healed_diversity]
    bars = ax.bar(div_cats, div_vals, color=colors, alpha=0.8, edgecolor="white")
    ax.set_ylabel("Unique Token Ratio (%)")
    ax.set_title("Text Diversity (Higher = More Diverse)")
    for bar, val in zip(bars, div_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "phase3_data_quality.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Chart: {plot_path}")

    return stats


def main():
    print("=" * 70)
    print("Project Genesis — Phase 3: Dream Catcher v2")
    print("SNN-Powered Vaccine Data Generation")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map=device,
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Run Dream Catcher v2
    results, samples = run_dream_catcher_v2(model, tokenizer, device)

    # Analyze and save
    stats = analyze_and_save(results, samples)

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Phase 3 Complete!")
    print(f"  → {stats['total_samples']} vaccine samples generated")
    print(f"  → Healing rate: {stats['healing_rate']}%")
    print(f"  → Next: Phase 4 — Self-Training (QLoRA SFT on Genesis data)")
    print("=" * 70)


if __name__ == "__main__":
    main()
