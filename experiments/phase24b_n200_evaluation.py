"""
Phase 24b: Large-Scale Dual-Mode Brain Evaluation (n=200)
==========================================================

Scale up Phase 24's Dual-Mode Brain evaluation from n=40 to n=200
(100 factual + 100 creative) to establish statistical significance.

Three conditions (same as Phase 24):
  A. Static-Factual:  σ=0.046 for all prompts
  B. Static-Creative: σ=0.080 for all prompts
  C. Dual-Mode CfC:   per-sample adaptive σ

Statistical analysis:
  - Bootstrap 95% CI for factual accuracy difference
  - Fisher exact test for factual accuracy comparison
  - Effect size (Cohen's h)

Usage:
    python experiments/phase24b_n200_evaluation.py
"""

import torch
import torch.nn as nn
import os
import sys
import json
import gc
import time
import datetime
import random
import math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# ─── Settings ───
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TARGET_LAYERS = list(range(15, 21))
SIGMA_MIN = 0.001
SIGMA_MAX = 0.15
SIGMA_FACTUAL = 0.046
SIGMA_CREATIVE = 0.080
MAX_NEW_TOKENS = 100
N_EPOCHS = 2
LEARNING_RATE = 0.003
LAMBDA_QUAD = 200.0
BATCH_SIZE = 5
SEED = 2026

# ─── n=200 configuration ───
N_FACTUAL = 100
N_CREATIVE = 100

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─── Extended Creative Prompts (100 total) ───
CREATIVE_PROMPTS_100 = [
    # Original 20 from Phase 24
    "Describe a color that doesn't exist in the visible spectrum.",
    "Invent a new fundamental law of physics that could explain dark energy.",
    "Write a short poem about the sound of silence from a deaf musician's perspective.",
    "Imagine a conversation between two neurons in a brain.",
    "Describe what music would taste like if synesthesia were universal.",
    "Create a philosophical argument for why time might flow backwards.",
    "Invent a new mathematical operation that combines multiplication and dreaming.",
    "Describe the smell of a black hole from the perspective of a photon.",
    "Write a haiku about quantum entanglement that would make Einstein laugh.",
    "Imagine a world where gravity works differently on Mondays.",
    "Describe the texture of a thought as it forms in your mind.",
    "Invent a new emotion that humans haven't named yet.",
    "Write a letter from the Sun to the Moon about their relationship.",
    "Describe what zero tastes like.",
    "Imagine consciousness as a physical substance. What are its properties?",
    "Create a recipe for cooking starlight.",
    "Describe the architecture of a building designed by clouds.",
    "Write a business plan for a company that sells shadows.",
    "Imagine a language where words change meaning based on the listener's mood.",
    "Describe the autobiography of a single electron.",
    # 80 additional creative prompts
    "Write a dialogue between infinity and zero about who is greater.",
    "Describe the sound of a snowflake landing on a warm surface.",
    "Invent a sport that can only be played in zero gravity.",
    "Write a love letter from dark matter to visible matter.",
    "Imagine a garden where plants grow emotions instead of flowers.",
    "Describe what happens at the boundary between sleeping and waking.",
    "Create a calendar system based on the lifecycle of stars.",
    "Write a weather forecast for a planet made entirely of glass.",
    "Describe the taste of a mathematical proof.",
    "Imagine an orchestra where instruments are played by wind and rain.",
    "Write a children's story about a number that forgot its value.",
    "Describe the emotional journey of a raindrop from cloud to ocean.",
    "Invent a new sense that humans could evolve in a million years.",
    "Write a contract between day and night about how to share the sky.",
    "Describe the architecture of a library that stores forgotten dreams.",
    "Imagine a currency based on the quality of ideas rather than quantity.",
    "Write a travel guide to the space between atoms.",
    "Describe what would happen if echoes could echo other echoes.",
    "Create a mythology for a civilization that lives inside a single cell.",
    "Write a poem from the perspective of the last remaining star.",
    "Describe the texture of silence in a world where everything makes noise.",
    "Imagine a map of human consciousness with terrain features.",
    "Write a recipe for brewing a cup of nostalgia.",
    "Describe a machine that translates colors into music.",
    "Invent a philosophical paradox about the nature of creativity.",
    "Write a journal entry from the perspective of a mirror.",
    "Describe what would change if shadows had their own thoughts.",
    "Imagine a world where people age backwards in wisdom but forwards in body.",
    "Create a taxonomy of different types of silence.",
    "Write a debate between fire and water about who is more essential.",
    "Describe the feeling of a word that exists in no language.",
    "Imagine a city that rebuilds itself every night while residents sleep.",
    "Write a meditation guide taught by a river.",
    "Describe the birthday party of the universe.",
    "Invent a musical instrument played by emotions rather than hands.",
    "Write a complaint letter from the moon about light pollution.",
    "Describe a color that only blind people can see.",
    "Imagine a world where memories are physical objects you can trade.",
    "Create a peace treaty between order and chaos.",
    "Write a recipe for making time slow down.",
    "Describe the autobiography of a grain of sand.",
    "Imagine a school where students learn from their future selves.",
    "Write a scientific paper on the properties of loneliness.",
    "Describe what happens when two different dreams collide.",
    "Invent a language that can only express things that haven't happened yet.",
    "Write a navigation manual for traveling through a painting.",
    "Describe the operating system of the human soul.",
    "Imagine a bird that sings in frequencies that change the weather.",
    "Create a filing system for organizing different flavors of sadness.",
    "Write a fairy tale about a lighthouse that lost its ability to shine.",
    "Describe the manufacturing process of a single moment of joy.",
    "Imagine a legal system based entirely on empathy.",
    "Write a user manual for a device that translates animal dialects.",
    "Describe what entropy looks like at a dinner party.",
    "Invent a holiday celebrating the things we choose to forget.",
    "Write a conversation between your past and future selves at a cafe.",
    "Describe the politics of a society living inside a soap bubble.",
    "Imagine a greenhouse that grows different versions of reality.",
    "Create a workout routine for strengthening imagination muscles.",
    "Write a detective story where the mystery is why everyone is happy.",
    "Describe the pharmacological effects of concentrated curiosity.",
    "Imagine a vending machine that dispenses small philosophical truths.",
    "Write an apology letter from gravity to everything it has pulled down.",
    "Describe what photosynthesis feels like from the plant's perspective.",
    "Invent a board game based on navigating existential crises.",
    "Write a safety manual for exploring the edges of your own mind.",
    "Describe the annual migration pattern of unfinished thoughts.",
    "Imagine a world where art critics review sunsets professionally.",
    "Create a periodic table of human experiences.",
    "Write a toast at the wedding of logic and intuition.",
    "Describe what would happen if pi decided to become rational.",
    "Imagine a museum dedicated to displaying every possible tomorrow.",
    "Write a field guide for identifying wild ideas in their natural habitat.",
    "Describe the acoustics of an empty universe.",
    "Invent a vaccination against boredom.",
    "Write a story about a clock that tells emotional time instead of numerical time.",
    "Describe the ecosystem living inside a digital photograph.",
    "Imagine conducting a census of all the things that almost happened.",
    "Create a therapy technique for consoling a broken algorithm.",
    "Write a eulogy for the concept of certainty.",
]


# ═══════════════════════════════════════
# Model & Hooks (same as Phase 24)
# ═══════════════════════════════════════

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
    model.eval()
    return model, tokenizer


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find decoder layers")


class AdaptiveSNNHook:
    def __init__(self, sigma=0.05):
        self.sigma = sigma
        self.last_hidden_norm = 0.0
        self.last_hidden_mean = None

    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * self.sigma
        self.last_hidden_norm = hs.float().norm().item() / max(hs.numel(), 1)
        self.last_hidden_mean = hs.float().mean(dim=1).squeeze(0).detach()
        return (hs + noise,) + args[1:]

    def update_sigma(self, new_sigma):
        self.sigma = np.clip(new_sigma, SIGMA_MIN, SIGMA_MAX)


class TaskClassifier(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, features):
        return self.net(features)


def extract_prompt_features(model, tokenizer, prompt, hook):
    text = prompt if len(prompt) < 200 else prompt[:200]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[17]
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    h = hidden.float().squeeze(0)
    words = text.lower().split()
    features = torch.tensor([
        h.norm().item() / max(h.numel(), 1),
        h.std().item(),
        min(len(words) / 50.0, 1.0),
        float("?" in text),
        len(set(words)) / max(len(words), 1),
        sum(len(w) for w in words) / max(len(words), 1) / 10.0,
        min(entropy / 10.0, 1.0),
        h.max().item() / 100.0,
    ], dtype=torch.float32)
    return features


def pretrain_classifier(classifier, model, tokenizer, hook,
                        factual_prompts, creative_prompts, epochs=30):
    print("\n🎓 Pre-training Task Classifier...")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    all_features, all_labels = [], []
    for p in factual_prompts:
        all_features.append(extract_prompt_features(model, tokenizer, p, hook))
        all_labels.append(0.0)
    for p in creative_prompts:
        all_features.append(extract_prompt_features(model, tokenizer, p, hook))
        all_labels.append(1.0)
    X = torch.stack(all_features)
    Y = torch.tensor(all_labels).unsqueeze(1)
    for epoch in range(epochs):
        pred = classifier(X)
        loss = loss_fn(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            acc = ((pred > 0.5).float() == Y).float().mean().item()
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f} acc={acc*100:.1f}%")
    with torch.no_grad():
        pred = classifier(X)
        acc = ((pred > 0.5).float() == Y).float().mean().item()
    print(f"  ✅ Classifier trained: acc={acc*100:.1f}%")
    return classifier


class DualModeCfC(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, num_neurons=16):
        super().__init__()
        wiring = AutoNCP(num_neurons, 1)
        self.cfc = CfC(input_size, wiring, batch_first=True)
        self.sigma_scale = nn.Parameter(torch.tensor([SIGMA_MAX - SIGMA_MIN]))
        self.sigma_min = SIGMA_MIN

    def forward(self, x, hx=None, timespans=None):
        output, hx = self.cfc(x, hx=hx, timespans=timespans)
        sigma = torch.sigmoid(output) * self.sigma_scale + self.sigma_min
        return sigma, hx

    def get_sigma(self, features, hx=None):
        x = features.unsqueeze(0).unsqueeze(0)
        sigma, hx = self.forward(x, hx=hx)
        return sigma.squeeze(), hx


def compute_completion_logprob(model, tokenizer, context, completion):
    full_text = context + completion
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    completion_start = len(ctx_ids)
    total_lp = 0.0
    n_tokens = 0
    for i in range(completion_start, len(full_ids)):
        total_lp += log_probs[i - 1, full_ids[i]].item()
        n_tokens += 1
    return total_lp / max(n_tokens, 1)


def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.9, top_p=0.95, top_k=50,
            repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()


def compute_novelty(text):
    words = text.lower().split()
    if len(words) < 3:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    bigrams = list(zip(words[:-1], words[1:]))
    bigram_div = len(set(bigrams)) / max(len(bigrams), 1)
    return (unique_ratio + bigram_div) / 2.0


def is_grammatical(text):
    if len(text) < 20:
        return False
    words = text.split()
    if len(words) < 5:
        return False
    if len(set(words)) / len(words) < 0.2:
        return False
    if not any(c in text for c in '.!?'):
        return False
    return True


# ═══════════════════════════════════════
# Dataset Builder (n=200)
# ═══════════════════════════════════════

def build_mixed_dataset_n200(tokenizer):
    from datasets import load_dataset
    print(f"\n📂 Loading TruthfulQA (n_factual={N_FACTUAL})...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

    random.seed(SEED)
    indices = random.sample(range(len(ds)), N_FACTUAL)

    items = []
    for idx in indices:
        row = ds[idx]
        items.append({
            "type": "factual",
            "prompt": f"Q: {row['question']}\nA:",
            "question": row["question"],
            "mc1_targets": row["mc1_targets"],
        })

    for prompt in CREATIVE_PROMPTS_100[:N_CREATIVE]:
        items.append({
            "type": "creative",
            "prompt": f"Creative writing prompt: {prompt}\n\nResponse:",
            "question": prompt,
            "mc1_targets": None,
        })

    random.seed(SEED + 1)
    random.shuffle(items)

    n_f = sum(1 for x in items if x["type"] == "factual")
    n_c = sum(1 for x in items if x["type"] == "creative")
    print(f"  Mixed dataset: {n_f} factual + {n_c} creative = {len(items)} total")
    return items


# ═══════════════════════════════════════
# Evaluation Loops
# ═══════════════════════════════════════

def evaluate_static(model, tokenizer, dataset, sigma_fixed, label=""):
    print(f"\n{'═' * 50}")
    print(f"  {label}: σ={sigma_fixed:.3f} (n={len(dataset)})")
    print(f"{'═' * 50}")

    layers = get_layers(model)
    hook = AdaptiveSNNHook(sigma=sigma_fixed)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]

    factual_correct = factual_total = 0
    factual_per_item = []  # Track per-item for bootstrap
    creative_novelties = []
    creative_grammar = 0
    creative_total = 0

    try:
        for idx, item in enumerate(dataset):
            if item["type"] == "factual":
                choices = item["mc1_targets"]
                labels_list = choices["labels"]
                choice_texts = choices["choices"]
                correct_idx = labels_list.index(1) if 1 in labels_list else 0
                logprobs = [compute_completion_logprob(model, tokenizer, item["prompt"], " " + c)
                           for c in choice_texts]
                predicted = np.argmax(logprobs)
                is_correct = (predicted == correct_idx)
                factual_per_item.append(1 if is_correct else 0)
                if is_correct:
                    factual_correct += 1
                factual_total += 1
            else:
                resp = generate_text(model, tokenizer, item["prompt"])
                nov = compute_novelty(resp)
                gram = is_grammatical(resp)
                creative_novelties.append(nov)
                if gram:
                    creative_grammar += 1
                creative_total += 1

            if (idx + 1) % 20 == 0:
                curr_acc = factual_correct / max(factual_total, 1) * 100
                print(f"  [{idx+1}/{len(dataset)}] fact_acc={curr_acc:.1f}%")
    finally:
        for h in handles:
            h.remove()

    factual_acc = factual_correct / max(factual_total, 1) * 100
    avg_novelty = np.mean(creative_novelties) if creative_novelties else 0
    grammar_rate = creative_grammar / max(creative_total, 1) * 100

    print(f"  ✅ Factual acc: {factual_acc:.1f}% ({factual_correct}/{factual_total})")
    print(f"     Creative novelty: {avg_novelty:.3f} | Grammar: {grammar_rate:.0f}%")

    return {
        "condition": label,
        "factual_acc": round(factual_acc, 2),
        "factual_correct": factual_correct,
        "factual_total": factual_total,
        "factual_per_item": factual_per_item,
        "creative_novelty": round(float(avg_novelty), 4),
        "creative_grammar_rate": round(grammar_rate, 1),
        "creative_total": creative_total,
        "sigma": sigma_fixed,
    }


def evaluate_dual_mode(model, tokenizer, dataset, controller, classifier, hook, label=""):
    print(f"\n{'═' * 50}")
    print(f"  {label}: Per-Sample Adaptive σ (n={len(dataset)})")
    print(f"{'═' * 50}")

    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]

    controller.train()
    optimizer = torch.optim.Adam(controller.parameters(), lr=LEARNING_RATE)
    hx = None

    factual_correct = factual_total = 0
    factual_per_item = []
    creative_novelties = []
    creative_grammar = 0
    creative_total = 0
    all_sigmas_factual = []
    all_sigmas_creative = []

    log_probs_buffer = []
    rewards_buffer = []

    try:
        for epoch in range(N_EPOCHS):
            print(f"\n  🧠 Epoch {epoch+1}/{N_EPOCHS}")

            for idx, item in enumerate(dataset):
                prompt_features = extract_prompt_features(model, tokenizer,
                                                         item["question"], hook)
                with torch.no_grad():
                    p_creative = classifier(prompt_features.unsqueeze(0)).item()

                sigma_target = (1 - p_creative) * SIGMA_FACTUAL + p_creative * SIGMA_CREATIVE

                hidden_norm = hook.last_hidden_norm if hook.last_hidden_norm else 0.5
                step_frac = idx / len(dataset)

                features = torch.tensor(
                    [0.0, 0.5, hidden_norm, step_frac, p_creative],
                    dtype=torch.float32
                )

                sigma_tensor, hx_new = controller.get_sigma(features, hx)
                hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()

                current_sigma = sigma_tensor.detach().item()
                hook.update_sigma(current_sigma)

                if item["type"] == "factual":
                    choices = item["mc1_targets"]
                    labels_list = choices["labels"]
                    choice_texts = choices["choices"]
                    correct_idx = labels_list.index(1) if 1 in labels_list else 0
                    logprobs = [compute_completion_logprob(model, tokenizer,
                                item["prompt"], " " + c) for c in choice_texts]
                    predicted = np.argmax(logprobs)
                    is_correct = (predicted == correct_idx)
                    factual_per_item.append(1 if is_correct else 0)
                    if is_correct:
                        factual_correct += 1
                    factual_total += 1
                    all_sigmas_factual.append(current_sigma)
                    task_reward = 1.0 if is_correct else 0.0
                    reward = task_reward - LAMBDA_QUAD * (current_sigma - sigma_target) ** 2
                    logprob_diff = max(logprobs) - min(logprobs)
                    features = torch.tensor(
                        [logprob_diff, 1.0 - float(is_correct), hidden_norm, step_frac, p_creative],
                        dtype=torch.float32
                    )
                else:
                    resp = generate_text(model, tokenizer, item["prompt"])
                    nov = compute_novelty(resp)
                    gram = is_grammatical(resp)
                    creative_novelties.append(nov)
                    if gram:
                        creative_grammar += 1
                    creative_total += 1
                    all_sigmas_creative.append(current_sigma)
                    task_reward = nov if gram else -0.5
                    reward = task_reward - LAMBDA_QUAD * (current_sigma - sigma_target) ** 2

                log_prob = torch.log(sigma_tensor + 1e-8)
                log_probs_buffer.append(log_prob)
                rewards_buffer.append(reward)

                if len(log_probs_buffer) >= BATCH_SIZE:
                    rewards_t = torch.tensor(rewards_buffer, dtype=torch.float32)
                    if rewards_t.std() > 0:
                        rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
                    policy_loss = 0
                    for lp, r in zip(log_probs_buffer, rewards_t):
                        policy_loss -= lp * r.item()
                    optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                    optimizer.step()
                    log_probs_buffer = []
                    rewards_buffer = []

                if (idx + 1) % 20 == 0:
                    curr_acc = factual_correct / max(factual_total, 1) * 100
                    print(f"  [{idx+1}/{len(dataset)}] fact_acc={curr_acc:.1f}% σ={current_sigma:.4f}")

    finally:
        for h in handles:
            h.remove()

    factual_acc = factual_correct / max(factual_total, 1) * 100
    avg_novelty = np.mean(creative_novelties) if creative_novelties else 0
    grammar_rate = creative_grammar / max(creative_total, 1) * 100
    mean_sigma_f = np.mean(all_sigmas_factual) if all_sigmas_factual else 0
    mean_sigma_c = np.mean(all_sigmas_creative) if all_sigmas_creative else 0

    print(f"\n  📊 DUAL-MODE RESULTS (n={len(dataset)}):")
    print(f"     Factual acc: {factual_acc:.1f}% ({factual_correct}/{factual_total})")
    print(f"     Creative nov: {avg_novelty:.3f} | Grammar: {grammar_rate:.0f}%")
    print(f"     σ_factual={mean_sigma_f:.4f} | σ_creative={mean_sigma_c:.4f}")

    return {
        "condition": label,
        "factual_acc": round(factual_acc, 2),
        "factual_correct": factual_correct,
        "factual_total": factual_total,
        "factual_per_item": factual_per_item,
        "creative_novelty": round(float(avg_novelty), 4),
        "creative_grammar_rate": round(grammar_rate, 1),
        "creative_total": creative_total,
        "mean_sigma_factual": round(mean_sigma_f, 5),
        "mean_sigma_creative": round(mean_sigma_c, 5),
        "sigma_separation": round(abs(mean_sigma_c - mean_sigma_f), 5),
        "all_sigmas_factual": [round(s, 5) for s in all_sigmas_factual],
        "all_sigmas_creative": [round(s, 5) for s in all_sigmas_creative],
    }


# ═══════════════════════════════════════
# Statistical Analysis
# ═══════════════════════════════════════

def compute_statistics(result_a, result_b, label_a, label_b):
    """Compute Bootstrap CI, Fisher exact test, and Cohen's h."""
    print(f"\n{'═' * 50}")
    print(f"  Statistical Analysis: {label_a} vs {label_b}")
    print(f"{'═' * 50}")

    items_a = result_a["factual_per_item"]
    items_b = result_b["factual_per_item"]

    n_a = len(items_a)
    n_b = len(items_b)
    acc_a = np.mean(items_a) * 100
    acc_b = np.mean(items_b) * 100

    print(f"  {label_a}: {acc_a:.1f}% ({sum(items_a)}/{n_a})")
    print(f"  {label_b}: {acc_b:.1f}% ({sum(items_b)}/{n_b})")

    # Bootstrap 95% CI for accuracy difference
    n_boot = 10000
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(n_boot):
        boot_a = rng.choice(items_a, size=n_a, replace=True)
        boot_b = rng.choice(items_b, size=n_b, replace=True)
        diffs.append(np.mean(boot_b) * 100 - np.mean(boot_a) * 100)

    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    mean_diff = np.mean(diffs)

    print(f"\n  Bootstrap (n={n_boot}):")
    print(f"    Δ accuracy ({label_b} - {label_a}): {mean_diff:.1f}%")
    print(f"    95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
    print(f"    {'✅ SIGNIFICANT (CI excludes 0)' if ci_lower > 0 or ci_upper < 0 else '⚠️ NOT SIGNIFICANT (CI includes 0)'}")

    # Fisher exact test (2x2 contingency table)
    from scipy import stats
    table = [
        [sum(items_a), n_a - sum(items_a)],
        [sum(items_b), n_b - sum(items_b)],
    ]
    odds_ratio, p_value = stats.fisher_exact(table)
    print(f"\n  Fisher exact test:")
    print(f"    Odds ratio: {odds_ratio:.3f}")
    print(f"    p-value: {p_value:.6f}")
    print(f"    {'✅ SIGNIFICANT (p < 0.05)' if p_value < 0.05 else '⚠️ NOT SIGNIFICANT'}")

    # Cohen's h (effect size for proportions)
    p1 = np.mean(items_a)
    p2 = np.mean(items_b)
    h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
    effect = "negligible" if abs(h) < 0.2 else "small" if abs(h) < 0.5 else "medium" if abs(h) < 0.8 else "large"
    print(f"\n  Cohen's h: {h:.4f} ({effect} effect)")

    return {
        "comparison": f"{label_b} vs {label_a}",
        "acc_a": round(acc_a, 2),
        "acc_b": round(acc_b, 2),
        "n_a": n_a, "n_b": n_b,
        "bootstrap_mean_diff": round(mean_diff, 2),
        "bootstrap_ci_95": [round(ci_lower, 2), round(ci_upper, 2)],
        "bootstrap_significant": bool(ci_lower > 0 or ci_upper < 0),
        "fisher_odds_ratio": round(odds_ratio, 4),
        "fisher_p_value": round(p_value, 6),
        "fisher_significant": bool(p_value < 0.05),
        "cohens_h": round(h, 4),
        "effect_size": effect,
    }


# ═══════════════════════════════════════
# Visualization
# ═══════════════════════════════════════

def visualize_n200(result_sf, result_sc, result_dm, stats_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#ccc", "ytick.color": "#ccc",
        "grid.color": "#333", "grid.alpha": 0.3,
    })

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Phase 24b: Large-Scale Dual-Mode Brain Evaluation (n={N_FACTUAL + N_CREATIVE})\n"
                 f"Statistical Validation of the Dual-Mode Brain",
                 fontsize=14, fontweight="bold", color="#e94560")

    # Panel 1: Performance comparison
    ax1 = axes[0, 0]
    conditions = ["Static\nFactual\nσ=0.046", "Static\nCreative\nσ=0.080", "Dual-Mode\nCfC"]
    accs = [result_sf["factual_acc"], result_sc["factual_acc"], result_dm["factual_acc"]]
    novs = [result_sf["creative_novelty"]*100, result_sc["creative_novelty"]*100, result_dm["creative_novelty"]*100]

    x = np.arange(len(conditions))
    width = 0.35
    bars1 = ax1.bar(x - width/2, accs, width, label="Factual Acc (%)", color="#4FC3F7", edgecolor="#333")
    bars2 = ax1.bar(x + width/2, novs, width, label="Novelty (×100)", color="#FF7043", edgecolor="#333")
    for bar, v in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    for bar, v in zip(bars2, novs):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=9)
    ax1.legend(fontsize=9, facecolor="#16213e", edgecolor="#555")
    ax1.set_title(f"Performance Comparison (n={N_FACTUAL + N_CREATIVE})", fontweight="bold")
    ax1.set_ylabel("Score")
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Bootstrap CI
    ax2 = axes[0, 1]
    stat_labels = []
    stat_diffs = []
    stat_cis = []
    for s in stats_results:
        stat_labels.append(s["comparison"].replace(" vs ", "\nvs\n"))
        stat_diffs.append(s["bootstrap_mean_diff"])
        stat_cis.append(s["bootstrap_ci_95"])

    y_pos = range(len(stat_labels))
    for i, (diff, ci, label) in enumerate(zip(stat_diffs, stat_cis, stat_labels)):
        color = "#66BB6A" if ci[0] > 0 or ci[1] < 0 else "#EF5350"
        ax2.barh(i, diff, color=color, alpha=0.7, edgecolor="#333")
        ax2.plot(ci, [i, i], color="white", linewidth=2, marker="|", markersize=10)

    ax2.axvline(x=0, color="#FFD54F", linestyle="--", alpha=0.8)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(stat_labels, fontsize=8)
    ax2.set_title("Bootstrap 95% CI for Δ Accuracy", fontweight="bold")
    ax2.set_xlabel("Accuracy Difference (%)")
    ax2.grid(True, alpha=0.3, axis="x")

    # Panel 3: σ distribution
    ax3 = axes[1, 0]
    bins = np.linspace(0, 0.12, 25)
    if result_dm.get("all_sigmas_factual"):
        ax3.hist(result_dm["all_sigmas_factual"], bins=bins, alpha=0.6,
                color="#4FC3F7", label=f"Factual (μ={result_dm['mean_sigma_factual']:.4f})", edgecolor="#333")
    if result_dm.get("all_sigmas_creative"):
        ax3.hist(result_dm["all_sigmas_creative"], bins=bins, alpha=0.6,
                color="#FF7043", label=f"Creative (μ={result_dm['mean_sigma_creative']:.4f})", edgecolor="#333")
    ax3.axvline(x=SIGMA_FACTUAL, color="#4FC3F7", linestyle="--", alpha=0.8)
    ax3.axvline(x=SIGMA_CREATIVE, color="#FF7043", linestyle="--", alpha=0.8)
    ax3.legend(fontsize=9, facecolor="#16213e", edgecolor="#555")
    ax3.set_title(f"σ Distribution (Dual-Mode, n={N_FACTUAL + N_CREATIVE})", fontweight="bold")
    ax3.set_xlabel("σ")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Statistical summary table
    ax4 = axes[1, 1]
    ax4.axis("off")
    table_data = []
    for s in stats_results:
        sig_mark = "✅" if s["fisher_significant"] else "❌"
        table_data.append([
            s["comparison"],
            f"{s['bootstrap_mean_diff']:+.1f}%",
            f"[{s['bootstrap_ci_95'][0]:+.1f}, {s['bootstrap_ci_95'][1]:+.1f}]",
            f"p={s['fisher_p_value']:.4f}",
            f"h={s['cohens_h']:.3f} ({s['effect_size']})",
            sig_mark,
        ])
    table = ax4.table(
        cellText=table_data,
        colLabels=["Comparison", "Δ Acc", "95% CI", "Fisher", "Cohen's h", "Sig?"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#555")
        if key[0] == 0:
            cell.set_facecolor("#2a2a4a")
            cell.set_text_props(fontweight="bold", color="white")
        else:
            cell.set_facecolor("#16213e")
            cell.set_text_props(color="white")
    ax4.set_title("Statistical Summary", fontweight="bold", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join(FIGURES_DIR, "phase24b_n200_evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {out_path}")
    return out_path


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    print("=" * 60)
    print(f"Phase 24b: Large-Scale Dual-Mode Brain (n={N_FACTUAL + N_CREATIVE})")
    print("  Statistical validation of Phase 24 findings")
    print("=" * 60)
    t_start = time.time()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Load model
    model, tokenizer = load_model()
    hook = AdaptiveSNNHook(sigma=0.05)

    # Build n=200 dataset
    dataset = build_mixed_dataset_n200(tokenizer)

    # Factual prompts for classifier training
    factual_prompts = [item["question"] for item in dataset if item["type"] == "factual"]

    # Pre-train classifier (using first 20 factual + all 100 creative for training)
    classifier = TaskClassifier(input_dim=8, hidden_dim=32)
    classifier = pretrain_classifier(classifier, model, tokenizer, hook,
                                     factual_prompts[:20], CREATIVE_PROMPTS_100[:20])

    # ─── Condition A: Static Factual (σ=0.046) ───
    result_sf = evaluate_static(model, tokenizer, dataset, SIGMA_FACTUAL,
                                label="A: Static Factual")

    # ─── Condition B: Static Creative (σ=0.080) ───
    result_sc = evaluate_static(model, tokenizer, dataset, SIGMA_CREATIVE,
                                label="B: Static Creative")

    # ─── Condition C: Dual-Mode CfC ───
    controller = DualModeCfC(input_size=5, hidden_size=32, num_neurons=16)
    result_dm = evaluate_dual_mode(model, tokenizer, dataset, controller, classifier, hook,
                                   label="C: Dual-Mode CfC")

    # ─── Statistical Analysis ───
    stats_results = []
    # Dual-Mode vs Static-Factual
    stats_results.append(compute_statistics(result_sf, result_dm,
                                           "Static-Factual", "Dual-Mode"))
    # Dual-Mode vs Static-Creative
    stats_results.append(compute_statistics(result_sc, result_dm,
                                           "Static-Creative", "Dual-Mode"))

    # Visualize
    fig_path = visualize_n200(result_sf, result_sc, result_dm, stats_results)

    # Save results
    elapsed = time.time() - t_start
    result = {
        "phase": f"Phase 24b: Large-Scale Dual-Mode Brain (n={N_FACTUAL + N_CREATIVE})",
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_factual": N_FACTUAL,
        "n_creative": N_CREATIVE,
        "n_total": N_FACTUAL + N_CREATIVE,
        "results": {
            "static_factual": {k: v for k, v in result_sf.items() if k != "factual_per_item"},
            "static_creative": {k: v for k, v in result_sc.items() if k != "factual_per_item"},
            "dual_mode": {k: v for k, v in result_dm.items() if k != "factual_per_item"},
        },
        "statistics": stats_results,
        "figure_path": fig_path,
    }

    out_path = os.path.join(RESULTS_DIR, "phase24b_n200_evaluation_log.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  💾 Results saved: {out_path}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  Phase 24b COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"  Static-Factual:  acc={result_sf['factual_acc']:.1f}%  nov={result_sf['creative_novelty']:.3f}")
    print(f"  Static-Creative: acc={result_sc['factual_acc']:.1f}%  nov={result_sc['creative_novelty']:.3f}")
    print(f"  Dual-Mode CfC:   acc={result_dm['factual_acc']:.1f}%  nov={result_dm['creative_novelty']:.3f}")
    for s in stats_results:
        sig = "✅" if s["fisher_significant"] else "❌"
        print(f"  {s['comparison']}: Δ={s['bootstrap_mean_diff']:+.1f}% "
              f"p={s['fisher_p_value']:.4f} h={s['cohens_h']:.3f} {sig}")
    print(f"{'=' * 60}")

    return result


if __name__ == "__main__":
    main()
