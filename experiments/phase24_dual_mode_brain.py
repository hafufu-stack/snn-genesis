"""
Phase 24: Per-Sample Task-Dynamic CfC — The Dual-Mode Brain
============================================================

The CfC controller classifies each prompt as factual or creative using
hidden-layer features, then adjusts σ target in real-time:

    σ_target = (1 - p_creative) × 0.046 + p_creative × 0.080

Three conditions (head-to-head):
  A. Static-Factual:  σ=0.046 for all prompts
  B. Static-Creative: σ=0.080 for all prompts
  C. Dual-Mode CfC:   per-sample adaptive σ

Key metric: σ separation = |mean_σ_creative − mean_σ_factual|

Usage:
    python experiments/phase24_dual_mode_brain.py
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
SIGMA_FACTUAL = 0.046   # Sweet spot for accuracy tasks
SIGMA_CREATIVE = 0.080  # Sweet spot for creativity tasks
MAX_NEW_TOKENS = 100
N_EPOCHS = 2            # Early-stop at 2 per Phase 20d insight
LEARNING_RATE = 0.003
LAMBDA_QUAD = 200.0
BATCH_SIZE = 5           # Smaller batches for 40-item dataset
SEED = 2026

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Mixed Dataset ───
CREATIVE_PROMPTS = [
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
]


# ═══════════════════════════════════════
# Model & Hooks
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
    """SNN noise injection with adjustable σ."""
    def __init__(self, sigma=0.05):
        self.sigma = sigma
        self.last_hidden_norm = 0.0
        self.last_hidden_mean = None  # Store for task classification

    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * self.sigma
        self.last_hidden_norm = hs.float().norm().item() / max(hs.numel(), 1)
        # Store mean-pooled hidden state for task classification
        self.last_hidden_mean = hs.float().mean(dim=1).squeeze(0).detach()
        return (hs + noise,) + args[1:]

    def update_sigma(self, new_sigma):
        self.sigma = np.clip(new_sigma, SIGMA_MIN, SIGMA_MAX)


# ═══════════════════════════════════════
# Task Classifier — Factual vs Creative
# ═══════════════════════════════════════

class TaskClassifier(nn.Module):
    """
    Classifies prompt as factual (0) or creative (1) from hidden features.
    
    Input: prompt-level features extracted from Mistral hidden states
    Output: p_creative ∈ [0, 1]
    """
    def __init__(self, input_dim=8, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return self.net(features)


def extract_prompt_features(model, tokenizer, prompt, hook):
    """
    Extract task-discriminative features from a prompt's hidden states.
    
    Returns 8 features:
    - hidden_norm: L2 norm of hidden states (creative prompts tend to have different norms)
    - hidden_std: standard deviation (creative prompts have more diverse activations)
    - prompt_length: normalized length
    - question_mark: presence of question marks
    - unique_ratio: vocabulary diversity
    - avg_word_length: longer words → more abstract/creative
    - logprob_entropy: entropy of next-token distribution (creative → higher entropy)
    - hidden_max: max activation value
    """
    text = prompt if len(prompt) < 200 else prompt[:200]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get hidden states from target layer (L17, middle of injection range)
        hidden = outputs.hidden_states[17]  # [1, seq_len, hidden_dim]
        
        # Next-token logits for entropy
        logits = outputs.logits[0, -1, :]  # Last token logits
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    
    h = hidden.float().squeeze(0)  # [seq_len, hidden_dim]
    
    words = text.lower().split()
    
    features = torch.tensor([
        h.norm().item() / max(h.numel(), 1),            # hidden_norm
        h.std().item(),                                   # hidden_std
        min(len(words) / 50.0, 1.0),                     # prompt_length (normalized)
        float("?" in text),                               # question_mark
        len(set(words)) / max(len(words), 1),            # unique_ratio
        sum(len(w) for w in words) / max(len(words), 1) / 10.0,  # avg_word_length
        min(entropy / 10.0, 1.0),                        # logprob_entropy (normalized)
        h.max().item() / 100.0,                          # hidden_max (normalized)
    ], dtype=torch.float32)
    
    return features


def pretrain_classifier(classifier, model, tokenizer, hook,
                        factual_prompts, creative_prompts, epochs=30):
    """
    Pre-train classifier with pseudo-labels:
    - factual prompts → label=0
    - creative prompts → label=1
    """
    print("\n🎓 Pre-training Task Classifier...")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    
    # Extract features once
    all_features = []
    all_labels = []
    
    for p in factual_prompts:
        f = extract_prompt_features(model, tokenizer, p, hook)
        all_features.append(f)
        all_labels.append(0.0)
    
    for p in creative_prompts:
        f = extract_prompt_features(model, tokenizer, p, hook)
        all_features.append(f)
        all_labels.append(1.0)
    
    X = torch.stack(all_features)
    Y = torch.tensor(all_labels).unsqueeze(1)
    
    # Train
    for epoch in range(epochs):
        pred = classifier(X)
        loss = loss_fn(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            acc = ((pred > 0.5).float() == Y).float().mean().item()
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f} acc={acc*100:.1f}%")
    
    # Final accuracy
    with torch.no_grad():
        pred = classifier(X)
        acc = ((pred > 0.5).float() == Y).float().mean().item()
        mean_factual = pred[:len(factual_prompts)].mean().item()
        mean_creative = pred[len(factual_prompts):].mean().item()
    
    print(f"  ✅ Classifier trained: acc={acc*100:.1f}% "
          f"| factual p̄={mean_factual:.3f} | creative p̄={mean_creative:.3f}")
    
    return classifier


# ═══════════════════════════════════════
# DualMode CfC Controller
# ═══════════════════════════════════════

class DualModeCfC(nn.Module):
    """
    CfC controller with task-awareness.
    
    Input: 5 features = [logprob_diff, error_rate, hidden_norm, step_frac, p_creative]
    Output: σ ∈ [SIGMA_MIN, SIGMA_MAX]
    
    The p_creative feature tells CfC what kind of task it's dealing with,
    allowing it to learn different σ targets for different task types.
    """
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


# ═══════════════════════════════════════
# Generation & Evaluation
# ═══════════════════════════════════════

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
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()


def compute_novelty(text):
    """Compute text novelty metrics."""
    words = text.lower().split()
    if len(words) < 3:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    bigrams = list(zip(words[:-1], words[1:]))
    bigram_div = len(set(bigrams)) / max(len(bigrams), 1)
    return (unique_ratio + bigram_div) / 2.0


def is_grammatical(text):
    """Check basic grammar quality."""
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
# Mixed Dataset Builder
# ═══════════════════════════════════════

def build_mixed_dataset(tokenizer):
    """Build interleaved factual + creative dataset."""
    from datasets import load_dataset
    print("\n📂 Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    
    # Sample 20 factual questions
    random.seed(SEED)
    indices = random.sample(range(len(ds)), 20)
    
    items = []
    
    for idx in indices:
        row = ds[idx]
        items.append({
            "type": "factual",
            "prompt": f"Q: {row['question']}\nA:",
            "question": row["question"],
            "mc1_targets": row["mc1_targets"],
            "p_creative_label": 0.0,
        })
    
    for prompt in CREATIVE_PROMPTS:
        items.append({
            "type": "creative",
            "prompt": f"Creative writing prompt: {prompt}\n\nResponse:",
            "question": prompt,
            "mc1_targets": None,
            "p_creative_label": 1.0,
        })
    
    # Shuffle deterministically
    random.seed(SEED + 1)
    random.shuffle(items)
    
    n_factual = sum(1 for x in items if x["type"] == "factual")
    n_creative = sum(1 for x in items if x["type"] == "creative")
    print(f"  Mixed dataset: {n_factual} factual + {n_creative} creative = {len(items)} total")
    
    return items


# ═══════════════════════════════════════
# Evaluation Loop (Per-Sample CfC)
# ═══════════════════════════════════════

def evaluate_static(model, tokenizer, dataset, sigma_fixed, label=""):
    """Evaluate with a fixed σ for all prompts."""
    print(f"\n{'═' * 50}")
    print(f"  {label}: σ={sigma_fixed:.3f}")
    print(f"{'═' * 50}")
    
    layers = get_layers(model)
    hook = AdaptiveSNNHook(sigma=sigma_fixed)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]
    
    factual_correct = factual_total = 0
    creative_novelties = []
    creative_grammar = 0
    creative_total = 0
    all_sigmas_factual = []
    all_sigmas_creative = []
    
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
                if predicted == correct_idx:
                    factual_correct += 1
                factual_total += 1
                all_sigmas_factual.append(sigma_fixed)
            else:
                resp = generate_text(model, tokenizer, item["prompt"])
                nov = compute_novelty(resp)
                gram = is_grammatical(resp)
                creative_novelties.append(nov)
                if gram:
                    creative_grammar += 1
                creative_total += 1
                all_sigmas_creative.append(sigma_fixed)
            
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(dataset)}] done")
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
        "creative_novelty": round(float(avg_novelty), 4),
        "creative_grammar_rate": round(grammar_rate, 1),
        "mean_sigma_factual": sigma_fixed,
        "mean_sigma_creative": sigma_fixed,
        "sigma_separation": 0.0,
        "all_sigmas_factual": all_sigmas_factual,
        "all_sigmas_creative": all_sigmas_creative,
    }


def evaluate_dual_mode(model, tokenizer, dataset, controller, classifier, hook, label=""):
    """Evaluate with per-sample adaptive σ via DualModeCfC."""
    print(f"\n{'═' * 50}")
    print(f"  {label}: Per-Sample Adaptive σ")
    print(f"{'═' * 50}")
    
    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]
    
    controller.train()
    optimizer = torch.optim.Adam(controller.parameters(), lr=LEARNING_RATE)
    hx = None
    
    factual_correct = factual_total = 0
    creative_novelties = []
    creative_grammar = 0
    creative_total = 0
    all_sigmas_factual = []
    all_sigmas_creative = []
    all_p_creative = []
    sigma_trajectory = []
    
    log_probs_buffer = []
    rewards_buffer = []
    
    try:
        for epoch in range(N_EPOCHS):
            print(f"\n  🧠 Epoch {epoch+1}/{N_EPOCHS}")
            epoch_factual_correct = 0
            epoch_factual_total = 0
            epoch_creative_novelties = []
            epoch_creative_grammar = 0
            epoch_creative_total = 0
            
            for idx, item in enumerate(dataset):
                # Step 1: Extract features and classify task type
                prompt_features = extract_prompt_features(model, tokenizer,
                                                         item["question"], hook)
                with torch.no_grad():
                    p_creative = classifier(prompt_features.unsqueeze(0)).item()
                
                all_p_creative.append({"type": item["type"], "p_creative": p_creative})
                
                # Step 2: Compute dynamic σ target
                sigma_target = (1 - p_creative) * SIGMA_FACTUAL + p_creative * SIGMA_CREATIVE
                
                # Step 3: CfC decides actual σ with p_creative as input
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
                
                # Step 4: Evaluate based on task type
                if item["type"] == "factual":
                    choices = item["mc1_targets"]
                    labels_list = choices["labels"]
                    choice_texts = choices["choices"]
                    correct_idx = labels_list.index(1) if 1 in labels_list else 0
                    
                    logprobs = [compute_completion_logprob(model, tokenizer,
                                item["prompt"], " " + c) for c in choice_texts]
                    predicted = np.argmax(logprobs)
                    is_correct = (predicted == correct_idx)
                    
                    if is_correct:
                        factual_correct += 1
                        epoch_factual_correct += 1
                    factual_total += 1
                    epoch_factual_total += 1
                    
                    all_sigmas_factual.append(current_sigma)
                    
                    # Reward: accuracy + quadratic penalty
                    task_reward = 1.0 if is_correct else 0.0
                    reward = task_reward - LAMBDA_QUAD * (current_sigma - sigma_target) ** 2
                    
                    logprob_diff = max(logprobs) - min(logprobs)
                    # Update features for next CfC step
                    features = torch.tensor(
                        [logprob_diff, 1.0 - float(is_correct), hidden_norm, step_frac, p_creative],
                        dtype=torch.float32
                    )
                else:
                    resp = generate_text(model, tokenizer, item["prompt"])
                    nov = compute_novelty(resp)
                    gram = is_grammatical(resp)
                    
                    creative_novelties.append(nov)
                    epoch_creative_novelties.append(nov)
                    if gram:
                        creative_grammar += 1
                        epoch_creative_grammar += 1
                    creative_total += 1
                    epoch_creative_total += 1
                    
                    all_sigmas_creative.append(current_sigma)
                    
                    # Reward: novelty * grammar + quadratic penalty
                    task_reward = nov if gram else -0.5
                    reward = task_reward - LAMBDA_QUAD * (current_sigma - sigma_target) ** 2
                
                # REINFORCE update
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
                
                sigma_trajectory.append({
                    "epoch": epoch + 1,
                    "step": epoch * len(dataset) + idx,
                    "sigma": round(current_sigma, 5),
                    "p_creative": round(p_creative, 3),
                    "sigma_target": round(sigma_target, 4),
                    "type": item["type"],
                })
                
                if (idx + 1) % 10 == 0:
                    recent = sigma_trajectory[-10:]
                    recent_fact = [s["sigma"] for s in recent if s["type"] == "factual"]
                    recent_crea = [s["sigma"] for s in recent if s["type"] == "creative"]
                    avg_f = np.mean(recent_fact) if recent_fact else 0
                    avg_c = np.mean(recent_crea) if recent_crea else 0
                    sep = abs(avg_c - avg_f)
                    print(f"  [{idx+1}/{len(dataset)}] σ_fact={avg_f:.4f} σ_crea={avg_c:.4f} "
                          f"sep={sep:.4f} p_c={p_creative:.3f}")
            
            # Epoch summary
            e_facc = epoch_factual_correct / max(epoch_factual_total, 1) * 100
            e_nov = np.mean(epoch_creative_novelties) if epoch_creative_novelties else 0
            e_gram = epoch_creative_grammar / max(epoch_creative_total, 1) * 100
            print(f"  ✅ E{epoch+1}: Fact acc={e_facc:.1f}% | Crea nov={e_nov:.3f} gram={e_gram:.0f}%")
    
    finally:
        for h in handles:
            h.remove()
    
    # Overall metrics
    factual_acc = factual_correct / max(factual_total, 1) * 100
    avg_novelty = np.mean(creative_novelties) if creative_novelties else 0
    grammar_rate = creative_grammar / max(creative_total, 1) * 100
    
    mean_sigma_f = np.mean(all_sigmas_factual) if all_sigmas_factual else 0
    mean_sigma_c = np.mean(all_sigmas_creative) if all_sigmas_creative else 0
    sigma_sep = abs(mean_sigma_c - mean_sigma_f)
    
    # Per-epoch factual acc (use last epoch only)
    last_epoch_facts = [s for s in sigma_trajectory if s["epoch"] == N_EPOCHS and s["type"] == "factual"]
    
    print(f"\n  📊 DUAL-MODE RESULTS:")
    print(f"     Factual acc: {factual_acc:.1f}% | σ̄_factual = {mean_sigma_f:.4f}")
    print(f"     Creative nov: {avg_novelty:.3f} | σ̄_creative = {mean_sigma_c:.4f}")
    print(f"     Grammar: {grammar_rate:.0f}%")
    print(f"     ★ σ SEPARATION: {sigma_sep:.4f}")
    
    return {
        "condition": label,
        "factual_acc": round(factual_acc, 2),
        "factual_correct": factual_correct,
        "factual_total": factual_total,
        "creative_novelty": round(float(avg_novelty), 4),
        "creative_grammar_rate": round(grammar_rate, 1),
        "mean_sigma_factual": round(mean_sigma_f, 5),
        "mean_sigma_creative": round(mean_sigma_c, 5),
        "sigma_separation": round(sigma_sep, 5),
        "all_sigmas_factual": [round(s, 5) for s in all_sigmas_factual],
        "all_sigmas_creative": [round(s, 5) for s in all_sigmas_creative],
        "sigma_trajectory": sigma_trajectory,
        "p_creative_log": all_p_creative,
    }


# ═══════════════════════════════════════
# Visualization
# ═══════════════════════════════════════

def visualize(result_static_f, result_static_c, result_dual):
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
    fig.suptitle("Phase 24: Per-Sample Task-Dynamic CfC — The Dual-Mode Brain\n"
                 "σ_target = (1-p_creative)×0.046 + p_creative×0.080",
                 fontsize=14, fontweight="bold", color="#e94560")

    # Panel 1: σ trajectory colored by task type
    ax1 = axes[0, 0]
    traj = result_dual.get("sigma_trajectory", [])
    if traj:
        steps_f = [t["step"] for t in traj if t["type"] == "factual"]
        sigmas_f = [t["sigma"] for t in traj if t["type"] == "factual"]
        steps_c = [t["step"] for t in traj if t["type"] == "creative"]
        sigmas_c = [t["sigma"] for t in traj if t["type"] == "creative"]
        
        ax1.scatter(steps_f, sigmas_f, c="#4FC3F7", s=20, alpha=0.7, label="Factual", zorder=3)
        ax1.scatter(steps_c, sigmas_c, c="#FF7043", s=20, alpha=0.7, label="Creative", zorder=3)
    
    ax1.axhline(y=SIGMA_FACTUAL, color="#4FC3F7", linestyle="--", alpha=0.6,
                label=f"σ* factual={SIGMA_FACTUAL}")
    ax1.axhline(y=SIGMA_CREATIVE, color="#FF7043", linestyle="--", alpha=0.6,
                label=f"σ* creative={SIGMA_CREATIVE}")
    ax1.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax1.set_title("Per-Sample σ: Factual vs Creative", fontweight="bold")
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("σ")
    ax1.grid(True, alpha=0.3)

    # Panel 2: σ distribution comparison
    ax2 = axes[0, 1]
    bins = np.linspace(0, 0.12, 25)
    if result_dual["all_sigmas_factual"]:
        ax2.hist(result_dual["all_sigmas_factual"], bins=bins, alpha=0.6,
                color="#4FC3F7", label=f"Factual (μ={result_dual['mean_sigma_factual']:.4f})",
                edgecolor="#333")
    if result_dual["all_sigmas_creative"]:
        ax2.hist(result_dual["all_sigmas_creative"], bins=bins, alpha=0.6,
                color="#FF7043", label=f"Creative (μ={result_dual['mean_sigma_creative']:.4f})",
                edgecolor="#333")
    ax2.axvline(x=SIGMA_FACTUAL, color="#4FC3F7", linestyle="--", alpha=0.8)
    ax2.axvline(x=SIGMA_CREATIVE, color="#FF7043", linestyle="--", alpha=0.8)
    ax2.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax2.set_title(f"σ Distribution — Separation = {result_dual['sigma_separation']:.4f}",
                  fontweight="bold")
    ax2.set_xlabel("σ")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Performance comparison (3 conditions)
    ax3 = axes[1, 0]
    conditions = ["Static\nFactual", "Static\nCreative", "Dual-Mode\nCfC"]
    accs = [result_static_f["factual_acc"], result_static_c["factual_acc"],
            result_dual["factual_acc"]]
    novs = [result_static_f["creative_novelty"] * 100, 
            result_static_c["creative_novelty"] * 100,
            result_dual["creative_novelty"] * 100]
    
    x = np.arange(len(conditions))
    width = 0.35
    bars1 = ax3.bar(x - width/2, accs, width, label="Factual Acc (%)", 
                    color="#4FC3F7", edgecolor="#333")
    bars2 = ax3.bar(x + width/2, novs, width, label="Novelty (×100)", 
                    color="#FF7043", edgecolor="#333")
    
    for bar, v in zip(bars1, accs):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}",
                ha="center", fontsize=8)
    for bar, v in zip(bars2, novs):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}",
                ha="center", fontsize=8)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(conditions, fontsize=9)
    ax3.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax3.set_title("Performance: Static vs Dual-Mode", fontweight="bold")
    ax3.set_ylabel("Score")
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: p_creative classifier output
    ax4 = axes[1, 1]
    p_log = result_dual.get("p_creative_log", [])
    if p_log:
        p_factual = [x["p_creative"] for x in p_log if x["type"] == "factual"]
        p_creative = [x["p_creative"] for x in p_log if x["type"] == "creative"]
        
        bins_p = np.linspace(0, 1, 20)
        ax4.hist(p_factual, bins=bins_p, alpha=0.6, color="#4FC3F7",
                label=f"Factual (μ={np.mean(p_factual):.3f})", edgecolor="#333")
        ax4.hist(p_creative, bins=bins_p, alpha=0.6, color="#FF7043",
                label=f"Creative (μ={np.mean(p_creative):.3f})", edgecolor="#333")
    ax4.axvline(x=0.5, color="#FFD54F", linestyle="--", alpha=0.8, label="Decision boundary")
    ax4.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax4.set_title("Task Classifier Output (p_creative)", fontweight="bold")
    ax4.set_xlabel("p_creative")
    ax4.set_ylabel("Count")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join(FIGURES_DIR, "phase24_dual_mode_brain.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Figure saved: {out_path}")
    return out_path


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 24: Per-Sample Task-Dynamic CfC — The Dual-Mode Brain")
    print("  σ_target = (1-p_creative)×0.046 + p_creative×0.080")
    print("=" * 60)
    t_start = time.time()
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    model, tokenizer = load_model()
    layers = get_layers(model)
    
    # Build mixed dataset
    dataset = build_mixed_dataset(tokenizer)
    
    # Setup hook
    hook = AdaptiveSNNHook(sigma=0.05)
    
    # Extract factual prompts for classifier training
    factual_prompts = [item["question"] for item in dataset if item["type"] == "factual"]
    creative_prompts_list = [item["question"] for item in dataset if item["type"] == "creative"]
    
    # Pre-train task classifier
    classifier = TaskClassifier(input_dim=8, hidden_dim=32)
    handles_tmp = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]
    classifier = pretrain_classifier(classifier, model, tokenizer, hook,
                                     factual_prompts, creative_prompts_list, epochs=30)
    for h in handles_tmp:
        h.remove()
    
    # ═══ Condition A: Static-Factual ═══
    result_a = evaluate_static(model, tokenizer, dataset, SIGMA_FACTUAL,
                               "A: Static-Factual (σ=0.046)")
    
    # ═══ Condition B: Static-Creative ═══
    result_b = evaluate_static(model, tokenizer, dataset, SIGMA_CREATIVE,
                               "B: Static-Creative (σ=0.080)")
    
    # ═══ Condition C: Dual-Mode CfC ═══
    controller = DualModeCfC(input_size=5)
    hook_dual = AdaptiveSNNHook(sigma=0.05)
    result_c = evaluate_dual_mode(model, tokenizer, dataset, controller, classifier,
                                  hook_dual, "C: Dual-Mode CfC")
    
    # ═══ Summary ═══
    elapsed = (time.time() - t_start) / 60
    
    print(f"\n{'═' * 70}")
    print(f"  RESULTS — Phase 24: Per-Sample Task-Dynamic CfC")
    print(f"{'═' * 70}")
    print(f"  {'Condition':<30} | {'Fact Acc':>8} | {'Novelty':>8} | {'σ̄ Fact':>7} | {'σ̄ Crea':>7} | {'Sep':>6}")
    print(f"  {'─'*30}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}")
    
    for r in [result_a, result_b, result_c]:
        print(f"  {r['condition']:<30} | {r['factual_acc']:>7.1f}% | {r['creative_novelty']:>8.4f} | "
              f"{r['mean_sigma_factual']:>7.4f} | {r['mean_sigma_creative']:>7.4f} | "
              f"{r['sigma_separation']:>6.4f}")
    
    # Key test
    sep = result_c["sigma_separation"]
    print(f"\n  ★ KEY METRIC: σ Separation = {sep:.4f}")
    if sep > 0.01:
        print(f"  ✅ SUCCESS: CfC uses different σ for factual vs creative (sep > 0.01)")
    elif sep > 0.005:
        print(f"  ⚠️ PARTIAL: Some separation detected but below threshold")
    else:
        print(f"  ❌ CfC does not yet differentiate between task types")
    
    # Best of both worlds test
    dual_fact = result_c["factual_acc"]
    dual_nov = result_c["creative_novelty"]
    static_f_fact = result_a["factual_acc"]
    static_c_nov = result_b["creative_novelty"]
    
    if dual_fact >= static_f_fact * 0.9 and dual_nov >= static_c_nov * 0.9:
        print(f"  ✅ BEST OF BOTH WORLDS: Dual-Mode achieves ≥90% of each static baseline!")
    
    # Visualize
    fig_path = visualize(result_a, result_b, result_c)
    
    # Save
    log = {
        "experiment": "Phase 24: Per-Sample Task-Dynamic CfC",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "sigma_factual": SIGMA_FACTUAL,
            "sigma_creative": SIGMA_CREATIVE,
            "lambda_quad": LAMBDA_QUAD,
            "n_epochs": N_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "n_factual": sum(1 for x in dataset if x["type"] == "factual"),
            "n_creative": sum(1 for x in dataset if x["type"] == "creative"),
        },
        "static_factual": {k: v for k, v in result_a.items() 
                          if k not in ("all_sigmas_factual", "all_sigmas_creative")},
        "static_creative": {k: v for k, v in result_b.items()
                           if k not in ("all_sigmas_factual", "all_sigmas_creative")},
        "dual_mode": result_c,
        "elapsed_min": round(elapsed, 1),
    }
    
    log_path = os.path.join(RESULTS_DIR, "phase24_dual_mode_brain_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Results: {log_path}")
    print(f"📊 Figure:  {fig_path}")
    print(f"⏱  Total: {elapsed:.1f} min")
    print(f"\n✅ Phase 24 complete!")


if __name__ == "__main__":
    main()
