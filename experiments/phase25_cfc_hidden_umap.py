"""
Phase 25: CfC Hidden State UMAP — Decoding the Dual-Mode Brain
================================================================

Hypothesis: The Dual-Mode CfC controller's Unified Regime (σ ≈ 0.075 for both
tasks) is NOT because it fails to distinguish tasks. Instead, its 16D internal
hidden state encodes task context, acting as a "Soft MoE" — even when σ output
is nearly identical, the internal representation differs by task type.

Experiment:
    1. Re-run the Dual-Mode CfC evaluation (same as Phase 24)
    2. Record the 16D hidden state vector (hx) at every step
    3. Apply PCA and UMAP to project 16D → 2D
    4. Color-code by task type (factual vs creative)

Expected: Two distinct clusters in UMAP space, proving CfC internally
differentiates tasks even when σ output is nearly identical.

Usage:
    python experiments/phase25_cfc_hidden_umap.py
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
    """SNN noise injection with adjustable σ."""
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


# ═══════════════════════════════════════
# Task Classifier (same as Phase 24)
# ═══════════════════════════════════════

class TaskClassifier(nn.Module):
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


# ═══════════════════════════════════════
# DualMode CfC Controller (same as Phase 24)
# ═══════════════════════════════════════

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


# ═══════════════════════════════════════
# Generation & Evaluation helpers
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


def build_mixed_dataset(tokenizer):
    from datasets import load_dataset
    print("\n📂 Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
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
    random.seed(SEED + 1)
    random.shuffle(items)
    n_factual = sum(1 for x in items if x["type"] == "factual")
    n_creative = sum(1 for x in items if x["type"] == "creative")
    print(f"  Mixed dataset: {n_factual} factual + {n_creative} creative = {len(items)} total")
    return items


# ═══════════════════════════════════════
# Phase 25: Dual-Mode with Hidden State Recording
# ═══════════════════════════════════════

def evaluate_dual_mode_with_hidden_states(model, tokenizer, dataset,
                                          controller, classifier, hook):
    """
    Run Dual-Mode CfC evaluation while recording:
    - 16D hidden state (hx) at every step
    - σ output at every step
    - Task type (factual/creative)
    - p_creative
    """
    print(f"\n{'═' * 60}")
    print(f"  Phase 25: Dual-Mode CfC with Hidden State Recording")
    print(f"{'═' * 60}")

    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]

    controller.train()
    optimizer = torch.optim.Adam(controller.parameters(), lr=LEARNING_RATE)
    hx = None

    # Storage for hidden states
    hidden_states_log = []  # List of {hx: 16D, type, sigma, p_creative, step}
    
    factual_correct = factual_total = 0
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
                # Feature extraction
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
                
                # ★ RECORD THE HIDDEN STATE ★
                if isinstance(hx_new, tuple):
                    hx_flat = torch.cat([h.detach().cpu().flatten() for h in hx_new]).numpy()
                else:
                    hx_flat = hx_new.detach().cpu().flatten().numpy()

                hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()
                current_sigma = sigma_tensor.detach().item()
                hook.update_sigma(current_sigma)

                hidden_states_log.append({
                    "hx": hx_flat.tolist(),
                    "type": item["type"],
                    "sigma": round(current_sigma, 6),
                    "p_creative": round(p_creative, 4),
                    "epoch": epoch + 1,
                    "step": epoch * len(dataset) + idx,
                    "prompt_snippet": item["question"][:60],
                })

                # Evaluate
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

                # REINFORCE
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

                if (idx + 1) % 10 == 0:
                    print(f"  [{idx+1}/{len(dataset)}] σ={current_sigma:.4f} "
                          f"p_c={p_creative:.3f} type={item['type']}")

            # Epoch summary
            e_facc = factual_correct / max(factual_total, 1) * 100
            e_nov = np.mean(creative_novelties) if creative_novelties else 0
            e_gram = creative_grammar / max(creative_total, 1) * 100
            print(f"  ✅ E{epoch+1}: Fact acc={e_facc:.1f}% | Crea nov={e_nov:.3f} gram={e_gram:.0f}%")

    finally:
        for h in handles:
            h.remove()

    # Compute metrics
    factual_acc = factual_correct / max(factual_total, 1) * 100
    avg_novelty = np.mean(creative_novelties) if creative_novelties else 0
    grammar_rate = creative_grammar / max(creative_total, 1) * 100
    mean_sigma_f = np.mean(all_sigmas_factual) if all_sigmas_factual else 0
    mean_sigma_c = np.mean(all_sigmas_creative) if all_sigmas_creative else 0

    print(f"\n  📊 Performance: Fact acc={factual_acc:.1f}% | Nov={avg_novelty:.3f} | Gram={grammar_rate:.0f}%")
    print(f"     σ_factual={mean_sigma_f:.4f} | σ_creative={mean_sigma_c:.4f}")
    print(f"     Hidden states recorded: {len(hidden_states_log)} vectors of dim {len(hidden_states_log[0]['hx'])}")

    return hidden_states_log


# ═══════════════════════════════════════
# UMAP / PCA Visualization
# ═══════════════════════════════════════

def visualize_hidden_states(hidden_states_log):
    """Create PCA + UMAP scatter plots of CfC hidden states, colored by task type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    try:
        import umap
        has_umap = True
    except ImportError:
        print("  ⚠️ umap-learn not available, using t-SNE as fallback")
        from sklearn.manifold import TSNE
        has_umap = False

    # Extract data
    hx_matrix = np.array([h["hx"] for h in hidden_states_log])
    types = [h["type"] for h in hidden_states_log]
    sigmas = [h["sigma"] for h in hidden_states_log]
    epochs = [h["epoch"] for h in hidden_states_log]

    # Separate by type
    mask_f = np.array([t == "factual" for t in types])
    mask_c = np.array([t == "creative" for t in types])

    # Also separate by epoch for temporal analysis
    mask_e1 = np.array([e == 1 for e in epochs])
    mask_e2 = np.array([e == 2 for e in epochs])

    dim = hx_matrix.shape[1]
    print(f"\n  📐 Hidden state dimensionality: {dim}D")
    print(f"     Factual samples: {mask_f.sum()} | Creative samples: {mask_c.sum()}")

    # ─── PCA ───
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(hx_matrix)
    pca_var = pca.explained_variance_ratio_
    print(f"     PCA explained variance: PC1={pca_var[0]:.3f} PC2={pca_var[1]:.3f} "
          f"(total={sum(pca_var):.3f})")

    # ─── UMAP or t-SNE ───
    if has_umap:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED, metric='euclidean')
        proj_coords = reducer.fit_transform(hx_matrix)
        proj_name = "UMAP"
    else:
        reducer = TSNE(n_components=2, random_state=SEED, perplexity=min(15, len(hx_matrix)-1))
        proj_coords = reducer.fit_transform(hx_matrix)
        proj_name = "t-SNE"

    # ─── Compute cluster separation metrics ───
    centroid_f_pca = pca_coords[mask_f].mean(axis=0) if mask_f.any() else np.zeros(2)
    centroid_c_pca = pca_coords[mask_c].mean(axis=0) if mask_c.any() else np.zeros(2)
    cluster_dist_pca = np.linalg.norm(centroid_f_pca - centroid_c_pca)

    centroid_f_umap = proj_coords[mask_f].mean(axis=0) if mask_f.any() else np.zeros(2)
    centroid_c_umap = proj_coords[mask_c].mean(axis=0) if mask_c.any() else np.zeros(2)
    cluster_dist_umap = np.linalg.norm(centroid_f_umap - centroid_c_umap)

    # Intra-class spread (average distance from centroid)
    spread_f = np.mean(np.linalg.norm(proj_coords[mask_f] - centroid_f_umap, axis=1)) if mask_f.sum() > 1 else 0
    spread_c = np.mean(np.linalg.norm(proj_coords[mask_c] - centroid_c_umap, axis=1)) if mask_c.sum() > 1 else 0
    
    # Silhouette-like metric: inter-cluster / average intra-cluster
    avg_spread = (spread_f + spread_c) / 2
    separation_ratio = cluster_dist_umap / avg_spread if avg_spread > 0 else float('inf')

    print(f"\n  📊 Cluster Analysis ({proj_name}):")
    print(f"     Inter-cluster distance: {cluster_dist_umap:.4f}")
    print(f"     Factual spread: {spread_f:.4f} | Creative spread: {spread_c:.4f}")
    print(f"     Separation ratio (inter/intra): {separation_ratio:.3f}")
    print(f"     {'✅ CLEAR SEPARATION' if separation_ratio > 1.5 else '⚠️ Partial overlap' if separation_ratio > 0.8 else '❌ No clear separation'}")

    # ═══════════════════════════════════════
    # Visualization (2x2 plot)
    # ═══════════════════════════════════════

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#ccc", "ytick.color": "#ccc",
        "grid.color": "#333", "grid.alpha": 0.3,
    })

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Phase 25: CfC Hidden State Visualization — Decoding the Dual-Mode Brain\n"
                 f"16D → 2D projection | {dim}D hidden state | n={len(hidden_states_log)} samples",
                 fontsize=14, fontweight="bold", color="#e94560")

    # Panel 1: PCA colored by task type
    ax1 = axes[0, 0]
    ax1.scatter(pca_coords[mask_f, 0], pca_coords[mask_f, 1],
                c="#4FC3F7", s=50, alpha=0.7, label="Factual", zorder=3, edgecolors="#333", linewidths=0.5)
    ax1.scatter(pca_coords[mask_c, 0], pca_coords[mask_c, 1],
                c="#FF7043", s=50, alpha=0.7, label="Creative", zorder=3, edgecolors="#333", linewidths=0.5)
    # Draw centroid markers
    ax1.scatter(*centroid_f_pca, c="#4FC3F7", s=200, marker="*", edgecolors="white", linewidths=1.5, zorder=5)
    ax1.scatter(*centroid_c_pca, c="#FF7043", s=200, marker="*", edgecolors="white", linewidths=1.5, zorder=5)
    ax1.legend(fontsize=9, facecolor="#16213e", edgecolor="#555")
    ax1.set_title(f"PCA: Task Type (var={sum(pca_var)*100:.1f}%)\n"
                  f"Cluster dist={cluster_dist_pca:.3f}", fontweight="bold")
    ax1.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}%)")
    ax1.grid(True, alpha=0.3)

    # Panel 2: UMAP/t-SNE colored by task type
    ax2 = axes[0, 1]
    ax2.scatter(proj_coords[mask_f, 0], proj_coords[mask_f, 1],
                c="#4FC3F7", s=50, alpha=0.7, label="Factual", zorder=3, edgecolors="#333", linewidths=0.5)
    ax2.scatter(proj_coords[mask_c, 0], proj_coords[mask_c, 1],
                c="#FF7043", s=50, alpha=0.7, label="Creative", zorder=3, edgecolors="#333", linewidths=0.5)
    ax2.scatter(*centroid_f_umap, c="#4FC3F7", s=200, marker="*", edgecolors="white", linewidths=1.5, zorder=5)
    ax2.scatter(*centroid_c_umap, c="#FF7043", s=200, marker="*", edgecolors="white", linewidths=1.5, zorder=5)
    ax2.legend(fontsize=9, facecolor="#16213e", edgecolor="#555")
    verdict = "✅ SEPARATED" if separation_ratio > 1.5 else "⚠️ PARTIAL" if separation_ratio > 0.8 else "❌ MIXED"
    ax2.set_title(f"{proj_name}: Task Type — {verdict}\n"
                  f"Separation ratio={separation_ratio:.2f}", fontweight="bold")
    ax2.set_xlabel(f"{proj_name}-1")
    ax2.set_ylabel(f"{proj_name}-2")
    ax2.grid(True, alpha=0.3)

    # Panel 3: UMAP/t-SNE colored by σ value
    ax3 = axes[1, 0]
    sc = ax3.scatter(proj_coords[:, 0], proj_coords[:, 1],
                     c=sigmas, cmap="plasma", s=50, alpha=0.7, edgecolors="#333", linewidths=0.5)
    plt.colorbar(sc, ax=ax3, label="σ value")
    ax3.set_title(f"{proj_name}: Colored by σ Output", fontweight="bold")
    ax3.set_xlabel(f"{proj_name}-1")
    ax3.set_ylabel(f"{proj_name}-2")
    ax3.grid(True, alpha=0.3)

    # Panel 4: UMAP/t-SNE colored by epoch (temporal evolution)
    ax4 = axes[1, 1]
    if mask_e1.any():
        ax4.scatter(proj_coords[mask_e1 & mask_f, 0], proj_coords[mask_e1 & mask_f, 1],
                    c="#81D4FA", s=30, alpha=0.5, label="E1 Factual", marker="o")
        ax4.scatter(proj_coords[mask_e1 & mask_c, 0], proj_coords[mask_e1 & mask_c, 1],
                    c="#FFAB91", s=30, alpha=0.5, label="E1 Creative", marker="o")
    if mask_e2.any():
        ax4.scatter(proj_coords[mask_e2 & mask_f, 0], proj_coords[mask_e2 & mask_f, 1],
                    c="#0288D1", s=60, alpha=0.8, label="E2 Factual", marker="^")
        ax4.scatter(proj_coords[mask_e2 & mask_c, 0], proj_coords[mask_e2 & mask_c, 1],
                    c="#E64A19", s=60, alpha=0.8, label="E2 Creative", marker="^")
    ax4.legend(fontsize=7, facecolor="#16213e", edgecolor="#555", ncol=2)
    ax4.set_title(f"{proj_name}: Temporal Evolution (E1→E2)\n"
                  "Light=Epoch1, Dark=Epoch2", fontweight="bold")
    ax4.set_xlabel(f"{proj_name}-1")
    ax4.set_ylabel(f"{proj_name}-2")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join(FIGURES_DIR, "phase25_cfc_hidden_umap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {out_path}")

    return {
        "hidden_dim": dim,
        "n_samples": len(hidden_states_log),
        "n_factual": int(mask_f.sum()),
        "n_creative": int(mask_c.sum()),
        "pca_variance_explained": [round(float(v), 4) for v in pca_var],
        "cluster_distance_pca": round(float(cluster_dist_pca), 4),
        "cluster_distance_umap": round(float(cluster_dist_umap), 4),
        "factual_spread": round(float(spread_f), 4),
        "creative_spread": round(float(spread_c), 4),
        "separation_ratio": round(float(separation_ratio), 4),
        "projection_method": proj_name,
        "verdict": verdict,
        "figure_path": out_path,
    }


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 25: CfC Hidden State UMAP — Decoding the Dual-Mode Brain")
    print("  Question: Does the 16D hidden state encode task context?")
    print("=" * 60)
    t_start = time.time()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Load model
    model, tokenizer = load_model()
    hook = AdaptiveSNNHook(sigma=0.05)

    # Build dataset (same as Phase 24)
    dataset = build_mixed_dataset(tokenizer)

    # Factual prompts for classifier training
    factual_prompts = [item["question"] for item in dataset if item["type"] == "factual"]

    # Pre-train classifier
    classifier = TaskClassifier(input_dim=8, hidden_dim=32)
    classifier = pretrain_classifier(classifier, model, tokenizer, hook,
                                     factual_prompts, CREATIVE_PROMPTS)

    # Initialize CfC controller
    controller = DualModeCfC(input_size=5, hidden_size=32, num_neurons=16)

    # Run evaluation with hidden state recording
    hidden_states_log = evaluate_dual_mode_with_hidden_states(
        model, tokenizer, dataset, controller, classifier, hook
    )

    # Visualize
    analysis = visualize_hidden_states(hidden_states_log)

    # Save results
    elapsed = time.time() - t_start
    result = {
        "phase": "Phase 25: CfC Hidden State UMAP",
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "analysis": analysis,
        "hidden_states": hidden_states_log,  # Full data for reproducibility
    }

    out_path = os.path.join(RESULTS_DIR, "phase25_cfc_hidden_umap_log.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  💾 Results saved: {out_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Phase 25 COMPLETE — {elapsed:.0f}s")
    print(f"{'=' * 60}")
    print(f"  Hidden state dim: {analysis['hidden_dim']}D")
    print(f"  Projection method: {analysis['projection_method']}")
    print(f"  Cluster distance: {analysis['cluster_distance_umap']:.4f}")
    print(f"  Separation ratio: {analysis['separation_ratio']:.3f}")
    print(f"  Verdict: {analysis['verdict']}")
    print(f"{'=' * 60}")

    return result


if __name__ == "__main__":
    main()
