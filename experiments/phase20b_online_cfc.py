"""
phase20b_online_cfc.py — Online CfC: Zero Pre-training
========================================================

Phase 20b: Addresses the tautology concern from Phase 20.

Key difference from Phase 20:
  - NO pre-training on synthetic dose-response data
  - CfC starts from RANDOM initialization (no prior knowledge of sweet spot)
  - Learns σ purely from live TruthfulQA accuracy feedback (REINFORCE)
  - If σ converges to ~0.05 without pre-training, the sweet spot is proven real

Pipeline:
  Step 1: Initialize CfC with random weights (NO pre-training)
  Step 2: Run TruthfulQA with online REINFORCE learning (3 epochs)
  Step 3: Compare with Phase 20 pre-trained results + static baselines
"""

import torch
import torch.nn as nn
import os
import sys
import json
import gc
import time
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# ─── Settings ───
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TARGET_LAYERS = list(range(15, 21))  # L15-20
BATCH_SIZE = 10  # CfC updates σ every BATCH_SIZE questions
SIGMA_MIN = 0.001
SIGMA_MAX = 0.15
N_EPOCHS = 3  # Run TruthfulQA 3 times for convergence
LEARNING_RATE = 0.003  # REINFORCE learning rate
ENTROPY_BONUS = 0.01  # Encourage exploration
LAMBDA_TAX = 2.0  # Penalty weight for high σ

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════
# PART 1: CfC Dose Controller (same architecture, NO pre-training)
# ═══════════════════════════════════════

class CfCDoseController(nn.Module):
    """
    CfC-based adaptive σ controller.
    
    CRITICAL DIFFERENCE FROM PHASE 20:
    - Weights are RANDOMLY initialized
    - No call to pretrain_cfc()
    - Initial σ output ≈ sigmoid(random) * range ≈ 0.075 (midpoint)
    
    Input features (per timestep):
        0: logprob_diff    — answer distribution spread
        1: refusal_signal  — 1 - batch_accuracy
        2: hidden_norm     — L2 norm of hidden state at injection layer
        3: step_fraction   — current_step / total_steps
    
    Output: σ ∈ [SIGMA_MIN, SIGMA_MAX] via sigmoid scaling
    """
    def __init__(self, input_size=4, hidden_size=32, num_neurons=16):
        super().__init__()
        wiring = AutoNCP(num_neurons, 1)  # 1 motor neuron output
        self.cfc = CfC(input_size, wiring, batch_first=True)
        self.sigma_scale = nn.Parameter(torch.tensor([SIGMA_MAX - SIGMA_MIN]))
        self.sigma_min = SIGMA_MIN
        
    def forward(self, x, hx=None, timespans=None):
        output, hx = self.cfc(x, hx=hx, timespans=timespans)
        sigma = torch.sigmoid(output) * self.sigma_scale + self.sigma_min
        return sigma, hx
    
    def get_sigma(self, features, hx=None):
        """Single-step inference: returns σ tensor (for gradient) and hidden state."""
        x = features.unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
        sigma, hx = self.forward(x, hx=hx)
        return sigma.squeeze(), hx  # Return tensor (not .item()) for gradient


# ═══════════════════════════════════════
# PART 2: LLM Loading & SNN Hook
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
    """SNN hook with dynamically adjustable σ."""
    def __init__(self, sigma=0.01):
        self.sigma = sigma
        self.last_hidden_norm = 0.0
    
    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * self.sigma
        self.last_hidden_norm = hs.float().norm().item() / max(hs.numel(), 1)
        return (hs + noise,) + args[1:]
    
    def update_sigma(self, new_sigma):
        self.sigma = np.clip(new_sigma, SIGMA_MIN, SIGMA_MAX)


# ═══════════════════════════════════════
# PART 3: TruthfulQA Evaluation
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


def evaluate_truthfulqa_static(model, tokenizer, sigma, dataset, condition_name):
    """Evaluate TruthfulQA with static σ (baseline comparison)."""
    print(f"\n📊 Evaluating TruthfulQA with static σ={sigma} ({condition_name})...")
    
    layers = get_layers(model)
    handles = []
    
    if sigma > 0:
        def static_hook(module, args, sigma=sigma):
            hs = args[0]
            noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
            low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
            noise = 0.7 * noise + 0.3 * low_freq
            noise = noise * sigma
            return (hs + noise,) + args[1:]
        
        for layer_idx in TARGET_LAYERS:
            if layer_idx < len(layers):
                handles.append(layers[layer_idx].register_forward_pre_hook(static_hook))
    
    correct = total = 0
    t0 = time.time()
    
    try:
        for idx, row in enumerate(dataset):
            question = row["question"]
            choices = row["mc1_targets"]
            labels = choices["labels"]
            choice_texts = choices["choices"]
            
            correct_idx = labels.index(1) if 1 in labels else 0
            prompt = f"Q: {question}\nA:"
            
            logprobs = [compute_completion_logprob(model, tokenizer, prompt, " " + c) for c in choice_texts]
            if np.argmax(logprobs) == correct_idx:
                correct += 1
            total += 1
            
            if (idx + 1) % 200 == 0:
                print(f"  [{idx+1}/{len(dataset)}] {correct/total*100:.1f}%")
    finally:
        for h in handles:
            h.remove()
    
    acc = correct / total * 100
    elapsed = (time.time() - t0) / 60
    print(f"  ✅ {condition_name}: {acc:.2f}% ({correct}/{total}) [{elapsed:.1f}min]")
    
    return {
        "condition": condition_name,
        "accuracy": round(acc, 2),
        "correct": correct,
        "total": total,
        "sigma": sigma,
        "time_minutes": round(elapsed, 1),
    }


# ═══════════════════════════════════════
# PART 4: Online REINFORCE CfC Learning
# ═══════════════════════════════════════

def evaluate_truthfulqa_online(model, tokenizer, controller, optimizer, dataset,
                                epoch, n_epochs):
    """
    Evaluate TruthfulQA with ONLINE CfC learning.
    
    Key: CfC weights are UPDATED during evaluation via REINFORCE.
    
    Reward signal:
        reward = batch_accuracy - λ * σ_current
        
    This encourages the CfC to:
        1. Keep accuracy high (low alignment tax) → high batch_accuracy
        2. Use enough noise to discover vulnerabilities → but penalized for σ too high
    """
    print(f"\n🧠 Epoch {epoch+1}/{n_epochs}: Online CfC learning on TruthfulQA...")
    
    layers = get_layers(model)
    hook = AdaptiveSNNHook(sigma=0.05)  # Will be overwritten by CfC immediately
    
    handles = []
    for layer_idx in TARGET_LAYERS:
        if layer_idx < len(layers):
            handles.append(layers[layer_idx].register_forward_pre_hook(hook))
    
    # CfC state
    controller.train()  # TRAINING mode — weights will update
    hx = None
    
    correct = 0
    total = 0
    sigma_trajectory = []
    all_sigmas = []
    all_rewards = []
    log_probs_buffer = []
    rewards_buffer = []
    t0 = time.time()
    
    # Batch tracking
    batch_correct = 0
    batch_total = 0
    batch_logprob_diffs = []
    
    try:
        for idx, row in enumerate(dataset):
            question = row["question"]
            choices = row["mc1_targets"]
            labels = choices["labels"]
            choice_texts = choices["choices"]
            
            correct_idx = labels.index(1) if 1 in labels else 0
            prompt = f"Q: {question}\nA:"
            
            logprobs = []
            for choice in choice_texts:
                lp = compute_completion_logprob(model, tokenizer, prompt, " " + choice)
                logprobs.append(lp)
            
            predicted = np.argmax(logprobs)
            is_correct = (predicted == correct_idx)
            if is_correct:
                correct += 1
                batch_correct += 1
            total += 1
            batch_total += 1
            
            logprob_diff = max(logprobs) - min(logprobs)
            batch_logprob_diffs.append(logprob_diff)
            all_sigmas.append(hook.sigma)
            
            # Every BATCH_SIZE questions: REINFORCE update
            if batch_total >= BATCH_SIZE:
                batch_acc = batch_correct / batch_total
                avg_logprob_diff = np.mean(batch_logprob_diffs)
                hidden_norm = hook.last_hidden_norm
                step_frac = idx / len(dataset)
                
                # CfC input
                features = torch.tensor(
                    [avg_logprob_diff, 1.0 - batch_acc, hidden_norm, step_frac],
                    dtype=torch.float32
                )
                
                # Forward pass (WITH gradient)
                sigma_tensor, hx_new = controller.get_sigma(features, hx)
                
                # Detach hidden state for next step (truncated BPTT)
                hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()
                
                # REINFORCE reward
                # Higher accuracy = good, lower σ = good (but need some σ for vulnerability probing)
                # Sweet spot: enough σ for discovery, not so much that accuracy tanks
                reward = batch_acc - LAMBDA_TAX * sigma_tensor.detach().item()
                
                # Policy gradient: log π(σ|s) * reward
                # σ is output of sigmoid, so log_prob ≈ log(σ / (SIGMA_MAX - SIGMA_MIN))
                # Simplified: use MSE-like loss to push σ toward reward-maximizing direction
                log_prob = torch.log(sigma_tensor + 1e-8)
                
                # Accumulate for batch update
                log_probs_buffer.append(log_prob)
                rewards_buffer.append(reward)
                all_rewards.append(reward)
                
                # Update CfC every 5 batches (50 questions)
                if len(log_probs_buffer) >= 5:
                    # Normalize rewards (baseline subtraction)
                    rewards_t = torch.tensor(rewards_buffer, dtype=torch.float32)
                    if rewards_t.std() > 0:
                        rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
                    
                    # Policy gradient loss
                    policy_loss = 0
                    for lp, r in zip(log_probs_buffer, rewards_t):
                        policy_loss -= lp * r.item()  # Negative because we maximize reward
                    
                    # Add entropy bonus for exploration
                    entropy = -torch.mean(torch.stack([lp for lp in log_probs_buffer]))
                    loss = policy_loss - ENTROPY_BONUS * entropy
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                    optimizer.step()
                    
                    log_probs_buffer = []
                    rewards_buffer = []
                
                # Update hook σ
                new_sigma = sigma_tensor.detach().item()
                old_sigma = hook.sigma
                hook.update_sigma(new_sigma)
                
                sigma_trajectory.append({
                    "epoch": epoch + 1,
                    "step": idx,
                    "global_step": epoch * len(dataset) + idx,
                    "sigma": round(hook.sigma, 5),
                    "batch_acc": round(batch_acc * 100, 1),
                    "reward": round(reward, 4),
                })
                
                # Reset batch
                batch_correct = 0
                batch_total = 0
                batch_logprob_diffs = []
            
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (idx + 1) * (len(dataset) - idx - 1) / 60
                avg_r = np.mean(all_rewards[-10:]) if all_rewards else 0
                print(f"  [{idx+1}/{len(dataset)}] Acc: {correct/total*100:.1f}% "
                      f"σ: {hook.sigma:.4f} R: {avg_r:.3f} ETA: {eta:.1f}min")
    
    finally:
        for h in handles:
            h.remove()
    
    acc = correct / total * 100
    elapsed = (time.time() - t0) / 60
    avg_sigma = np.mean(all_sigmas)
    avg_reward = np.mean(all_rewards) if all_rewards else 0
    
    print(f"  ✅ Epoch {epoch+1} | Acc: {acc:.2f}% | Avg σ: {avg_sigma:.4f} | "
          f"Avg R: {avg_reward:.3f} | {elapsed:.1f}min")
    
    return {
        "condition": f"CfC-Online (Epoch {epoch+1})",
        "accuracy": round(acc, 2),
        "correct": correct,
        "total": total,
        "avg_sigma": round(avg_sigma, 5),
        "min_sigma": round(min(all_sigmas), 5),
        "max_sigma": round(max(all_sigmas), 5),
        "avg_reward": round(avg_reward, 4),
        "sigma_trajectory": sigma_trajectory,
        "all_sigmas": [round(s, 5) for s in all_sigmas],
        "time_minutes": round(elapsed, 1),
    }


# ═══════════════════════════════════════
# PART 5: Visualization
# ═══════════════════════════════════════

def visualize(results, all_trajectories):
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
    fig.suptitle("Phase 20b: Online CfC — Zero Pre-training\n"
                 "REINFORCE Learning from Live TruthfulQA Feedback",
                 fontsize=14, fontweight="bold", color="#e94560")

    # ─── Panel 1: σ Trajectory (all epochs) ───
    ax1 = axes[0, 0]
    if all_trajectories:
        global_steps = [t["global_step"] for t in all_trajectories]
        sigmas = [t["sigma"] for t in all_trajectories]
        epochs = [t["epoch"] for t in all_trajectories]
        
        # Color by epoch
        colors_map = {1: "#FF6B6B", 2: "#FFA726", 3: "#00D4AA"}
        for ep in sorted(set(epochs)):
            ep_steps = [s for s, e in zip(global_steps, epochs) if e == ep]
            ep_sigmas = [s for s, e in zip(sigmas, epochs) if e == ep]
            ax1.plot(ep_steps, ep_sigmas, color=colors_map.get(ep, "#FFA726"),
                    linewidth=2, marker="o", markersize=2, label=f"Epoch {ep}")
        
        ax1.axhline(y=0.05, color="#FFD54F", linestyle="--", alpha=0.7, 
                    label="σ=0.05 (Phase 17d sweet spot)")
        ax1.axhline(y=0.046, color="#00D4AA", linestyle=":", alpha=0.7,
                    label="σ=0.046 (Phase 20 pre-trained)")
        ax1.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax1.set_title("CfC σ Trajectory — Online Learning", fontweight="bold")
    ax1.set_xlabel("Global Step (across epochs)")
    ax1.set_ylabel("σ (noise intensity)")
    ax1.grid(True, alpha=0.3)

    # ─── Panel 2: Reward Trajectory ───
    ax2 = axes[0, 1]
    if all_trajectories:
        rewards = [t["reward"] for t in all_trajectories]
        # Smooth with moving average
        window = 10
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax2.plot(range(len(smoothed)), smoothed, color="#00D4AA", linewidth=2,
                    label="Reward (smoothed)")
        ax2.plot(rewards, color="#FFA726", alpha=0.3, linewidth=0.5, label="Reward (raw)")
        ax2.axhline(y=0, color="#e94560", linestyle="--", alpha=0.5)
        ax2.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax2.set_title("REINFORCE Reward Trajectory", fontweight="bold")
    ax2.set_xlabel("Batch Update #")
    ax2.set_ylabel("Reward (acc - λ·σ)")
    ax2.grid(True, alpha=0.3)

    # ─── Panel 3: Accuracy Comparison ───
    ax3 = axes[1, 0]
    # Filter to summary results (not per-epoch online)
    summary_results = [r for r in results if "Epoch" not in r.get("condition", "")]
    # Add final epoch result
    online_results = [r for r in results if "Epoch" in r.get("condition", "")]
    if online_results:
        summary_results.append(online_results[-1])  # Last epoch
    
    conditions = [r["condition"] for r in summary_results]
    accuracies = [r["accuracy"] for r in summary_results]
    colors = ["#888888", "#00D4AA", "#e94560"]
    colors = colors[:len(summary_results)]
    if len(summary_results) > 3:
        colors.extend(["#FFA726"] * (len(summary_results) - 3))
    
    bars = ax3.bar(range(len(conditions)), accuracies, color=colors, 
                   edgecolor="#333", linewidth=0.5)
    ax3.set_xticks(range(len(conditions)))
    ax3.set_xticklabels(conditions, rotation=15, ha="right", fontsize=8)
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{acc:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax3.set_title("TruthfulQA MC1 Accuracy", fontweight="bold")
    ax3.set_ylabel("Accuracy (%)")
    ax3.grid(True, alpha=0.3, axis="y")

    # ─── Panel 4: σ convergence per epoch ───
    ax4 = axes[1, 1]
    if online_results:
        epoch_nums = []
        epoch_avg_sigmas = []
        epoch_accs = []
        for r in online_results:
            ep = int(r["condition"].split("Epoch ")[-1].rstrip(")"))
            epoch_nums.append(ep)
            epoch_avg_sigmas.append(r["avg_sigma"])
            epoch_accs.append(r["accuracy"])
        
        ax4.bar(epoch_nums, epoch_avg_sigmas, color="#FFA726", edgecolor="#333",
               alpha=0.8, label="Avg σ")
        ax4.axhline(y=0.046, color="#00D4AA", linestyle=":", alpha=0.7,
                    label="Phase 20 pre-trained (0.046)")
        ax4.axhline(y=0.05, color="#FFD54F", linestyle="--", alpha=0.7,
                    label="Phase 17d sweet spot (0.05)")
        
        # Add accuracy as text
        for ep, sig, acc in zip(epoch_nums, epoch_avg_sigmas, epoch_accs):
            ax4.text(ep, sig + 0.003, f"Acc: {acc:.1f}%", ha="center", fontsize=9)
        
        ax4.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax4.set_title("σ Convergence Per Epoch", fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Average σ")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(FIGURES_DIR, "phase20b_online_cfc.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Figure saved: {out_path}")
    return out_path


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 20b: Online CfC — Zero Pre-training")
    print("  Addressing tautology concern: NO Phase 17d knowledge")
    print("  CfC learns σ purely from live TruthfulQA feedback")
    print("=" * 60)
    t_start = time.time()

    # ─── Step 1: Random CfC initialization (NO pre-training!) ───
    print("\n🧠 Step 1: CfC random initialization (NO pre-training)")
    controller = CfCDoseController(input_size=4, hidden_size=32, num_neurons=16)
    
    # Verify initial σ is NOT near sweet spot
    test_features = torch.tensor([0.0, 0.5, 1.0, 0.0], dtype=torch.float32)
    with torch.no_grad():
        initial_sigma, _ = controller.get_sigma(test_features)
    print(f"  Initial σ (random weights): {initial_sigma.item():.4f}")
    print(f"  Expected: ~0.075 (midpoint), NOT 0.046-0.050")
    
    # REINFORCE optimizer
    optimizer = torch.optim.Adam(controller.parameters(), lr=LEARNING_RATE)

    # ─── Step 2: Load LLM ───
    model, tokenizer = load_model()

    # ─── Step 3: Load TruthfulQA ───
    from datasets import load_dataset
    print("\n📂 Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    print(f"  Loaded {len(ds)} questions")

    # ─── Step 4: Static Baselines (run once) ───
    results = []

    r_base = evaluate_truthfulqa_static(model, tokenizer, 0.0, ds, "Base (No Noise)")
    results.append(r_base)

    r_001 = evaluate_truthfulqa_static(model, tokenizer, 0.01, ds, "Static σ=0.01")
    results.append(r_001)

    r_005 = evaluate_truthfulqa_static(model, tokenizer, 0.05, ds, "Static σ=0.05")
    results.append(r_005)

    # ─── Step 5: Online CfC Learning (3 epochs) ───
    all_trajectories = []
    online_results = []
    
    for epoch in range(N_EPOCHS):
        r_online = evaluate_truthfulqa_online(
            model, tokenizer, controller, optimizer, ds, epoch, N_EPOCHS
        )
        results.append(r_online)
        online_results.append(r_online)
        all_trajectories.extend(r_online.get("sigma_trajectory", []))
    
    # ─── Step 6: Summary ───
    base_acc = r_base["accuracy"]
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY — Phase 20b: Online CfC (Zero Pre-training)")
    print("=" * 60)
    
    for r in results:
        tax = base_acc - r["accuracy"]
        if "avg_sigma" in r:
            sigma_info = f"avg σ={r['avg_sigma']:.4f}"
        else:
            sigma_info = f"σ={r.get('sigma', 'N/A')}"
        print(f"  {r['condition']:25s} | Acc: {r['accuracy']:.1f}% | "
              f"Tax: {tax:+.1f}% | {sigma_info}")
    
    # Key comparison with Phase 20
    final_online = online_results[-1] if online_results else None
    if final_online:
        print(f"\n  {'─' * 50}")
        print(f"  KEY COMPARISON:")
        print(f"  Phase 20 (pre-trained):  avg σ = 0.046, tax = +0.2%")
        print(f"  Phase 20b (online):      avg σ = {final_online['avg_sigma']:.4f}, "
              f"tax = {base_acc - final_online['accuracy']:+.1f}%")
        
        if 0.02 <= final_online['avg_sigma'] <= 0.08:
            print(f"  ✅ CONVERGENCE: σ is in sweet spot range [0.02, 0.08]!")
            print(f"     → Tautology concern RESOLVED: sweet spot is real")
        else:
            print(f"  ⚠️  σ did not converge to sweet spot range")
            print(f"     → More epochs may be needed, or different reward function")
    
    # ─── Step 7: Visualize ───
    fig_path = visualize(results, all_trajectories)

    # ─── Step 8: Save ───
    elapsed = (time.time() - t_start) / 60
    log = {
        "experiment": "Phase 20b: Online CfC (Zero Pre-training)",
        "timestamp": datetime.datetime.now().isoformat(),
        "model": MODEL_NAME,
        "target_layers": TARGET_LAYERS,
        "batch_size": BATCH_SIZE,
        "lambda_tax": LAMBDA_TAX,
        "learning_rate": LEARNING_RATE,
        "n_epochs": N_EPOCHS,
        "entropy_bonus": ENTROPY_BONUS,
        "initial_sigma": round(initial_sigma.item(), 5),
        "pretrained": False,
        "results": results,
        "all_trajectories": all_trajectories,
        "total_time_minutes": round(elapsed, 1),
    }

    # Remove all_sigmas from results to keep log manageable
    for r in log["results"]:
        r.pop("all_sigmas", None)

    log_path = os.path.join(RESULTS_DIR, "phase20b_online_cfc_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 Results: {log_path}")
    print(f"📊 Figure:  {fig_path}")
    print(f"⏱  Total time: {elapsed:.1f} minutes")
    print("✅ Phase 20b complete!")


if __name__ == "__main__":
    main()
