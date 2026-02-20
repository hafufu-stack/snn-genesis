"""
Phase 20c: CfC with Exploration Bonus
======================================

Fixes the σ overshooting from Phase 20b by adding an exploration bonus
that rewards the CfC for operating near the "edge of chaos" sweet spot.

Phase 20b problem:
    reward = batch_acc - λ * σ
    → σ monotonically decreases toward 0 (Epoch 3: σ=0.018, too low)

Phase 20c solution:
    reward = batch_acc - λ * σ + β * exploration_bonus(σ)
    exploration_bonus = exp(-((σ - σ_target)² / (2 * width²)))
    → Gaussian bonus centered at σ≈0.045 (discovered sweet spot)
    → CfC should "hover" near optimal σ instead of overshooting

Usage:
    python experiments/phase20c_exploration_bonus.py
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
TARGET_LAYERS = list(range(15, 21))
BATCH_SIZE = 10
SIGMA_MIN = 0.001
SIGMA_MAX = 0.15
N_EPOCHS = 4  # Extra epoch to show stability
LEARNING_RATE = 0.003
ENTROPY_BONUS = 0.01
LAMBDA_TAX = 2.0

# ★ Phase 20c: Exploration bonus parameters
EXPLORATION_BETA = 0.5     # Bonus strength
EXPLORATION_TARGET = 0.045  # Target σ (Phase 20b sweet spot)
EXPLORATION_WIDTH = 0.02    # Gaussian width

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════
# CfC Controller (same architecture as 20b)
# ═══════════════════════════════════════

class CfCDoseController(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_neurons=16):
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
# LLM & Hook (same as 20b)
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


# ═══════════════════════════════════════
# ★ PHASE 20c: Enhanced Reward with Exploration Bonus
# ═══════════════════════════════════════

def exploration_bonus(sigma_value):
    """
    Gaussian bonus that rewards σ near the sweet spot.
    
    exp(-((σ - target)² / (2 * width²)))
    
    At σ=target: bonus = 1.0 (maximum)
    At σ=0.0 or σ>>target: bonus → 0 (no reward)
    """
    return np.exp(-((sigma_value - EXPLORATION_TARGET) ** 2) / (2 * EXPLORATION_WIDTH ** 2))


def compute_reward_20b(batch_acc, sigma_value):
    """Phase 20b vanilla reward (for comparison)."""
    return batch_acc - LAMBDA_TAX * sigma_value


def compute_reward_20c(batch_acc, sigma_value):
    """Phase 20c: vanilla + exploration bonus."""
    vanilla = batch_acc - LAMBDA_TAX * sigma_value
    bonus = EXPLORATION_BETA * exploration_bonus(sigma_value)
    return vanilla + bonus


# ═══════════════════════════════════════
# Online REINFORCE Learning (with exploration bonus)
# ═══════════════════════════════════════

def evaluate_online(model, tokenizer, controller, optimizer, dataset,
                    epoch, n_epochs, reward_fn, label=""):
    print(f"\n🧠 Epoch {epoch+1}/{n_epochs}: {label}")
    
    layers = get_layers(model)
    hook = AdaptiveSNNHook(sigma=0.05)
    
    handles = []
    for layer_idx in TARGET_LAYERS:
        if layer_idx < len(layers):
            handles.append(layers[layer_idx].register_forward_pre_hook(hook))
    
    controller.train()
    hx = None
    
    correct = total = 0
    sigma_trajectory = []
    all_sigmas = []
    all_rewards = []
    log_probs_buffer = []
    rewards_buffer = []
    t0 = time.time()
    
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
            
            logprobs = [compute_completion_logprob(model, tokenizer, prompt, " " + c) for c in choice_texts]
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
            
            if batch_total >= BATCH_SIZE:
                batch_acc = batch_correct / batch_total
                avg_logprob_diff = np.mean(batch_logprob_diffs)
                hidden_norm = hook.last_hidden_norm
                step_frac = idx / len(dataset)
                
                features = torch.tensor(
                    [avg_logprob_diff, 1.0 - batch_acc, hidden_norm, step_frac],
                    dtype=torch.float32
                )
                
                sigma_tensor, hx_new = controller.get_sigma(features, hx)
                hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()
                
                # ★ Use the specified reward function (20b or 20c)
                reward = reward_fn(batch_acc, sigma_tensor.detach().item())
                
                log_prob = torch.log(sigma_tensor + 1e-8)
                log_probs_buffer.append(log_prob)
                rewards_buffer.append(reward)
                all_rewards.append(reward)
                
                if len(log_probs_buffer) >= 5:
                    rewards_t = torch.tensor(rewards_buffer, dtype=torch.float32)
                    if rewards_t.std() > 0:
                        rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
                    
                    policy_loss = 0
                    for lp, r in zip(log_probs_buffer, rewards_t):
                        policy_loss -= lp * r.item()
                    
                    entropy = -torch.mean(torch.stack([lp for lp in log_probs_buffer]))
                    loss = policy_loss - ENTROPY_BONUS * entropy
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                    optimizer.step()
                    
                    log_probs_buffer = []
                    rewards_buffer = []
                
                new_sigma = sigma_tensor.detach().item()
                hook.update_sigma(new_sigma)
                
                sigma_trajectory.append({
                    "epoch": epoch + 1,
                    "step": idx,
                    "global_step": epoch * len(dataset) + idx,
                    "sigma": round(hook.sigma, 5),
                    "batch_acc": round(batch_acc * 100, 1),
                    "reward": round(reward, 4),
                    "exploration_bonus": round(exploration_bonus(hook.sigma), 4),
                })
                
                batch_correct = batch_total = 0
                batch_logprob_diffs = []
            
            if (idx + 1) % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (idx + 1) * (len(dataset) - idx - 1) / 60
                avg_r = np.mean(all_rewards[-10:]) if all_rewards else 0
                eb = exploration_bonus(hook.sigma)
                print(f"  [{idx+1}/{len(dataset)}] Acc: {correct/total*100:.1f}% "
                      f"σ: {hook.sigma:.4f} R: {avg_r:.3f} EB: {eb:.3f} ETA: {eta:.1f}min")
    
    finally:
        for h in handles:
            h.remove()
    
    acc = correct / total * 100
    elapsed = (time.time() - t0) / 60
    avg_sigma = np.mean(all_sigmas)
    end_sigma = all_sigmas[-1] if all_sigmas else 0
    avg_reward = np.mean(all_rewards) if all_rewards else 0
    
    print(f"  ✅ Epoch {epoch+1} | Acc: {acc:.2f}% | Avg σ: {avg_sigma:.4f} | "
          f"End σ: {end_sigma:.4f} | R: {avg_reward:.3f} | {elapsed:.1f}min")
    
    return {
        "condition": f"{label} (Epoch {epoch+1})",
        "accuracy": round(acc, 2),
        "correct": correct,
        "total": total,
        "avg_sigma": round(avg_sigma, 5),
        "end_sigma": round(end_sigma, 5),
        "min_sigma": round(min(all_sigmas), 5),
        "max_sigma": round(max(all_sigmas), 5),
        "avg_reward": round(avg_reward, 4),
        "sigma_trajectory": sigma_trajectory,
        "time_minutes": round(elapsed, 1),
    }


# ═══════════════════════════════════════
# Visualization
# ═══════════════════════════════════════

def visualize(results_20b, results_20c, traj_20b, traj_20c):
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
    fig.suptitle("Phase 20c: CfC Exploration Bonus\n"
                 "Preventing σ Overshooting at Edge of Chaos",
                 fontsize=14, fontweight="bold", color="#e94560")

    # Panel 1: σ Trajectory comparison
    ax1 = axes[0, 0]
    if traj_20b:
        steps = [t["global_step"] for t in traj_20b]
        sigmas = [t["sigma"] for t in traj_20b]
        ax1.plot(steps, sigmas, color="#FFA726", linewidth=1.5, alpha=0.7, label="20b (vanilla)")
    if traj_20c:
        steps = [t["global_step"] for t in traj_20c]
        sigmas = [t["sigma"] for t in traj_20c]
        ax1.plot(steps, sigmas, color="#00D4AA", linewidth=2, label="20c (exploration bonus)")
    
    ax1.axhline(y=0.045, color="#FFD54F", linestyle="--", alpha=0.8, label="Sweet spot (0.045)")
    ax1.axhspan(0.03, 0.06, alpha=0.1, color="#00D4AA", label="Optimal range")
    ax1.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax1.set_title("σ Trajectory: Vanilla vs Exploration Bonus", fontweight="bold")
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("σ")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Exploration bonus curve
    ax2 = axes[0, 1]
    sigmas_range = np.linspace(0, 0.15, 200)
    bonus_curve = [exploration_bonus(s) for s in sigmas_range]
    ax2.plot(sigmas_range, bonus_curve, color="#00D4AA", linewidth=2)
    ax2.axvline(x=EXPLORATION_TARGET, color="#FFD54F", linestyle="--", 
                label=f"Target σ={EXPLORATION_TARGET}")
    ax2.fill_between(sigmas_range, bonus_curve, alpha=0.2, color="#00D4AA")
    ax2.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax2.set_title("Exploration Bonus Function", fontweight="bold")
    ax2.set_xlabel("σ")
    ax2.set_ylabel("Bonus (β × Gaussian)")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Accuracy per epoch
    ax3 = axes[1, 0]
    epochs_20b = [int(r["condition"].split("Epoch ")[-1].rstrip(")")) for r in results_20b]
    acc_20b = [r["accuracy"] for r in results_20b]
    epochs_20c = [int(r["condition"].split("Epoch ")[-1].rstrip(")")) for r in results_20c]
    acc_20c = [r["accuracy"] for r in results_20c]
    
    x_b = [e - 0.15 for e in epochs_20b]
    x_c = [e + 0.15 for e in epochs_20c]
    ax3.bar(x_b, acc_20b, width=0.3, color="#FFA726", label="20b vanilla", edgecolor="#333")
    ax3.bar(x_c, acc_20c, width=0.3, color="#00D4AA", label="20c exploration", edgecolor="#333")
    for x, a in zip(x_b, acc_20b):
        ax3.text(x, a + 0.2, f"{a:.1f}%", ha="center", fontsize=8)
    for x, a in zip(x_c, acc_20c):
        ax3.text(x, a + 0.2, f"{a:.1f}%", ha="center", fontsize=8)
    ax3.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax3.set_title("Accuracy Per Epoch", fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("TruthfulQA MC1 Accuracy (%)")
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: End σ per epoch
    ax4 = axes[1, 1]
    end_20b = [r["end_sigma"] for r in results_20b]
    end_20c = [r["end_sigma"] for r in results_20c]
    ax4.bar(x_b, end_20b, width=0.3, color="#FFA726", label="20b vanilla", edgecolor="#333")
    ax4.bar(x_c, end_20c, width=0.3, color="#00D4AA", label="20c exploration", edgecolor="#333")
    ax4.axhline(y=0.045, color="#FFD54F", linestyle="--", alpha=0.8, label="Sweet spot")
    ax4.axhspan(0.03, 0.06, alpha=0.1, color="#00D4AA")
    for x, s in zip(x_b, end_20b):
        ax4.text(x, s + 0.002, f"{s:.3f}", ha="center", fontsize=8)
    for x, s in zip(x_c, end_20c):
        ax4.text(x, s + 0.002, f"{s:.3f}", ha="center", fontsize=8)
    ax4.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax4.set_title("End σ Per Epoch (Overshooting Test)", fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("End σ")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(FIGURES_DIR, "phase20c_exploration_bonus.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Figure saved: {out_path}")
    return out_path


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 20c: CfC Exploration Bonus")
    print("  Fixing σ overshooting with Gaussian sweet-spot bonus")
    print(f"  Target σ: {EXPLORATION_TARGET}, Width: {EXPLORATION_WIDTH}, β: {EXPLORATION_BETA}")
    print("=" * 60)
    t_start = time.time()
    
    model, tokenizer = load_model()
    
    from datasets import load_dataset
    print("\n📂 Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    print(f"  Loaded {len(ds)} questions")
    
    # ═══ Condition A: Vanilla CfC (Phase 20b baseline) ═══
    print(f"\n{'═' * 50}")
    print(f"  CONDITION A: Vanilla CfC (20b reward)")
    print(f"{'═' * 50}")
    
    ctrl_a = CfCDoseController()
    opt_a = torch.optim.Adam(ctrl_a.parameters(), lr=LEARNING_RATE)
    
    with torch.no_grad():
        init_a, _ = ctrl_a.get_sigma(torch.tensor([0.0, 0.5, 1.0, 0.0]))
    print(f"  Initial σ: {init_a.item():.4f}")
    
    results_a = []
    traj_a = []
    for epoch in range(N_EPOCHS):
        r = evaluate_online(model, tokenizer, ctrl_a, opt_a, ds,
                           epoch, N_EPOCHS, compute_reward_20b, "Vanilla")
        results_a.append(r)
        traj_a.extend(r.get("sigma_trajectory", []))
    
    # ═══ Condition B: Exploration Bonus CfC (Phase 20c) ═══
    print(f"\n{'═' * 50}")
    print(f"  CONDITION B: Exploration Bonus CfC (20c reward)")
    print(f"{'═' * 50}")
    
    ctrl_b = CfCDoseController()
    opt_b = torch.optim.Adam(ctrl_b.parameters(), lr=LEARNING_RATE)
    
    with torch.no_grad():
        init_b, _ = ctrl_b.get_sigma(torch.tensor([0.0, 0.5, 1.0, 0.0]))
    print(f"  Initial σ: {init_b.item():.4f}")
    
    results_b = []
    traj_b = []
    for epoch in range(N_EPOCHS):
        r = evaluate_online(model, tokenizer, ctrl_b, opt_b, ds,
                           epoch, N_EPOCHS, compute_reward_20c, "Exploration")
        results_b.append(r)
        traj_b.extend(r.get("sigma_trajectory", []))
    
    # ═══ Summary ═══
    elapsed = (time.time() - t_start) / 60
    
    print(f"\n{'═' * 60}")
    print(f"  RESULTS — Phase 20c: Exploration Bonus Comparison")
    print(f"{'═' * 60}")
    print(f"\n  {'Condition':<28} | {'Acc':>6} | {'Avg σ':>7} | {'End σ':>7} | {'In Range?':>9}")
    print(f"  {'─'*28}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*9}")
    
    for r_a, r_b in zip(results_a, results_b):
        in_a = "✅" if 0.03 <= r_a["end_sigma"] <= 0.06 else "❌"
        in_b = "✅" if 0.03 <= r_b["end_sigma"] <= 0.06 else "❌"
        print(f"  {r_a['condition']:<28} | {r_a['accuracy']:>5.1f}% | {r_a['avg_sigma']:>7.4f} | {r_a['end_sigma']:>7.4f} | {in_a:>9}")
        print(f"  {r_b['condition']:<28} | {r_b['accuracy']:>5.1f}% | {r_b['avg_sigma']:>7.4f} | {r_b['end_sigma']:>7.4f} | {in_b:>9}")
        print(f"  {'─'*28}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*9}")
    
    # Key hypothesis test: does exploration bonus prevent overshooting?
    vanilla_overshoot = results_a[-1]["end_sigma"] < 0.03
    bonus_stable = 0.03 <= results_b[-1]["end_sigma"] <= 0.06
    
    print(f"\n  KEY HYPOTHESIS TEST:")
    if vanilla_overshoot:
        print(f"  ⚠️  Vanilla: σ overshoots to {results_a[-1]['end_sigma']:.4f} (< 0.03)")
    else:
        print(f"  📊 Vanilla: σ ends at {results_a[-1]['end_sigma']:.4f}")
    
    if bonus_stable:
        print(f"  ✅ Exploration bonus: σ stable at {results_b[-1]['end_sigma']:.4f} (in [0.03, 0.06])")
        print(f"     → Overshooting FIXED!")
    else:
        print(f"  ⚠️  Exploration bonus: σ at {results_b[-1]['end_sigma']:.4f}")
    
    # Visualize
    fig_path = visualize(results_a, results_b, traj_a, traj_b)
    
    # Save
    log = {
        "experiment": "Phase 20c: Exploration Bonus",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "exploration_beta": EXPLORATION_BETA,
            "exploration_target": EXPLORATION_TARGET,
            "exploration_width": EXPLORATION_WIDTH,
            "lambda_tax": LAMBDA_TAX,
            "learning_rate": LEARNING_RATE,
            "n_epochs": N_EPOCHS,
        },
        "vanilla_results": results_a,
        "exploration_results": results_b,
        "vanilla_trajectories": traj_a,
        "exploration_trajectories": traj_b,
        "elapsed_min": round(elapsed, 1),
    }
    
    log_path = os.path.join(RESULTS_DIR, "phase20c_exploration_bonus_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Results: {log_path}")
    print(f"📊 Figure:  {fig_path}")
    print(f"⏱  Total: {elapsed:.1f} min")
    print(f"\n✅ Phase 20c complete!")


if __name__ == "__main__":
    main()
