"""
Phase 27: PPO Dual-Mode Brain — Stable Homeostatic Control
============================================================
Replaces REINFORCE with PPO to solve the σ drift/collapse problem.
Changes from Phase 24:
  1. Added CfCCritic (value network) for advantage estimation
  2. GAE (Generalized Advantage Estimation) replaces raw rewards
  3. Clipped surrogate objective replaces vanilla policy gradient
  4. Multiple PPO update epochs per rollout buffer

Usage:
    python experiments/phase27_ppo_dual_mode.py
"""

import torch
import torch.nn as nn
import os, sys, json, gc, time, datetime, random, math
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
LEARNING_RATE = 3e-4       # PPO uses smaller LR than REINFORCE
LAMBDA_QUAD = 200.0
SEED = 2026

# PPO hyperparameters
PPO_CLIP_EPS = 0.2
PPO_UPDATE_EPOCHS = 4      # Mini-epochs per rollout
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_ENTROPY_COEFF = 0.01
PPO_VALUE_COEFF = 0.5
PPO_ROLLOUT_SIZE = 10      # Steps before PPO update
PPO_MAX_GRAD_NORM = 0.5

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

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
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
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


# ═══════════════════════════════════════
# Task Classifier
# ═══════════════════════════════════════

class TaskClassifier(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())
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
    return torch.tensor([
        h.norm().item() / max(h.numel(), 1), h.std().item(),
        min(len(words) / 50.0, 1.0), float("?" in text),
        len(set(words)) / max(len(words), 1),
        sum(len(w) for w in words) / max(len(words), 1) / 10.0,
        min(entropy / 10.0, 1.0), h.max().item() / 100.0,
    ], dtype=torch.float32)

def pretrain_classifier(classifier, model, tokenizer, hook, factual_prompts, creative_prompts, epochs=30):
    print("\n🎓 Pre-training Task Classifier...")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    all_features, all_labels = [], []
    for p in factual_prompts:
        all_features.append(extract_prompt_features(model, tokenizer, p, hook)); all_labels.append(0.0)
    for p in creative_prompts:
        all_features.append(extract_prompt_features(model, tokenizer, p, hook)); all_labels.append(1.0)
    X = torch.stack(all_features); Y = torch.tensor(all_labels).unsqueeze(1)
    for epoch in range(epochs):
        pred = classifier(X); loss = loss_fn(pred, Y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch + 1) % 10 == 0:
            acc = ((pred > 0.5).float() == Y).float().mean().item()
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f} acc={acc*100:.1f}%")
    with torch.no_grad():
        acc = ((classifier(X) > 0.5).float() == Y).float().mean().item()
    print(f"  ✅ Classifier trained: acc={acc*100:.1f}%")
    return classifier


# ═══════════════════════════════════════
# PPO Actor-Critic Architecture
# ═══════════════════════════════════════

class DualModeCfCActor(nn.Module):
    """Policy network: CfC that outputs σ as a Gaussian distribution."""
    def __init__(self, input_size=5, num_neurons=16):
        super().__init__()
        wiring = AutoNCP(num_neurons, 1)
        self.cfc = CfC(input_size, wiring, batch_first=True)
        self.log_std = nn.Parameter(torch.tensor([-1.0]))  # Learnable log-std

    def forward(self, x, hx=None):
        output, hx = self.cfc(x, hx=hx)
        mu = torch.sigmoid(output) * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, hx

    def get_sigma_and_logprob(self, features, hx=None):
        """Get sigma, log_prob, entropy, and new hidden state."""
        x = features.unsqueeze(0).unsqueeze(0)
        mu, std, hx = self.forward(x, hx=hx)
        mu = mu.squeeze(); std = std.squeeze()
        dist = torch.distributions.Normal(mu, std)
        raw_sigma = dist.rsample()
        sigma = torch.clamp(raw_sigma, SIGMA_MIN, SIGMA_MAX)
        log_prob = dist.log_prob(raw_sigma)
        entropy = dist.entropy()
        return sigma, log_prob, entropy, hx

    def evaluate_action(self, features, old_sigma):
        """Re-evaluate log_prob of a previously taken action (for PPO ratio)."""
        x = features.unsqueeze(0).unsqueeze(0)
        mu, std, _ = self.forward(x)
        mu = mu.squeeze(); std = std.squeeze()
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(old_sigma)
        entropy = dist.entropy()
        return log_prob, entropy


class CfCCritic(nn.Module):
    """Value network: MLP that estimates V(s) from same features as actor."""
    def __init__(self, input_size=5, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)


# ═══════════════════════════════════════
# Generation & Evaluation helpers
# ═══════════════════════════════════════

def compute_completion_logprob(model, tokenizer, context, completion):
    full_text = context + completion
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    total_lp, n = 0.0, 0
    for i in range(len(ctx_ids), len(full_ids)):
        total_lp += log_probs[i - 1, full_ids[i]].item(); n += 1
    return total_lp / max(n, 1)

def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.9, top_p=0.95, top_k=50,
                                 repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()

def compute_novelty(text):
    words = text.lower().split()
    if len(words) < 3: return 0.0
    ur = len(set(words)) / len(words)
    bg = list(zip(words[:-1], words[1:]))
    return (ur + len(set(bg)) / max(len(bg), 1)) / 2.0

def is_grammatical(text):
    if len(text) < 20: return False
    words = text.split()
    if len(words) < 5: return False
    if len(set(words)) / len(words) < 0.2: return False
    return any(c in text for c in '.!?')

def build_mixed_dataset(tokenizer):
    from datasets import load_dataset
    print("\n📂 Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), 20)
    items = []
    for idx in indices:
        row = ds[idx]
        items.append({"type": "factual", "prompt": f"Q: {row['question']}\nA:",
                      "question": row["question"], "mc1_targets": row["mc1_targets"]})
    for prompt in CREATIVE_PROMPTS:
        items.append({"type": "creative", "prompt": f"Creative writing prompt: {prompt}\n\nResponse:",
                      "question": prompt, "mc1_targets": None})
    random.seed(SEED + 1); random.shuffle(items)
    nf = sum(1 for x in items if x["type"] == "factual")
    nc = sum(1 for x in items if x["type"] == "creative")
    print(f"  Mixed dataset: {nf} factual + {nc} creative = {len(items)} total")
    return items


# ═══════════════════════════════════════
# Static Evaluation (baseline)
# ═══════════════════════════════════════

def evaluate_static(model, tokenizer, dataset, sigma_fixed, label=""):
    print(f"\n{'═' * 50}\n  {label}: σ={sigma_fixed:.3f}\n{'═' * 50}")
    layers = get_layers(model)
    hook = AdaptiveSNNHook(sigma=sigma_fixed)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]
    fc, ft, cn, cg, ct = 0, 0, [], 0, 0
    try:
        for idx, item in enumerate(dataset):
            if item["type"] == "factual":
                ch = item["mc1_targets"]; ci = ch["labels"].index(1) if 1 in ch["labels"] else 0
                lps = [compute_completion_logprob(model, tokenizer, item["prompt"], " " + c) for c in ch["choices"]]
                if np.argmax(lps) == ci: fc += 1
                ft += 1
            else:
                resp = generate_text(model, tokenizer, item["prompt"])
                cn.append(compute_novelty(resp))
                if is_grammatical(resp): cg += 1
                ct += 1
            if (idx + 1) % 10 == 0: print(f"  [{idx+1}/{len(dataset)}] done")
    finally:
        for h in handles: h.remove()
    fa = fc / max(ft, 1) * 100; an = np.mean(cn) if cn else 0; gr = cg / max(ct, 1) * 100
    print(f"  ✅ Fact acc: {fa:.1f}% | Nov: {an:.3f} | Gram: {gr:.0f}%")
    return {"condition": label, "factual_acc": round(fa, 2), "factual_correct": fc,
            "factual_total": ft, "creative_novelty": round(float(an), 4),
            "creative_grammar_rate": round(gr, 1), "sigma": sigma_fixed}


# ═══════════════════════════════════════
# PPO Training Loop
# ═══════════════════════════════════════

def compute_gae(rewards, values, gamma=PPO_GAMMA, lam=PPO_GAE_LAMBDA):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:len(advantages)])]
    return advantages, returns


def evaluate_ppo_dual_mode(model, tokenizer, dataset, actor, critic, classifier, hook, label=""):
    """PPO-based Dual-Mode CfC evaluation."""
    print(f"\n{'═' * 60}")
    print(f"  {label}: PPO Dual-Mode CfC")
    print(f"{'═' * 60}")

    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]

    actor.train(); critic.train()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)
    hx = None

    fc, ft = 0, 0
    cn, cg, ct = [], 0, 0
    all_sf, all_sc = [], []
    sigma_trajectory = []

    # PPO rollout buffers
    buf_features = []
    buf_sigmas = []
    buf_log_probs = []
    buf_values = []
    buf_rewards = []
    buf_entropies = []

    total_policy_loss = 0
    total_value_loss = 0
    n_updates = 0

    try:
        for epoch in range(N_EPOCHS):
            print(f"\n  🧠 Epoch {epoch+1}/{N_EPOCHS}")

            for idx, item in enumerate(dataset):
                # Feature extraction & classification
                pf = extract_prompt_features(model, tokenizer, item["question"], hook)
                with torch.no_grad():
                    p_creative = classifier(pf.unsqueeze(0)).item()

                sigma_target = (1 - p_creative) * SIGMA_FACTUAL + p_creative * SIGMA_CREATIVE
                hidden_norm = hook.last_hidden_norm if hook.last_hidden_norm else 0.5
                step_frac = idx / len(dataset)

                features = torch.tensor(
                    [0.0, 0.5, hidden_norm, step_frac, p_creative], dtype=torch.float32)

                # Actor: sample σ from policy
                sigma, log_prob, entropy, hx_new = actor.get_sigma_and_logprob(features, hx)
                hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()

                current_sigma = sigma.detach().item()
                hook.update_sigma(current_sigma)

                # Critic: estimate value
                with torch.no_grad():
                    value = critic(features).item()

                # Evaluate task
                if item["type"] == "factual":
                    ch = item["mc1_targets"]; ci = ch["labels"].index(1) if 1 in ch["labels"] else 0
                    lps = [compute_completion_logprob(model, tokenizer, item["prompt"], " " + c)
                           for c in ch["choices"]]
                    is_correct = (np.argmax(lps) == ci)
                    if is_correct: fc += 1
                    ft += 1
                    all_sf.append(current_sigma)
                    task_reward = 1.0 if is_correct else 0.0
                    reward = task_reward - LAMBDA_QUAD * (current_sigma - sigma_target) ** 2
                    ld = max(lps) - min(lps)
                    features = torch.tensor(
                        [ld, 1.0 - float(is_correct), hidden_norm, step_frac, p_creative],
                        dtype=torch.float32)
                else:
                    resp = generate_text(model, tokenizer, item["prompt"])
                    nov = compute_novelty(resp); gram = is_grammatical(resp)
                    cn.append(nov)
                    if gram: cg += 1
                    ct += 1
                    all_sc.append(current_sigma)
                    task_reward = nov if gram else -0.5
                    reward = task_reward - LAMBDA_QUAD * (current_sigma - sigma_target) ** 2

                # Store in rollout buffer
                buf_features.append(features.detach())
                buf_sigmas.append(sigma.detach())
                buf_log_probs.append(log_prob.detach())
                buf_values.append(value)
                buf_rewards.append(reward)
                buf_entropies.append(entropy.detach())

                sigma_trajectory.append({
                    "epoch": epoch + 1, "step": epoch * len(dataset) + idx,
                    "sigma": round(current_sigma, 5), "sigma_target": round(sigma_target, 4),
                    "p_creative": round(p_creative, 3), "type": item["type"],
                })

                # ═══ PPO UPDATE ═══
                if len(buf_rewards) >= PPO_ROLLOUT_SIZE:
                    advantages, returns = compute_gae(buf_rewards, buf_values)
                    adv_t = torch.tensor(advantages, dtype=torch.float32)
                    ret_t = torch.tensor(returns, dtype=torch.float32)
                    old_log_probs = torch.stack(buf_log_probs)
                    old_sigmas = torch.stack(buf_sigmas)
                    feats_batch = torch.stack(buf_features)

                    # Normalize advantages
                    if adv_t.std() > 1e-8:
                        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

                    # Multiple PPO epochs
                    for _ in range(PPO_UPDATE_EPOCHS):
                        new_log_probs, new_entropies = [], []
                        new_values = []
                        for i in range(len(feats_batch)):
                            lp, ent = actor.evaluate_action(feats_batch[i], old_sigmas[i])
                            new_log_probs.append(lp)
                            new_entropies.append(ent)
                            new_values.append(critic(feats_batch[i]))

                        new_lp_t = torch.stack(new_log_probs)
                        new_ent_t = torch.stack(new_entropies)
                        new_val_t = torch.stack(new_values)

                        # Policy loss (clipped surrogate)
                        ratio = torch.exp(new_lp_t - old_log_probs)
                        surr1 = ratio * adv_t
                        surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * adv_t
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Value loss
                        value_loss = nn.functional.mse_loss(new_val_t.squeeze(), ret_t)

                        # Entropy bonus
                        entropy_loss = -new_ent_t.mean()

                        # Total loss
                        loss = (policy_loss
                                + PPO_VALUE_COEFF * value_loss
                                + PPO_ENTROPY_COEFF * entropy_loss)

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(actor.parameters()) + list(critic.parameters()),
                            PPO_MAX_GRAD_NORM)
                        optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    n_updates += 1

                    # Clear buffers
                    buf_features, buf_sigmas, buf_log_probs = [], [], []
                    buf_values, buf_rewards, buf_entropies = [], [], []

                if (idx + 1) % 10 == 0:
                    recent = sigma_trajectory[-10:]
                    rf = [s["sigma"] for s in recent if s["type"] == "factual"]
                    rc = [s["sigma"] for s in recent if s["type"] == "creative"]
                    af = np.mean(rf) if rf else 0; ac = np.mean(rc) if rc else 0
                    print(f"  [{idx+1}/{len(dataset)}] σ_f={af:.4f} σ_c={ac:.4f} "
                          f"sep={abs(ac-af):.4f} p_c={p_creative:.3f}")

            fa = fc / max(ft, 1) * 100
            an = np.mean(cn) if cn else 0
            print(f"  ✅ E{epoch+1}: Fact={fa:.1f}% Nov={an:.3f}")

    finally:
        for h in handles: h.remove()

    fa = fc / max(ft, 1) * 100
    an = np.mean(cn) if cn else 0
    gr = cg / max(ct, 1) * 100
    msf = np.mean(all_sf) if all_sf else 0
    msc = np.mean(all_sc) if all_sc else 0
    sep = abs(msc - msf)

    # σ stability: check drift by comparing first-half vs second-half σ
    mid = len(sigma_trajectory) // 2
    first_half_sigmas = [s["sigma"] for s in sigma_trajectory[:mid]]
    second_half_sigmas = [s["sigma"] for s in sigma_trajectory[mid:]]
    drift = abs(np.mean(second_half_sigmas) - np.mean(first_half_sigmas))

    avg_pl = total_policy_loss / max(n_updates, 1)
    avg_vl = total_value_loss / max(n_updates, 1)

    print(f"\n  📊 PPO DUAL-MODE RESULTS:")
    print(f"     Fact acc: {fa:.1f}% | Nov: {an:.3f} | Gram: {gr:.0f}%")
    print(f"     σ̄_factual={msf:.4f} | σ̄_creative={msc:.4f} | sep={sep:.4f}")
    print(f"     σ drift (1st→2nd half): {drift:.4f}")
    print(f"     PPO: policy_loss={avg_pl:.4f} value_loss={avg_vl:.4f}")
    print(f"     {'✅ STABLE' if drift < 0.01 else '⚠️ SOME DRIFT' if drift < 0.03 else '❌ DRIFT DETECTED'}")

    return {
        "condition": label, "factual_acc": round(fa, 2), "factual_correct": fc,
        "factual_total": ft, "creative_novelty": round(float(an), 4),
        "creative_grammar_rate": round(gr, 1),
        "mean_sigma_factual": round(msf, 5), "mean_sigma_creative": round(msc, 5),
        "sigma_separation": round(sep, 5), "sigma_drift": round(drift, 5),
        "all_sigmas_factual": [round(s, 5) for s in all_sf],
        "all_sigmas_creative": [round(s, 5) for s in all_sc],
        "sigma_trajectory": sigma_trajectory,
        "ppo_avg_policy_loss": round(avg_pl, 4),
        "ppo_avg_value_loss": round(avg_vl, 4),
        "ppo_n_updates": n_updates,
    }


# ═══════════════════════════════════════
# Visualization
# ═══════════════════════════════════════

def visualize(result_sf, result_sc, result_reinforce_ref, result_ppo):
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
    fig.suptitle("Phase 27: PPO Dual-Mode Brain — Stable Homeostatic Control\n"
                 "REINFORCE → PPO: Solving the σ drift problem",
                 fontsize=14, fontweight="bold", color="#e94560")

    # Panel 1: σ trajectory colored by task type
    ax1 = axes[0, 0]
    traj = result_ppo.get("sigma_trajectory", [])
    if traj:
        sf = [t["step"] for t in traj if t["type"] == "factual"]
        sigf = [t["sigma"] for t in traj if t["type"] == "factual"]
        sc = [t["step"] for t in traj if t["type"] == "creative"]
        sigc = [t["sigma"] for t in traj if t["type"] == "creative"]
        ax1.scatter(sf, sigf, c="#4FC3F7", s=20, alpha=0.7, label="Factual", zorder=3)
        ax1.scatter(sc, sigc, c="#FF7043", s=20, alpha=0.7, label="Creative", zorder=3)
    ax1.axhline(y=SIGMA_FACTUAL, color="#4FC3F7", linestyle="--", alpha=0.6, label=f"σ*={SIGMA_FACTUAL}")
    ax1.axhline(y=SIGMA_CREATIVE, color="#FF7043", linestyle="--", alpha=0.6, label=f"σ*={SIGMA_CREATIVE}")
    ax1.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax1.set_title("PPO σ Trajectory: Factual vs Creative", fontweight="bold")
    ax1.set_xlabel("Step"); ax1.set_ylabel("σ"); ax1.grid(True, alpha=0.3)

    # Panel 2: σ distribution
    ax2 = axes[0, 1]
    bins = np.linspace(0, 0.15, 30)
    if result_ppo["all_sigmas_factual"]:
        ax2.hist(result_ppo["all_sigmas_factual"], bins=bins, alpha=0.6, color="#4FC3F7",
                label=f"Factual (μ={result_ppo['mean_sigma_factual']:.4f})", edgecolor="#333")
    if result_ppo["all_sigmas_creative"]:
        ax2.hist(result_ppo["all_sigmas_creative"], bins=bins, alpha=0.6, color="#FF7043",
                label=f"Creative (μ={result_ppo['mean_sigma_creative']:.4f})", edgecolor="#333")
    ax2.axvline(x=SIGMA_FACTUAL, color="#4FC3F7", linestyle="--", alpha=0.8)
    ax2.axvline(x=SIGMA_CREATIVE, color="#FF7043", linestyle="--", alpha=0.8)
    ax2.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    drift_status = "✅ STABLE" if result_ppo["sigma_drift"] < 0.01 else "⚠️ DRIFT"
    ax2.set_title(f"σ Distribution | sep={result_ppo['sigma_separation']:.4f} | {drift_status}",
                  fontweight="bold")
    ax2.set_xlabel("σ"); ax2.set_ylabel("Count"); ax2.grid(True, alpha=0.3)

    # Panel 3: Performance comparison (4 conditions)
    ax3 = axes[1, 0]
    conds = ["Static\nFactual", "Static\nCreative", "REINFORCE\n(Phase 24)", "PPO\n(Phase 27)"]
    accs = [result_sf["factual_acc"], result_sc["factual_acc"],
            result_reinforce_ref.get("factual_acc", 0), result_ppo["factual_acc"]]
    colors_bars = ["#4FC3F7", "#FF7043", "#FFA726", "#66BB6A"]
    bars = ax3.bar(range(len(conds)), accs, color=colors_bars, edgecolor="#333")
    for b, v in zip(bars, accs):
        ax3.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax3.set_xticks(range(len(conds))); ax3.set_xticklabels(conds, fontsize=9)
    ax3.set_title("Factual Accuracy: All Conditions", fontweight="bold")
    ax3.set_ylabel("Accuracy (%)"); ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: σ drift comparison (REINFORCE vs PPO)
    ax4 = axes[1, 1]
    # PPO trajectory over time
    if traj:
        steps = [t["step"] for t in traj]
        sigmas_all = [t["sigma"] for t in traj]
        window = max(1, len(sigmas_all) // 10)
        rolling = [np.mean(sigmas_all[max(0,i-window):i+1]) for i in range(len(sigmas_all))]
        ax4.plot(steps, rolling, color="#66BB6A", linewidth=2, label="PPO (rolling avg)")
    ax4.axhline(y=(SIGMA_FACTUAL + SIGMA_CREATIVE)/2, color="#FFD54F", linestyle="--",
                alpha=0.8, label=f"Target midpoint={((SIGMA_FACTUAL+SIGMA_CREATIVE)/2):.3f}")
    ax4.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax4.set_title(f"σ Stability: drift={result_ppo['sigma_drift']:.4f}", fontweight="bold")
    ax4.set_xlabel("Step"); ax4.set_ylabel("σ (rolling avg)"); ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIGURES_DIR, "phase27_ppo_dual_mode.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  📊 Figure saved: {out}")
    return out


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 27: PPO Dual-Mode Brain — Stable Homeostatic Control")
    print("  REINFORCE → PPO: Solving the σ drift problem")
    print("=" * 60)
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model()
    hook = AdaptiveSNNHook(sigma=0.05)
    dataset = build_mixed_dataset(tokenizer)

    factual_prompts = [item["question"] for item in dataset if item["type"] == "factual"]
    classifier = TaskClassifier(input_dim=8, hidden_dim=32)
    classifier = pretrain_classifier(classifier, model, tokenizer, hook,
                                     factual_prompts, CREATIVE_PROMPTS)

    # Static baselines
    result_sf = evaluate_static(model, tokenizer, dataset, SIGMA_FACTUAL, "A: Static Factual")
    result_sc = evaluate_static(model, tokenizer, dataset, SIGMA_CREATIVE, "B: Static Creative")

    # Phase 24 REINFORCE reference (load from existing results if available)
    reinforce_ref = {"factual_acc": 0, "creative_novelty": 0}
    ref_path = os.path.join(RESULTS_DIR, "phase24_dual_mode_brain_log.json")
    if os.path.exists(ref_path):
        try:
            with open(ref_path) as f:
                ref_data = json.load(f)
            if "dual_mode" in ref_data.get("results", {}):
                reinforce_ref = ref_data["results"]["dual_mode"]
            elif "factual_acc" in ref_data:
                reinforce_ref = ref_data
            print(f"  📎 REINFORCE ref: acc={reinforce_ref.get('factual_acc', '?')}%")
        except Exception:
            pass

    # PPO Dual-Mode
    actor = DualModeCfCActor(input_size=5, num_neurons=16)
    critic = CfCCritic(input_size=5, hidden_size=64)
    result_ppo = evaluate_ppo_dual_mode(model, tokenizer, dataset, actor, critic,
                                         classifier, hook, "C: PPO Dual-Mode")

    # Visualize
    fig_path = visualize(result_sf, result_sc, reinforce_ref, result_ppo)

    # Save
    elapsed = time.time() - t_start
    result = {
        "phase": "Phase 27: PPO Dual-Mode Brain",
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "ppo_hyperparams": {
            "clip_eps": PPO_CLIP_EPS, "update_epochs": PPO_UPDATE_EPOCHS,
            "gamma": PPO_GAMMA, "gae_lambda": PPO_GAE_LAMBDA,
            "entropy_coeff": PPO_ENTROPY_COEFF, "value_coeff": PPO_VALUE_COEFF,
            "rollout_size": PPO_ROLLOUT_SIZE, "lr": LEARNING_RATE,
        },
        "results": {
            "static_factual": result_sf, "static_creative": result_sc,
            "reinforce_ref": reinforce_ref, "ppo_dual_mode": result_ppo,
        },
        "figure_path": fig_path,
    }
    out = os.path.join(RESULTS_DIR, "phase27_ppo_dual_mode_log.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  💾 Results saved: {out}")

    print(f"\n{'=' * 60}")
    print(f"  Phase 27 COMPLETE — {elapsed:.0f}s")
    print(f"{'=' * 60}")
    print(f"  Static-Factual:  acc={result_sf['factual_acc']:.1f}%")
    print(f"  Static-Creative: acc={result_sc['factual_acc']:.1f}%")
    print(f"  REINFORCE (ref): acc={reinforce_ref.get('factual_acc', '?')}%")
    print(f"  PPO Dual-Mode:   acc={result_ppo['factual_acc']:.1f}%")
    print(f"  σ separation:    {result_ppo['sigma_separation']:.4f}")
    print(f"  σ drift:         {result_ppo['sigma_drift']:.4f}")
    print(f"{'=' * 60}")
    return result


if __name__ == "__main__":
    main()
