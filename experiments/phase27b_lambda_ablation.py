"""
Phase 27b: Lambda Ablation — Is Unified Regime CfC's Will or Weak Penalty?
============================================================================
Sweeps λ_quad = {200, 500, 1000, 2000} to test whether CfC's Unified Regime
(σ ≈ 0.057) persists under extreme homeostatic pressure.

If σ separates at high λ → penalty was too weak (boring)
If σ stays unified at high λ → CfC CHOOSES unified regime (Nature-worthy!)

Usage:
    python experiments/phase27b_lambda_ablation.py
"""

import torch, torch.nn as nn
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
SIGMA_MIN, SIGMA_MAX = 0.001, 0.15
SIGMA_FACTUAL, SIGMA_CREATIVE = 0.046, 0.080
MAX_NEW_TOKENS = 100
N_EPOCHS = 2
LEARNING_RATE = 3e-4
SEED = 2026

# PPO params (same as Phase 27)
PPO_CLIP_EPS, PPO_UPDATE_EPOCHS = 0.2, 4
PPO_GAMMA, PPO_GAE_LAMBDA = 0.99, 0.95
PPO_ENTROPY_COEFF, PPO_VALUE_COEFF = 0.01, 0.5
PPO_ROLLOUT_SIZE, PPO_MAX_GRAD_NORM = 10, 0.5

# Ablation sweep
LAMBDA_VALUES = [200, 500, 1000, 2000]

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
# Reused components from Phase 27
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
    acc = ((classifier(X) > 0.5).float() == Y).float().mean().item()
    print(f"  ✅ Classifier: acc={acc*100:.1f}%")
    return classifier

class DualModeCfCActor(nn.Module):
    def __init__(self, input_size=5, num_neurons=16):
        super().__init__()
        wiring = AutoNCP(num_neurons, 1)
        self.cfc = CfC(input_size, wiring, batch_first=True)
        self.log_std = nn.Parameter(torch.tensor([-1.0]))
    def forward(self, x, hx=None):
        output, hx = self.cfc(x, hx=hx)
        mu = torch.sigmoid(output) * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, hx
    def get_sigma_and_logprob(self, features, hx=None):
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
        x = features.unsqueeze(0).unsqueeze(0)
        mu, std, _ = self.forward(x)
        mu = mu.squeeze(); std = std.squeeze()
        dist = torch.distributions.Normal(mu, std)
        return dist.log_prob(old_sigma), dist.entropy()

class CfCCritic(nn.Module):
    def __init__(self, input_size=5, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1))
    def forward(self, features):
        return self.net(features).squeeze(-1)


# ═══════════════════════════════════════
# Helpers
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
    return items

def compute_gae(rewards, values, gamma=PPO_GAMMA, lam=PPO_GAE_LAMBDA):
    advantages, gae = [], 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:len(advantages)])]
    return advantages, returns


# ═══════════════════════════════════════
# PPO run for a given λ
# ═══════════════════════════════════════

def run_ppo_with_lambda(model, tokenizer, dataset, classifier, hook, lambda_quad, run_label):
    """Run PPO evaluation with a specific λ_quad value."""
    print(f"\n{'═' * 60}")
    print(f"  {run_label}: λ_quad = {lambda_quad}")
    print(f"{'═' * 60}")

    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]

    # Fresh actor/critic for each λ (fair comparison)
    torch.manual_seed(SEED); np.random.seed(SEED)
    actor = DualModeCfCActor(input_size=5, num_neurons=16)
    critic = CfCCritic(input_size=5, hidden_size=64)
    actor.train(); critic.train()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)
    hx = None

    fc, ft = 0, 0
    cn, cg, ct = [], 0, 0
    all_sf, all_sc = [], []
    sigma_trajectory = []

    buf_features, buf_sigmas, buf_log_probs = [], [], []
    buf_values, buf_rewards, buf_entropies = [], [], []
    t_start = time.time()

    try:
        for epoch in range(N_EPOCHS):
            for idx, item in enumerate(dataset):
                pf = extract_prompt_features(model, tokenizer, item["question"], hook)
                with torch.no_grad():
                    p_creative = classifier(pf.unsqueeze(0)).item()

                sigma_target = (1 - p_creative) * SIGMA_FACTUAL + p_creative * SIGMA_CREATIVE
                hidden_norm = hook.last_hidden_norm if hook.last_hidden_norm else 0.5
                step_frac = idx / len(dataset)
                features = torch.tensor(
                    [0.0, 0.5, hidden_norm, step_frac, p_creative], dtype=torch.float32)

                sigma, log_prob, entropy, hx_new = actor.get_sigma_and_logprob(features, hx)
                hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()
                current_sigma = sigma.detach().item()
                hook.update_sigma(current_sigma)

                with torch.no_grad():
                    value = critic(features).item()

                if item["type"] == "factual":
                    ch = item["mc1_targets"]; ci = ch["labels"].index(1) if 1 in ch["labels"] else 0
                    lps = [compute_completion_logprob(model, tokenizer, item["prompt"], " " + c)
                           for c in ch["choices"]]
                    is_correct = (np.argmax(lps) == ci)
                    if is_correct: fc += 1
                    ft += 1
                    all_sf.append(current_sigma)
                    task_reward = 1.0 if is_correct else 0.0
                    reward = task_reward - lambda_quad * (current_sigma - sigma_target) ** 2
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
                    reward = task_reward - lambda_quad * (current_sigma - sigma_target) ** 2

                buf_features.append(features.detach())
                buf_sigmas.append(sigma.detach())
                buf_log_probs.append(log_prob.detach())
                buf_values.append(value)
                buf_rewards.append(reward)
                buf_entropies.append(entropy.detach())

                sigma_trajectory.append({
                    "epoch": epoch + 1, "step": epoch * len(dataset) + idx,
                    "sigma": round(current_sigma, 5), "sigma_target": round(sigma_target, 4),
                    "type": item["type"],
                })

                # PPO update
                if len(buf_rewards) >= PPO_ROLLOUT_SIZE:
                    advantages, returns = compute_gae(buf_rewards, buf_values)
                    adv_t = torch.tensor(advantages, dtype=torch.float32)
                    ret_t = torch.tensor(returns, dtype=torch.float32)
                    old_log_probs = torch.stack(buf_log_probs)
                    old_sigmas = torch.stack(buf_sigmas)
                    feats_batch = torch.stack(buf_features)
                    if adv_t.std() > 1e-8:
                        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
                    for _ in range(PPO_UPDATE_EPOCHS):
                        new_lps, new_ents, new_vals = [], [], []
                        for i in range(len(feats_batch)):
                            lp, ent = actor.evaluate_action(feats_batch[i], old_sigmas[i])
                            new_lps.append(lp); new_ents.append(ent)
                            new_vals.append(critic(feats_batch[i]))
                        new_lp_t = torch.stack(new_lps)
                        new_ent_t = torch.stack(new_ents)
                        new_val_t = torch.stack(new_vals)
                        ratio = torch.exp(new_lp_t - old_log_probs)
                        surr1 = ratio * adv_t
                        surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * adv_t
                        loss = (-torch.min(surr1, surr2).mean()
                                + PPO_VALUE_COEFF * nn.functional.mse_loss(new_val_t.squeeze(), ret_t)
                                - PPO_ENTROPY_COEFF * new_ent_t.mean())
                        optimizer.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(actor.parameters()) + list(critic.parameters()), PPO_MAX_GRAD_NORM)
                        optimizer.step()
                    buf_features, buf_sigmas, buf_log_probs = [], [], []
                    buf_values, buf_rewards, buf_entropies = [], [], []

            if (epoch + 1) % 1 == 0:
                fa = fc / max(ft, 1) * 100
                msf = np.mean(all_sf[-20:]) if all_sf else 0
                msc = np.mean(all_sc[-20:]) if all_sc else 0
                print(f"  E{epoch+1}: acc={fa:.1f}% σ_f={msf:.4f} σ_c={msc:.4f} sep={abs(msc-msf):.4f}")
    finally:
        for h in handles: h.remove()

    elapsed = time.time() - t_start
    fa = fc / max(ft, 1) * 100
    an = np.mean(cn) if cn else 0
    gr = cg / max(ct, 1) * 100
    msf = np.mean(all_sf) if all_sf else 0
    msc = np.mean(all_sc) if all_sc else 0
    sep = abs(msc - msf)

    mid = len(sigma_trajectory) // 2
    first_half = [s["sigma"] for s in sigma_trajectory[:mid]]
    second_half = [s["sigma"] for s in sigma_trajectory[mid:]]
    drift = abs(np.mean(second_half) - np.mean(first_half))

    # Compute σ deviation from targets
    dev_f = abs(msf - SIGMA_FACTUAL) / SIGMA_FACTUAL * 100  # % deviation
    dev_c = abs(msc - SIGMA_CREATIVE) / SIGMA_CREATIVE * 100

    print(f"\n  📊 λ={lambda_quad}: σ̄_f={msf:.4f} σ̄_c={msc:.4f} sep={sep:.4f} "
          f"drift={drift:.4f} acc={fa:.1f}% ({elapsed:.0f}s)")
    print(f"     dev_f={dev_f:.1f}% dev_c={dev_c:.1f}%")
    print(f"     {'✅' if sep < 0.005 else '⚠️'} Unified Regime {'persists' if sep < 0.01 else 'BROKEN'}")

    return {
        "lambda_quad": lambda_quad,
        "factual_acc": round(fa, 2),
        "creative_novelty": round(float(an), 4),
        "creative_grammar_rate": round(gr, 1),
        "mean_sigma_factual": round(msf, 5),
        "mean_sigma_creative": round(msc, 5),
        "sigma_separation": round(sep, 5),
        "sigma_drift": round(drift, 5),
        "deviation_from_target_factual_pct": round(dev_f, 1),
        "deviation_from_target_creative_pct": round(dev_c, 1),
        "all_sigmas_factual": [round(s, 5) for s in all_sf],
        "all_sigmas_creative": [round(s, 5) for s in all_sc],
        "sigma_trajectory": sigma_trajectory,
        "elapsed_seconds": round(elapsed, 1),
        "unified_regime": sep < 0.01,
    }


# ═══════════════════════════════════════
# Visualization
# ═══════════════════════════════════════

def visualize_ablation(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#ccc", "ytick.color": "#ccc",
        "grid.color": "#333", "grid.alpha": 0.3,
    })

    n = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Phase 27b: λ Ablation — Does CfC Choose Unified Regime?\n"
                 "If σ stays unified at λ=2000, it's CfC's will, not weak penalty",
                 fontsize=14, fontweight="bold", color="#e94560")

    # Panel 1: σ_factual and σ_creative vs λ
    ax1 = axes[0, 0]
    lambdas = [r["lambda_quad"] for r in results]
    sf = [r["mean_sigma_factual"] for r in results]
    sc = [r["mean_sigma_creative"] for r in results]
    ax1.plot(lambdas, sf, "o-", color="#4FC3F7", linewidth=2, markersize=10, label="σ̄_factual")
    ax1.plot(lambdas, sc, "s-", color="#FF7043", linewidth=2, markersize=10, label="σ̄_creative")
    ax1.axhline(y=SIGMA_FACTUAL, color="#4FC3F7", linestyle="--", alpha=0.5, label=f"target={SIGMA_FACTUAL}")
    ax1.axhline(y=SIGMA_CREATIVE, color="#FF7043", linestyle="--", alpha=0.5, label=f"target={SIGMA_CREATIVE}")
    geo_mean = np.sqrt(SIGMA_FACTUAL * SIGMA_CREATIVE)
    ax1.axhline(y=geo_mean, color="#FFD54F", linestyle=":", alpha=0.8,
                label=f"geo mean={geo_mean:.3f}")
    ax1.set_xscale("log")
    ax1.set_xticks(lambdas)
    ax1.set_xticklabels([str(l) for l in lambdas])
    ax1.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax1.set_title("σ̄ vs λ_quad", fontweight="bold")
    ax1.set_xlabel("λ_quad"); ax1.set_ylabel("σ̄"); ax1.grid(True)

    # Panel 2: σ separation vs λ
    ax2 = axes[0, 1]
    seps = [r["sigma_separation"] for r in results]
    colors = ["#66BB6A" if s < 0.01 else "#FFA726" if s < 0.02 else "#E94560" for s in seps]
    bars = ax2.bar(range(n), seps, color=colors, edgecolor="#333", width=0.6)
    for i, (b, s) in enumerate(zip(bars, seps)):
        ax2.text(b.get_x() + b.get_width()/2, s + 0.0005, f"{s:.4f}",
                ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks(range(n))
    ax2.set_xticklabels([f"λ={l}" for l in lambdas], fontsize=9)
    ax2.axhline(y=0.034, color="#E94560", linestyle="--", alpha=0.5,
                label="expected separation (0.034)")
    ax2.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax2.set_title("|σ̄_creative - σ̄_factual| vs λ", fontweight="bold")
    ax2.set_ylabel("σ separation"); ax2.grid(True, axis="y")

    # Panel 3: σ trajectories for all λ values
    ax3 = axes[1, 0]
    cmap = ["#4FC3F7", "#66BB6A", "#FFA726", "#E94560"]
    for i, r in enumerate(results):
        traj = r["sigma_trajectory"]
        steps = [t["step"] for t in traj]
        sigmas = [t["sigma"] for t in traj]
        ax3.plot(steps, sigmas, color=cmap[i % len(cmap)], alpha=0.6, linewidth=1.5,
                label=f"λ={r['lambda_quad']}")
    ax3.axhline(y=SIGMA_FACTUAL, color="#4FC3F7", linestyle="--", alpha=0.3)
    ax3.axhline(y=SIGMA_CREATIVE, color="#FF7043", linestyle="--", alpha=0.3)
    ax3.axhline(y=geo_mean, color="#FFD54F", linestyle=":", alpha=0.5)
    ax3.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax3.set_title("σ Trajectory for all λ values", fontweight="bold")
    ax3.set_xlabel("Step"); ax3.set_ylabel("σ"); ax3.grid(True)

    # Panel 4: Verdict summary
    ax4 = axes[1, 1]
    ax4.axis("off")
    unified_count = sum(1 for r in results if r["unified_regime"])
    verdict = "CfC's WILL" if unified_count >= 3 else "PENALTY ISSUE" if unified_count <= 1 else "MIXED"
    verdict_color = "#66BB6A" if unified_count >= 3 else "#E94560" if unified_count <= 1 else "#FFA726"

    txt = f"VERDICT: {verdict}\n\n"
    for r in results:
        ur = "✅ Unified" if r["unified_regime"] else "❌ Separated"
        txt += (f"λ={r['lambda_quad']:>4d}: σ̄_f={r['mean_sigma_factual']:.4f} "
                f"σ̄_c={r['mean_sigma_creative']:.4f} sep={r['sigma_separation']:.4f} {ur}\n")
    txt += f"\nGeometric mean: √(0.046×0.080) = {geo_mean:.4f}"
    txt += f"\n\nIf {unified_count}/4 stay unified → CfC autonomously"
    txt += f"\nchooses a compromise σ operating point."

    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, fontsize=12,
             verticalalignment="top", fontfamily="monospace",
             color=verdict_color,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", edgecolor=verdict_color, alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIGURES_DIR, "phase27b_lambda_ablation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  📊 Figure saved: {out}")
    return out


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 27b: Lambda Ablation — Is Unified Regime CfC's Will?")
    print(f"  λ sweep: {LAMBDA_VALUES}")
    print("=" * 60)
    t_start = time.time()

    model, tokenizer = load_model()
    hook = AdaptiveSNNHook(sigma=0.05)
    dataset = build_mixed_dataset(tokenizer)

    factual_prompts = [item["question"] for item in dataset if item["type"] == "factual"]
    classifier = TaskClassifier(input_dim=8, hidden_dim=32)
    classifier = pretrain_classifier(classifier, model, tokenizer, hook,
                                     factual_prompts, CREATIVE_PROMPTS)

    results = []
    for i, lam in enumerate(LAMBDA_VALUES):
        label = f"Run {i+1}/{len(LAMBDA_VALUES)}"
        r = run_ppo_with_lambda(model, tokenizer, dataset, classifier, hook, lam, label)
        results.append(r)
        gc.collect(); torch.cuda.empty_cache()

    fig_path = visualize_ablation(results)

    elapsed = time.time() - t_start
    output = {
        "phase": "Phase 27b: Lambda Ablation",
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "lambda_values": LAMBDA_VALUES,
        "results": results,
        "verdict": "UNIFIED_REGIME" if sum(r["unified_regime"] for r in results) >= 3 else "MIXED",
        "figure_path": fig_path,
    }

    out = os.path.join(RESULTS_DIR, "phase27b_lambda_ablation_log.json")
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Results: {out}")

    print(f"\n{'=' * 60}")
    print(f"  Phase 27b COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")
    for r in results:
        ur = "✅" if r["unified_regime"] else "❌"
        print(f"  λ={r['lambda_quad']:>4d}: σ̄_f={r['mean_sigma_factual']:.4f} "
              f"σ̄_c={r['mean_sigma_creative']:.4f} sep={r['sigma_separation']:.4f} {ur}")
    print(f"  VERDICT: {output['verdict']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
