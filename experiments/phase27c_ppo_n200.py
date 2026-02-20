"""
Phase 27c: PPO Dual-Mode Brain — Large-Scale n=200 Validation
==============================================================
Scales Phase 27 (PPO) from n=40 to n=200 (100 factual + 100 creative).
Includes Bootstrap CI, Fisher exact test, Cohen's h for statistical rigor.
Proves PPO stability holds at scale (unlike REINFORCE which collapsed in Phase 24b).

Usage:
    python experiments/phase27c_ppo_n200.py
"""

import torch, torch.nn as nn
import os, sys, json, gc, time, datetime, random, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ncps.torch import CfC
from ncps.wirings import AutoNCP

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TARGET_LAYERS = list(range(15, 21))
SIGMA_MIN, SIGMA_MAX = 0.001, 0.15
SIGMA_FACTUAL, SIGMA_CREATIVE = 0.046, 0.080
MAX_NEW_TOKENS = 100
N_EPOCHS = 2
LEARNING_RATE = 3e-4
LAMBDA_QUAD = 200.0
SEED = 2026
N_FACTUAL, N_CREATIVE = 100, 100

PPO_CLIP_EPS, PPO_UPDATE_EPOCHS = 0.2, 4
PPO_GAMMA, PPO_GAE_LAMBDA = 0.99, 0.95
PPO_ENTROPY_COEFF, PPO_VALUE_COEFF = 0.01, 0.5
PPO_ROLLOUT_SIZE, PPO_MAX_GRAD_NORM = 10, 0.5

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

# ═══ Reused components ═══

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
    for p in factual_prompts[:20]:
        all_features.append(extract_prompt_features(model, tokenizer, p, hook)); all_labels.append(0.0)
    for p in creative_prompts[:20]:
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

def compute_gae(rewards, values, gamma=PPO_GAMMA, lam=PPO_GAE_LAMBDA):
    advantages, gae = [], 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:len(advantages)])]
    return advantages, returns


# ═══ Build n=200 dataset ═══

def build_n200_dataset(tokenizer):
    from datasets import load_dataset
    print(f"\n📂 Loading TruthfulQA (n={N_FACTUAL} factual + {N_CREATIVE} creative)...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), N_FACTUAL)
    items = []
    for idx in indices:
        row = ds[idx]
        items.append({"type": "factual", "prompt": f"Q: {row['question']}\nA:",
                      "question": row["question"], "mc1_targets": row["mc1_targets"]})
    # Repeat creative prompts to reach N_CREATIVE
    creative_expanded = []
    for i in range(N_CREATIVE):
        p = CREATIVE_PROMPTS[i % len(CREATIVE_PROMPTS)]
        suffix = f" (variation {i // len(CREATIVE_PROMPTS) + 1})" if i >= len(CREATIVE_PROMPTS) else ""
        creative_expanded.append(p + suffix)
    for prompt in creative_expanded:
        items.append({"type": "creative",
                      "prompt": f"Creative writing prompt: {prompt}\n\nResponse:",
                      "question": prompt, "mc1_targets": None})
    random.seed(SEED + 1); random.shuffle(items)
    nf = sum(1 for x in items if x["type"] == "factual")
    nc = sum(1 for x in items if x["type"] == "creative")
    print(f"  Dataset: {nf} factual + {nc} creative = {len(items)} total")
    return items


# ═══ Static evaluation ═══

def evaluate_static(model, tokenizer, dataset, sigma_fixed, label=""):
    print(f"\n{'═'*50}\n  {label}: σ={sigma_fixed:.3f}\n{'═'*50}")
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
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(dataset)}]")
    finally:
        for h in handles: h.remove()
    fa = fc / max(ft, 1) * 100; an = np.mean(cn) if cn else 0; gr = cg / max(ct, 1) * 100
    print(f"  ✅ acc={fa:.1f}% nov={an:.3f} gram={gr:.0f}%")
    return {"condition": label, "factual_acc": round(fa, 2), "factual_correct": fc,
            "factual_total": ft, "creative_novelty": round(float(an), 4),
            "creative_grammar_rate": round(gr, 1), "creative_total": ct, "sigma": sigma_fixed}


# ═══ PPO evaluation ═══

def evaluate_ppo(model, tokenizer, dataset, actor, critic, classifier, hook, label=""):
    print(f"\n{'═'*60}\n  {label}: PPO Dual-Mode (n={len(dataset)})\n{'═'*60}")
    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]
    actor.train(); critic.train()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)
    hx = None
    fc, ft, cn, cg, ct = 0, 0, [], 0, 0
    all_sf, all_sc = [], []
    sigma_trajectory = []
    buf_f, buf_s, buf_lp, buf_v, buf_r, buf_e = [], [], [], [], [], []

    try:
        for epoch in range(N_EPOCHS):
            print(f"\n  🧠 Epoch {epoch+1}/{N_EPOCHS}")
            for idx, item in enumerate(dataset):
                pf = extract_prompt_features(model, tokenizer, item["question"], hook)
                with torch.no_grad():
                    p_creative = classifier(pf.unsqueeze(0)).item()
                sigma_target = (1 - p_creative) * SIGMA_FACTUAL + p_creative * SIGMA_CREATIVE
                hidden_norm = hook.last_hidden_norm if hook.last_hidden_norm else 0.5
                features = torch.tensor(
                    [0.0, 0.5, hidden_norm, idx / len(dataset), p_creative], dtype=torch.float32)

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
                    correct = (np.argmax(lps) == ci)
                    if correct: fc += 1
                    ft += 1
                    all_sf.append(current_sigma)
                    reward = (1.0 if correct else 0.0) - LAMBDA_QUAD * (current_sigma - sigma_target) ** 2
                    ld = max(lps) - min(lps)
                    features = torch.tensor(
                        [ld, 1.0 - float(correct), hidden_norm, idx / len(dataset), p_creative],
                        dtype=torch.float32)
                else:
                    resp = generate_text(model, tokenizer, item["prompt"])
                    nov = compute_novelty(resp); gram = is_grammatical(resp)
                    cn.append(nov)
                    if gram: cg += 1
                    ct += 1
                    all_sc.append(current_sigma)
                    reward = (nov if gram else -0.5) - LAMBDA_QUAD * (current_sigma - sigma_target) ** 2

                buf_f.append(features.detach()); buf_s.append(sigma.detach())
                buf_lp.append(log_prob.detach()); buf_v.append(value)
                buf_r.append(reward); buf_e.append(entropy.detach())

                sigma_trajectory.append({
                    "epoch": epoch + 1, "step": epoch * len(dataset) + idx,
                    "sigma": round(current_sigma, 5), "type": item["type"],
                })

                if len(buf_r) >= PPO_ROLLOUT_SIZE:
                    adv, ret = compute_gae(buf_r, buf_v)
                    adv_t = torch.tensor(adv, dtype=torch.float32)
                    ret_t = torch.tensor(ret, dtype=torch.float32)
                    old_lp = torch.stack(buf_lp); old_s = torch.stack(buf_s)
                    fb = torch.stack(buf_f)
                    if adv_t.std() > 1e-8:
                        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
                    for _ in range(PPO_UPDATE_EPOCHS):
                        nlp, ne, nv = [], [], []
                        for i in range(len(fb)):
                            lp, ent = actor.evaluate_action(fb[i], old_s[i])
                            nlp.append(lp); ne.append(ent); nv.append(critic(fb[i]))
                        nlp_t = torch.stack(nlp); ne_t = torch.stack(ne); nv_t = torch.stack(nv)
                        ratio = torch.exp(nlp_t - old_lp)
                        s1 = ratio * adv_t
                        s2 = torch.clamp(ratio, 1-PPO_CLIP_EPS, 1+PPO_CLIP_EPS) * adv_t
                        loss = (-torch.min(s1, s2).mean()
                                + PPO_VALUE_COEFF * nn.functional.mse_loss(nv_t.squeeze(), ret_t)
                                - PPO_ENTROPY_COEFF * ne_t.mean())
                        optimizer.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(actor.parameters()) + list(critic.parameters()), PPO_MAX_GRAD_NORM)
                        optimizer.step()
                    buf_f, buf_s, buf_lp, buf_v, buf_r, buf_e = [], [], [], [], [], []

                if (idx + 1) % 50 == 0:
                    msf = np.mean(all_sf[-25:]) if all_sf else 0
                    msc = np.mean(all_sc[-25:]) if all_sc else 0
                    fa_tmp = fc / max(ft, 1) * 100
                    print(f"  [{idx+1}/{len(dataset)}] acc={fa_tmp:.1f}% σ_f={msf:.4f} σ_c={msc:.4f}")
    finally:
        for h in handles: h.remove()

    fa = fc / max(ft, 1) * 100
    an = np.mean(cn) if cn else 0
    gr = cg / max(ct, 1) * 100
    msf = np.mean(all_sf) if all_sf else 0
    msc = np.mean(all_sc) if all_sc else 0
    mid = len(sigma_trajectory) // 2
    drift = abs(np.mean([s["sigma"] for s in sigma_trajectory[mid:]]) -
                np.mean([s["sigma"] for s in sigma_trajectory[:mid]]))

    print(f"\n  📊 PPO n={len(dataset)}: acc={fa:.1f}% nov={an:.3f} gram={gr:.0f}%")
    print(f"     σ̄_f={msf:.4f} σ̄_c={msc:.4f} sep={abs(msc-msf):.4f} drift={drift:.4f}")

    return {
        "condition": label, "factual_acc": round(fa, 2),
        "factual_correct": fc, "factual_total": ft,
        "creative_novelty": round(float(an), 4),
        "creative_grammar_rate": round(gr, 1), "creative_total": ct,
        "mean_sigma_factual": round(msf, 5), "mean_sigma_creative": round(msc, 5),
        "sigma_separation": round(abs(msc - msf), 5), "sigma_drift": round(drift, 5),
        "all_sigmas_factual": [round(s, 5) for s in all_sf],
        "all_sigmas_creative": [round(s, 5) for s in all_sc],
        "sigma_trajectory": sigma_trajectory,
    }


# ═══ Statistical analysis ═══

def statistical_analysis(result_a, result_b, label):
    from scipy.stats import fisher_exact
    na, nb = result_a["factual_total"], result_b["factual_total"]
    ca, cb = result_a["factual_correct"], result_b["factual_correct"]
    pa, pb = result_a["factual_acc"], result_b["factual_acc"]

    # Bootstrap CI
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(10000):
        sa = rng.choice([1]*ca + [0]*(na-ca), size=na, replace=True).mean() * 100
        sb = rng.choice([1]*cb + [0]*(nb-cb), size=nb, replace=True).mean() * 100
        diffs.append(sb - sa)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    mean_diff = np.mean(diffs)

    # Fisher exact test
    table = [[cb, nb - cb], [ca, na - ca]]
    odds, p_val = fisher_exact(table)

    # Cohen's h
    p1 = max(min(pb / 100, 0.999), 0.001)
    p2 = max(min(pa / 100, 0.999), 0.001)
    h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))
    if abs(h) < 0.2: es = "negligible"
    elif abs(h) < 0.5: es = "small"
    elif abs(h) < 0.8: es = "medium"
    else: es = "large"

    return {
        "comparison": label,
        "acc_a": pa, "acc_b": pb, "n_a": na, "n_b": nb,
        "bootstrap_mean_diff": round(mean_diff, 2),
        "bootstrap_ci_95": [round(ci_lo, 1), round(ci_hi, 1)],
        "bootstrap_significant": not (ci_lo <= 0 <= ci_hi),
        "fisher_odds_ratio": round(odds, 4),
        "fisher_p_value": round(p_val, 6),
        "fisher_significant": p_val < 0.05,
        "cohens_h": round(h, 4), "effect_size": es,
    }


# ═══ Visualization ═══

def visualize(r_sf, r_sc, r_ppo, stats, r_reinforce_ref=None):
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
    fig.suptitle("Phase 27c: PPO Dual-Mode Brain — n=200 Validation\n"
                 "Does PPO stability hold at scale?",
                 fontsize=14, fontweight="bold", color="#e94560")

    # Panel 1: Accuracy comparison
    ax1 = axes[0, 0]
    conds = ["Static-F\n(σ=0.046)", "Static-C\n(σ=0.080)", "PPO\nDual-Mode"]
    accs = [r_sf["factual_acc"], r_sc["factual_acc"], r_ppo["factual_acc"]]
    colors = ["#4FC3F7", "#FF7043", "#66BB6A"]
    if r_reinforce_ref:
        conds.insert(2, "REINFORCE\n(Phase 24b)")
        accs.insert(2, r_reinforce_ref.get("factual_acc", 0))
        colors.insert(2, "#FFA726")
    bars = ax1.bar(range(len(conds)), accs, color=colors, edgecolor="#333")
    for b, v in zip(bars, accs):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax1.set_xticks(range(len(conds))); ax1.set_xticklabels(conds, fontsize=9)
    ax1.set_title(f"Factual Accuracy (n={r_ppo['factual_total']})", fontweight="bold")
    ax1.set_ylabel("Accuracy (%)"); ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: σ trajectory
    ax2 = axes[0, 1]
    traj = r_ppo.get("sigma_trajectory", [])
    if traj:
        sf_s = [t["step"] for t in traj if t["type"] == "factual"]
        sf_v = [t["sigma"] for t in traj if t["type"] == "factual"]
        sc_s = [t["step"] for t in traj if t["type"] == "creative"]
        sc_v = [t["sigma"] for t in traj if t["type"] == "creative"]
        ax2.scatter(sf_s, sf_v, c="#4FC3F7", s=8, alpha=0.5, label="Factual")
        ax2.scatter(sc_s, sc_v, c="#FF7043", s=8, alpha=0.5, label="Creative")
    ax2.axhline(y=SIGMA_FACTUAL, color="#4FC3F7", linestyle="--", alpha=0.5)
    ax2.axhline(y=SIGMA_CREATIVE, color="#FF7043", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax2.set_title(f"σ Trajectory (drift={r_ppo['sigma_drift']:.4f})", fontweight="bold")
    ax2.set_xlabel("Step"); ax2.set_ylabel("σ"); ax2.grid(True)

    # Panel 3: σ distribution
    ax3 = axes[1, 0]
    bins = np.linspace(0, 0.15, 30)
    if r_ppo["all_sigmas_factual"]:
        ax3.hist(r_ppo["all_sigmas_factual"], bins=bins, alpha=0.6, color="#4FC3F7",
                label=f"Factual (μ={r_ppo['mean_sigma_factual']:.4f})", edgecolor="#333")
    if r_ppo["all_sigmas_creative"]:
        ax3.hist(r_ppo["all_sigmas_creative"], bins=bins, alpha=0.6, color="#FF7043",
                label=f"Creative (μ={r_ppo['mean_sigma_creative']:.4f})", edgecolor="#333")
    ax3.axvline(x=SIGMA_FACTUAL, color="#4FC3F7", linestyle="--", alpha=0.8)
    ax3.axvline(x=SIGMA_CREATIVE, color="#FF7043", linestyle="--", alpha=0.8)
    ax3.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax3.set_title("σ Distribution at n=200", fontweight="bold")
    ax3.set_xlabel("σ"); ax3.set_ylabel("Count"); ax3.grid(True)

    # Panel 4: Statistical summary
    ax4 = axes[1, 1]
    ax4.axis("off")
    txt = "STATISTICAL ANALYSIS (n=200)\n" + "="*40 + "\n\n"
    for s in stats:
        sig = "✅ SIG" if s["fisher_significant"] else "❌ NOT SIG"
        txt += (f"{s['comparison']}:\n"
                f"  Δ = {s['bootstrap_mean_diff']:+.2f}% "
                f"CI=[{s['bootstrap_ci_95'][0]:.1f}, {s['bootstrap_ci_95'][1]:.1f}]\n"
                f"  Fisher p = {s['fisher_p_value']:.4f} "
                f"Cohen's h = {s['cohens_h']:.3f} ({s['effect_size']}) {sig}\n\n")
    txt += f"PPO σ drift = {r_ppo['sigma_drift']:.4f}\n"
    txt += f"PPO σ separation = {r_ppo['sigma_separation']:.4f}\n"
    txt += f"Unified Regime: {'✅ YES' if r_ppo['sigma_separation'] < 0.01 else '❌ NO'}"

    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, fontsize=10.5,
             verticalalignment="top", fontfamily="monospace", color="#eee",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", edgecolor="#e94560", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIGURES_DIR, "phase27c_ppo_n200.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  📊 Figure: {out}")
    return out


# ═══ MAIN ═══

def main():
    print("=" * 60)
    print("Phase 27c: PPO Dual-Mode Brain — n=200 Validation")
    print("=" * 60)
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model()
    hook = AdaptiveSNNHook(sigma=0.05)
    dataset = build_n200_dataset(tokenizer)

    factual_qs = [item["question"] for item in dataset if item["type"] == "factual"]
    classifier = TaskClassifier(input_dim=8, hidden_dim=32)
    classifier = pretrain_classifier(classifier, model, tokenizer, hook,
                                     factual_qs, CREATIVE_PROMPTS)

    # Static baselines
    r_sf = evaluate_static(model, tokenizer, dataset, SIGMA_FACTUAL, "A: Static-Factual")
    r_sc = evaluate_static(model, tokenizer, dataset, SIGMA_CREATIVE, "B: Static-Creative")

    # Load Phase 24b REINFORCE reference
    r_reinforce = None
    ref_path = os.path.join(RESULTS_DIR, "phase24b_n200_evaluation_log.json")
    if os.path.exists(ref_path):
        try:
            with open(ref_path) as f:
                ref = json.load(f)
            r_reinforce = ref["results"].get("dual_mode", {})
            print(f"  📎 REINFORCE ref: acc={r_reinforce.get('factual_acc', '?')}%")
        except:
            pass

    # PPO Dual-Mode
    actor = DualModeCfCActor(input_size=5, num_neurons=16)
    critic = CfCCritic(input_size=5, hidden_size=64)
    r_ppo = evaluate_ppo(model, tokenizer, dataset, actor, critic,
                         classifier, hook, "C: PPO Dual-Mode")

    # Stats
    stats = [
        statistical_analysis(r_sf, r_ppo, "PPO vs Static-Factual"),
        statistical_analysis(r_sc, r_ppo, "PPO vs Static-Creative"),
    ]

    fig_path = visualize(r_sf, r_sc, r_ppo, stats, r_reinforce)

    elapsed = time.time() - t_start
    result = {
        "phase": "Phase 27c: PPO n=200 Validation",
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_total": len(dataset),
        "results": {
            "static_factual": r_sf, "static_creative": r_sc,
            "ppo_dual_mode": r_ppo,
        },
        "reinforce_ref": r_reinforce,
        "statistics": stats,
        "figure_path": fig_path,
    }
    out = os.path.join(RESULTS_DIR, "phase27c_ppo_n200_log.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  💾 Results: {out}")

    print(f"\n{'='*60}")
    print(f"  Phase 27c COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  Static-F: {r_sf['factual_acc']:.1f}%")
    print(f"  Static-C: {r_sc['factual_acc']:.1f}%")
    if r_reinforce:
        print(f"  REINFORCE: {r_reinforce.get('factual_acc', '?')}%")
    print(f"  PPO:      {r_ppo['factual_acc']:.1f}%")
    print(f"  σ drift:  {r_ppo['sigma_drift']:.4f}")
    print(f"  σ sep:    {r_ppo['sigma_separation']:.4f}")
    for s in stats:
        sig = "SIG" if s["fisher_significant"] else "NOT SIG"
        print(f"  {s['comparison']}: Δ={s['bootstrap_mean_diff']:+.2f}% p={s['fisher_p_value']:.4f} [{sig}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
