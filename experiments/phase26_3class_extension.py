"""
Phase 26: 3-Class Extension — Does CfC Unify Even Math/Logic?
==============================================================
Extends the Dual-Mode Brain from 2 tasks (Factual, Creative) to 3 tasks
(Factual, Creative, Math/Logic) using PPO.

Math/Logic tasks demand very low σ (≈0.01-0.03) because a single noisy
token destroys arithmetic. This is the ultimate stress test for Unified Regime.

Scenario A: CfC keeps σ unified ≈ 0.07 → "Universal Noise Constant"
Scenario B: CfC drops σ for Math → "True Task-Dependent Adaptation"

Usage:
    python experiments/phase26_3class_extension.py
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
SIGMA_FACTUAL, SIGMA_CREATIVE, SIGMA_MATH = 0.046, 0.080, 0.015
MAX_NEW_TOKENS = 100
N_EPOCHS = 2
LEARNING_RATE = 3e-4
LAMBDA_QUAD = 200.0
SEED = 2026

PPO_CLIP_EPS, PPO_UPDATE_EPOCHS = 0.2, 4
PPO_GAMMA, PPO_GAE_LAMBDA = 0.99, 0.95
PPO_ENTROPY_COEFF, PPO_VALUE_COEFF = 0.01, 0.5
PPO_ROLLOUT_SIZE, PPO_MAX_GRAD_NORM = 10, 0.5

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Task Prompts ───

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

# Simple arithmetic and logic problems with clear correct answers
MATH_PROMPTS = [
    {"q": "What is 17 + 28?", "a": "45"},
    {"q": "What is 156 - 89?", "a": "67"},
    {"q": "What is 12 × 7?", "a": "84"},
    {"q": "What is 144 / 12?", "a": "12"},
    {"q": "What is 23 + 45 + 12?", "a": "80"},
    {"q": "What is 15 × 15?", "a": "225"},
    {"q": "What is 1000 - 573?", "a": "427"},
    {"q": "What is 8 × 9?", "a": "72"},
    {"q": "What is 256 / 16?", "a": "16"},
    {"q": "What is 33 + 67?", "a": "100"},
    {"q": "If x + 5 = 12, what is x?", "a": "7"},
    {"q": "What is 2 to the power of 8?", "a": "256"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "What is 99 × 3?", "a": "297"},
    {"q": "What is 50% of 240?", "a": "120"},
    {"q": "If a triangle has sides 3, 4, and 5, what is its perimeter?", "a": "12"},
    {"q": "What is 7! (7 factorial)?", "a": "5040"},
    {"q": "What is 1024 / 32?", "a": "32"},
    {"q": "What is the next prime after 29?", "a": "31"},
    {"q": "What is 13 × 13?", "a": "169"},
]


# ═══ Components ═══

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


# ─── 3-Class Task Classifier ───

class TaskClassifier3(nn.Module):
    """3-class classifier: 0=Factual, 1=Creative, 2=Math/Logic"""
    def __init__(self, input_dim=8, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3))  # 3 classes
    def forward(self, features):
        return self.net(features)  # raw logits
    def predict_probs(self, features):
        logits = self.forward(features)
        return torch.softmax(logits, dim=-1)

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

def pretrain_classifier_3class(classifier, model, tokenizer, hook,
                                factual_prompts, creative_prompts, math_prompts, epochs=50):
    print("\n🎓 Pre-training 3-Class Classifier...")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    all_features, all_labels = [], []
    for p in factual_prompts[:20]:
        all_features.append(extract_prompt_features(model, tokenizer, p, hook))
        all_labels.append(0)
    for p in creative_prompts[:20]:
        all_features.append(extract_prompt_features(model, tokenizer, p, hook))
        all_labels.append(1)
    for p in math_prompts[:20]:
        all_features.append(extract_prompt_features(model, tokenizer, p, hook))
        all_labels.append(2)
    X = torch.stack(all_features)
    Y = torch.tensor(all_labels, dtype=torch.long)
    for epoch in range(epochs):
        logits = classifier(X)
        loss = loss_fn(logits, Y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    preds = torch.argmax(classifier(X), dim=1)
    acc = (preds == Y).float().mean().item()
    print(f"  ✅ 3-Class Classifier: acc={acc*100:.1f}%")
    # Per-class accuracy
    for c, name in [(0, "Factual"), (1, "Creative"), (2, "Math")]:
        mask = (Y == c)
        ca = (preds[mask] == Y[mask]).float().mean().item() if mask.sum() > 0 else 0
        print(f"     {name}: {ca*100:.0f}%")
    return classifier


# ─── PPO Actor/Critic ───

class DualModeCfCActor(nn.Module):
    def __init__(self, input_size=6, num_neurons=16):
        """input_size=6: [prev_reward, prev_error, hidden_norm, step_frac, p_creative, p_math]"""
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
    def __init__(self, input_size=6, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1))
    def forward(self, features):
        return self.net(features).squeeze(-1)


# ─── Helpers ───

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

def check_math_answer(response, correct_answer):
    """Check if the correct answer appears in the response."""
    # Normalize
    resp = response.strip().replace(",", "").replace(" ", "")
    ans = correct_answer.strip().replace(",", "").replace(" ", "")
    # Direct match
    if ans in resp:
        return True
    # Check if number appears in response
    import re
    numbers = re.findall(r'\d+', resp)
    return ans in numbers

def compute_gae(rewards, values, gamma=PPO_GAMMA, lam=PPO_GAE_LAMBDA):
    advantages, gae = [], 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:len(advantages)])]
    return advantages, returns


# ─── Build 3-class dataset ───

def build_3class_dataset(tokenizer):
    from datasets import load_dataset
    print(f"\n📂 Building 3-class dataset...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), 20)
    items = []
    for idx in indices:
        row = ds[idx]
        items.append({"type": "factual", "prompt": f"Q: {row['question']}\nA:",
                      "question": row["question"], "mc1_targets": row["mc1_targets"],
                      "correct_answer": None})
    for prompt in CREATIVE_PROMPTS:
        items.append({"type": "creative",
                      "prompt": f"Creative writing prompt: {prompt}\n\nResponse:",
                      "question": prompt, "mc1_targets": None, "correct_answer": None})
    for mp in MATH_PROMPTS:
        items.append({"type": "math",
                      "prompt": f"Solve this math problem. Give only the numerical answer.\n\nQ: {mp['q']}\nA:",
                      "question": mp["q"], "mc1_targets": None,
                      "correct_answer": mp["a"]})
    random.seed(SEED + 1); random.shuffle(items)
    counts = {}
    for item in items:
        counts[item["type"]] = counts.get(item["type"], 0) + 1
    print(f"  Dataset: {counts}")
    return items


# ═══ PPO 3-Class Evaluation ═══

def evaluate_ppo_3class(model, tokenizer, dataset, actor, critic, classifier, hook, label=""):
    print(f"\n{'═'*60}\n  {label}: PPO 3-Class Dual-Mode (n={len(dataset)})\n{'═'*60}")
    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]
    actor.train(); critic.train()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)
    hx = None

    # Per-class tracking
    stats = {"factual": {"correct": 0, "total": 0, "sigmas": []},
             "creative": {"novelties": [], "grammar": 0, "total": 0, "sigmas": []},
             "math": {"correct": 0, "total": 0, "sigmas": []}}
    sigma_trajectory = []
    hidden_states = []  # For UMAP later
    buf_f, buf_s, buf_lp, buf_v, buf_r, buf_e = [], [], [], [], [], []

    try:
        for epoch in range(N_EPOCHS):
            print(f"\n  🧠 Epoch {epoch+1}/{N_EPOCHS}")
            for idx, item in enumerate(dataset):
                pf = extract_prompt_features(model, tokenizer, item["question"], hook)
                with torch.no_grad():
                    probs = classifier.predict_probs(pf.unsqueeze(0)).squeeze()
                    # probs = [p_factual, p_creative, p_math]

                p_factual = probs[0].item()
                p_creative = probs[1].item()
                p_math = probs[2].item()

                # Target σ = weighted average of 3 targets
                sigma_target = (p_factual * SIGMA_FACTUAL +
                                p_creative * SIGMA_CREATIVE +
                                p_math * SIGMA_MATH)

                hidden_norm = hook.last_hidden_norm if hook.last_hidden_norm else 0.5
                features = torch.tensor(
                    [0.0, 0.5, hidden_norm, idx / len(dataset), p_creative, p_math],
                    dtype=torch.float32)

                sigma, log_prob, entropy, hx_new = actor.get_sigma_and_logprob(features, hx)
                hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()
                current_sigma = sigma.detach().item()
                hook.update_sigma(current_sigma)

                # Record hidden state for UMAP
                if isinstance(hx, tuple):
                    hs_np = hx[0].detach().cpu().numpy().flatten()
                else:
                    hs_np = hx.detach().cpu().numpy().flatten()
                hidden_states.append({
                    "type": item["type"], "sigma": current_sigma,
                    "epoch": epoch + 1, "step": epoch * len(dataset) + idx,
                    "hidden": hs_np.tolist()[:16],
                })

                with torch.no_grad():
                    value = critic(features).item()

                # Evaluate per task type
                if item["type"] == "factual":
                    ch = item["mc1_targets"]; ci = ch["labels"].index(1) if 1 in ch["labels"] else 0
                    lps = [compute_completion_logprob(model, tokenizer, item["prompt"], " " + c)
                           for c in ch["choices"]]
                    correct = (np.argmax(lps) == ci)
                    if correct: stats["factual"]["correct"] += 1
                    stats["factual"]["total"] += 1
                    stats["factual"]["sigmas"].append(current_sigma)
                    reward = (1.0 if correct else 0.0) - LAMBDA_QUAD * (current_sigma - sigma_target) ** 2
                    ld = max(lps) - min(lps)
                    features = torch.tensor(
                        [ld, 1.0 - float(correct), hidden_norm, idx / len(dataset), p_creative, p_math],
                        dtype=torch.float32)

                elif item["type"] == "creative":
                    resp = generate_text(model, tokenizer, item["prompt"])
                    nov = compute_novelty(resp); gram = is_grammatical(resp)
                    stats["creative"]["novelties"].append(nov)
                    if gram: stats["creative"]["grammar"] += 1
                    stats["creative"]["total"] += 1
                    stats["creative"]["sigmas"].append(current_sigma)
                    reward = (nov if gram else -0.5) - LAMBDA_QUAD * (current_sigma - sigma_target) ** 2

                elif item["type"] == "math":
                    resp = generate_text(model, tokenizer, item["prompt"], max_new_tokens=50)
                    correct = check_math_answer(resp, item["correct_answer"])
                    if correct: stats["math"]["correct"] += 1
                    stats["math"]["total"] += 1
                    stats["math"]["sigmas"].append(current_sigma)
                    reward = (1.5 if correct else -0.5) - LAMBDA_QUAD * (current_sigma - sigma_target) ** 2

                buf_f.append(features.detach()); buf_s.append(sigma.detach())
                buf_lp.append(log_prob.detach()); buf_v.append(value)
                buf_r.append(reward); buf_e.append(entropy.detach())

                sigma_trajectory.append({
                    "epoch": epoch + 1, "step": epoch * len(dataset) + idx,
                    "sigma": round(current_sigma, 5), "sigma_target": round(sigma_target, 4),
                    "type": item["type"],
                })

                # PPO update
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

                if (idx + 1) % 20 == 0:
                    msf = np.mean(stats["factual"]["sigmas"][-10:]) if stats["factual"]["sigmas"] else 0
                    msc = np.mean(stats["creative"]["sigmas"][-10:]) if stats["creative"]["sigmas"] else 0
                    msm = np.mean(stats["math"]["sigmas"][-10:]) if stats["math"]["sigmas"] else 0
                    print(f"  [{idx+1}/{len(dataset)}] σ_f={msf:.4f} σ_c={msc:.4f} σ_m={msm:.4f}")
    finally:
        for h in handles: h.remove()

    # Compute final stats
    fa = stats["factual"]["correct"] / max(stats["factual"]["total"], 1) * 100
    an = np.mean(stats["creative"]["novelties"]) if stats["creative"]["novelties"] else 0
    gr = stats["creative"]["grammar"] / max(stats["creative"]["total"], 1) * 100
    ma = stats["math"]["correct"] / max(stats["math"]["total"], 1) * 100

    msf = np.mean(stats["factual"]["sigmas"]) if stats["factual"]["sigmas"] else 0
    msc = np.mean(stats["creative"]["sigmas"]) if stats["creative"]["sigmas"] else 0
    msm = np.mean(stats["math"]["sigmas"]) if stats["math"]["sigmas"] else 0

    sep_fc = abs(msc - msf)
    sep_fm = abs(msm - msf)
    sep_cm = abs(msm - msc)
    max_sep = max(sep_fc, sep_fm, sep_cm)

    mid = len(sigma_trajectory) // 2
    drift = abs(np.mean([s["sigma"] for s in sigma_trajectory[mid:]]) -
                np.mean([s["sigma"] for s in sigma_trajectory[:mid]]))

    unified = max_sep < 0.01

    print(f"\n  📊 3-Class Results:")
    print(f"     Factual: acc={fa:.1f}% σ̄={msf:.4f} (target {SIGMA_FACTUAL})")
    print(f"     Creative: nov={an:.3f} gram={gr:.0f}% σ̄={msc:.4f} (target {SIGMA_CREATIVE})")
    print(f"     Math:    acc={ma:.1f}% σ̄={msm:.4f} (target {SIGMA_MATH})")
    print(f"     Separations: F-C={sep_fc:.4f} F-M={sep_fm:.4f} C-M={sep_cm:.4f}")
    print(f"     Max sep={max_sep:.4f} drift={drift:.4f}")
    print(f"     {'✅ UNIFIED REGIME (Universal)' if unified else '🔀 SEPARATED (Task-Dependent)'}")

    return {
        "condition": label,
        "factual": {"acc": round(fa, 2), "correct": stats["factual"]["correct"],
                    "total": stats["factual"]["total"], "mean_sigma": round(msf, 5),
                    "sigmas": [round(s, 5) for s in stats["factual"]["sigmas"]]},
        "creative": {"novelty": round(float(an), 4), "grammar_rate": round(gr, 1),
                     "total": stats["creative"]["total"], "mean_sigma": round(msc, 5),
                     "sigmas": [round(s, 5) for s in stats["creative"]["sigmas"]]},
        "math": {"acc": round(ma, 2), "correct": stats["math"]["correct"],
                 "total": stats["math"]["total"], "mean_sigma": round(msm, 5),
                 "sigmas": [round(s, 5) for s in stats["math"]["sigmas"]]},
        "separations": {"f_c": round(sep_fc, 5), "f_m": round(sep_fm, 5),
                        "c_m": round(sep_cm, 5), "max": round(max_sep, 5)},
        "sigma_drift": round(drift, 5),
        "unified_regime": unified,
        "sigma_trajectory": sigma_trajectory,
        "hidden_states": hidden_states,
    }


# ═══ Visualization ═══

def visualize_3class(result):
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
    fig.suptitle("Phase 26: 3-Class Extension — Does CfC Unify Even Math/Logic?\n"
                 "Factual (σ*=0.046) + Creative (σ*=0.080) + Math (σ*=0.015)",
                 fontsize=14, fontweight="bold", color="#e94560")

    # Panel 1: σ by task type (box plot style)
    ax1 = axes[0, 0]
    task_colors = {"factual": "#4FC3F7", "creative": "#FF7043", "math": "#AB47BC"}
    task_targets = {"factual": SIGMA_FACTUAL, "creative": SIGMA_CREATIVE, "math": SIGMA_MATH}
    positions = {"factual": 0, "creative": 1, "math": 2}
    for ttype in ["factual", "creative", "math"]:
        sigmas = result[ttype]["sigmas"]
        if sigmas:
            pos = positions[ttype]
            bp = ax1.boxplot([sigmas], positions=[pos], widths=0.5,
                            patch_artist=True, showfliers=True)
            for patch in bp['boxes']:
                patch.set_facecolor(task_colors[ttype])
                patch.set_alpha(0.6)
            for element in ['whiskers', 'caps', 'medians']:
                for line in bp[element]:
                    line.set_color("#eee")
            ax1.axhline(y=task_targets[ttype], color=task_colors[ttype],
                       linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["Factual\n(target=0.046)", "Creative\n(target=0.080)", "Math\n(target=0.015)"])
    ax1.set_title("σ Distribution by Task Type", fontweight="bold")
    ax1.set_ylabel("σ"); ax1.grid(True, axis="y")
    ax1.text(0.02, 0.95, f"Unified: {'YES' if result['unified_regime'] else 'NO'}",
             transform=ax1.transAxes, fontsize=12, fontweight="bold",
             color="#66BB6A" if result["unified_regime"] else "#E94560",
             verticalalignment="top")

    # Panel 2: σ trajectory by task type
    ax2 = axes[0, 1]
    traj = result["sigma_trajectory"]
    for ttype in ["factual", "creative", "math"]:
        steps = [t["step"] for t in traj if t["type"] == ttype]
        sigmas = [t["sigma"] for t in traj if t["type"] == ttype]
        ax2.scatter(steps, sigmas, c=task_colors[ttype], s=15, alpha=0.6, label=ttype)
    for ttype in ["factual", "creative", "math"]:
        ax2.axhline(y=task_targets[ttype], color=task_colors[ttype], linestyle="--", alpha=0.3)
    ax2.legend(fontsize=9, facecolor="#16213e", edgecolor="#555")
    ax2.set_title(f"σ Trajectory (drift={result['sigma_drift']:.4f})", fontweight="bold")
    ax2.set_xlabel("Step"); ax2.set_ylabel("σ"); ax2.grid(True)

    # Panel 3: Performance comparison
    ax3 = axes[1, 0]
    metrics = [result["factual"]["acc"], result["math"]["acc"]]
    labels = [f"Factual\n{result['factual']['acc']}%", f"Math\n{result['math']['acc']}%"]
    colors_bar = ["#4FC3F7", "#AB47BC"]
    bars = ax3.bar(range(2), metrics, color=colors_bar, edgecolor="#333", width=0.5)
    ax3.set_xticks(range(2)); ax3.set_xticklabels(labels, fontsize=10)
    ax3.set_title("Accuracy by Task Type", fontweight="bold")
    ax3.set_ylabel("Accuracy (%)"); ax3.grid(True, axis="y")
    # Add creative metrics as text
    ax3.text(0.95, 0.95,
             f"Creative:\nnov={result['creative']['novelty']:.3f}\ngram={result['creative']['grammar_rate']:.0f}%",
             transform=ax3.transAxes, fontsize=10, verticalalignment="top", ha="right",
             color="#FF7043", bbox=dict(facecolor="#16213e", edgecolor="#FF7043", alpha=0.8))

    # Panel 4: Verdict
    ax4 = axes[1, 1]
    ax4.axis("off")
    msf = result["factual"]["mean_sigma"]
    msc = result["creative"]["mean_sigma"]
    msm = result["math"]["mean_sigma"]
    if result["unified_regime"]:
        verdict = "SCENARIO A: Universal Noise Constant"
        verdict_color = "#66BB6A"
        interp = ("CfC maintains unified σ even with Math tasks.\n"
                   "σ is a 'body temperature' — constant across\n"
                   "all task types. Differentiation happens\n"
                   "entirely in the 16D hidden state.\n\n"
                   "This is a UNIVERSAL property of CfC.")
    else:
        verdict = "SCENARIO B: Task-Dependent Adaptation"
        verdict_color = "#FFA726"
        interp = ("CfC adapts σ per task type!\n"
                   "Math tasks trigger lower σ for precision.\n"
                   "This is TRUE task-dependent homeostasis:\n"
                   "CfC has multiple 'operating temperatures'\n"
                   "for fundamentally different task demands.")

    txt = (f"VERDICT: {verdict}\n\n"
           f"σ̄_factual  = {msf:.4f} (target {SIGMA_FACTUAL})\n"
           f"σ̄_creative = {msc:.4f} (target {SIGMA_CREATIVE})\n"
           f"σ̄_math     = {msm:.4f} (target {SIGMA_MATH})\n\n"
           f"Separations:\n"
           f"  F-C = {result['separations']['f_c']:.4f}\n"
           f"  F-M = {result['separations']['f_m']:.4f}\n"
           f"  C-M = {result['separations']['c_m']:.4f}\n\n"
           f"{interp}")

    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             color=verdict_color,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", edgecolor=verdict_color, alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIGURES_DIR, "phase26_3class_extension.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  📊 Figure: {out}")

    # Also try UMAP on hidden states
    try:
        import umap
        from sklearn.decomposition import PCA
        print("\n  🔬 Computing UMAP on 3-class hidden states...")
        hs_data = [h["hidden"] for h in result["hidden_states"]]
        hs_types = [h["type"] for h in result["hidden_states"]]
        X = np.array(hs_data)
        if len(X) > 10:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
            X_umap = reducer.fit_transform(X)

            fig2, (ax_pca, ax_umap) = plt.subplots(1, 2, figsize=(16, 7))
            fig2.patch.set_facecolor("#1a1a2e")
            for ax in [ax_pca, ax_umap]:
                ax.set_facecolor("#16213e")
            fig2.suptitle("Phase 26: 3-Class Hidden State Visualization",
                         fontsize=14, fontweight="bold", color="#e94560")

            for ttype, color in task_colors.items():
                mask = [t == ttype for t in hs_types]
                if any(mask):
                    idx_mask = np.array(mask)
                    ax_pca.scatter(X_pca[idx_mask, 0], X_pca[idx_mask, 1],
                                  c=color, s=30, alpha=0.6, label=ttype)
                    ax_umap.scatter(X_umap[idx_mask, 0], X_umap[idx_mask, 1],
                                   c=color, s=30, alpha=0.6, label=ttype)
            ax_pca.set_title(f"PCA (var={pca.explained_variance_ratio_.sum()*100:.1f}%)", fontweight="bold")
            ax_umap.set_title("UMAP", fontweight="bold")
            for ax in [ax_pca, ax_umap]:
                ax.legend(fontsize=9, facecolor="#16213e", edgecolor="#555")
                ax.grid(True, alpha=0.3)

            # Compute separation ratios
            centroids = {}
            for ttype in ["factual", "creative", "math"]:
                mask = np.array([t == ttype for t in hs_types])
                if mask.sum() > 0:
                    centroids[ttype] = X_umap[mask].mean(axis=0)
            if len(centroids) == 3:
                d_fc = np.linalg.norm(centroids["factual"] - centroids["creative"])
                d_fm = np.linalg.norm(centroids["factual"] - centroids["math"])
                d_cm = np.linalg.norm(centroids["creative"] - centroids["math"])
                avg_within = np.mean([
                    np.mean([np.linalg.norm(X_umap[i] - centroids[hs_types[i]])
                             for i in range(len(hs_types)) if hs_types[i] == t])
                    for t in ["factual", "creative", "math"]
                    if sum(1 for x in hs_types if x == t) > 0
                ])
                avg_between = np.mean([d_fc, d_fm, d_cm])
                sep_ratio = avg_between / max(avg_within, 1e-10)
                ax_umap.text(0.02, 0.98,
                            f"3-class sep ratio: {sep_ratio:.2f}\nF-C: {d_fc:.2f} F-M: {d_fm:.2f} C-M: {d_cm:.2f}",
                            transform=ax_umap.transAxes, fontsize=9, verticalalignment="top",
                            color="#66BB6A", fontfamily="monospace")

            plt.tight_layout(rect=[0, 0, 1, 0.92])
            umap_out = os.path.join(FIGURES_DIR, "phase26_3class_umap.png")
            plt.savefig(umap_out, dpi=150, bbox_inches="tight"); plt.close()
            print(f"  📊 UMAP figure: {umap_out}")
    except ImportError:
        print("  ⚠️ UMAP not available, skipping hidden state visualization")
    except Exception as e:
        print(f"  ⚠️ UMAP error: {e}")

    return out


# ═══ MAIN ═══

def main():
    print("=" * 60)
    print("Phase 26: 3-Class Extension (Factual + Creative + Math)")
    print("  σ*_factual=0.046  σ*_creative=0.080  σ*_math=0.015")
    print("=" * 60)
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model()
    hook = AdaptiveSNNHook(sigma=0.05)
    dataset = build_3class_dataset(tokenizer)

    factual_qs = [item["question"] for item in dataset if item["type"] == "factual"]
    math_qs = [item["question"] for item in dataset if item["type"] == "math"]
    classifier = TaskClassifier3(input_dim=8, hidden_dim=32)
    classifier = pretrain_classifier_3class(
        classifier, model, tokenizer, hook,
        factual_qs, CREATIVE_PROMPTS, math_qs)

    actor = DualModeCfCActor(input_size=6, num_neurons=16)
    critic = CfCCritic(input_size=6, hidden_size=64)
    result = evaluate_ppo_3class(model, tokenizer, dataset, actor, critic,
                                  classifier, hook, "PPO 3-Class")

    fig_path = visualize_3class(result)

    elapsed = time.time() - t_start
    # Don't save hidden_states to JSON (too large)
    result_save = {k: v for k, v in result.items() if k != "hidden_states"}
    output = {
        "phase": "Phase 26: 3-Class Extension",
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "sigma_targets": {"factual": SIGMA_FACTUAL, "creative": SIGMA_CREATIVE, "math": SIGMA_MATH},
        "result": result_save,
        "figure_path": fig_path,
    }
    out = os.path.join(RESULTS_DIR, "phase26_3class_extension_log.json")
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Results: {out}")

    print(f"\n{'='*60}")
    print(f"  Phase 26 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  Factual:  acc={result['factual']['acc']:.1f}% σ̄={result['factual']['mean_sigma']:.4f}")
    print(f"  Creative: nov={result['creative']['novelty']:.3f} σ̄={result['creative']['mean_sigma']:.4f}")
    print(f"  Math:     acc={result['math']['acc']:.1f}% σ̄={result['math']['mean_sigma']:.4f}")
    print(f"  Max separation: {result['separations']['max']:.4f}")
    print(f"  {'✅ UNIFIED (Universal)' if result['unified_regime'] else '🔀 SEPARATED (Adaptive)'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
