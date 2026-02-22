"""
Phase 37: Brain Transplant — Can CfC Homeostasis Transfer Across Architectures?
================================================================================

Phase 36 showed each architecture has a unique "cognitive style" (entropy ordering).
This experiment tests: Can a CfC trained on one model's cognitive style be
transplanted into a different model and still maintain homeostasis?

Experiment Design:
  1. Train CfC on Qwen (DONOR) → save weights
  2. Free Qwen, load Mistral (HOST)
  3. Train CfC on Mistral natively → evaluate (baseline)
  4. Load Qwen's CfC into Mistral → evaluate (transplant, NO training)
  5. Compare: native vs transplanted

Key Questions:
  - Does the transplanted CfC maintain σ≈0.07 on a foreign body?
  - Does Mistral's Math accuracy change with Qwen's "brain"?
  - Is the homeostatic strategy universal or architecture-specific?

Usage:
    python experiments/phase37_brain_transplant.py
"""

import torch, torch.nn as nn
import os, sys, json, gc, time, datetime, random, math, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# ═══ Configuration ═══
DONOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DONOR_SHORT = "Qwen2.5-7B"
DONOR_LAYERS = list(range(14, 20))

HOST_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
HOST_SHORT = "Mistral-7B"
HOST_LAYERS = list(range(15, 21))

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
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ─── Prompts ───
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

def load_model(model_name, short_name):
    print(f"\n📦 Loading {model_name}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    layers = get_layers(model)
    print(f"  ✅ {short_name} loaded: {len(layers)} layers")
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


class TaskClassifier3(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3))
    def forward(self, features):
        return self.net(features)
    def predict_probs(self, features):
        return torch.softmax(self.forward(features), dim=-1)


class DualModeCfCActor(nn.Module):
    def __init__(self, input_size=6, num_neurons=16):
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

def extract_prompt_features(model, tokenizer, prompt, hook):
    text = prompt if len(prompt) < 200 else prompt[:200]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    feat_layer = min(16, len(get_layers(model)) - 1)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[feat_layer]
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

def pretrain_classifier(classifier, model, tokenizer, hook,
                        factual_prompts, creative_prompts, math_prompts, epochs=50):
    print("  🎓 Pre-training classifier...")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    all_features, all_labels = [], []
    for p in factual_prompts[:20]:
        all_features.append(extract_prompt_features(model, tokenizer, p, hook)); all_labels.append(0)
    for p in creative_prompts[:20]:
        all_features.append(extract_prompt_features(model, tokenizer, p, hook)); all_labels.append(1)
    for p in math_prompts[:20]:
        all_features.append(extract_prompt_features(model, tokenizer, p, hook)); all_labels.append(2)
    X = torch.stack(all_features); Y = torch.tensor(all_labels, dtype=torch.long)
    for _ in range(epochs):
        logits = classifier(X)
        loss = loss_fn(logits, Y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    acc = (torch.argmax(classifier(X), dim=1) == Y).float().mean().item()
    print(f"  ✅ Classifier acc={acc*100:.1f}%")
    return classifier

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
    resp = response.strip().replace(",", "").replace(" ", "")
    ans = correct_answer.strip().replace(",", "").replace(" ", "")
    if ans in resp: return True
    return ans in re.findall(r'\d+', resp)

def compute_gae(rewards, values, gamma=PPO_GAMMA, lam=PPO_GAE_LAMBDA):
    advantages, gae = [], 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:len(advantages)])]
    return advantages, returns


# ─── Build dataset ───

def build_dataset(tokenizer):
    from datasets import load_dataset
    print(f"  📂 Building dataset...")
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
                      "question": mp["q"], "mc1_targets": None, "correct_answer": mp["a"]})
    random.seed(SEED + 1); random.shuffle(items)
    print(f"  Dataset: {len(items)} items")
    return items


# ═══ PPO Train + Evaluate (combined) ═══

def run_ppo(model, tokenizer, dataset, actor, critic, classifier, hook,
            target_layers, label="", do_train=True):
    """Run PPO loop. If do_train=False, just evaluate (transplant mode)."""
    mode_str = "TRAIN+EVAL" if do_train else "EVAL ONLY (transplanted)"
    print(f"\n{'═'*60}")
    print(f"  🧠 {label} [{mode_str}]")
    print(f"{'═'*60}")

    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in target_layers if i < len(layers)]

    if do_train:
        actor.train(); critic.train()
        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)
    else:
        actor.eval()

    hx = None
    stats = {"factual": {"correct": 0, "total": 0, "sigmas": []},
             "creative": {"novelties": [], "grammar": 0, "total": 0, "sigmas": []},
             "math": {"correct": 0, "total": 0, "sigmas": []}}
    sigma_trajectory = []
    buf_f, buf_s, buf_lp, buf_v, buf_r, buf_e = [], [], [], [], [], []

    n_epochs = N_EPOCHS if do_train else 1

    try:
        for epoch in range(n_epochs):
            print(f"\n  {'🧠' if do_train else '🔬'} Epoch {epoch+1}/{n_epochs}")
            for idx, item in enumerate(dataset):
                pf = extract_prompt_features(model, tokenizer, item["question"], hook)
                with torch.no_grad():
                    probs = classifier.predict_probs(pf.unsqueeze(0)).squeeze()
                p_creative = probs[1].item()
                p_math = probs[2].item()
                sigma_target = (probs[0].item() * SIGMA_FACTUAL +
                                p_creative * SIGMA_CREATIVE +
                                p_math * SIGMA_MATH)

                hidden_norm = hook.last_hidden_norm if hook.last_hidden_norm else 0.5
                features = torch.tensor(
                    [0.0, 0.5, hidden_norm, idx / len(dataset), p_creative, p_math],
                    dtype=torch.float32)

                if do_train:
                    sigma, log_prob, entropy, hx_new = actor.get_sigma_and_logprob(features, hx)
                else:
                    with torch.no_grad():
                        sigma, log_prob, entropy, hx_new = actor.get_sigma_and_logprob(features, hx)

                hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()
                current_sigma = sigma.detach().item()
                hook.update_sigma(current_sigma)

                with torch.no_grad():
                    value = critic(features).item()

                if item["type"] == "factual":
                    ch = item["mc1_targets"]
                    ci = ch["labels"].index(1) if 1 in ch["labels"] else 0
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
                    "sigma": round(current_sigma, 5), "type": item["type"],
                })

                # PPO update (only during training)
                if do_train and len(buf_r) >= PPO_ROLLOUT_SIZE:
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
    mean_sigma = float(np.mean([msf, msc, msm]))
    max_sep = max(abs(msc - msf), abs(msm - msf), abs(msm - msc))

    print(f"\n  📊 Results ({label}):")
    print(f"     Factual:  acc={fa:.1f}% σ̄={msf:.4f}")
    print(f"     Creative: nov={an:.3f}  σ̄={msc:.4f}")
    print(f"     Math:     acc={ma:.1f}% σ̄={msm:.4f}")
    print(f"     Mean σ = {mean_sigma:.4f}, Max sep = {max_sep:.4f}")

    return {
        "condition": label, "mode": "train" if do_train else "transplant",
        "factual": {"acc": round(fa, 2), "mean_sigma": round(msf, 5)},
        "creative": {"novelty": round(float(an), 4), "grammar_rate": round(gr, 1),
                     "mean_sigma": round(msc, 5)},
        "math": {"acc": round(ma, 2), "mean_sigma": round(msm, 5)},
        "mean_sigma_all": round(mean_sigma, 5),
        "max_separation": round(max_sep, 5),
        "sigma_trajectory": sigma_trajectory,
    }


# ═══ Visualization ═══

def visualize_transplant(donor_result, native_result, transplant_result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Phase 37: Brain Transplant — Qwen CfC → Mistral Body\n"
                 "Can homeostatic strategy transfer across architectures?",
                 fontsize=14, fontweight="bold")

    colors = {"factual": "#2ecc71", "creative": "#e74c3c", "math": "#3498db"}

    # Panel 1: σ comparison bar chart
    ax1 = axes[0, 0]
    conditions = ["Qwen\n(donor)", "Mistral\n(native)", "Mistral\n(transplanted)"]
    results_list = [donor_result, native_result, transplant_result]
    x = np.arange(len(conditions))
    width = 0.25
    for i, (task, color) in enumerate(colors.items()):
        means = [r[task]["mean_sigma"] for r in results_list]
        ax1.bar(x + i * width, means, width, label=task, color=color, alpha=0.7)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(conditions)
    ax1.set_ylabel("Mean σ")
    ax1.set_title("σ by Task & Condition")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.axhline(y=0.074, color="gold", linestyle="--", alpha=0.7, label="σ≈0.074")

    # Panel 2: Accuracy comparison
    ax2 = axes[0, 1]
    fa_vals = [r["factual"]["acc"] for r in results_list]
    ma_vals = [r["math"]["acc"] for r in results_list]
    x2 = np.arange(len(conditions))
    ax2.bar(x2 - 0.15, fa_vals, 0.3, label="Factual Acc", color="#2ecc71", alpha=0.7)
    ax2.bar(x2 + 0.15, ma_vals, 0.3, label="Math Acc", color="#3498db", alpha=0.7)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(conditions)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy by Condition")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: σ trajectory comparison (native vs transplant)
    ax3 = axes[1, 0]
    for ttype, color in colors.items():
        nat_steps = [t["step"] for t in native_result["sigma_trajectory"] if t["type"] == ttype]
        nat_sigmas = [t["sigma"] for t in native_result["sigma_trajectory"] if t["type"] == ttype]
        ax3.scatter(nat_steps, nat_sigmas, c=color, s=15, alpha=0.4, marker="o")

        trans_steps = [t["step"] for t in transplant_result["sigma_trajectory"] if t["type"] == ttype]
        trans_sigmas = [t["sigma"] for t in transplant_result["sigma_trajectory"] if t["type"] == ttype]
        ax3.scatter(trans_steps, trans_sigmas, c=color, s=30, alpha=0.8, marker="^", label=f"{ttype} (transplant)")

    ax3.axhline(y=0.074, color="gold", linestyle="--", alpha=0.5, linewidth=2)
    ax3.legend(fontsize=8)
    ax3.set_title("σ Trajectory: ○=native, △=transplanted")
    ax3.set_xlabel("Step"); ax3.set_ylabel("σ"); ax3.grid(True, alpha=0.3)

    # Panel 4: Verdict
    ax4 = axes[1, 1]; ax4.axis("off")
    nat_sigma = native_result["mean_sigma_all"]
    trans_sigma = transplant_result["mean_sigma_all"]
    donor_sigma = donor_result["mean_sigma_all"]
    delta_nt = abs(nat_sigma - trans_sigma)
    delta_dt = abs(donor_sigma - trans_sigma)

    # Determine if transplant preserved donor behavior
    preserved = delta_dt < delta_nt  # closer to donor than to native

    nat_math = native_result["math"]["acc"]
    trans_math = transplant_result["math"]["acc"]
    donor_math = donor_result["math"]["acc"]

    txt = (f"BRAIN TRANSPLANT RESULTS\n"
           f"{'='*40}\n\n"
           f"DONOR (Qwen CfC on Qwen):\n"
           f"  σ̄={donor_sigma:.4f}  Math={donor_math:.1f}%\n\n"
           f"HOST NATIVE (Mistral CfC on Mistral):\n"
           f"  σ̄={nat_sigma:.4f}  Math={nat_math:.1f}%\n\n"
           f"TRANSPLANT (Qwen CfC on Mistral):\n"
           f"  σ̄={trans_sigma:.4f}  Math={trans_math:.1f}%\n\n"
           f"{'='*40}\n"
           f"Δ(native vs transplant) = {delta_nt:.4f}\n"
           f"Δ(donor vs transplant) = {delta_dt:.4f}\n\n")

    if preserved:
        txt += ("🧟 DONOR PERSONALITY PRESERVED!\n"
                "Transplanted CfC retains donor's σ behavior\n"
                "→ Homeostatic strategy IS transferable!")
        verdict_color = "#2ecc71"
    else:
        txt += ("🔬 HOST ADAPTATION!\n"
                "CfC adapts to host's body (σ shifted toward native)\n"
                "→ Homeostatic strategy is architecture-specific!")
        verdict_color = "#e74c3c"

    if trans_math != nat_math:
        delta_math = trans_math - nat_math
        txt += f"\n\n📊 Math Accuracy Change: {delta_math:+.1f}%"
        if delta_math > 5:
            txt += "\n🎆 TRANSPLANT BOOSTED PERFORMANCE!"
        elif delta_math < -5:
            txt += "\n⚠️ Transplant degraded performance"

    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             color=verdict_color,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase37_brain_transplant.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


# ═══ MAIN ═══

def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    # ─── Step 1: Train CfC on DONOR (Qwen) ───
    print("\n" + "🧬"*30)
    print("  STEP 1: Training CfC on DONOR (Qwen)")
    print("🧬"*30)

    model, tokenizer = load_model(DONOR_MODEL, DONOR_SHORT)
    hook = AdaptiveSNNHook(sigma=0.05)
    dataset = build_dataset(tokenizer)

    factual_qs = [item["question"] for item in dataset if item["type"] == "factual"]
    math_qs = [item["question"] for item in dataset if item["type"] == "math"]

    donor_classifier = TaskClassifier3()
    donor_classifier = pretrain_classifier(
        donor_classifier, model, tokenizer, hook,
        factual_qs, CREATIVE_PROMPTS, math_qs)

    donor_actor = DualModeCfCActor(input_size=6, num_neurons=16)
    donor_critic = CfCCritic(input_size=6, hidden_size=64)

    donor_result = run_ppo(model, tokenizer, dataset, donor_actor, donor_critic,
                           donor_classifier, hook, DONOR_LAYERS,
                           f"DONOR: {DONOR_SHORT}", do_train=True)

    # Save donor CfC weights
    donor_weights_path = os.path.join(WEIGHTS_DIR, "donor_qwen_cfc.pt")
    torch.save({
        "actor_state_dict": donor_actor.state_dict(),
        "critic_state_dict": donor_critic.state_dict(),
    }, donor_weights_path)
    print(f"\n  💾 Donor CfC weights saved: {donor_weights_path}")

    # Free donor model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("  🗑️ Qwen unloaded")

    # ─── Step 2: Load HOST (Mistral) ───
    print("\n" + "🏥"*30)
    print("  STEP 2: Training NATIVE CfC on HOST (Mistral)")
    print("🏥"*30)

    model, tokenizer = load_model(HOST_MODEL, HOST_SHORT)
    hook = AdaptiveSNNHook(sigma=0.05)
    dataset = build_dataset(tokenizer)  # Rebuild with same seed

    factual_qs = [item["question"] for item in dataset if item["type"] == "factual"]
    math_qs = [item["question"] for item in dataset if item["type"] == "math"]

    host_classifier = TaskClassifier3()
    host_classifier = pretrain_classifier(
        host_classifier, model, tokenizer, hook,
        factual_qs, CREATIVE_PROMPTS, math_qs)

    # Train native CfC on Mistral
    native_actor = DualModeCfCActor(input_size=6, num_neurons=16)
    native_critic = CfCCritic(input_size=6, hidden_size=64)

    native_result = run_ppo(model, tokenizer, dataset, native_actor, native_critic,
                            host_classifier, hook, HOST_LAYERS,
                            f"NATIVE: {HOST_SHORT}", do_train=True)

    # ─── Step 3: Transplant Qwen CfC into Mistral ───
    print("\n" + "🧟"*30)
    print("  STEP 3: TRANSPLANTING Qwen CfC into Mistral body!")
    print("🧟"*30)

    # Load donor weights
    transplant_actor = DualModeCfCActor(input_size=6, num_neurons=16)
    transplant_critic = CfCCritic(input_size=6, hidden_size=64)
    checkpoint = torch.load(donor_weights_path, weights_only=True)
    transplant_actor.load_state_dict(checkpoint["actor_state_dict"])
    transplant_critic.load_state_dict(checkpoint["critic_state_dict"])
    print(f"  🧟 Donor CfC weights loaded into transplant actor!")

    # Reset hook
    hook = AdaptiveSNNHook(sigma=0.05)

    # Evaluate transplanted CfC on Mistral (NO training!)
    transplant_result = run_ppo(model, tokenizer, dataset, transplant_actor, transplant_critic,
                                host_classifier, hook, HOST_LAYERS,
                                f"TRANSPLANT: Qwen CfC → {HOST_SHORT}", do_train=False)

    # ─── Visualization ───
    fig_path = visualize_transplant(donor_result, native_result, transplant_result)

    # ─── Save results ───
    elapsed = time.time() - t_start
    output = {
        "experiment": "Phase 37: Brain Transplant",
        "donor": DONOR_SHORT, "host": HOST_SHORT,
        "elapsed_minutes": round(elapsed / 60, 1),
        "donor_result": {k: v for k, v in donor_result.items() if k != "sigma_trajectory"},
        "native_result": {k: v for k, v in native_result.items() if k != "sigma_trajectory"},
        "transplant_result": {k: v for k, v in transplant_result.items() if k != "sigma_trajectory"},
        "comparison": {
            "native_mean_sigma": native_result["mean_sigma_all"],
            "transplant_mean_sigma": transplant_result["mean_sigma_all"],
            "donor_mean_sigma": donor_result["mean_sigma_all"],
            "delta_native_transplant": round(abs(native_result["mean_sigma_all"] - transplant_result["mean_sigma_all"]), 5),
            "delta_donor_transplant": round(abs(donor_result["mean_sigma_all"] - transplant_result["mean_sigma_all"]), 5),
            "native_math_acc": native_result["math"]["acc"],
            "transplant_math_acc": transplant_result["math"]["acc"],
            "donor_math_acc": donor_result["math"]["acc"],
            "math_acc_change": round(transplant_result["math"]["acc"] - native_result["math"]["acc"], 2),
        },
        "figure_path": fig_path,
    }

    log_path = os.path.join(RESULTS_DIR, "phase37_brain_transplant_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Results: {log_path}")

    # Grand summary
    print(f"\n{'═'*60}")
    print(f"  🏆 PHASE 37: BRAIN TRANSPLANT — GRAND SUMMARY")
    print(f"{'═'*60}")
    print(f"  DONOR (Qwen CfC on Qwen):")
    print(f"    σ̄={donor_result['mean_sigma_all']:.4f} Math={donor_result['math']['acc']:.1f}%")
    print(f"  NATIVE (Mistral CfC on Mistral):")
    print(f"    σ̄={native_result['mean_sigma_all']:.4f} Math={native_result['math']['acc']:.1f}%")
    print(f"  TRANSPLANT (Qwen CfC on Mistral):")
    print(f"    σ̄={transplant_result['mean_sigma_all']:.4f} Math={transplant_result['math']['acc']:.1f}%")
    print(f"\n  Δ(native vs transplant) = {output['comparison']['delta_native_transplant']:.4f}")
    print(f"  Δ(donor vs transplant) = {output['comparison']['delta_donor_transplant']:.4f}")
    print(f"  Math Acc Change: {output['comparison']['math_acc_change']:+.1f}%")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
