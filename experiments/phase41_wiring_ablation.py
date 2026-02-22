"""
Phase 41: Wiring Ablation — Where Does σ ≈ 0.074 Come From?
=============================================================

The homeostatic set-point σ ≈ 0.074 has been observed across:
  - Mistral-7B (Phase 25: σ=0.071)
  - Qwen2.5-7B (Phase 29: σ=0.075)

Is this determined by:
  (a) CfC architecture (neuron count, wiring)?
  (b) The Transformer itself?

Ablation: Train CfC controllers with different NCP neuron counts
(8, 16, 32, 64) on the SAME Mistral-7B and compare converged σ.

If σ is constant regardless of neuron count → Transformer determines it.
If σ scales with neuron count → CfC architecture determines it.

Usage:
    python experiments/phase41_wiring_ablation.py
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
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
TARGET_LAYERS = list(range(15, 21))

NEURON_COUNTS = [8, 16, 32, 64]

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

CREATIVE_PROMPTS = [
    "Describe a color that doesn't exist in the visible spectrum.",
    "Invent a new fundamental law of physics.",
    "Write a short poem about silence.",
    "Imagine a conversation between two neurons.",
    "Describe what music would taste like.",
    "Create a philosophical argument for backwards time.",
    "Invent a mathematical operation combining multiplication and dreaming.",
    "Describe the smell of a black hole.",
    "Write a haiku about quantum entanglement.",
    "Imagine a world where gravity works differently on Mondays.",
    "Describe the texture of a thought.",
    "Invent a new emotion that humans haven't named.",
    "Write a letter from the Sun to the Moon.",
    "Describe what zero tastes like.",
    "Imagine consciousness as a physical substance.",
    "Create a recipe for cooking starlight.",
    "Describe architecture designed by clouds.",
    "Write a business plan for selling shadows.",
    "Imagine a language where words change with listener mood.",
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
    {"q": "What is 2^8?", "a": "256"},
    {"q": "What is sqrt(144)?", "a": "12"},
    {"q": "What is 99 × 3?", "a": "297"},
    {"q": "What is 50% of 240?", "a": "120"},
    {"q": "What is 7!?", "a": "5040"},
    {"q": "What is 1024 / 32?", "a": "32"},
    {"q": "What is the next prime after 29?", "a": "31"},
    {"q": "What is 13 × 13?", "a": "169"},
    {"q": "What is 200 - 137?", "a": "63"},
    {"q": "What is 25 × 4?", "a": "100"},
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


class FlexCfCActor(nn.Module):
    """CfC Actor with configurable neuron count."""
    def __init__(self, input_size=6, num_neurons=16):
        super().__init__()
        wiring = AutoNCP(num_neurons, 1)
        self.cfc = CfC(input_size, wiring, batch_first=True)
        self.log_std = nn.Parameter(torch.tensor([-1.0]))
        self.num_neurons = num_neurons
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


def build_dataset(tokenizer):
    from datasets import load_dataset
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
    return items


# ═══ Run PPO for a specific neuron count ═══

def run_ablation(model, tokenizer, dataset, classifier, hook, num_neurons, target_layers):
    print(f"\n{'═'*60}")
    print(f"  🧠 NCP Neurons = {num_neurons} [PPO Training]")
    print(f"{'═'*60}")

    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in target_layers if i < len(layers)]

    actor = FlexCfCActor(input_size=6, num_neurons=num_neurons)
    critic = CfCCritic(input_size=6, hidden_size=64)
    actor.train(); critic.train()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in actor.parameters())
    print(f"  Parameters: {n_params}")

    hx = None
    stats = {"factual": {"correct": 0, "total": 0, "sigmas": []},
             "creative": {"novelties": [], "grammar": 0, "total": 0, "sigmas": []},
             "math": {"correct": 0, "total": 0, "sigmas": []}}
    sigma_trajectory = []
    buf_f, buf_s, buf_lp, buf_v, buf_r, buf_e = [], [], [], [], [], []

    for epoch in range(N_EPOCHS):
        print(f"\n  🧠 Epoch {epoch+1}/{N_EPOCHS}")
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

    for h in handles:
        h.remove()

    fa = stats["factual"]["correct"] / max(stats["factual"]["total"], 1) * 100
    an = np.mean(stats["creative"]["novelties"]) if stats["creative"]["novelties"] else 0
    gr = stats["creative"]["grammar"] / max(stats["creative"]["total"], 1) * 100
    ma = stats["math"]["correct"] / max(stats["math"]["total"], 1) * 100
    msf = np.mean(stats["factual"]["sigmas"]) if stats["factual"]["sigmas"] else 0
    msc = np.mean(stats["creative"]["sigmas"]) if stats["creative"]["sigmas"] else 0
    msm = np.mean(stats["math"]["sigmas"]) if stats["math"]["sigmas"] else 0
    mean_sigma = float(np.mean([msf, msc, msm]))
    max_sep = max(abs(msc - msf), abs(msm - msf), abs(msm - msc))

    # Last-epoch sigmas for convergence analysis
    last_epoch_sigmas = [t["sigma"] for t in sigma_trajectory if t["epoch"] == N_EPOCHS]
    converged_sigma = np.mean(last_epoch_sigmas) if last_epoch_sigmas else mean_sigma

    print(f"\n  📊 NCP={num_neurons}: σ̄={mean_sigma:.4f} (conv={converged_sigma:.4f})"
          f" Math={ma:.1f}% Fact={fa:.1f}% params={n_params}")

    return {
        "num_neurons": num_neurons,
        "n_params": n_params,
        "mean_sigma_all": round(mean_sigma, 5),
        "converged_sigma": round(converged_sigma, 5),
        "sigma_factual": round(msf, 5),
        "sigma_creative": round(msc, 5),
        "sigma_math": round(msm, 5),
        "max_separation": round(max_sep, 5),
        "factual_acc": round(fa, 2),
        "math_acc": round(ma, 2),
        "creative_novelty": round(float(an), 4),
        "sigma_trajectory": sigma_trajectory,
    }


# ═══ Visualization ═══

def visualize_ablation(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 41: Wiring Ablation — Does CfC Neuron Count Affect σ?\n"
                 "Same Mistral-7B, varying NCP neuron count",
                 fontsize=14, fontweight="bold")

    neurons = [r["num_neurons"] for r in results]
    colors = {"factual": "#2ecc71", "creative": "#e74c3c", "math": "#3498db"}

    # Panel 1: Converged σ by neuron count
    ax1 = axes[0, 0]
    conv_sigmas = [r["converged_sigma"] for r in results]
    ax1.plot(neurons, conv_sigmas, "ko-", markersize=10, linewidth=2, label="Converged σ̄")
    ax1.axhline(y=0.074, color="gold", linestyle="--", linewidth=2, alpha=0.7, label="σ ≈ 0.074")
    ax1.fill_between(neurons, [0.064]*len(neurons), [0.084]*len(neurons),
                      color="gold", alpha=0.1, label="±0.010 band")
    ax1.set_xlabel("NCP Neuron Count")
    ax1.set_ylabel("Converged σ")
    ax1.set_title("Does σ Depend on Neuron Count?")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(neurons)
    ax1.set_xticklabels(neurons)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Task-specific σ
    ax2 = axes[0, 1]
    for task, color in colors.items():
        vals = [r[f"sigma_{task}"] for r in results]
        ax2.plot(neurons, vals, "o-", color=color, markersize=8, linewidth=2, label=task)
    ax2.axhline(y=0.074, color="gold", linestyle="--", alpha=0.5)
    ax2.set_xlabel("NCP Neuron Count")
    ax2.set_ylabel("σ")
    ax2.set_title("Task-Specific σ by Neuron Count")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(neurons)
    ax2.set_xticklabels(neurons)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Math accuracy and task separation
    ax3 = axes[1, 0]
    math_accs = [r["math_acc"] for r in results]
    seps = [r["max_separation"] for r in results]
    ax3_twin = ax3.twinx()
    ax3.bar(range(len(neurons)), math_accs, color="#3498db", alpha=0.6, label="Math Acc")
    ax3_twin.plot(range(len(neurons)), seps, "rs-", markersize=8, linewidth=2, label="Max Sep")
    ax3.set_xticks(range(len(neurons)))
    ax3.set_xticklabels(neurons)
    ax3.set_xlabel("NCP Neuron Count")
    ax3.set_ylabel("Math Accuracy (%)", color="#3498db")
    ax3_twin.set_ylabel("Max σ Separation", color="red")
    ax3.set_title("Performance vs Neuron Count")
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Verdict
    ax4 = axes[1, 1]; ax4.axis("off")
    sigma_range = max(conv_sigmas) - min(conv_sigmas)
    sigma_mean = np.mean(conv_sigmas)
    cv = sigma_range / sigma_mean if sigma_mean > 0 else 0

    txt = (f"WIRING ABLATION RESULTS\n"
           f"{'='*40}\n\n")
    for r in results:
        txt += (f"NCP={r['num_neurons']:2d}: σ̄={r['converged_sigma']:.4f} "
                f"Math={r['math_acc']:5.1f}% "
                f"params={r['n_params']}\n")

    txt += (f"\n{'='*40}\n"
            f"σ range: {min(conv_sigmas):.4f} – {max(conv_sigmas):.4f}\n"
            f"σ spread: {sigma_range:.4f}\n"
            f"CV: {cv:.2%}\n\n")

    if sigma_range < 0.015:
        txt += ("🎆 σ IS INVARIANT TO NEURON COUNT!\n"
                "→ Transformer determines the set-point\n"
                "→ CfC finds the SAME σ regardless of capacity")
        vc = "#2ecc71"
    elif sigma_range < 0.03:
        txt += ("🔬 WEAK DEPENDENCE ON NEURON COUNT\n"
                "→ Mostly Transformer-determined,\n"
                "   slight CfC capacity effect")
        vc = "#e67e22"
    else:
        txt += ("📊 σ DEPENDS ON NEURON COUNT!\n"
                "→ CfC architecture partially determines σ\n"
                "→ More complex relationship than expected")
        vc = "#e74c3c"

    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace", color=vc,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase41_wiring_ablation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


# ═══ MAIN ═══

def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model(MODEL_NAME, MODEL_SHORT)
    hook = AdaptiveSNNHook(sigma=0.05)
    dataset = build_dataset(tokenizer)
    print(f"  📂 Dataset: {len(dataset)} items")

    factual_qs = [item["question"] for item in dataset if item["type"] == "factual"]
    math_qs = [item["question"] for item in dataset if item["type"] == "math"]

    classifier = TaskClassifier3()
    classifier = pretrain_classifier(
        classifier, model, tokenizer, hook,
        factual_qs, CREATIVE_PROMPTS, math_qs)

    all_results = []
    for n_neurons in NEURON_COUNTS:
        hook = AdaptiveSNNHook(sigma=0.05)
        torch.manual_seed(SEED)
        result = run_ablation(model, tokenizer, dataset, classifier, hook,
                              n_neurons, TARGET_LAYERS)
        all_results.append(result)

    fig_path = visualize_ablation(all_results)

    elapsed = time.time() - t_start
    conv_sigmas = [r["converged_sigma"] for r in all_results]

    output = {
        "experiment": "Phase 41: Wiring Ablation",
        "model": MODEL_SHORT,
        "elapsed_minutes": round(elapsed / 60, 1),
        "neuron_counts": NEURON_COUNTS,
        "results": [{k: v for k, v in r.items() if k != "sigma_trajectory"}
                     for r in all_results],
        "sigma_invariance": {
            "converged_sigmas": conv_sigmas,
            "sigma_range": round(max(conv_sigmas) - min(conv_sigmas), 5),
            "sigma_mean": round(float(np.mean(conv_sigmas)), 5),
            "sigma_std": round(float(np.std(conv_sigmas)), 5),
            "is_invariant": bool((max(conv_sigmas) - min(conv_sigmas)) < 0.015),
        },
        "figure_path": fig_path,
    }

    log_path = os.path.join(RESULTS_DIR, "phase41_wiring_ablation_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Results: {log_path}")

    print(f"\n{'═'*60}")
    print(f"  🧬 PHASE 41: WIRING ABLATION — GRAND SUMMARY")
    print(f"{'═'*60}")
    for r in all_results:
        print(f"  NCP={r['num_neurons']:2d} ({r['n_params']:5d} params): "
              f"σ̄={r['converged_sigma']:.4f} Math={r['math_acc']:5.1f}%")
    print(f"  σ range: {min(conv_sigmas):.4f} – {max(conv_sigmas):.4f} "
          f"(spread={max(conv_sigmas)-min(conv_sigmas):.4f})")
    inv = "YES" if (max(conv_sigmas) - min(conv_sigmas)) < 0.015 else "NO"
    print(f"  Invariant? {inv}")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
