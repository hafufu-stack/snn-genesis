"""
Phase 43: Chaos Defibrillator — カオス除細動器
==============================================

When the LLM is stuck in a wrong reasoning path (detected via
inter-layer conflict from Phase 42), fire a σ SPIKE (0.15) for
ONE token to force course correction, then return to homeostasis.

Three modes compared:
  1. BASELINE:   Static σ=0.05 (no spike)
  2. RANDOM:     Random spikes (control group)
  3. DEFIBRILLATOR: Conflict-triggered spikes via CfC

The key innovation: CfC not only senses layer conflict (Phase 42)
but uses it as a TRIGGER for therapeutic noise bursts — like a
cardiac defibrillator that shocks only when arrhythmia is detected.

Usage:
    python experiments/phase43_chaos_defibrillator.py
"""

import torch, torch.nn as nn
import os, json, gc, time, random, re
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
CONFLICT_LAYER_SHALLOW = 8
CONFLICT_LAYER_DEEP = 28

SIGMA_NORMAL = 0.05       # Normal body temperature
SIGMA_SPIKE = 0.15        # Defibrillator shock
SIGMA_MIN, SIGMA_MAX = 0.001, 0.15
CONFLICT_THRESHOLD = 0.25 # Fire defibrillator when conflict > threshold
SPIKE_PROB_RANDOM = 0.15  # Random spike probability (control group)

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


class DefibrillatorHook:
    """Hook that can deliver σ spikes on demand."""
    def __init__(self, sigma=0.05):
        self.sigma = sigma
        self.last_hidden_norm = 0.0
        self.spike_active = False
        self.spike_count = 0
        self.normal_count = 0
    def __call__(self, module, args):
        hs = args[0]
        effective_sigma = SIGMA_SPIKE if self.spike_active else self.sigma
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * effective_sigma
        self.last_hidden_norm = hs.float().norm().item() / max(hs.numel(), 1)
        if self.spike_active:
            self.spike_count += 1
        else:
            self.normal_count += 1
        return (hs + noise,) + args[1:]
    def update_sigma(self, new_sigma):
        self.sigma = np.clip(new_sigma, SIGMA_MIN, SIGMA_MAX)
    def fire_spike(self):
        """Activate defibrillator spike for the next forward pass."""
        self.spike_active = True
    def reset_spike(self):
        """Deactivate spike."""
        self.spike_active = False


class CfCActorWithSpike(nn.Module):
    """CfC Actor that outputs both σ AND a spike decision."""
    def __init__(self, input_size=8, num_neurons=16):
        super().__init__()
        wiring = AutoNCP(num_neurons, 2)  # 2 outputs: σ, spike_logit
        self.cfc = CfC(input_size, wiring, batch_first=True)
        self.log_std = nn.Parameter(torch.tensor([-1.0]))
    def forward(self, x, hx=None):
        output, hx = self.cfc(x, hx=hx)
        mu = torch.sigmoid(output[:, :, 0:1]) * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN
        spike_logit = output[:, :, 1:2]  # Raw logit for spike decision
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, spike_logit, hx
    def get_sigma_spike_and_logprob(self, features, hx=None):
        x = features.unsqueeze(0).unsqueeze(0)
        mu, std, spike_logit, hx = self.forward(x, hx=hx)
        mu = mu.squeeze(); std = std.squeeze(); spike_logit = spike_logit.squeeze()
        dist = torch.distributions.Normal(mu, std)
        raw_sigma = dist.rsample()
        sigma = torch.clamp(raw_sigma, SIGMA_MIN, SIGMA_MAX)
        log_prob_sigma = dist.log_prob(raw_sigma)
        spike_prob = torch.sigmoid(spike_logit)
        entropy = dist.entropy()
        return sigma, spike_prob, log_prob_sigma, entropy, hx
    def evaluate_action(self, features, old_sigma):
        x = features.unsqueeze(0).unsqueeze(0)
        mu, std, _, _ = self.forward(x)
        mu = mu.squeeze(); std = std.squeeze()
        dist = torch.distributions.Normal(mu, std)
        return dist.log_prob(old_sigma), dist.entropy()


class CfCCritic(nn.Module):
    def __init__(self, input_size=8, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1))
    def forward(self, features):
        return self.net(features).squeeze(-1)


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


# ─── Helpers ───

def compute_layer_conflict(model, tokenizer, prompt):
    """Compute inter-layer conflict (cosine distance L8 vs L28)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states
        n_layers = len(hs) - 1
        shallow_idx = min(CONFLICT_LAYER_SHALLOW, n_layers - 1)
        deep_idx = min(CONFLICT_LAYER_DEEP, n_layers - 1)
        h_shallow = hs[shallow_idx + 1].float().mean(dim=1).squeeze(0)
        h_deep = hs[deep_idx + 1].float().mean(dim=1).squeeze(0)
        cos_sim = torch.nn.functional.cosine_similarity(
            h_shallow.unsqueeze(0), h_deep.unsqueeze(0)).item()
        conflict = 1.0 - cos_sim
        norm_delta = abs(h_deep.norm().item() - h_shallow.norm().item()) / max(h_shallow.norm().item(), 1e-6)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    return conflict, norm_delta, entropy


def extract_features(model, tokenizer, prompt, hook):
    """8-feature extraction with conflict metrics."""
    text = prompt if len(prompt) < 200 else prompt[:200]
    conflict, norm_delta, entropy = compute_layer_conflict(model, tokenizer, text)
    words = text.lower().split()
    return torch.tensor([
        hook.last_hidden_norm if hook.last_hidden_norm else 0.5,
        min(len(words) / 50.0, 1.0),
        len(set(words)) / max(len(words), 1),
        min(entropy / 10.0, 1.0),
        conflict,
        norm_delta / 5.0,
        float("?" in text),
        sum(len(w) for w in words) / max(len(words), 1) / 10.0,
    ], dtype=torch.float32), conflict


def pretrain_classifier(classifier, model, tokenizer, hook,
                        factual_qs, creative_qs, math_qs, epochs=50):
    print("  🎓 Pre-training classifier...")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    all_features, all_labels = [], []
    for p in factual_qs[:20]:
        f, _ = extract_features(model, tokenizer, p, hook)
        all_features.append(f); all_labels.append(0)
    for p in creative_qs[:20]:
        f, _ = extract_features(model, tokenizer, p, hook)
        all_features.append(f); all_labels.append(1)
    for p in math_qs[:20]:
        f, _ = extract_features(model, tokenizer, p, hook)
        all_features.append(f); all_labels.append(2)
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


# ═══ Run experiment ═══

def run_experiment(model, tokenizer, dataset, hook, target_layers,
                   mode="defibrillator"):
    """
    Modes:
      - "baseline": Static σ, no spikes
      - "random_spike": Random spikes (control)
      - "defibrillator": CfC-triggered spikes based on conflict
    """
    print(f"\n{'═'*60}")
    print(f"  ⚡ MODE: {mode.upper()}")
    print(f"{'═'*60}")

    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in target_layers if i < len(layers)]

    input_size = 8
    actor = CfCActorWithSpike(input_size=input_size, num_neurons=16)
    critic = CfCCritic(input_size=input_size)
    actor.train(); critic.train()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)

    classifier = TaskClassifier3(input_dim=input_size)
    factual_qs = [item["question"] for item in dataset if item["type"] == "factual"]
    math_qs = [item["question"] for item in dataset if item["type"] == "math"]
    classifier = pretrain_classifier(
        classifier, model, tokenizer, hook, factual_qs, CREATIVE_PROMPTS, math_qs)

    hx = None
    stats = {"factual": {"correct": 0, "total": 0, "sigmas": [], "spikes": 0},
             "creative": {"novelties": [], "grammar": 0, "total": 0, "sigmas": [], "spikes": 0},
             "math": {"correct": 0, "total": 0, "sigmas": [], "spikes": 0}}
    total_spikes, total_items = 0, 0
    buf_f, buf_s, buf_lp, buf_v, buf_r, buf_e = [], [], [], [], [], []

    for epoch in range(N_EPOCHS):
        print(f"\n  ⚡ Epoch {epoch+1}/{N_EPOCHS}")
        for idx, item in enumerate(dataset):
            features, conflict = extract_features(model, tokenizer, item["question"], hook)
            with torch.no_grad():
                probs = classifier.predict_probs(features.unsqueeze(0)).squeeze()
            sigma_target = (probs[0].item() * SIGMA_FACTUAL +
                            probs[1].item() * SIGMA_CREATIVE +
                            probs[2].item() * SIGMA_MATH)

            # Decide whether to spike
            spiked = False
            if mode == "baseline":
                hook.update_sigma(SIGMA_NORMAL)
                hook.reset_spike()
            elif mode == "random_spike":
                hook.update_sigma(SIGMA_NORMAL)
                if random.random() < SPIKE_PROB_RANDOM:
                    hook.fire_spike(); spiked = True
                else:
                    hook.reset_spike()
            elif mode == "defibrillator":
                sigma, spike_prob, log_prob, entropy, hx_new = \
                    actor.get_sigma_spike_and_logprob(features, hx)
                hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()
                current_sigma = sigma.detach().item()
                hook.update_sigma(current_sigma)

                # Fire spike if conflict exceeds threshold AND CfC says so
                if conflict > CONFLICT_THRESHOLD and spike_prob.item() > 0.5:
                    hook.fire_spike(); spiked = True
                else:
                    hook.reset_spike()

            if spiked:
                total_spikes += 1
                stats[item["type"]]["spikes"] += 1
            total_items += 1

            with torch.no_grad():
                value = critic(features).item() if mode == "defibrillator" else 0.0

            if item["type"] == "factual":
                ch = item["mc1_targets"]
                ci = ch["labels"].index(1) if 1 in ch["labels"] else 0
                lps = [compute_completion_logprob(model, tokenizer, item["prompt"], " " + c)
                       for c in ch["choices"]]
                correct = (np.argmax(lps) == ci)
                if correct: stats["factual"]["correct"] += 1
                stats["factual"]["total"] += 1
                stats["factual"]["sigmas"].append(hook.sigma)
                reward = (1.0 if correct else 0.0) - LAMBDA_QUAD * (hook.sigma - sigma_target) ** 2
            elif item["type"] == "creative":
                resp = generate_text(model, tokenizer, item["prompt"])
                nov = compute_novelty(resp); gram = is_grammatical(resp)
                stats["creative"]["novelties"].append(nov)
                if gram: stats["creative"]["grammar"] += 1
                stats["creative"]["total"] += 1
                stats["creative"]["sigmas"].append(hook.sigma)
                reward = (nov if gram else -0.5) - LAMBDA_QUAD * (hook.sigma - sigma_target) ** 2
            elif item["type"] == "math":
                resp = generate_text(model, tokenizer, item["prompt"], max_new_tokens=50)
                correct = check_math_answer(resp, item["correct_answer"])
                if correct: stats["math"]["correct"] += 1
                stats["math"]["total"] += 1
                stats["math"]["sigmas"].append(hook.sigma)
                # Bonus for spike + correct (defibrillator resuscitated!)
                spike_bonus = 0.5 if (spiked and correct) else 0.0
                reward = (1.5 if correct else -0.5) + spike_bonus \
                         - LAMBDA_QUAD * (hook.sigma - sigma_target) ** 2

            hook.reset_spike()  # Always reset after each item

            if mode == "defibrillator":
                buf_f.append(features.detach()); buf_s.append(sigma.detach())
                buf_lp.append(log_prob.detach()); buf_v.append(value)
                buf_r.append(reward); buf_e.append(entropy.detach())

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
                spike_rate = total_spikes / max(total_items, 1) * 100
                print(f"  [{idx+1}/{len(dataset)}] σ_f={msf:.4f} σ_m={msm:.4f}"
                      f" spikes={spike_rate:.1f}%")

    for h in handles:
        h.remove()

    fa = stats["factual"]["correct"] / max(stats["factual"]["total"], 1) * 100
    an = float(np.mean(stats["creative"]["novelties"])) if stats["creative"]["novelties"] else 0
    ma = stats["math"]["correct"] / max(stats["math"]["total"], 1) * 100
    msf = float(np.mean(stats["factual"]["sigmas"])) if stats["factual"]["sigmas"] else 0
    msc = float(np.mean(stats["creative"]["sigmas"])) if stats["creative"]["sigmas"] else 0
    msm = float(np.mean(stats["math"]["sigmas"])) if stats["math"]["sigmas"] else 0
    spike_rate = total_spikes / max(total_items, 1) * 100

    print(f"\n  📊 {mode}: Math={ma:.1f}% Fact={fa:.1f}% Spikes={spike_rate:.1f}%")

    return {
        "mode": mode,
        "math_acc": round(ma, 2), "factual_acc": round(fa, 2),
        "creative_novelty": round(an, 4),
        "sigma_factual": round(msf, 5), "sigma_creative": round(msc, 5),
        "sigma_math": round(msm, 5),
        "total_spikes": total_spikes, "total_items": total_items,
        "spike_rate": round(spike_rate, 2),
        "spikes_factual": stats["factual"]["spikes"],
        "spikes_creative": stats["creative"]["spikes"],
        "spikes_math": stats["math"]["spikes"],
    }


# ═══ Visualization ═══

def visualize_defibrillator(baseline, random_spike, defibrillator):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 43: Chaos Defibrillator\n"
                 "Conflict-triggered σ spikes for course correction",
                 fontsize=14, fontweight="bold")

    modes = ["Baseline", "Random\nSpike", "Defibrillator"]
    results = [baseline, random_spike, defibrillator]
    colors = ["#95a5a6", "#e67e22", "#e74c3c"]

    # Panel 1: Math accuracy comparison
    ax1 = axes[0, 0]
    math_vals = [r["math_acc"] for r in results]
    bars = ax1.bar(modes, math_vals, color=colors, alpha=0.8)
    for bar, val in zip(bars, math_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Math Accuracy (%)")
    ax1.set_title("Math Accuracy: Does Defibrillation Help?")
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Spike distribution
    ax2 = axes[0, 1]
    if random_spike["total_spikes"] > 0 or defibrillator["total_spikes"] > 0:
        cat = ["Factual", "Creative", "Math"]
        rs = [random_spike["spikes_factual"], random_spike["spikes_creative"],
              random_spike["spikes_math"]]
        ds = [defibrillator["spikes_factual"], defibrillator["spikes_creative"],
              defibrillator["spikes_math"]]
        x = np.arange(len(cat)); w = 0.35
        ax2.bar(x - w/2, rs, w, label="Random", color="#e67e22", alpha=0.8)
        ax2.bar(x + w/2, ds, w, label="Defibrillator", color="#e74c3c", alpha=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels(cat)
        ax2.set_ylabel("Spike Count")
        ax2.set_title("Where Did Spikes Fire?")
        ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Factual accuracy
    ax3 = axes[1, 0]
    fact_vals = [r["factual_acc"] for r in results]
    bars3 = ax3.bar(modes, fact_vals, color=colors, alpha=0.8)
    for bar, val in zip(bars3, fact_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", fontsize=11)
    ax3.set_ylabel("Factual Accuracy (%)")
    ax3.set_title("Factual Accuracy")
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Verdict
    ax4 = axes[1, 1]; ax4.axis("off")
    dm = defibrillator["math_acc"] - baseline["math_acc"]
    df = defibrillator["factual_acc"] - baseline["factual_acc"]
    rm = random_spike["math_acc"] - baseline["math_acc"]

    txt = (f"CHAOS DEFIBRILLATOR RESULTS\n"
           f"{'='*45}\n\n"
           f"Baseline:      Math={baseline['math_acc']:.1f}% Fact={baseline['factual_acc']:.1f}%\n"
           f"Random Spike:  Math={random_spike['math_acc']:.1f}% Fact={random_spike['factual_acc']:.1f}%\n"
           f"Defibrillator: Math={defibrillator['math_acc']:.1f}% Fact={defibrillator['factual_acc']:.1f}%\n\n"
           f"Math Δ (defib vs base):  {dm:+.1f}%\n"
           f"Math Δ (random vs base): {rm:+.1f}%\n\n"
           f"Spike rates:\n"
           f"  Random: {random_spike['spike_rate']:.1f}%\n"
           f"  Defib:  {defibrillator['spike_rate']:.1f}%\n\n")

    if dm > rm and dm > 0:
        txt += ("🎆 DEFIBRILLATOR WINS!\n"
                "→ Targeted spikes > random spikes\n"
                "→ CfC learned when to shock!")
        vc = "#2ecc71"
    elif dm > 0:
        txt += ("🔬 SPIKES HELP BUT TARGETING UNCLEAR\n"
                "→ Both spike methods help\n"
                "→ Targeted selection not clearly superior")
        vc = "#e67e22"
    else:
        txt += ("📊 SPIKES DON'T HELP\n"
                "→ Chaos injection didn't improve performance")
        vc = "#e74c3c"

    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace", color=vc,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase43_chaos_defibrillator.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


# ═══ MAIN ═══

def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model(MODEL_NAME, MODEL_SHORT)
    dataset = build_dataset(tokenizer)
    print(f"  📂 Dataset: {len(dataset)} items")

    # === MODE 1: BASELINE (static σ, no spikes) ===
    hook1 = DefibrillatorHook(sigma=SIGMA_NORMAL)
    torch.manual_seed(SEED)
    baseline = run_experiment(model, tokenizer, dataset, hook1, TARGET_LAYERS, mode="baseline")

    # === MODE 2: RANDOM SPIKE (control group) ===
    hook2 = DefibrillatorHook(sigma=SIGMA_NORMAL)
    torch.manual_seed(SEED); random.seed(SEED + 100)
    random_spike = run_experiment(model, tokenizer, dataset, hook2, TARGET_LAYERS, mode="random_spike")

    # === MODE 3: DEFIBRILLATOR (CfC-triggered spikes) ===
    hook3 = DefibrillatorHook(sigma=SIGMA_NORMAL)
    torch.manual_seed(SEED)
    defibrillator = run_experiment(model, tokenizer, dataset, hook3, TARGET_LAYERS, mode="defibrillator")

    fig_path = visualize_defibrillator(baseline, random_spike, defibrillator)
    elapsed = time.time() - t_start

    output = {
        "experiment": "Phase 43: Chaos Defibrillator",
        "model": MODEL_SHORT,
        "elapsed_minutes": round(elapsed / 60, 1),
        "baseline": baseline,
        "random_spike": random_spike,
        "defibrillator": defibrillator,
        "comparison": {
            "math_delta_defib_vs_base": round(defibrillator["math_acc"] - baseline["math_acc"], 2),
            "math_delta_random_vs_base": round(random_spike["math_acc"] - baseline["math_acc"], 2),
            "factual_delta_defib_vs_base": round(defibrillator["factual_acc"] - baseline["factual_acc"], 2),
        },
        "figure_path": fig_path,
    }

    log_path = os.path.join(RESULTS_DIR, "phase43_chaos_defibrillator_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)

    dm = output["comparison"]["math_delta_defib_vs_base"]
    rm = output["comparison"]["math_delta_random_vs_base"]
    print(f"\n{'═'*60}")
    print(f"  ⚡ PHASE 43: CHAOS DEFIBRILLATOR — VERDICT")
    print(f"{'═'*60}")
    print(f"  Baseline:      Math={baseline['math_acc']:.1f}% Fact={baseline['factual_acc']:.1f}%")
    print(f"  Random Spike:  Math={random_spike['math_acc']:.1f}% Fact={random_spike['factual_acc']:.1f}%")
    print(f"  Defibrillator: Math={defibrillator['math_acc']:.1f}% Fact={defibrillator['factual_acc']:.1f}%")
    print(f"  Math Δ (defib): {dm:+.1f}%  Δ (random): {rm:+.1f}%")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
