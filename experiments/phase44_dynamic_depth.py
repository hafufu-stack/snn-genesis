"""
Phase 44: Dynamic Depth Scanning — 動的レイヤー移動
===================================================

CfC outputs TWO control signals:
  1. σ — noise amplitude (as before)
  2. L — target layer depth (0-31, continuous → discretized)

This enables the "conscious focus" to move up and down the
Transformer layers in real-time, focusing noise injection where
it's most needed for each specific prompt.

Three modes:
  1. FIXED:    Noise at layers 15-20 (standard)
  2. RANDOM:   Noise at random single layer (control)
  3. DYNAMIC:  CfC selects the injection layer

Usage:
    python experiments/phase44_dynamic_depth.py
"""

import torch, torch.nn as nn
import os, json, gc, time, random, re, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# ═══ Configuration ═══
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
N_LAYERS_TOTAL = 32
FIXED_TARGET_LAYERS = list(range(15, 21))
CONFLICT_LAYER_SHALLOW = 8
CONFLICT_LAYER_DEEP = 28

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


class DynamicDepthHook:
    """Hook that dynamically selects which layers receive noise."""
    def __init__(self, sigma=0.05, active_layers=None):
        self.sigma = sigma
        self.active_layers = set(active_layers or FIXED_TARGET_LAYERS)
        self.last_hidden_norm = 0.0
        self.layer_index = 0  # Track which layer is being called
    def __call__(self, module, args):
        hs = args[0]
        if self.layer_index in self.active_layers:
            noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
            low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
            noise = 0.7 * noise + 0.3 * low_freq
            noise = noise * self.sigma
            self.last_hidden_norm = hs.float().norm().item() / max(hs.numel(), 1)
            return (hs + noise,) + args[1:]
        self.last_hidden_norm = hs.float().norm().item() / max(hs.numel(), 1)
        return args
    def update_sigma(self, new_sigma):
        self.sigma = np.clip(new_sigma, SIGMA_MIN, SIGMA_MAX)
    def set_active_layers(self, layers):
        """Set which layers receive noise."""
        self.active_layers = set(layers)


class DepthAwareCfCActor(nn.Module):
    """CfC Actor that outputs σ AND target layer depth."""
    def __init__(self, input_size=8, num_neurons=16):
        super().__init__()
        wiring = AutoNCP(num_neurons, 2)  # 2 outputs: σ, layer_depth
        self.cfc = CfC(input_size, wiring, batch_first=True)
        self.log_std_sigma = nn.Parameter(torch.tensor([-1.0]))
        self.log_std_depth = nn.Parameter(torch.tensor([-0.5]))
    def forward(self, x, hx=None):
        output, hx = self.cfc(x, hx=hx)
        mu_sigma = torch.sigmoid(output[:, :, 0:1]) * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN
        mu_depth = torch.sigmoid(output[:, :, 1:2]) * (N_LAYERS_TOTAL - 1)
        std_sigma = torch.exp(self.log_std_sigma).expand_as(mu_sigma)
        std_depth = torch.exp(self.log_std_depth).expand_as(mu_depth)
        return mu_sigma, std_sigma, mu_depth, std_depth, hx
    def get_sigma_depth_and_logprob(self, features, hx=None):
        x = features.unsqueeze(0).unsqueeze(0)
        mu_s, std_s, mu_d, std_d, hx = self.forward(x, hx=hx)
        mu_s = mu_s.squeeze(); std_s = std_s.squeeze()
        mu_d = mu_d.squeeze(); std_d = std_d.squeeze()
        dist_s = torch.distributions.Normal(mu_s, std_s)
        dist_d = torch.distributions.Normal(mu_d, std_d)
        raw_sigma = dist_s.rsample()
        raw_depth = dist_d.rsample()
        sigma = torch.clamp(raw_sigma, SIGMA_MIN, SIGMA_MAX)
        depth = torch.clamp(raw_depth, 0, N_LAYERS_TOTAL - 1)
        log_prob = dist_s.log_prob(raw_sigma) + dist_d.log_prob(raw_depth)
        entropy = dist_s.entropy() + dist_d.entropy()
        return sigma, depth, log_prob, entropy, hx
    def evaluate_action(self, features, old_sigma):
        x = features.unsqueeze(0).unsqueeze(0)
        mu_s, std_s, _, _, _ = self.forward(x)
        mu_s = mu_s.squeeze(); std_s = std_s.squeeze()
        dist_s = torch.distributions.Normal(mu_s, std_s)
        return dist_s.log_prob(old_sigma), dist_s.entropy()


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
        f, _ = extract_features(model, tokenizer, p, hook); all_features.append(f); all_labels.append(0)
    for p in creative_qs[:20]:
        f, _ = extract_features(model, tokenizer, p, hook); all_features.append(f); all_labels.append(1)
    for p in math_qs[:20]:
        f, _ = extract_features(model, tokenizer, p, hook); all_features.append(f); all_labels.append(2)
    X = torch.stack(all_features); Y = torch.tensor(all_labels, dtype=torch.long)
    for _ in range(epochs):
        logits = classifier(X); loss = loss_fn(logits, Y)
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

def run_experiment(model, tokenizer, dataset, target_layers, mode="dynamic"):
    """
    Modes:
      - "fixed": Noise at layers 15-20 (standard)
      - "random_layer": Random single layer (control)
      - "dynamic": CfC selects layer + σ
    """
    print(f"\n{'═'*60}")
    print(f"  🔭 MODE: {mode.upper()}")
    print(f"{'═'*60}")

    layers = get_layers(model)
    hook = DynamicDepthHook(sigma=0.05, active_layers=FIXED_TARGET_LAYERS)

    # Register hook on ALL layers
    handles = []
    for i in range(len(layers)):
        def make_hook(layer_idx):
            def layer_hook(module, args):
                hook.layer_index = layer_idx
                return hook(module, args)
            return layer_hook
        handles.append(layers[i].register_forward_pre_hook(make_hook(i)))

    input_size = 8
    actor = DepthAwareCfCActor(input_size=input_size, num_neurons=16)
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
    stats = {"factual": {"correct": 0, "total": 0, "sigmas": [], "depths": []},
             "creative": {"novelties": [], "grammar": 0, "total": 0, "sigmas": [], "depths": []},
             "math": {"correct": 0, "total": 0, "sigmas": [], "depths": []}}
    buf_f, buf_s, buf_lp, buf_v, buf_r, buf_e = [], [], [], [], [], []

    for epoch in range(N_EPOCHS):
        print(f"\n  🔭 Epoch {epoch+1}/{N_EPOCHS}")
        for idx, item in enumerate(dataset):
            features, conflict = extract_features(model, tokenizer, item["question"], hook)
            with torch.no_grad():
                probs = classifier.predict_probs(features.unsqueeze(0)).squeeze()
            sigma_target = (probs[0].item() * SIGMA_FACTUAL +
                            probs[1].item() * SIGMA_CREATIVE +
                            probs[2].item() * SIGMA_MATH)

            if mode == "fixed":
                hook.set_active_layers(FIXED_TARGET_LAYERS)
                hook.update_sigma(0.05)
                depth_val = 17.5  # midpoint of 15-20
            elif mode == "random_layer":
                rand_layer = random.randint(0, N_LAYERS_TOTAL - 1)
                # Spread noise across 3 layers centered on rand_layer
                spread = [max(0, rand_layer-1), rand_layer, min(N_LAYERS_TOTAL-1, rand_layer+1)]
                hook.set_active_layers(spread)
                hook.update_sigma(0.05)
                depth_val = float(rand_layer)
            elif mode == "dynamic":
                sigma, depth, log_prob, entropy, hx_new = \
                    actor.get_sigma_depth_and_logprob(features, hx)
                hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()
                current_sigma = sigma.detach().item()
                depth_val = depth.detach().item()
                layer_center = int(round(depth_val))
                layer_center = max(1, min(N_LAYERS_TOTAL - 2, layer_center))
                active = [layer_center - 1, layer_center, layer_center + 1]
                hook.set_active_layers(active)
                hook.update_sigma(current_sigma)

            with torch.no_grad():
                value = critic(features).item() if mode == "dynamic" else 0.0

            if item["type"] == "factual":
                ch = item["mc1_targets"]
                ci = ch["labels"].index(1) if 1 in ch["labels"] else 0
                lps = [compute_completion_logprob(model, tokenizer, item["prompt"], " " + c)
                       for c in ch["choices"]]
                correct = (np.argmax(lps) == ci)
                if correct: stats["factual"]["correct"] += 1
                stats["factual"]["total"] += 1
                stats["factual"]["sigmas"].append(hook.sigma)
                stats["factual"]["depths"].append(depth_val)
                reward = (1.0 if correct else 0.0) - LAMBDA_QUAD * (hook.sigma - sigma_target) ** 2
            elif item["type"] == "creative":
                resp = generate_text(model, tokenizer, item["prompt"])
                nov = compute_novelty(resp); gram = is_grammatical(resp)
                stats["creative"]["novelties"].append(nov)
                if gram: stats["creative"]["grammar"] += 1
                stats["creative"]["total"] += 1
                stats["creative"]["sigmas"].append(hook.sigma)
                stats["creative"]["depths"].append(depth_val)
                reward = (nov if gram else -0.5) - LAMBDA_QUAD * (hook.sigma - sigma_target) ** 2
            elif item["type"] == "math":
                resp = generate_text(model, tokenizer, item["prompt"], max_new_tokens=50)
                correct = check_math_answer(resp, item["correct_answer"])
                if correct: stats["math"]["correct"] += 1
                stats["math"]["total"] += 1
                stats["math"]["sigmas"].append(hook.sigma)
                stats["math"]["depths"].append(depth_val)
                reward = (1.5 if correct else -0.5) - LAMBDA_QUAD * (hook.sigma - sigma_target) ** 2

            if mode == "dynamic":
                buf_f.append(features.detach()); buf_s.append(sigma.detach())
                buf_lp.append(log_prob.detach()); buf_v.append(value)
                buf_r.append(reward); buf_e.append(entropy.detach())
                if len(buf_r) >= PPO_ROLLOUT_SIZE:
                    adv, ret = compute_gae(buf_r, buf_v)
                    adv_t = torch.tensor(adv, dtype=torch.float32)
                    ret_t = torch.tensor(ret, dtype=torch.float32)
                    old_lp = torch.stack(buf_lp); old_s = torch.stack(buf_s); fb = torch.stack(buf_f)
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
                msm = np.mean(stats["math"]["sigmas"][-10:]) if stats["math"]["sigmas"] else 0
                md = np.mean(stats["math"]["depths"][-10:]) if stats["math"]["depths"] else 0
                print(f"  [{idx+1}/{len(dataset)}] σ_f={msf:.4f} σ_m={msm:.4f}"
                      f" depth={md:.1f}")

    for h in handles:
        h.remove()

    fa = stats["factual"]["correct"] / max(stats["factual"]["total"], 1) * 100
    an = float(np.mean(stats["creative"]["novelties"])) if stats["creative"]["novelties"] else 0
    ma = stats["math"]["correct"] / max(stats["math"]["total"], 1) * 100
    msf = float(np.mean(stats["factual"]["sigmas"])) if stats["factual"]["sigmas"] else 0
    msc = float(np.mean(stats["creative"]["sigmas"])) if stats["creative"]["sigmas"] else 0
    msm = float(np.mean(stats["math"]["sigmas"])) if stats["math"]["sigmas"] else 0
    df = float(np.mean(stats["factual"]["depths"])) if stats["factual"]["depths"] else 0
    dc = float(np.mean(stats["creative"]["depths"])) if stats["creative"]["depths"] else 0
    dm = float(np.mean(stats["math"]["depths"])) if stats["math"]["depths"] else 0

    print(f"\n  📊 {mode}: Math={ma:.1f}% Fact={fa:.1f}%"
          f" depth_f={df:.1f} depth_c={dc:.1f} depth_m={dm:.1f}")

    return {
        "mode": mode,
        "math_acc": round(ma, 2), "factual_acc": round(fa, 2),
        "creative_novelty": round(an, 4),
        "sigma_factual": round(msf, 5), "sigma_creative": round(msc, 5),
        "sigma_math": round(msm, 5),
        "depth_factual": round(df, 2), "depth_creative": round(dc, 2),
        "depth_math": round(dm, 2),
    }


# ═══ Visualization ═══

def visualize_depth(fixed, random_layer, dynamic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 44: Dynamic Depth Scanning\n"
                 "CfC controls WHERE noise goes (layer selection)",
                 fontsize=14, fontweight="bold")

    modes = ["Fixed\n(L15-20)", "Random\nLayer", "Dynamic\nCfC"]
    results = [fixed, random_layer, dynamic]
    colors = ["#95a5a6", "#e67e22", "#9b59b6"]

    # Panel 1: Math accuracy
    ax1 = axes[0, 0]
    math_vals = [r["math_acc"] for r in results]
    bars = ax1.bar(modes, math_vals, color=colors, alpha=0.8)
    for bar, val in zip(bars, math_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Math Accuracy (%)")
    ax1.set_title("Math Accuracy by Depth Strategy")
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Average depth by task (dynamic only)
    ax2 = axes[0, 1]
    tasks = ["Factual", "Creative", "Math"]
    depths = [dynamic["depth_factual"], dynamic["depth_creative"], dynamic["depth_math"]]
    task_colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars2 = ax2.bar(tasks, depths, color=task_colors, alpha=0.8)
    for bar, val in zip(bars2, depths):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"L{val:.1f}", ha="center", fontsize=11)
    ax2.axhline(y=17.5, color="gold", linestyle="--", alpha=0.7, label="Fixed midpoint (L17.5)")
    ax2.set_ylabel("Average Layer Depth")
    ax2.set_title("Where Does CfC Focus? (Dynamic Mode)")
    ax2.set_ylim(0, 32)
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
    ax3.set_title("Factual Accuracy by Depth Strategy")
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Verdict
    ax4 = axes[1, 1]; ax4.axis("off")
    dm_math = dynamic["math_acc"] - fixed["math_acc"]
    dm_fact = dynamic["factual_acc"] - fixed["factual_acc"]
    depth_diff = abs(dynamic["depth_factual"] - dynamic["depth_math"])

    txt = (f"DYNAMIC DEPTH SCANNING RESULTS\n"
           f"{'='*45}\n\n"
           f"Fixed (L15-20): Math={fixed['math_acc']:.1f}% Fact={fixed['factual_acc']:.1f}%\n"
           f"Random Layer:   Math={random_layer['math_acc']:.1f}% Fact={random_layer['factual_acc']:.1f}%\n"
           f"Dynamic CfC:    Math={dynamic['math_acc']:.1f}% Fact={dynamic['factual_acc']:.1f}%\n\n"
           f"Math Δ (dynamic vs fixed): {dm_math:+.1f}%\n"
           f"Fact Δ (dynamic vs fixed): {dm_fact:+.1f}%\n\n"
           f"CfC selected depths:\n"
           f"  Factual:  L{dynamic['depth_factual']:.1f}\n"
           f"  Creative: L{dynamic['depth_creative']:.1f}\n"
           f"  Math:     L{dynamic['depth_math']:.1f}\n"
           f"  Task depth separation: {depth_diff:.1f} layers\n\n")

    if depth_diff > 2.0:
        txt += ("🎆 CfC LEARNED DEPTH PREFERENCE!\n"
                "→ Different tasks → different layers\n"
                "→ 'Conscious focus' moves in the brain!")
        vc = "#2ecc71"
    elif dm_math > 0 or dm_fact > 0:
        txt += ("🔬 DYNAMIC HELPS SOMEWHAT\n"
                "→ Performance improves but depths similar")
        vc = "#e67e22"
    else:
        txt += ("📊 FIXED DEPTH IS SUFFICIENT\n"
                "→ L15-20 is already near-optimal")
        vc = "#e74c3c"

    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace", color=vc,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase44_dynamic_depth.png")
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

    # === MODE 1: FIXED (L15-20) ===
    torch.manual_seed(SEED)
    fixed = run_experiment(model, tokenizer, dataset, FIXED_TARGET_LAYERS, mode="fixed")

    # === MODE 2: RANDOM LAYER ===
    torch.manual_seed(SEED); random.seed(SEED + 200)
    rand = run_experiment(model, tokenizer, dataset, FIXED_TARGET_LAYERS, mode="random_layer")

    # === MODE 3: DYNAMIC CfC ===
    torch.manual_seed(SEED)
    dynamic = run_experiment(model, tokenizer, dataset, FIXED_TARGET_LAYERS, mode="dynamic")

    fig_path = visualize_depth(fixed, rand, dynamic)
    elapsed = time.time() - t_start

    output = {
        "experiment": "Phase 44: Dynamic Depth Scanning",
        "model": MODEL_SHORT,
        "elapsed_minutes": round(elapsed / 60, 1),
        "fixed": fixed, "random_layer": rand, "dynamic": dynamic,
        "comparison": {
            "math_delta_dynamic_vs_fixed": round(dynamic["math_acc"] - fixed["math_acc"], 2),
            "factual_delta_dynamic_vs_fixed": round(dynamic["factual_acc"] - fixed["factual_acc"], 2),
            "depth_separation": round(abs(dynamic["depth_factual"] - dynamic["depth_math"]), 2),
        },
        "figure_path": fig_path,
    }

    log_path = os.path.join(RESULTS_DIR, "phase44_dynamic_depth_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)

    dm = output["comparison"]["math_delta_dynamic_vs_fixed"]
    df = output["comparison"]["factual_delta_dynamic_vs_fixed"]
    print(f"\n{'═'*60}")
    print(f"  🔭 PHASE 44: DYNAMIC DEPTH — VERDICT")
    print(f"{'═'*60}")
    print(f"  Fixed:   Math={fixed['math_acc']:.1f}% Fact={fixed['factual_acc']:.1f}%")
    print(f"  Random:  Math={rand['math_acc']:.1f}% Fact={rand['factual_acc']:.1f}%")
    print(f"  Dynamic: Math={dynamic['math_acc']:.1f}% Fact={dynamic['factual_acc']:.1f}%")
    print(f"  Math Δ: {dm:+.1f}%  Fact Δ: {df:+.1f}%")
    print(f"  Depth sep: {output['comparison']['depth_separation']:.2f}")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
