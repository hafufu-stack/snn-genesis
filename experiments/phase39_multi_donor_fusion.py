"""
Phase 39: Multi-Donor Fusion — Can We Build a "Best of Breed" CfC?
==================================================================

Phase 37-38 established bidirectional causal asymmetry:
  - Qwen CfC → Mistral: Math +15%
  - Mistral CfC → Qwen: Math -5%

This experiment tests: Can we FUSE donor CfC weights to get the best of both?

Fusion strategies:
  1. SIMPLE AVERAGE: w_fused = (w_qwen + w_mistral) / 2
  2. WEIGHTED AVERAGE: w_fused = α·w_qwen + (1-α)·w_mistral, α=0.7 (favor Math-strong)
  3. SELECTIVE: Use Qwen's CfC core + Mistral's log_std (best sigma control)

Evaluated on BOTH Mistral and Qwen hosts.

Usage:
    python experiments/phase39_multi_donor_fusion.py
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
MODELS = [
    {"name": "mistralai/Mistral-7B-Instruct-v0.3", "short": "Mistral-7B",
     "layers": list(range(15, 21))},
    {"name": "Qwen/Qwen2.5-7B-Instruct", "short": "Qwen2.5-7B",
     "layers": list(range(14, 20))},
]

SIGMA_MIN, SIGMA_MAX = 0.001, 0.15
SIGMA_FACTUAL, SIGMA_CREATIVE, SIGMA_MATH = 0.046, 0.080, 0.015
MAX_NEW_TOKENS = 100
SEED = 2026

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

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


# ═══ Evaluate (no training) ═══

def evaluate_cfc(model, tokenizer, dataset, actor, critic, classifier, hook,
                 target_layers, label=""):
    print(f"\n{'═'*60}")
    print(f"  🔬 {label} [EVAL ONLY]")
    print(f"{'═'*60}")

    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in target_layers if i < len(layers)]
    actor.eval()

    hx = None
    stats = {"factual": {"correct": 0, "total": 0, "sigmas": []},
             "creative": {"novelties": [], "grammar": 0, "total": 0, "sigmas": []},
             "math": {"correct": 0, "total": 0, "sigmas": []}}

    try:
        for idx, item in enumerate(dataset):
            pf = extract_prompt_features(model, tokenizer, item["question"], hook)
            with torch.no_grad():
                probs = classifier.predict_probs(pf.unsqueeze(0)).squeeze()
            p_creative = probs[1].item()
            p_math = probs[2].item()

            hidden_norm = hook.last_hidden_norm if hook.last_hidden_norm else 0.5
            features = torch.tensor(
                [0.0, 0.5, hidden_norm, idx / len(dataset), p_creative, p_math],
                dtype=torch.float32)

            with torch.no_grad():
                sigma, _, _, hx_new = actor.get_sigma_and_logprob(features, hx)

            hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()
            current_sigma = sigma.detach().item()
            hook.update_sigma(current_sigma)

            if item["type"] == "factual":
                ch = item["mc1_targets"]
                ci = ch["labels"].index(1) if 1 in ch["labels"] else 0
                lps = [compute_completion_logprob(model, tokenizer, item["prompt"], " " + c)
                       for c in ch["choices"]]
                correct = (np.argmax(lps) == ci)
                if correct: stats["factual"]["correct"] += 1
                stats["factual"]["total"] += 1
                stats["factual"]["sigmas"].append(current_sigma)

            elif item["type"] == "creative":
                resp = generate_text(model, tokenizer, item["prompt"])
                nov = compute_novelty(resp); gram = is_grammatical(resp)
                stats["creative"]["novelties"].append(nov)
                if gram: stats["creative"]["grammar"] += 1
                stats["creative"]["total"] += 1
                stats["creative"]["sigmas"].append(current_sigma)

            elif item["type"] == "math":
                resp = generate_text(model, tokenizer, item["prompt"], max_new_tokens=50)
                correct = check_math_answer(resp, item["correct_answer"])
                if correct: stats["math"]["correct"] += 1
                stats["math"]["total"] += 1
                stats["math"]["sigmas"].append(current_sigma)

            if (idx + 1) % 20 == 0:
                msf = np.mean(stats["factual"]["sigmas"][-10:]) if stats["factual"]["sigmas"] else 0
                msc = np.mean(stats["creative"]["sigmas"][-10:]) if stats["creative"]["sigmas"] else 0
                msm = np.mean(stats["math"]["sigmas"][-10:]) if stats["math"]["sigmas"] else 0
                print(f"  [{idx+1}/{len(dataset)}] σ_f={msf:.4f} σ_c={msc:.4f} σ_m={msm:.4f}")
    finally:
        for h in handles: h.remove()

    fa = stats["factual"]["correct"] / max(stats["factual"]["total"], 1) * 100
    an = np.mean(stats["creative"]["novelties"]) if stats["creative"]["novelties"] else 0
    gr = stats["creative"]["grammar"] / max(stats["creative"]["total"], 1) * 100
    ma = stats["math"]["correct"] / max(stats["math"]["total"], 1) * 100
    msf = np.mean(stats["factual"]["sigmas"]) if stats["factual"]["sigmas"] else 0
    msc = np.mean(stats["creative"]["sigmas"]) if stats["creative"]["sigmas"] else 0
    msm = np.mean(stats["math"]["sigmas"]) if stats["math"]["sigmas"] else 0
    mean_sigma = float(np.mean([msf, msc, msm]))

    print(f"\n  📊 {label}:")
    print(f"     Factual={fa:.1f}% Math={ma:.1f}% σ̄={mean_sigma:.4f}")

    return {
        "condition": label,
        "factual": {"acc": round(fa, 2), "mean_sigma": round(msf, 5)},
        "creative": {"novelty": round(float(an), 4), "grammar_rate": round(gr, 1),
                     "mean_sigma": round(msc, 5)},
        "math": {"acc": round(ma, 2), "mean_sigma": round(msm, 5)},
        "mean_sigma_all": round(mean_sigma, 5),
    }


# ═══ Fusion Strategies ═══

def create_fused_actors(qwen_path, mistral_path):
    """Create fused CfC actors from saved weights."""
    qwen_ckpt = torch.load(qwen_path, weights_only=True)
    mistral_ckpt = torch.load(mistral_path, weights_only=True)

    qwen_sd = qwen_ckpt["actor_state_dict"]
    mistral_sd = mistral_ckpt["actor_state_dict"]

    fusions = {}

    # Strategy 1: Simple Average
    avg_sd = {}
    for key in qwen_sd:
        avg_sd[key] = (qwen_sd[key].float() + mistral_sd[key].float()) / 2.0
    actor_avg = DualModeCfCActor(input_size=6, num_neurons=16)
    actor_avg.load_state_dict(avg_sd)
    fusions["simple_avg"] = actor_avg

    # Strategy 2: Qwen-weighted (70/30 favoring Math-strong Qwen)
    weighted_sd = {}
    for key in qwen_sd:
        weighted_sd[key] = 0.7 * qwen_sd[key].float() + 0.3 * mistral_sd[key].float()
    actor_weighted = DualModeCfCActor(input_size=6, num_neurons=16)
    actor_weighted.load_state_dict(weighted_sd)
    fusions["qwen_weighted_70"] = actor_weighted

    # Strategy 3: Selective — Qwen CfC core + Mistral log_std
    selective_sd = {}
    for key in qwen_sd:
        if "log_std" in key:
            selective_sd[key] = mistral_sd[key]  # Mistral's sigma variance
        else:
            selective_sd[key] = qwen_sd[key]  # Qwen's CfC weights
    actor_selective = DualModeCfCActor(input_size=6, num_neurons=16)
    actor_selective.load_state_dict(selective_sd)
    fusions["qwen_core_mistral_std"] = actor_selective

    # Also prepare pure donors for reference
    actor_qwen = DualModeCfCActor(input_size=6, num_neurons=16)
    actor_qwen.load_state_dict(qwen_sd)
    fusions["pure_qwen"] = actor_qwen

    actor_mistral = DualModeCfCActor(input_size=6, num_neurons=16)
    actor_mistral.load_state_dict(mistral_sd)
    fusions["pure_mistral"] = actor_mistral

    print(f"\n  🧪 Created {len(fusions)} CfC variants:")
    for name in fusions:
        print(f"     - {name}")

    return fusions


# ═══ Visualization ═══

def visualize_fusion(all_results, p37_data, p38_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Phase 39: Multi-Donor Fusion — Best of Breed CfC\n"
                 "Evaluating fused CfC weights on Mistral host",
                 fontsize=14, fontweight="bold")

    # Panel 1: Math Accuracy by fusion strategy
    ax1 = axes[0, 0]
    labels = [r["condition"].replace("Mistral: ", "") for r in all_results]
    math_accs = [r["math"]["acc"] for r in all_results]
    colors_bar = ["#e74c3c" if "mistral" in l.lower() else
                  "#3498db" if "qwen" in l.lower() else
                  "#2ecc71" if "avg" in l.lower() else
                  "#9b59b6" if "weighted" in l.lower() else
                  "#e67e22" for l in labels]
    bars = ax1.barh(range(len(labels)), math_accs, color=colors_bar, alpha=0.7)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Math Accuracy (%)")
    ax1.set_title("Math Accuracy by Fusion Strategy (Mistral Host)")
    ax1.grid(True, alpha=0.3, axis="x")
    # Add reference lines
    if p37_data:
        ax1.axvline(x=p37_data["native_result"]["math"]["acc"],
                     color="gray", linestyle="--", alpha=0.5, label=f"P37 native={p37_data['native_result']['math']['acc']}%")
    for bar, acc in zip(bars, math_accs):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{acc:.0f}%', va='center', fontsize=9)

    # Panel 2: σ comparison
    ax2 = axes[0, 1]
    sigmas = [r["mean_sigma_all"] for r in all_results]
    ax2.barh(range(len(labels)), sigmas, color=colors_bar, alpha=0.7)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("Mean σ")
    ax2.set_title("Mean σ by Fusion Strategy")
    ax2.axvline(x=0.074, color="gold", linestyle="--", alpha=0.7, label="σ≈0.074")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.legend(fontsize=9)

    # Panel 3: Factual Accuracy
    ax3 = axes[1, 0]
    fact_accs = [r["factual"]["acc"] for r in all_results]
    ax3.barh(range(len(labels)), fact_accs, color=colors_bar, alpha=0.7)
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlabel("Factual Accuracy (%)")
    ax3.set_title("Factual Accuracy by Fusion Strategy")
    ax3.grid(True, alpha=0.3, axis="x")

    # Panel 4: Grand verdict
    ax4 = axes[1, 1]; ax4.axis("off")

    best_math = max(all_results, key=lambda r: r["math"]["acc"])
    best_fact = max(all_results, key=lambda r: r["factual"]["acc"])

    txt = (f"MULTI-DONOR FUSION RESULTS\n"
           f"{'='*40}\n\n")
    for r in all_results:
        flag = " ★" if r == best_math else ""
        txt += (f"{r['condition'].replace('Mistral: ', '')}\n"
                f"  Math={r['math']['acc']:.0f}% Fact={r['factual']['acc']:.0f}% "
                f"σ̄={r['mean_sigma_all']:.4f}{flag}\n")

    txt += (f"\n{'='*40}\n"
            f"★ Best Math: {best_math['condition'].replace('Mistral: ', '')} "
            f"({best_math['math']['acc']:.0f}%)\n"
            f"★ Best Factual: {best_fact['condition'].replace('Mistral: ', '')} "
            f"({best_fact['factual']['acc']:.0f}%)\n\n")

    # Check if any fusion beats pure donors
    pure_q_math = next(r["math"]["acc"] for r in all_results if "pure_qwen" in r["condition"].lower())
    pure_m_math = next(r["math"]["acc"] for r in all_results if "pure_mistral" in r["condition"].lower())
    best_fusion_math = max(r["math"]["acc"] for r in all_results
                           if "pure" not in r["condition"].lower())
    if best_fusion_math > max(pure_q_math, pure_m_math):
        txt += "🎆 FUSION BEATS PURE DONORS!\n→ Chimera is superior to either parent!"
        vc = "#2ecc71"
    elif best_fusion_math >= max(pure_q_math, pure_m_math):
        txt += "🔬 FUSION MATCHES BEST DONOR\n→ No degradation from averaging"
        vc = "#3498db"
    else:
        txt += "📊 PURE DONOR IS BEST\n→ Fusion averages out specialization"
        vc = "#e67e22"

    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace", color=vc,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase39_multi_donor_fusion.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


# ═══ MAIN ═══

def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    # ─── Load previous results ───
    p37_path = os.path.join(RESULTS_DIR, "phase37_brain_transplant_log.json")
    p38_path = os.path.join(RESULTS_DIR, "phase38_reverse_transplant_log.json")
    p37_data = json.load(open(p37_path)) if os.path.exists(p37_path) else None
    p38_data = json.load(open(p38_path)) if os.path.exists(p38_path) else None

    # ─── Create fused CfC actors ───
    qwen_weights = os.path.join(WEIGHTS_DIR, "donor_qwen_cfc.pt")
    mistral_weights = os.path.join(WEIGHTS_DIR, "donor_mistral_cfc.pt")

    if not os.path.exists(qwen_weights) or not os.path.exists(mistral_weights):
        print("❌ Missing donor weights! Run Phase 37 and Phase 38 first.")
        return

    fusions = create_fused_actors(qwen_weights, mistral_weights)

    # ─── Evaluate on Mistral (primary host) ───
    print("\n" + "🧬"*30)
    print("  Evaluating ALL fusion strategies on Mistral host")
    print("🧬"*30)

    model, tokenizer = load_model(MODELS[0]["name"], MODELS[0]["short"])
    hook = AdaptiveSNNHook(sigma=0.05)
    dataset = build_dataset(tokenizer)

    factual_qs = [item["question"] for item in dataset if item["type"] == "factual"]
    math_qs = [item["question"] for item in dataset if item["type"] == "math"]

    classifier = TaskClassifier3()
    classifier = pretrain_classifier(
        classifier, model, tokenizer, hook,
        factual_qs, CREATIVE_PROMPTS, math_qs)

    all_results = []
    target_layers = MODELS[0]["layers"]

    for name, actor in fusions.items():
        critic = CfCCritic(input_size=6, hidden_size=64)
        hook = AdaptiveSNNHook(sigma=0.05)
        result = evaluate_cfc(model, tokenizer, dataset, actor, critic,
                              classifier, hook, target_layers,
                              f"Mistral: {name}")
        all_results.append(result)

    # ─── Visualization ───
    fig_path = visualize_fusion(all_results, p37_data, p38_data)

    # ─── Save results ───
    elapsed = time.time() - t_start

    # Find best
    best_math = max(all_results, key=lambda r: r["math"]["acc"])
    best_fact = max(all_results, key=lambda r: r["factual"]["acc"])

    output = {
        "experiment": "Phase 39: Multi-Donor Fusion",
        "host": MODELS[0]["short"],
        "elapsed_minutes": round(elapsed / 60, 1),
        "fusion_results": all_results,
        "best_math": {"strategy": best_math["condition"], "acc": best_math["math"]["acc"]},
        "best_factual": {"strategy": best_fact["condition"], "acc": best_fact["factual"]["acc"]},
        "figure_path": fig_path,
    }

    log_path = os.path.join(RESULTS_DIR, "phase39_multi_donor_fusion_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Results: {log_path}")

    # Grand summary
    print(f"\n{'═'*60}")
    print(f"  🏆 PHASE 39: MULTI-DONOR FUSION — GRAND SUMMARY")
    print(f"{'═'*60}")
    for r in all_results:
        flag = " ★" if r == best_math else ""
        print(f"  {r['condition']:40s} Math={r['math']['acc']:5.1f}% Fact={r['factual']['acc']:5.1f}% σ̄={r['mean_sigma_all']:.4f}{flag}")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
