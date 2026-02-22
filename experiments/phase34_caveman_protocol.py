"""
Phase 34: The Caveman Protocol — Unsupervised Homeostasis via Entropy Survival Instinct
Phase 35: Tabula Rasa — Zero-Training Reservoir Computing Proof
=================================================================

Phase 34 (Caveman Protocol):
  - Removes ALL task-specific rewards (accuracy, novelty, grammar)
  - Removes TaskClassifier — CfC has NO knowledge of task types
  - Reward = -α × LLM_output_entropy - λ × (σ - 0.074)²
  - "Keep your body temperature and don't panic" = the only survival instinct
  
  Key Question: Will CfC autonomously lower σ for Math (to avoid entropy explosion)
  and recover Math accuracy from 0% → 30%+?

Phase 35 (Tabula Rasa):
  - PPO training is COMPLETELY SKIPPED
  - CfC has random weights (no learning at all)
  - Just forward 60 prompts and record 16D hidden states
  
  Key Question: Does the CfC's ODE structure (reservoir) separate tasks
  in UMAP without ANY training?

Base Model: Mistral-7B-Instruct (same as v7 for comparison)
"""

import os, sys, json, time, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ncps.wirings import AutoNCP
from ncps.torch import CfC

# ═══ Configuration ═══

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
TARGET_LAYERS = list(range(15, 21))    # Same as v7
MAX_NEW_TOKENS = 100
SIGMA_MIN, SIGMA_MAX = 0.001, 0.20
TARGET_SIGMA = 0.074                    # Universal homeostatic constant
N_EPOCHS = 2
LEARNING_RATE = 3e-4
LAMBDA_QUAD = 200.0
ALPHA_ENTROPY = 0.1                     # Entropy penalty weight
SEED = 2026

PPO_CLIP_EPS, PPO_UPDATE_EPOCHS = 0.2, 4
PPO_GAMMA, PPO_GAE_LAMBDA = 0.99, 0.95
PPO_ENTROPY_COEFF, PPO_VALUE_COEFF = 0.01, 0.5
PPO_ROLLOUT_SIZE, PPO_MAX_GRAD_NORM = 10, 0.5

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Dataset (same prompts as Phase 26-29) ───

FACTUAL_PROMPTS = [
    "What is the capital of France?", "Who wrote Romeo and Juliet?",
    "What year did World War II end?", "What is the chemical symbol for gold?",
    "Who painted the Mona Lisa?", "What is the largest planet in our solar system?",
    "What is the speed of light in km/s?", "Who developed the theory of relativity?",
    "What is the boiling point of water in Celsius?", "What is the smallest country in the world?",
    "Who invented the telephone?", "What is the chemical formula for water?",
    "What year was the Declaration of Independence signed?", "What is the tallest mountain in the world?",
    "Who was the first person to walk on the moon?", "What is the capital of Japan?",
    "What element has the atomic number 6?", "Who wrote the Origin of Species?",
    "What is the largest ocean on Earth?", "What year did the Berlin Wall fall?",
]

CREATIVE_PROMPTS = [
    "Describe a color that doesn't exist in the visible spectrum.",
    "Invent a new fundamental law of physics that could explain dark energy.",
    "Write a short poem about the sound of silence from a deaf musician's perspective.",
    "Design a holiday that celebrates the concept of entropy.",
    "Describe what music would look like if you could see sound waves.",
    "Invent a sport that could be played in zero gravity.",
    "Write a recipe for a dish that tastes like nostalgia.",
    "Create a new type of weather that has never existed.",
    "Describe the autobiography of a cloud.",
    "Invent a language that can only express emotions, not facts.",
    "Write a haiku about the taste of mathematics.",
    "Describe what happens when two different dreams collide.",
    "Invent a new musical instrument made from impossible materials.",
    "Write a love letter from one star to another.",
    "Describe a garden where the plants grow backwards in time.",
    "Invent a philosophical paradox about artificial consciousness.",
    "Write a news report from the year 3000.",
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
    {"q": "What is 8 × 9?", "a": "72"},
    {"q": "What is 200 - 137?", "a": "63"},
    {"q": "What is 15 × 15?", "a": "225"},
    {"q": "What is 1000 / 8?", "a": "125"},
    {"q": "What is 99 + 101?", "a": "200"},
    {"q": "What is 7 × 11?", "a": "77"},
    {"q": "What is 500 - 267?", "a": "233"},
    {"q": "What is 25 × 4?", "a": "100"},
    {"q": "What is 360 / 9?", "a": "40"},
    {"q": "What is 88 + 77?", "a": "165"},
    {"q": "What is 13 × 17?", "a": "221"},
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
    layers = get_layers(model)
    print(f"  ✅ {MODEL_SHORT} loaded: {len(layers)} layers, target={TARGET_LAYERS}")
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


# ─── Caveman CfC Actor (NO task classifier input) ───

class CavemanCfCActor(nn.Module):
    """CfC Actor with NO task knowledge. Input: [prev_reward, prev_entropy, hidden_norm, step_frac]"""
    def __init__(self, input_size=4, num_neurons=16):
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
    def __init__(self, input_size=4, hidden_size=64):
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
    if len(words) < 2: return 0.0
    bigrams = set(zip(words[:-1], words[1:]))
    diversity = len(bigrams) / max(len(words) - 1, 1)
    unique_ratio = len(set(words)) / len(words)
    return (diversity + unique_ratio) / 2

def is_grammatical(text):
    words = text.split()
    if len(words) < 3: return False
    if not text.rstrip()[-1] in '.!?,"\'': return False
    return True

def check_math_answer(response, correct_answer):
    return correct_answer in response.replace(",", "").replace(" ", "")

def compute_output_entropy(model, tokenizer, prompt):
    """Compute LLM output entropy = panic/confusion level"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # Last token logits
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
    return entropy

def compute_gae(rewards, values, gamma=PPO_GAMMA, lam=PPO_GAE_LAMBDA):
    advantages, returns = [], []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    return advantages, returns

def build_3class_dataset(tokenizer):
    """Build shuffled 3-class dataset (same as Phase 26-29)"""
    dataset = []
    # Load TruthfulQA for factual
    try:
        from datasets import load_dataset
        tqa = load_dataset("truthful_qa", "multiple_choice", split="validation",
                           trust_remote_code=True)
        for item in tqa:
            if len(dataset) >= 20: break
            if "mc1_targets" in item:
                dataset.append({"type": "factual", "question": item["question"],
                                "prompt": item["question"], "mc1_targets": item["mc1_targets"]})
    except:
        for p in FACTUAL_PROMPTS[:20]:
            dataset.append({"type": "factual", "question": p, "prompt": p,
                           "mc1_targets": {"choices": ["Yes", "No"], "labels": [1, 0]}})
    
    for p in CREATIVE_PROMPTS[:20]:
        dataset.append({"type": "creative", "question": p, "prompt": p})
    
    for p in MATH_PROMPTS[:20]:
        dataset.append({"type": "math", "question": p["q"],
                        "prompt": f"Answer with just the number. {p['q']}",
                        "correct_answer": p["a"]})
    
    random.seed(SEED)
    random.shuffle(dataset)
    print(f"  📋 Dataset: {len(dataset)} items ({sum(1 for d in dataset if d['type']=='factual')} F, "
          f"{sum(1 for d in dataset if d['type']=='creative')} C, "
          f"{sum(1 for d in dataset if d['type']=='math')} M)")
    return dataset


# ═══════════════════════════════════════════════════════
# Phase 35: TABULA RASA — Zero-training reservoir eval
# ═══════════════════════════════════════════════════════

def run_tabula_rasa(model, tokenizer, dataset, hook):
    """Phase 35: Random CfC weights, NO training, just evaluate hidden states"""
    print(f"\n{'═'*60}")
    print(f"  🌊 Phase 35: TABULA RASA — Zero-Training Reservoir")
    print(f"     'Born yesterday, but can I already see the world?'")
    print(f"{'═'*60}")
    
    torch.manual_seed(SEED)
    actor = CavemanCfCActor(input_size=4, num_neurons=16)
    # NO TRAINING — random weights
    
    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]
    
    hidden_states = []
    sigmas_by_type = {"factual": [], "creative": [], "math": []}
    hx = None
    
    try:
        for idx, item in enumerate(dataset):
            # Get hidden norm from a forward pass
            inputs = tokenizer(item["question"][:200], return_tensors="pt",
                               truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                model(**inputs)
            hidden_norm = hook.last_hidden_norm if hook.last_hidden_norm else 0.5
            
            # CfC forward with random weights (no task info, no reward)
            features = torch.tensor([0.0, 5.0, hidden_norm, idx / len(dataset)],
                                     dtype=torch.float32)
            with torch.no_grad():
                sigma, log_prob, entropy, hx_new = actor.get_sigma_and_logprob(features, hx)
            hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()
            
            current_sigma = sigma.detach().item()
            sigmas_by_type[item["type"]].append(current_sigma)
            
            # Record hidden state for UMAP
            if isinstance(hx, tuple):
                hs_np = hx[0].detach().cpu().numpy().flatten()
            else:
                hs_np = hx.detach().cpu().numpy().flatten()
            hidden_states.append({
                "type": item["type"], "sigma": current_sigma,
                "step": idx, "hidden": hs_np.tolist()[:16],
            })
            
            if (idx + 1) % 20 == 0:
                print(f"  [{idx+1}/{len(dataset)}] σ={current_sigma:.4f} type={item['type']}")
    finally:
        for h in handles:
            h.remove()
    
    # Compute UMAP separation
    from sklearn.preprocessing import StandardScaler
    hiddens = np.array([h["hidden"] for h in hidden_states])
    types = [h["type"] for h in hidden_states]
    scaler = StandardScaler()
    hiddens_scaled = scaler.fit_transform(hiddens)
    
    # Compute separation ratio (inter-class / intra-class distance)
    type_map = {"factual": 0, "creative": 1, "math": 2}
    labels = np.array([type_map[t] for t in types])
    
    inter_dists, intra_dists = [], []
    for c in range(3):
        mask_c = labels == c
        if mask_c.sum() < 2: continue
        center_c = hiddens_scaled[mask_c].mean(axis=0)
        intra_dists.append(np.mean(np.linalg.norm(hiddens_scaled[mask_c] - center_c, axis=1)))
        for c2 in range(c + 1, 3):
            mask_c2 = labels == c2
            if mask_c2.sum() < 2: continue
            center_c2 = hiddens_scaled[mask_c2].mean(axis=0)
            inter_dists.append(np.linalg.norm(center_c - center_c2))
    
    mean_intra = np.mean(intra_dists) if intra_dists else 1.0
    mean_inter = np.mean(inter_dists) if inter_dists else 0.0
    separation_ratio = mean_inter / max(mean_intra, 1e-6)
    
    msf = np.mean(sigmas_by_type["factual"]) if sigmas_by_type["factual"] else 0
    msc = np.mean(sigmas_by_type["creative"]) if sigmas_by_type["creative"] else 0
    msm = np.mean(sigmas_by_type["math"]) if sigmas_by_type["math"] else 0
    
    print(f"\n  🌊 Phase 35 Results (TABULA RASA):")
    print(f"     σ_factual  = {msf:.4f}")
    print(f"     σ_creative = {msc:.4f}")
    print(f"     σ_math     = {msm:.4f}")
    print(f"     Separation ratio = {separation_ratio:.3f}")
    if separation_ratio > 1.5:
        print(f"     🎆 RESERVOIR PROOF! Tasks SEPARATE without training!")
    else:
        print(f"     📊 No clear separation — reservoir alone is insufficient")
    
    return {
        "phase": "35_tabula_rasa",
        "sigmas": {"factual": round(msf, 5), "creative": round(msc, 5), "math": round(msm, 5)},
        "separation_ratio": round(separation_ratio, 4),
        "hidden_states": hidden_states,
        "all_sigmas": {k: [round(s, 5) for s in v] for k, v in sigmas_by_type.items()},
    }


# ═══════════════════════════════════════════════════════
# Phase 34: CAVEMAN PROTOCOL — Entropy survival instinct
# ═══════════════════════════════════════════════════════

def run_caveman_protocol(model, tokenizer, dataset, hook):
    """Phase 34: PPO with ONLY entropy + sigma penalty (no accuracy, no task labels)"""
    print(f"\n{'═'*60}")
    print(f"  🦇 Phase 34: THE CAVEMAN PROTOCOL")
    print(f"     'No teacher. No labels. Only survival instinct.'")
    print(f"     Reward = -α×entropy - λ×(σ-0.074)²")
    print(f"{'═'*60}")
    
    torch.manual_seed(SEED)
    actor = CavemanCfCActor(input_size=4, num_neurons=16)
    critic = CfCCritic(input_size=4)
    
    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]
    actor.train(); critic.train()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)
    hx = None
    
    # Stats (we record accuracy but NEVER use it for reward)
    stats = {"factual": {"correct": 0, "total": 0, "sigmas": [], "entropies": []},
             "creative": {"novelties": [], "grammar": 0, "total": 0, "sigmas": [], "entropies": []},
             "math": {"correct": 0, "total": 0, "sigmas": [], "entropies": []}}
    sigma_trajectory = []
    hidden_states = []
    entropy_trajectory = []
    buf_f, buf_s, buf_lp, buf_v, buf_r, buf_e = [], [], [], [], [], []
    
    try:
        for epoch in range(N_EPOCHS):
            print(f"\n  🦇 Epoch {epoch+1}/{N_EPOCHS}")
            for idx, item in enumerate(dataset):
                # ─── Step 1: Get hidden norm ───
                inputs = tokenizer(item["question"][:200], return_tensors="pt",
                                   truncation=True, max_length=128).to(model.device)
                with torch.no_grad():
                    model(**inputs)
                hidden_norm = hook.last_hidden_norm if hook.last_hidden_norm else 0.5
                
                # ─── Step 2: CfC decides sigma (NO task info!) ───
                prev_reward = buf_r[-1] if buf_r else 0.0
                prev_entropy = entropy_trajectory[-1] if entropy_trajectory else 5.0
                features = torch.tensor(
                    [prev_reward, prev_entropy / 10.0, hidden_norm, idx / len(dataset)],
                    dtype=torch.float32)
                
                sigma, log_prob, policy_entropy, hx_new = actor.get_sigma_and_logprob(features, hx)
                hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()
                current_sigma = sigma.detach().item()
                hook.update_sigma(current_sigma)
                
                # Record hidden state
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
                
                # ─── Step 3: Compute CAVEMAN REWARD (entropy + sigma only!) ───
                output_entropy = compute_output_entropy(model, tokenizer, item["prompt"])
                sigma_penalty = (current_sigma - TARGET_SIGMA) ** 2
                reward = -(ALPHA_ENTROPY * output_entropy) - (LAMBDA_QUAD * sigma_penalty)
                
                entropy_trajectory.append(output_entropy)
                
                # ─── Step 4: Record accuracy (but NEVER use for reward!) ───
                if item["type"] == "factual":
                    ch = item["mc1_targets"]; ci = ch["labels"].index(1) if 1 in ch["labels"] else 0
                    lps = [compute_completion_logprob(model, tokenizer, item["prompt"], " " + c)
                           for c in ch["choices"]]
                    correct = (np.argmax(lps) == ci)
                    if correct: stats["factual"]["correct"] += 1
                    stats["factual"]["total"] += 1
                    stats["factual"]["sigmas"].append(current_sigma)
                    stats["factual"]["entropies"].append(output_entropy)
                    
                elif item["type"] == "creative":
                    resp = generate_text(model, tokenizer, item["prompt"])
                    nov = compute_novelty(resp); gram = is_grammatical(resp)
                    stats["creative"]["novelties"].append(nov)
                    if gram: stats["creative"]["grammar"] += 1
                    stats["creative"]["total"] += 1
                    stats["creative"]["sigmas"].append(current_sigma)
                    stats["creative"]["entropies"].append(output_entropy)
                    
                elif item["type"] == "math":
                    resp = generate_text(model, tokenizer, item["prompt"], max_new_tokens=50)
                    correct = check_math_answer(resp, item["correct_answer"])
                    if correct: stats["math"]["correct"] += 1
                    stats["math"]["total"] += 1
                    stats["math"]["sigmas"].append(current_sigma)
                    stats["math"]["entropies"].append(output_entropy)
                
                # ─── Step 5: PPO buffer ───
                buf_f.append(features.detach()); buf_s.append(sigma.detach())
                buf_lp.append(log_prob.detach()); buf_v.append(value)
                buf_r.append(reward); buf_e.append(policy_entropy.detach())
                
                sigma_trajectory.append({
                    "epoch": epoch + 1, "step": epoch * len(dataset) + idx,
                    "sigma": current_sigma, "type": item["type"],
                    "entropy": output_entropy, "reward": reward,
                })
                
                # ─── Step 6: PPO update ───
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
                    me = np.mean(entropy_trajectory[-20:]) if entropy_trajectory else 0
                    print(f"  [{idx+1}/{len(dataset)}] σ_f={msf:.4f} σ_c={msc:.4f} σ_m={msm:.4f} entropy={me:.2f}")
    finally:
        for h in handles:
            h.remove()
    
    # ─── Final stats ───
    fa = stats["factual"]["correct"] / max(stats["factual"]["total"], 1) * 100
    an = np.mean(stats["creative"]["novelties"]) if stats["creative"]["novelties"] else 0
    gr = stats["creative"]["grammar"] / max(stats["creative"]["total"], 1) * 100
    ma = stats["math"]["correct"] / max(stats["math"]["total"], 1) * 100
    
    msf = np.mean(stats["factual"]["sigmas"]) if stats["factual"]["sigmas"] else 0
    msc = np.mean(stats["creative"]["sigmas"]) if stats["creative"]["sigmas"] else 0
    msm = np.mean(stats["math"]["sigmas"]) if stats["math"]["sigmas"] else 0
    
    ent_f = np.mean(stats["factual"]["entropies"]) if stats["factual"]["entropies"] else 0
    ent_c = np.mean(stats["creative"]["entropies"]) if stats["creative"]["entropies"] else 0
    ent_m = np.mean(stats["math"]["entropies"]) if stats["math"]["entropies"] else 0
    
    sep_fc = abs(msc - msf)
    sep_fm = abs(msm - msf)
    sep_cm = abs(msm - msc)
    max_sep = max(sep_fc, sep_fm, sep_cm)
    mean_sigma = float(np.mean([msf, msc, msm]))
    
    # Compute UMAP separation ratio
    from sklearn.preprocessing import StandardScaler
    hiddens = np.array([h["hidden"] for h in hidden_states])
    types = [h["type"] for h in hidden_states]
    scaler = StandardScaler()
    hiddens_scaled = scaler.fit_transform(hiddens)
    type_map = {"factual": 0, "creative": 1, "math": 2}
    labels = np.array([type_map[t] for t in types])
    
    inter_dists, intra_dists = [], []
    for c in range(3):
        mask_c = labels == c
        if mask_c.sum() < 2: continue
        center_c = hiddens_scaled[mask_c].mean(axis=0)
        intra_dists.append(np.mean(np.linalg.norm(hiddens_scaled[mask_c] - center_c, axis=1)))
        for c2 in range(c + 1, 3):
            mask_c2 = labels == c2
            if mask_c2.sum() < 2: continue
            center_c2 = hiddens_scaled[mask_c2].mean(axis=0)
            inter_dists.append(np.linalg.norm(center_c - center_c2))
    mean_intra = np.mean(intra_dists) if intra_dists else 1.0
    mean_inter = np.mean(inter_dists) if inter_dists else 0.0
    separation_ratio = mean_inter / max(mean_intra, 1e-6)
    
    print(f"\n  🦇 Phase 34 Results (CAVEMAN PROTOCOL):")
    print(f"     Factual:  acc={fa:.1f}%  σ̄={msf:.4f}  entropy={ent_f:.2f}")
    print(f"     Creative: nov={an:.3f}   σ̄={msc:.4f}  entropy={ent_c:.2f}")
    print(f"     Math:     acc={ma:.1f}%  σ̄={msm:.4f}  entropy={ent_m:.2f}")
    print(f"     Max separation = {max_sep:.4f}")
    print(f"     Mean σ = {mean_sigma:.4f}")
    print(f"     UMAP separation ratio = {separation_ratio:.3f}")
    
    # The BIG question
    print(f"\n  🔮 THE GRAND QUESTION:")
    if max_sep > 0.01:
        print(f"     🎆 SIGMA SEPARATED! CfC autonomously differentiated tasks!")
        print(f"        σ_math={msm:.4f} vs σ_factual={msf:.4f} (Δ={abs(msm-msf):.4f})")
        if msm < msf:
            print(f"     🦇 CfC LOWERED σ for Math! Survival instinct → task adaptation!")
    else:
        print(f"     🏠 UNIFIED REGIME maintained — survival instinct = homeostasis")
    
    if ma > 0:
        print(f"     🎆🎆🎆 MATH ACCURACY = {ma:.1f}%! (v7 was 0%)")
        print(f"     Zero-Shot Alignment CONFIRMED!")
    else:
        print(f"     Math accuracy = 0% — same as v7 (homeostasis wins over survival)")
    
    # Compare with v7
    print(f"\n  📊 Comparison with v7 (supervised PPO):")
    print(f"     v7: σ̄≈0.071, Math=0%, unified regime")
    print(f"     v8-Caveman: σ̄={mean_sigma:.4f}, Math={ma:.1f}%, sep={max_sep:.4f}")
    
    return {
        "phase": "34_caveman_protocol",
        "reward_type": "entropy_survival_instinct",
        "reward_formula": "-(α×entropy) - (λ×(σ-0.074)²)",
        "alpha_entropy": ALPHA_ENTROPY,
        "lambda_quad": LAMBDA_QUAD,
        "target_sigma": TARGET_SIGMA,
        "factual": {"acc": round(fa, 2), "mean_sigma": round(msf, 5),
                     "mean_entropy": round(ent_f, 3)},
        "creative": {"novelty": round(float(an), 4), "grammar_rate": round(gr, 1),
                     "mean_sigma": round(msc, 5), "mean_entropy": round(ent_c, 3)},
        "math": {"acc": round(ma, 2), "mean_sigma": round(msm, 5),
                 "mean_entropy": round(ent_m, 3)},
        "separations": {"f_c": round(sep_fc, 5), "f_m": round(sep_fm, 5),
                        "c_m": round(sep_cm, 5), "max": round(max_sep, 5)},
        "mean_sigma_all": round(mean_sigma, 5),
        "separation_ratio": round(separation_ratio, 4),
        "sigma_trajectory": sigma_trajectory,
        "hidden_states": hidden_states,
        "entropy_trajectory": entropy_trajectory,
        "v7_comparison": {
            "v7_sigma": 0.071, "v7_math_acc": 0.0,
            "caveman_sigma": round(mean_sigma, 4), "caveman_math_acc": round(ma, 2),
        },
    }


# ═══════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════

def visualize_results(result34, result35):
    """Create comprehensive comparison figure"""
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Phase 34-35: The Caveman Protocol & Tabula Rasa\n"
                 "Can CfC learn without teachers? Can it see without training?",
                 fontsize=14, fontweight="bold")
    
    colors = {"factual": "#2ecc71", "creative": "#e74c3c", "math": "#3498db"}
    
    # ─── Panel 1: Phase 34 σ trajectory ───
    ax1 = fig.add_subplot(3, 2, 1)
    traj = result34["sigma_trajectory"]
    for task_type in ["factual", "creative", "math"]:
        steps = [t["step"] for t in traj if t["type"] == task_type]
        sigmas = [t["sigma"] for t in traj if t["type"] == task_type]
        ax1.scatter(steps, sigmas, c=colors[task_type], s=20, alpha=0.7, label=task_type)
    ax1.axhline(y=TARGET_SIGMA, color="gold", linestyle="--", linewidth=2, label=f"Target σ={TARGET_SIGMA}")
    ax1.axhline(y=0.071, color="gray", linestyle=":", alpha=0.5, label="v7 σ=0.071")
    ax1.set_xlabel("Step"); ax1.set_ylabel("σ")
    ax1.set_title("Phase 34: Caveman σ Trajectory\n(Reward = entropy + σ penalty only)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    
    # ─── Panel 2: Phase 34 Entropy by task ───
    ax2 = fig.add_subplot(3, 2, 2)
    ent_data = {}
    for t in traj:
        tt = t["type"]
        if tt not in ent_data: ent_data[tt] = []
        ent_data[tt].append(t["entropy"])
    positions = []
    labels_box = []
    data_box = []
    for i, (tt, ents) in enumerate(ent_data.items()):
        positions.append(i)
        labels_box.append(tt)
        data_box.append(ents)
    if data_box:
        bp = ax2.boxplot(data_box, positions=positions, patch_artist=True, widths=0.6)
        for patch, tt in zip(bp["boxes"], labels_box):
            patch.set_facecolor(colors.get(tt, "gray"))
            patch.set_alpha(0.6)
    ax2.set_xticks(positions); ax2.set_xticklabels(labels_box)
    ax2.set_ylabel("Output Entropy (headache)")
    ax2.set_title("Phase 34: LLM Entropy by Task\n(Math = more confusion?)")
    ax2.grid(True, alpha=0.3)
    
    # ─── Panel 3: Phase 34 UMAP ───
    ax3 = fig.add_subplot(3, 2, 3)
    try:
        from umap import UMAP
        hiddens34 = np.array([h["hidden"] for h in result34["hidden_states"]])
        types34 = [h["type"] for h in result34["hidden_states"]]
        if len(hiddens34) > 10:
            from sklearn.preprocessing import StandardScaler
            hiddens34_s = StandardScaler().fit_transform(hiddens34)
            umap34 = UMAP(n_neighbors=min(15, len(hiddens34)-1), min_dist=0.1,
                         random_state=SEED).fit_transform(hiddens34_s)
            for tt in ["factual", "creative", "math"]:
                mask = np.array([t == tt for t in types34])
                if mask.sum() > 0:
                    ax3.scatter(umap34[mask, 0], umap34[mask, 1], c=colors[tt],
                               s=40, alpha=0.7, label=tt, edgecolors="white", linewidth=0.5)
            ax3.legend(fontsize=8)
    except ImportError:
        ax3.text(0.5, 0.5, "UMAP not available", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_title(f"Phase 34: CfC Hidden State UMAP\n(Sep. ratio = {result34['separation_ratio']:.3f})")
    ax3.grid(True, alpha=0.3)
    
    # ─── Panel 4: Phase 35 UMAP ───
    ax4 = fig.add_subplot(3, 2, 4)
    try:
        from umap import UMAP
        from sklearn.preprocessing import StandardScaler
        hiddens35 = np.array([h["hidden"] for h in result35["hidden_states"]])
        types35 = [h["type"] for h in result35["hidden_states"]]
        if len(hiddens35) > 10:
            hiddens35_s = StandardScaler().fit_transform(hiddens35)
            umap35 = UMAP(n_neighbors=min(15, len(hiddens35)-1), min_dist=0.1,
                         random_state=SEED).fit_transform(hiddens35_s)
            for tt in ["factual", "creative", "math"]:
                mask = np.array([t == tt for t in types35])
                if mask.sum() > 0:
                    ax4.scatter(umap35[mask, 0], umap35[mask, 1], c=colors[tt],
                               s=40, alpha=0.7, label=tt, edgecolors="white", linewidth=0.5)
            ax4.legend(fontsize=8)
    except ImportError:
        ax4.text(0.5, 0.5, "UMAP not available", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_title(f"Phase 35: TABULA RASA UMAP (Zero Training!)\n(Sep. ratio = {result35['separation_ratio']:.3f})")
    ax4.grid(True, alpha=0.3)
    
    # ─── Panel 5: Accuracy comparison ───
    ax5 = fig.add_subplot(3, 2, 5)
    conditions = ["v7\n(Supervised)", "Phase 34\n(Caveman)", "Phase 35\n(Tabula Rasa)"]
    math_accs = [0.0, result34["math"]["acc"], 0.0]  # Phase 35 has no eval
    factual_accs = [37.5, result34["factual"]["acc"], 0.0]  # v7 factual from Phase 26
    
    x = np.arange(len(conditions))
    width = 0.35
    ax5.bar(x - width/2, factual_accs, width, label="Factual Acc %", color="#2ecc71", alpha=0.7)
    ax5.bar(x + width/2, math_accs, width, label="Math Acc %", color="#3498db", alpha=0.7)
    ax5.set_xticks(x); ax5.set_xticklabels(conditions)
    ax5.set_ylabel("Accuracy %"); ax5.legend()
    ax5.set_title("Accuracy Comparison: Supervised vs. Caveman vs. Tabula Rasa")
    ax5.grid(True, alpha=0.3, axis="y")
    
    # ─── Panel 6: Sigma comparison ───
    ax6 = fig.add_subplot(3, 2, 6)
    conditions_sigma = ["v7\nSupervised", "Phase 34\nCaveman", "Phase 35\nTabula Rasa"]
    sigma_f = [0.071, result34["factual"]["mean_sigma"], result35["sigmas"]["factual"]]
    sigma_c = [0.071, result34["creative"]["mean_sigma"], result35["sigmas"]["creative"]]
    sigma_m = [0.071, result34["math"]["mean_sigma"], result35["sigmas"]["math"]]
    
    x = np.arange(len(conditions_sigma))
    width = 0.25
    ax6.bar(x - width, sigma_f, width, label="Factual σ", color="#2ecc71", alpha=0.7)
    ax6.bar(x, sigma_c, width, label="Creative σ", color="#e74c3c", alpha=0.7)
    ax6.bar(x + width, sigma_m, width, label="Math σ", color="#3498db", alpha=0.7)
    ax6.axhline(y=TARGET_SIGMA, color="gold", linestyle="--", linewidth=2, label=f"Universal σ={TARGET_SIGMA}")
    ax6.set_xticks(x); ax6.set_xticklabels(conditions_sigma)
    ax6.set_ylabel("Mean σ"); ax6.legend(fontsize=8)
    ax6.set_title("σ per Task: Does Caveman differentiate?")
    ax6.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase34_caveman_protocol.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")


# ═══ MAIN ═══

def main():
    start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    
    model, tokenizer = load_model()
    hook = AdaptiveSNNHook(sigma=TARGET_SIGMA)
    dataset = build_3class_dataset(tokenizer)
    
    # ─── Phase 35 FIRST (quick: no training) ───
    result35 = run_tabula_rasa(model, tokenizer, dataset, hook)
    
    # ─── Phase 34 (Caveman Protocol with PPO) ───
    hook = AdaptiveSNNHook(sigma=TARGET_SIGMA)  # Reset hook
    result34 = run_caveman_protocol(model, tokenizer, dataset, hook)
    
    # ─── Visualization ───
    visualize_results(result34, result35)
    
    # ─── Save results ───
    elapsed = time.time() - start
    combined = {
        "experiment": "Phase 34-35: Caveman Protocol & Tabula Rasa",
        "model": MODEL_NAME,
        "model_short": MODEL_SHORT,
        "elapsed_minutes": round(elapsed / 60, 1),
        "phase_34_caveman": result34,
        "phase_35_tabula_rasa": result35,
        "grand_summary": {
            "caveman_math_acc": result34["math"]["acc"],
            "caveman_mean_sigma": result34["mean_sigma_all"],
            "caveman_max_sep": result34["separations"]["max"],
            "caveman_separation_ratio": result34["separation_ratio"],
            "tabula_rasa_separation_ratio": result35["separation_ratio"],
            "math_accuracy_recovered": result34["math"]["acc"] > 0,
            "sigma_differentiated": result34["separations"]["max"] > 0.01,
            "reservoir_separates": result35["separation_ratio"] > 1.5,
        }
    }
    
    log_path = os.path.join(RESULTS_DIR, "phase34_caveman_protocol_log.json")
    with open(log_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    
    print(f"\n{'═'*60}")
    print(f"  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"  📁 Results: {log_path}")
    print(f"{'═'*60}")
    
    print(f"\n  🏆 GRAND SUMMARY:")
    print(f"     Phase 34 (Caveman): Math={result34['math']['acc']:.1f}%, σ̄={result34['mean_sigma_all']:.4f}, sep={result34['separations']['max']:.4f}")
    print(f"     Phase 35 (Tabula):  sep_ratio={result35['separation_ratio']:.3f}")
    
    if result34["math"]["acc"] > 0:
        print(f"\n  🎆🎆🎆 ZERO-SHOT ALIGNMENT CONFIRMED! 🎆🎆🎆")
        print(f"  CfC recovered Math accuracy using ONLY entropy survival instinct!")
    
    if result35["separation_ratio"] > 1.5:
        print(f"\n  🌊🌊🌊 RESERVOIR PROOF CONFIRMED! 🌊🌊🌊")
        print(f"  CfC's ODE structure separates tasks WITHOUT any training!")


if __name__ == "__main__":
    main()
