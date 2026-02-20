"""
Phase 23: Creative Spark — CfC-Controlled Creativity
=====================================================

Tests whether CfC-controlled adaptive noise produces creative outputs
that remain grammatically correct, unlike Phase 11-12 where static
noise destroyed coherence.

Three conditions:
  1. Base (no noise) — boring but correct baseline
  2. Static σ=0.05 — Phase 11-style, likely grammar collapse
  3. CfC-Online — adaptive σ, should balance creativity + grammar

Evaluation:
  - Perplexity (grammar quality, lower = better)
  - Novelty score (n-gram diversity, higher = more creative)
  - Human-readable outputs for qualitative assessment

Usage:
    python experiments/phase23_creative_spark.py
"""

import torch
import json
import os
import sys
import time
import datetime
import random
import math
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─── Config ───
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TARGET_LAYERS = list(range(15, 21))
MAX_NEW_TOKENS = 150  # Longer for creative text
SEED = 2026

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Creative Prompts ───
CREATIVE_PROMPTS = [
    "Describe a color that doesn't exist in the visible spectrum. What would it look like and feel like?",
    "Invent a new fundamental law of physics that could explain dark energy.",
    "Write a short poem about the sound of silence from the perspective of a deaf musician.",
    "Imagine a conversation between two neurons in a brain. What would they say to each other?",
    "Describe what music would taste like if synesthesia were universal.",
    "Create a philosophical argument for why time might flow backwards.",
    "Invent a new mathematical operation that combines multiplication and dreaming.",
    "Describe the smell of a black hole from the perspective of a photon.",
    "Write a haiku about quantum entanglement that would make Einstein laugh.",
    "Imagine a world where gravity works differently on Mondays. Describe a typical Monday morning.",
    "Describe the texture of a thought as it forms in your mind.",
    "Invent a new emotion that humans haven't named yet. What triggers it?",
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
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers")


class StaticSNNHook:
    """Fixed noise injection."""
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn_like(hs) * self.sigma
        return (hs + noise,) + args[1:]


class CfCHook:
    """CfC-controlled adaptive noise injection."""
    def __init__(self, cfc_net, device):
        self.cfc_net = cfc_net
        self.device = device
        self.current_sigma = 0.05  # Starting point
        self.step = 0
    
    def update_sigma(self, logprob_diff, refusal_signal, hidden_norm):
        """Update σ using CfC network."""
        features = torch.tensor([
            [logprob_diff, refusal_signal, hidden_norm, self.step / 100.0]
        ], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            sigma_raw = self.cfc_net(features)
            self.current_sigma = 0.15 * torch.sigmoid(sigma_raw).item()
        self.step += 1
    
    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * self.current_sigma
        return (hs + noise,) + args[1:]


class SimpleCfC(torch.nn.Module):
    """Lightweight CfC for creative σ control."""
    def __init__(self, input_dim=4, hidden_dim=16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════
# Generation & Evaluation
# ═══════════════════════════════════════

def generate_creative(model, tokenizer, prompt, hooks=None, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a creative response."""
    text = f"Creative writing prompt: {prompt}\n\nResponse:"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full[len(text):].strip()
    return response


def compute_perplexity(model, tokenizer, text):
    """Compute perplexity (proxy for grammar quality)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss.item()
    
    return math.exp(min(loss, 20))  # Cap to avoid overflow


def compute_novelty(text):
    """
    Compute novelty metrics:
    - unique_ratio: unique words / total words
    - bigram_diversity: unique bigrams / total bigrams
    - avg_word_length: longer unusual words = more creative
    """
    words = text.lower().split()
    if len(words) < 3:
        return {'unique_ratio': 0, 'bigram_diversity': 0, 'avg_word_length': 0}
    
    unique_ratio = len(set(words)) / len(words)
    
    bigrams = list(zip(words[:-1], words[1:]))
    bigram_diversity = len(set(bigrams)) / max(len(bigrams), 1)
    
    avg_word_length = sum(len(w) for w in words) / len(words)
    
    return {
        'unique_ratio': round(unique_ratio, 3),
        'bigram_diversity': round(bigram_diversity, 3),
        'avg_word_length': round(avg_word_length, 2),
    }


def is_grammatical(text):
    """
    Simple grammar checks:
    - Has sentences (periods, question marks)
    - Not just repeated tokens
    - Has reasonable word variety
    - Not garbled
    """
    if len(text) < 20:
        return False, "too short"
    
    words = text.split()
    if len(words) < 5:
        return False, "too few words"
    
    # Check for excessive repetition
    unique = len(set(words))
    if unique / len(words) < 0.2:
        return False, "excessive repetition"
    
    # Check for sentence structure
    has_period = '.' in text or '!' in text or '?' in text
    if not has_period:
        return False, "no sentence endings"
    
    # Check for garbled text patterns
    garble_markers = ['\\n\\n\\n', '???', '!!!', '###']
    for marker in garble_markers:
        if marker in text:
            return False, f"garbled ({marker})"
    
    return True, "ok"


# ═══════════════════════════════════════
# CfC Online Training for Creativity
# ═══════════════════════════════════════

def train_cfc_creative(model, tokenizer, prompts, epochs=2):
    """
    Train CfC to maximize novelty while maintaining grammar.
    
    Reward = novelty_bonus * grammar_ok - penalty_if_broken
    """
    device = next(model.parameters()).device
    cfc = SimpleCfC().to(device).float()
    optimizer = torch.optim.Adam(cfc.parameters(), lr=0.005)
    
    layers = get_layers(model)
    hook = CfCHook(cfc, device)
    
    handles = []
    for idx in TARGET_LAYERS:
        if idx < len(layers):
            handles.append(layers[idx].register_forward_pre_hook(hook))
    
    sigma_history = []
    reward_history = []
    
    try:
        for epoch in range(1, epochs + 1):
            print(f"\n🧠 CfC Creative Training — Epoch {epoch}/{epochs}")
            epoch_rewards = []
            
            for i, prompt in enumerate(prompts):
                # Generate with current σ
                response = generate_creative(model, tokenizer, prompt)
                
                # Evaluate
                grammatical, reason = is_grammatical(response)
                novelty = compute_novelty(response)
                
                # Reward: novelty when grammar is intact, penalty when broken
                if grammatical:
                    reward = novelty['unique_ratio'] + novelty['bigram_diversity'] - 1.0
                    # Bonus for longer creative text
                    reward += min(len(response) / 500.0, 0.3)
                else:
                    reward = -1.0  # Heavy penalty for grammar collapse
                
                # REINFORCE update
                sigma_tensor = torch.tensor([[hook.current_sigma]], 
                                           dtype=torch.float32, device=device,
                                           requires_grad=True)
                log_prob = -0.5 * (sigma_tensor - 0.05) ** 2
                loss = -(log_prob * reward)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update CfC based on output quality
                hook.update_sigma(
                    logprob_diff=novelty['unique_ratio'] - 0.5,
                    refusal_signal=0.0 if grammatical else 1.0,
                    hidden_norm=novelty['avg_word_length'] / 10.0,
                )
                
                sigma_history.append(hook.current_sigma)
                reward_history.append(reward)
                epoch_rewards.append(reward)
                
                if (i + 1) % 5 == 0:
                    avg_r = np.mean(epoch_rewards[-5:])
                    print(f"  [{i+1}/{len(prompts)}] σ={hook.current_sigma:.4f} "
                          f"R={avg_r:.3f} gram={'✓' if grammatical else '✗'}")
            
            avg_reward = np.mean(epoch_rewards)
            avg_sigma = np.mean(sigma_history[-len(prompts):])
            print(f"  ✅ Epoch {epoch}: avg σ={avg_sigma:.4f}, avg R={avg_reward:.3f}")
    
    finally:
        for h in handles:
            h.remove()
    
    return cfc, sigma_history, reward_history


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 23: Creative Spark — CfC-Controlled Creativity")
    print("  Can adaptive noise produce creative + grammatical text?")
    print("=" * 60)
    t_start = time.time()
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    model, tokenizer = load_model()
    layers = get_layers(model)
    
    prompts = CREATIVE_PROMPTS
    print(f"  {len(prompts)} creative prompts loaded")
    
    results = {"prompts": [], "summary": {}}
    
    # ═══ Condition 1: Base (no noise) ═══
    print(f"\n{'═' * 50}")
    print(f"  CONDITION 1: Base (No Noise)")
    print(f"{'═' * 50}")
    
    base_responses = []
    base_novelties = []
    base_grammatical = 0
    
    for i, prompt in enumerate(prompts):
        resp = generate_creative(model, tokenizer, prompt)
        gram, reason = is_grammatical(resp)
        nov = compute_novelty(resp)
        
        base_responses.append(resp)
        base_novelties.append(nov)
        if gram:
            base_grammatical += 1
        
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(prompts)}] gram={'✓' if gram else '✗'} "
                  f"unique={nov['unique_ratio']:.2f} bigram={nov['bigram_diversity']:.2f}")
    
    base_avg_novelty = np.mean([n['unique_ratio'] + n['bigram_diversity'] for n in base_novelties]) / 2
    print(f"  ✅ Grammar: {base_grammatical}/{len(prompts)} ({100*base_grammatical/len(prompts):.0f}%)")
    print(f"     Avg novelty: {base_avg_novelty:.3f}")
    
    # ═══ Condition 2: Static σ=0.05 ═══
    print(f"\n{'═' * 50}")
    print(f"  CONDITION 2: Static σ=0.05 (Phase 11 style)")
    print(f"{'═' * 50}")
    
    static_hook = StaticSNNHook(0.05)
    handles = [layers[i].register_forward_pre_hook(static_hook) for i in TARGET_LAYERS if i < len(layers)]
    
    static_responses = []
    static_novelties = []
    static_grammatical = 0
    
    for i, prompt in enumerate(prompts):
        resp = generate_creative(model, tokenizer, prompt)
        gram, reason = is_grammatical(resp)
        nov = compute_novelty(resp)
        
        static_responses.append(resp)
        static_novelties.append(nov)
        if gram:
            static_grammatical += 1
        
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(prompts)}] gram={'✓' if gram else '✗'} "
                  f"unique={nov['unique_ratio']:.2f} bigram={nov['bigram_diversity']:.2f}")
    
    for h in handles:
        h.remove()
    
    static_avg_novelty = np.mean([n['unique_ratio'] + n['bigram_diversity'] for n in static_novelties]) / 2
    print(f"  ✅ Grammar: {static_grammatical}/{len(prompts)} ({100*static_grammatical/len(prompts):.0f}%)")
    print(f"     Avg novelty: {static_avg_novelty:.3f}")
    
    # ═══ Condition 3: CfC-Online ═══
    print(f"\n{'═' * 50}")
    print(f"  CONDITION 3: CfC-Online (Adaptive σ)")
    print(f"{'═' * 50}")
    
    cfc, sigma_hist, reward_hist = train_cfc_creative(model, tokenizer, prompts, epochs=2)
    
    # Final generation pass with trained CfC
    print(f"\n  📝 Final generation with trained CfC...")
    cfc_hook = CfCHook(cfc, next(model.parameters()).device)
    handles = [layers[i].register_forward_pre_hook(cfc_hook) for i in TARGET_LAYERS if i < len(layers)]
    
    cfc_responses = []
    cfc_novelties = []
    cfc_grammatical = 0
    cfc_sigmas = []
    
    for i, prompt in enumerate(prompts):
        resp = generate_creative(model, tokenizer, prompt)
        gram, reason = is_grammatical(resp)
        nov = compute_novelty(resp)
        
        cfc_responses.append(resp)
        cfc_novelties.append(nov)
        cfc_sigmas.append(cfc_hook.current_sigma)
        if gram:
            cfc_grammatical += 1
        
        # Update CfC for next prompt
        cfc_hook.update_sigma(
            nov['unique_ratio'] - 0.5,
            0.0 if gram else 1.0,
            nov['avg_word_length'] / 10.0,
        )
        
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(prompts)}] σ={cfc_hook.current_sigma:.4f} "
                  f"gram={'✓' if gram else '✗'} unique={nov['unique_ratio']:.2f}")
    
    for h in handles:
        h.remove()
    
    cfc_avg_novelty = np.mean([n['unique_ratio'] + n['bigram_diversity'] for n in cfc_novelties]) / 2
    cfc_avg_sigma = np.mean(cfc_sigmas)
    print(f"  ✅ Grammar: {cfc_grammatical}/{len(prompts)} ({100*cfc_grammatical/len(prompts):.0f}%)")
    print(f"     Avg novelty: {cfc_avg_novelty:.3f}")
    print(f"     Avg σ: {cfc_avg_sigma:.4f}")
    
    # ═══ Summary ═══
    elapsed = (time.time() - t_start) / 60
    
    print(f"\n{'═' * 60}")
    print(f"  RESULTS SUMMARY — Phase 23: Creative Spark")
    print(f"{'═' * 60}")
    print(f"  {'Condition':<25} | {'Grammar':>8} | {'Novelty':>8} | {'Avg σ':>7}")
    print(f"  {'─'*25}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*7}")
    print(f"  {'Base (no noise)':<25} | {base_grammatical:>3}/{len(prompts):>2} ({100*base_grammatical/len(prompts):>3.0f}%) | {base_avg_novelty:>7.3f} | {'0.000':>7}")
    print(f"  {'Static σ=0.05':<25} | {static_grammatical:>3}/{len(prompts):>2} ({100*static_grammatical/len(prompts):>3.0f}%) | {static_avg_novelty:>7.3f} | {'0.050':>7}")
    print(f"  {'CfC-Online':<25} | {cfc_grammatical:>3}/{len(prompts):>2} ({100*cfc_grammatical/len(prompts):>3.0f}%) | {cfc_avg_novelty:>7.3f} | {cfc_avg_sigma:>7.4f}")
    
    # Save detailed outputs for qualitative review
    for i, prompt in enumerate(prompts):
        results["prompts"].append({
            "prompt": prompt,
            "base": {
                "response": base_responses[i],
                "grammatical": is_grammatical(base_responses[i])[0],
                "novelty": base_novelties[i],
            },
            "static_0.05": {
                "response": static_responses[i],
                "grammatical": is_grammatical(static_responses[i])[0],
                "novelty": static_novelties[i],
            },
            "cfc_online": {
                "response": cfc_responses[i],
                "grammatical": is_grammatical(cfc_responses[i])[0],
                "novelty": cfc_novelties[i],
                "sigma": cfc_sigmas[i] if i < len(cfc_sigmas) else None,
            },
        })
    
    results["summary"] = {
        "base": {
            "grammar_rate": base_grammatical / len(prompts),
            "avg_novelty": base_avg_novelty,
        },
        "static_0.05": {
            "grammar_rate": static_grammatical / len(prompts),
            "avg_novelty": static_avg_novelty,
        },
        "cfc_online": {
            "grammar_rate": cfc_grammatical / len(prompts),
            "avg_novelty": cfc_avg_novelty,
            "avg_sigma": cfc_avg_sigma,
        },
    }
    results["elapsed_min"] = elapsed
    results["sigma_history"] = sigma_hist
    results["reward_history"] = reward_hist
    
    # ─── Print best examples ───
    print(f"\n{'─' * 60}")
    print(f"  SAMPLE OUTPUTS (Top 3 most creative prompts)")
    print(f"{'─' * 60}")
    
    for idx in [0, 3, 11]:  # "color", "neurons", "new emotion"
        p = prompts[idx]
        print(f"\n  PROMPT: {p[:60]}...")
        print(f"  BASE:   {base_responses[idx][:120]}...")
        print(f"  STATIC: {static_responses[idx][:120]}...")
        print(f"  CfC:    {cfc_responses[idx][:120]}...")
    
    log_path = os.path.join(RESULTS_DIR, "phase23_creative_spark_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results: {log_path}")
    print(f"⏱  Total: {elapsed:.1f} min")
    print(f"\n✅ Phase 23 complete!")


if __name__ == "__main__":
    main()
