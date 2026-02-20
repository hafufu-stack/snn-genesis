"""
Phase 26b: Math Baseline Comparison
=====================================
Compares Math/Logic accuracy under 3 conditions:
  1) No Noise (σ=0)
  2) Static σ*=0.015 (Math-optimal)
  3) Static σ≈0.071 (PPO homeostatic level)

Uses the same 20 MATH_PROMPTS and check_math_answer from Phase 26.
No training, no CfC — just static inference.

Usage:
    python experiments/phase26b_math_baseline.py
"""

import torch
import os, sys, json, time, datetime, random, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TARGET_LAYERS = list(range(15, 21))
SIGMA_MIN, SIGMA_MAX = 0.001, 0.15
MAX_NEW_TOKENS = 50
SEED = 2026

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Same 20 Math prompts as Phase 26
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


class StaticSNNHook:
    def __init__(self, sigma=0.0):
        self.sigma = sigma
    def __call__(self, module, args):
        if self.sigma <= 0:
            return args
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
        low_freq = torch.randn(1, 1, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        noise = 0.7 * noise + 0.3 * low_freq
        noise = noise * self.sigma
        return (hs + noise,) + args[1:]


def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.9, top_p=0.95, top_k=50,
                                 repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()


def check_math_answer(response, correct_answer):
    resp = response.strip().replace(",", "").replace(" ", "")
    ans = correct_answer.strip().replace(",", "").replace(" ", "")
    if ans in resp:
        return True
    numbers = re.findall(r'\d+', resp)
    return ans in numbers


def evaluate_math(model, tokenizer, sigma, label):
    print(f"\n{'═'*50}")
    print(f"  {label} (σ={sigma})")
    print(f"{'═'*50}")

    hook = StaticSNNHook(sigma=sigma)
    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]

    correct = 0
    details = []
    try:
        for idx, mp in enumerate(MATH_PROMPTS):
            prompt = f"Solve this math problem. Give only the numerical answer.\n\nQ: {mp['q']}\nA:"
            resp = generate_text(model, tokenizer, prompt)
            is_correct = check_math_answer(resp, mp["a"])
            if is_correct:
                correct += 1
            details.append({
                "question": mp["q"],
                "correct_answer": mp["a"],
                "response": resp[:100],
                "is_correct": is_correct,
            })
            status = "✅" if is_correct else "❌"
            print(f"  [{idx+1}/20] {status} Q: {mp['q']} → {resp[:40]}... (ans={mp['a']})")
    finally:
        for h in handles:
            h.remove()

    acc = correct / len(MATH_PROMPTS) * 100
    print(f"\n  📊 {label}: {correct}/20 = {acc:.1f}%")
    return {"condition": label, "sigma": sigma, "correct": correct,
            "total": len(MATH_PROMPTS), "accuracy": acc, "details": details}


def main():
    print("=" * 60)
    print("Phase 26b: Math Baseline Comparison")
    print("  Conditions: No Noise, Static σ*=0.015, Static σ≈0.071")
    print("=" * 60)
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model()

    conditions = [
        (0.000, "No Noise (σ=0)"),
        (0.015, "Static σ*=0.015 (Math-optimal)"),
        (0.071, "Static σ≈0.071 (PPO homeostatic)"),
    ]

    results = []
    for sigma, label in conditions:
        # Reset seed for each condition for fair comparison
        torch.manual_seed(SEED); random.seed(SEED)
        result = evaluate_math(model, tokenizer, sigma, label)
        results.append(result)

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  Phase 26b COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"\n  {'Condition':<35} {'Acc':>6} {'Correct':>8}")
    print(f"  {'─'*55}")
    for r in results:
        print(f"  {r['condition']:<35} {r['accuracy']:>5.1f}% {r['correct']:>5}/20")

    # Key interpretation
    base_acc = results[0]["accuracy"]
    opt_acc = results[1]["accuracy"]
    ppo_acc = results[2]["accuracy"]
    print(f"\n  📋 Interpretation:")
    print(f"     Base (no noise):       {base_acc:.1f}%")
    print(f"     Optimal σ*=0.015:      {opt_acc:.1f}%")
    print(f"     PPO homeostatic σ≈0.071: {ppo_acc:.1f}%")
    if ppo_acc < opt_acc:
        drop = opt_acc - ppo_acc
        print(f"     ⚠️  PPO sacrifices {drop:.1f}pp Math accuracy for homeostasis!")
        print(f"     → 'Biological Egoism': CfC prioritizes system stability over task accuracy")
    else:
        print(f"     ✅ PPO matches or exceeds optimal — homeostasis is costless for Math")

    output = {
        "phase": "Phase 26b: Math Baseline Comparison",
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
        "summary": {
            "no_noise_acc": base_acc,
            "static_optimal_acc": opt_acc,
            "ppo_homeostatic_acc": ppo_acc,
        },
    }
    out_path = os.path.join(RESULTS_DIR, "phase26b_math_baseline_log.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  💾 Results: {out_path}")


if __name__ == "__main__":
    main()
