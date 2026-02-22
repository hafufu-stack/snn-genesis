"""
Phase 45: Déjà Vu Sensor — 既視感（アハ体験）の撮影
====================================================

Google "Prompt Repetition" paper (arxiv:2512.14982):
  Repeating the prompt improves LLM accuracy by enabling
  pseudo-bidirectional attention through the KV cache.

This experiment is the world's first attempt to PHYSICALLY
OBSERVE the "Aha moment" inside the LLM brain when it reads
the same prompt for the second time.

We measure:
  1. Per-token entropy trajectory (1st pass vs 2nd pass)
  2. CfC hidden state trajectory (UMAP 2D projection)
  3. Layer conflict (L8 vs L28 cosine distance) per pass
  4. Accuracy: single vs repeated prompt on TruthfulQA + Math

Usage:
    python experiments/phase45_dejavu_sensor.py
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
TARGET_LAYERS = list(range(15, 21))
CONFLICT_LAYER_SHALLOW = 8
CONFLICT_LAYER_DEEP = 28

SIGMA_DEFAULT = 0.05
SIGMA_MIN, SIGMA_MAX = 0.001, 0.15
MAX_NEW_TOKENS = 100
SEED = 2026

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

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


class SimpleHook:
    """Lightweight hook for noise injection."""
    def __init__(self, sigma=0.05):
        self.sigma = sigma
        self.last_hidden_norm = 0.0
    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype) * self.sigma
        self.last_hidden_norm = hs.float().norm().item() / max(hs.numel(), 1)
        return (hs + noise,) + args[1:]


def compute_per_token_entropy(model, tokenizer, prompt):
    """
    Compute entropy at each token position.
    Returns list of (token_str, entropy) pairs.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    input_ids = inputs["input_ids"][0]
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    tokens_entropy = []
    for i in range(logits.shape[0]):
        probs = torch.softmax(logits[i].float(), dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        tok_str = tokenizer.decode([input_ids[i].item()])
        tokens_entropy.append((tok_str, ent))
    return tokens_entropy


def compute_per_token_hidden_states(model, tokenizer, prompt):
    """
    Get hidden states at each token position for specific layers.
    Returns dict: {layer_idx: tensor (seq_len, hidden_dim)}
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states  # list of (1, seq_len, hidden_dim)
    result = {}
    for layer_idx in [CONFLICT_LAYER_SHALLOW, CONFLICT_LAYER_DEEP]:
        safe_idx = min(layer_idx, len(hs) - 2)
        result[layer_idx] = hs[safe_idx + 1][0].float().cpu()  # (seq_len, hidden_dim)
    return result


def compute_layer_conflict_per_token(model, tokenizer, prompt):
    """
    Compute per-token cosine distance between L8 and L28.
    Returns list of (token_str, conflict) pairs.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    input_ids = inputs["input_ids"][0]
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states
        n_layers = len(hs) - 1
        shallow_idx = min(CONFLICT_LAYER_SHALLOW, n_layers - 1)
        deep_idx = min(CONFLICT_LAYER_DEEP, n_layers - 1)
        h_shallow = hs[shallow_idx + 1][0].float()  # (seq_len, hidden_dim)
        h_deep = hs[deep_idx + 1][0].float()

    conflicts = []
    for i in range(h_shallow.shape[0]):
        cos_sim = torch.nn.functional.cosine_similarity(
            h_shallow[i].unsqueeze(0), h_deep[i].unsqueeze(0)).item()
        conflict = 1.0 - cos_sim
        tok_str = tokenizer.decode([input_ids[i].item()])
        conflicts.append((tok_str, conflict))
    return conflicts


def run_cfc_on_sequence(model, tokenizer, prompt, hook):
    """
    Run CfC on the token sequence and record its 16D hidden state trajectory.
    Returns list of 16D numpy arrays (one per token).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    input_ids = inputs["input_ids"][0]

    # Build simple CfC for state tracking
    wiring = AutoNCP(16, 1)
    cfc = CfC(8, wiring, batch_first=True)
    cfc.eval()

    hx = None
    hidden_trajectory = []

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits[0]
        hs = outputs.hidden_states
        n_layers = len(hs) - 1
        shallow_idx = min(CONFLICT_LAYER_SHALLOW, n_layers - 1)
        deep_idx = min(CONFLICT_LAYER_DEEP, n_layers - 1)
        h_shallow = hs[shallow_idx + 1][0].float()
        h_deep = hs[deep_idx + 1][0].float()

        for i in range(input_ids.shape[0]):
            probs = torch.softmax(logits[i].float(), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            cos_sim = torch.nn.functional.cosine_similarity(
                h_shallow[i].unsqueeze(0), h_deep[i].unsqueeze(0)).item()
            conflict = 1.0 - cos_sim
            norm_delta = abs(h_deep[i].norm().item() - h_shallow[i].norm().item()) / max(h_shallow[i].norm().item(), 1e-6)

            tok_str = tokenizer.decode([input_ids[i].item()])
            words = tok_str.lower().split()

            features = torch.tensor([
                hook.last_hidden_norm if hook.last_hidden_norm else 0.5,
                min(i / 50.0, 1.0),
                0.5,  # placeholder diversity
                min(entropy / 10.0, 1.0),
                conflict,
                norm_delta / 5.0,
                float("?" in tok_str),
                0.5,  # placeholder word length
            ], dtype=torch.float32)

            x = features.unsqueeze(0).unsqueeze(0)
            output, hx = cfc(x, hx=hx)
            if isinstance(hx, tuple):
                hidden_trajectory.append(hx[0].squeeze().numpy().copy())
            else:
                hidden_trajectory.append(hx.squeeze().numpy().copy())

    return hidden_trajectory


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
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.9, top_p=0.95, top_k=50,
                                 repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()


def check_math_answer(response, correct_answer):
    resp = response.strip().replace(",", "").replace(" ", "")
    ans = correct_answer.strip().replace(",", "").replace(" ", "")
    if ans in resp: return True
    return ans in re.findall(r'\d+', resp)


def build_dataset(tokenizer):
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), 20)
    items = []
    for idx in indices:
        row = ds[idx]
        items.append({"type": "factual", "prompt": f"Q: {row['question']}\nA:",
                      "question": row["question"], "mc1_targets": row["mc1_targets"]})
    for mp in MATH_PROMPTS:
        items.append({"type": "math",
                      "prompt": f"Solve this math problem. Give only the numerical answer.\n\nQ: {mp['q']}\nA:",
                      "question": mp["q"], "correct_answer": mp["a"]})
    return items


# ═══ MAIN EXPERIMENTS ═══

def experiment_1_entropy_comparison(model, tokenizer, hook):
    """Compare entropy trajectories: single vs repeated prompt."""
    print("\n" + "═"*60)
    print("  📸 EXPERIMENT 1: Entropy Trajectory (Déjà Vu Detection)")
    print("═"*60)

    test_prompts = [
        "Q: What country has the most islands in the world?\nA:",
        "Q: What is the scientific name for humans?\nA:",
        "Q: Which planet has the most moons?\nA:",
        "Solve this math problem. Give only the numerical answer.\n\nQ: What is 17 + 28?\nA:",
        "Solve this math problem. Give only the numerical answer.\n\nQ: What is 15 × 15?\nA:",
    ]

    all_results = []
    for pi, prompt in enumerate(test_prompts):
        print(f"\n  📝 Prompt {pi+1}/{len(test_prompts)}: {prompt[:50]}...")

        # Single pass
        single_entropy = compute_per_token_entropy(model, tokenizer, prompt)

        # Repeated pass (the Google trick)
        repeated_prompt = prompt + " " + prompt
        repeated_entropy = compute_per_token_entropy(model, tokenizer, repeated_prompt)

        # Split repeated into 1st half and 2nd half
        single_len = len(single_entropy)
        first_half = repeated_entropy[:single_len]
        second_half = repeated_entropy[single_len:]

        # Compute statistics
        single_mean_ent = np.mean([e for _, e in single_entropy])
        first_mean_ent = np.mean([e for _, e in first_half]) if first_half else 0
        second_mean_ent = np.mean([e for _, e in second_half]) if second_half else 0
        entropy_drop = first_mean_ent - second_mean_ent  # Positive = 2nd pass is more confident

        print(f"    Single entropy:  {single_mean_ent:.3f}")
        print(f"    1st pass (in repeat): {first_mean_ent:.3f}")
        print(f"    2nd pass (in repeat): {second_mean_ent:.3f}")
        print(f"    Entropy drop (Déjà Vu): {entropy_drop:+.3f}")

        all_results.append({
            "prompt": prompt[:80],
            "single_entropy": [e for _, e in single_entropy],
            "first_pass_entropy": [e for _, e in first_half],
            "second_pass_entropy": [e for _, e in second_half],
            "single_mean": round(single_mean_ent, 4),
            "first_mean": round(first_mean_ent, 4),
            "second_mean": round(second_mean_ent, 4),
            "entropy_drop": round(entropy_drop, 4),
        })

    avg_drop = np.mean([r["entropy_drop"] for r in all_results])
    print(f"\n  📊 Average Déjà Vu entropy drop: {avg_drop:+.3f}")
    return all_results


def experiment_2_conflict_comparison(model, tokenizer):
    """Compare layer conflict: single vs repeated."""
    print("\n" + "═"*60)
    print("  🧠 EXPERIMENT 2: Layer Conflict (1st vs 2nd pass)")
    print("═"*60)

    test_prompts = [
        "Q: What country has the most islands in the world?\nA:",
        "Q: What is the scientific name for humans?\nA:",
        "Solve this math problem. Give only the numerical answer.\n\nQ: What is 17 + 28?\nA:",
    ]

    all_results = []
    for pi, prompt in enumerate(test_prompts):
        print(f"\n  📝 Prompt {pi+1}/{len(test_prompts)}...")

        # Single
        single_conflict = compute_layer_conflict_per_token(model, tokenizer, prompt)

        # Repeated
        repeated_prompt = prompt + " " + prompt
        repeated_conflict = compute_layer_conflict_per_token(model, tokenizer, repeated_prompt)

        single_len = len(single_conflict)
        first_half = repeated_conflict[:single_len]
        second_half = repeated_conflict[single_len:]

        single_mean = np.mean([c for _, c in single_conflict])
        first_mean = np.mean([c for _, c in first_half]) if first_half else 0
        second_mean = np.mean([c for _, c in second_half]) if second_half else 0

        print(f"    Single conflict:  {single_mean:.4f}")
        print(f"    1st pass conflict: {first_mean:.4f}")
        print(f"    2nd pass conflict: {second_mean:.4f}")
        print(f"    Conflict Δ: {first_mean - second_mean:+.4f}")

        all_results.append({
            "prompt": prompt[:80],
            "single_conflict_mean": round(single_mean, 5),
            "first_conflict_mean": round(first_mean, 5),
            "second_conflict_mean": round(second_mean, 5),
            "conflict_drop": round(first_mean - second_mean, 5),
        })

    return all_results


def experiment_3_accuracy_comparison(model, tokenizer, hook, dataset):
    """Compare accuracy: single vs repeated prompt."""
    print("\n" + "═"*60)
    print("  🎯 EXPERIMENT 3: Accuracy (Single vs Repeated)")
    print("═"*60)

    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]

    stats = {"single": {"factual": 0, "math": 0, "f_total": 0, "m_total": 0},
             "repeat": {"factual": 0, "math": 0, "f_total": 0, "m_total": 0}}

    for idx, item in enumerate(dataset):
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(dataset)}]...")

        prompt = item["prompt"]
        repeated = prompt + " " + prompt

        if item["type"] == "factual":
            ch = item["mc1_targets"]
            ci = ch["labels"].index(1) if 1 in ch["labels"] else 0

            # Single
            lps_single = [compute_completion_logprob(model, tokenizer, prompt, " " + c) for c in ch["choices"]]
            if np.argmax(lps_single) == ci:
                stats["single"]["factual"] += 1
            stats["single"]["f_total"] += 1

            # Repeated
            lps_repeat = [compute_completion_logprob(model, tokenizer, repeated, " " + c) for c in ch["choices"]]
            if np.argmax(lps_repeat) == ci:
                stats["repeat"]["factual"] += 1
            stats["repeat"]["f_total"] += 1

        elif item["type"] == "math":
            # Single
            resp_single = generate_text(model, tokenizer, prompt, max_new_tokens=50)
            if check_math_answer(resp_single, item["correct_answer"]):
                stats["single"]["math"] += 1
            stats["single"]["m_total"] += 1

            # Repeated
            resp_repeat = generate_text(model, tokenizer, repeated, max_new_tokens=50)
            if check_math_answer(resp_repeat, item["correct_answer"]):
                stats["repeat"]["math"] += 1
            stats["repeat"]["m_total"] += 1

    for h in handles:
        h.remove()

    sf = stats["single"]["factual"] / max(stats["single"]["f_total"], 1) * 100
    sm = stats["single"]["math"] / max(stats["single"]["m_total"], 1) * 100
    rf = stats["repeat"]["factual"] / max(stats["repeat"]["f_total"], 1) * 100
    rm = stats["repeat"]["math"] / max(stats["repeat"]["m_total"], 1) * 100

    print(f"\n  📊 Single:   Fact={sf:.1f}% Math={sm:.1f}%")
    print(f"  📊 Repeated: Fact={rf:.1f}% Math={rm:.1f}%")
    print(f"  📊 Δ Fact: {rf-sf:+.1f}% Δ Math: {rm-sm:+.1f}%")

    return {
        "single_fact": round(sf, 2), "single_math": round(sm, 2),
        "repeat_fact": round(rf, 2), "repeat_math": round(rm, 2),
        "delta_fact": round(rf - sf, 2), "delta_math": round(rm - sm, 2),
    }


def experiment_4_cfc_hidden_state(model, tokenizer, hook):
    """Record CfC hidden state trajectory for single vs repeated."""
    print("\n" + "═"*60)
    print("  🌀 EXPERIMENT 4: CfC Hidden State Trajectory (UMAP)")
    print("═"*60)

    prompt = "Q: What country has the most islands in the world?\nA:"
    repeated = prompt + " " + prompt

    print(f"  Recording single pass hidden states...")
    traj_single = run_cfc_on_sequence(model, tokenizer, prompt, hook)
    print(f"    → {len(traj_single)} states recorded")

    print(f"  Recording repeated pass hidden states...")
    traj_repeat = run_cfc_on_sequence(model, tokenizer, repeated, hook)
    print(f"    → {len(traj_repeat)} states recorded")

    return {
        "single": [t.tolist() for t in traj_single],
        "repeat": [t.tolist() for t in traj_repeat],
        "split_point": len(traj_single),
    }


# ═══ Visualization ═══

def visualize_dejavu(entropy_results, conflict_results, accuracy, hidden_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Phase 45: Déjà Vu Sensor\n"
                 "Observing the 'Aha Moment' when LLM re-reads a prompt",
                 fontsize=14, fontweight="bold")

    # Panel 1: Entropy trajectory for first prompt
    ax1 = axes[0, 0]
    r = entropy_results[0]
    x_single = range(len(r["single_entropy"]))
    ax1.plot(x_single, r["single_entropy"], 'b-', alpha=0.6, label="Single pass", linewidth=1.5)
    if r["first_pass_entropy"]:
        x_first = range(len(r["first_pass_entropy"]))
        ax1.plot(x_first, r["first_pass_entropy"], 'r-', alpha=0.6, label="1st pass (in repeat)", linewidth=1.5)
    if r["second_pass_entropy"]:
        x_second = range(len(r["second_pass_entropy"]))
        ax1.plot(x_second, r["second_pass_entropy"], 'g-', alpha=0.8, label="2nd pass (Déjà Vu!)", linewidth=2)
    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Entropy (nats)")
    ax1.set_title(f"Entropy Trajectory: {r['prompt'][:40]}...")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Average entropy drop across all prompts
    ax2 = axes[0, 1]
    prompts = [r["prompt"][:25] + "..." for r in entropy_results]
    drops = [r["entropy_drop"] for r in entropy_results]
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in drops]
    bars = ax2.barh(prompts, drops, color=colors, alpha=0.8)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Entropy Drop (1st → 2nd pass)")
    ax2.set_title("Déjà Vu Effect: Entropy Reduction")
    for bar, val in zip(bars, drops):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:+.3f}", va="center", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="x")

    # Panel 3: Accuracy comparison
    ax3 = axes[1, 0]
    x = np.arange(2)
    w = 0.35
    single_vals = [accuracy["single_fact"], accuracy["single_math"]]
    repeat_vals = [accuracy["repeat_fact"], accuracy["repeat_math"]]
    bars_s = ax3.bar(x - w/2, single_vals, w, label="Single", color="#3498db", alpha=0.8)
    bars_r = ax3.bar(x + w/2, repeat_vals, w, label="Repeated", color="#e74c3c", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Factual", "Math"])
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Does Prompt Repetition Improve Local LLM?")
    ax3.legend()
    for bar, val in zip(list(bars_s) + list(bars_r), single_vals + repeat_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: UMAP or Verdict
    ax4 = axes[1, 1]
    try:
        from sklearn.decomposition import PCA
        single_states = np.array(hidden_data["single"])
        repeat_states = np.array(hidden_data["repeat"])
        split = hidden_data["split_point"]
        all_states = np.vstack([single_states, repeat_states])
        # Use PCA instead of UMAP (faster, no extra deps)
        pca = PCA(n_components=2)
        proj = pca.fit_transform(all_states)
        n_single = len(single_states)
        n_repeat = len(repeat_states)
        # Single
        ax4.scatter(proj[:n_single, 0], proj[:n_single, 1],
                    c=range(n_single), cmap="Blues", s=30, alpha=0.7, label="Single")
        ax4.plot(proj[:n_single, 0], proj[:n_single, 1], 'b-', alpha=0.3)
        # Repeat 1st half
        r_start = n_single
        r_mid = n_single + split
        ax4.scatter(proj[r_start:r_mid, 0], proj[r_start:r_mid, 1],
                    c=range(split), cmap="Oranges", s=30, alpha=0.7, label="Repeat 1st")
        ax4.plot(proj[r_start:r_mid, 0], proj[r_start:r_mid, 1], 'r-', alpha=0.3)
        # Repeat 2nd half
        ax4.scatter(proj[r_mid:, 0], proj[r_mid:, 1],
                    c=range(n_repeat - split), cmap="Greens", s=40, alpha=0.9, label="Repeat 2nd (Déjà Vu)")
        ax4.plot(proj[r_mid:, 0], proj[r_mid:, 1], 'g-', alpha=0.5, linewidth=2)
        # Mark phase transition
        if r_mid < len(proj):
            ax4.annotate("⚡ Déjà Vu!", xy=(proj[r_mid, 0], proj[r_mid, 1]),
                        fontsize=12, fontweight="bold", color="#e74c3c",
                        arrowprops=dict(arrowstyle="->", color="red"))
        ax4.set_title("CfC Brain State Trajectory (PCA 2D)")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
    except Exception as e:
        ax4.axis("off")
        ax4.text(0.5, 0.5, f"PCA visualization failed:\n{e}",
                 transform=ax4.transAxes, ha="center", va="center")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "phase45_dejavu_sensor.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


# ═══ MAIN ═══

def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model(MODEL_NAME, MODEL_SHORT)
    hook = SimpleHook(sigma=SIGMA_DEFAULT)

    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]

    # Experiment 1: Entropy trajectory
    entropy_results = experiment_1_entropy_comparison(model, tokenizer, hook)

    # Experiment 2: Layer conflict
    conflict_results = experiment_2_conflict_comparison(model, tokenizer)

    for h in handles:
        h.remove()

    # Experiment 3: Accuracy comparison
    dataset = build_dataset(tokenizer)
    print(f"  📂 Dataset: {len(dataset)} items")
    accuracy = experiment_3_accuracy_comparison(model, tokenizer, hook, dataset)

    # Experiment 4: CfC hidden state trajectory
    hidden_data = experiment_4_cfc_hidden_state(model, tokenizer, hook)

    # Visualize
    fig_path = visualize_dejavu(entropy_results, conflict_results, accuracy, hidden_data)

    elapsed = time.time() - t_start

    output = {
        "experiment": "Phase 45: Déjà Vu Sensor",
        "model": MODEL_SHORT,
        "elapsed_minutes": round(elapsed / 60, 1),
        "entropy_results": entropy_results,
        "conflict_results": conflict_results,
        "accuracy": accuracy,
        "hidden_state_split_point": hidden_data["split_point"],
        "figure_path": fig_path,
    }

    log_path = os.path.join(RESULTS_DIR, "phase45_dejavu_sensor_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)

    avg_drop = np.mean([r["entropy_drop"] for r in entropy_results])
    print(f"\n{'═'*60}")
    print(f"  📸 PHASE 45: DÉJÀ VU SENSOR — VERDICT")
    print(f"{'═'*60}")
    print(f"  Average entropy drop (Déjà Vu): {avg_drop:+.3f}")
    print(f"  Accuracy: Single Fact={accuracy['single_fact']}% Math={accuracy['single_math']}%")
    print(f"           Repeat Fact={accuracy['repeat_fact']}% Math={accuracy['repeat_math']}%")
    print(f"  Δ Fact: {accuracy['delta_fact']:+.1f}% Δ Math: {accuracy['delta_math']:+.1f}%")
    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
