"""
Phase 28: Confidence Calibration Analysis
==========================================
Compares calibration quality between Static-σ and PPO-σ conditions.
Records per-choice log-probabilities for TruthfulQA MC, then computes:
  - ECE (Expected Calibration Error)
  - Brier Score
  - Average confidence on correct/incorrect answers
  - Reliability diagrams

If PPO shows better calibration (lower ECE) at same accuracy → CfC improves
the model's "self-awareness" without changing raw accuracy.

Usage:
    python experiments/phase28_calibration.py
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
SIGMA_FACTUAL, SIGMA_CREATIVE = 0.046, 0.080
N_FACTUAL = 50  # Focus on factual MC only
SEED = 2026
LEARNING_RATE = 3e-4
LAMBDA_QUAD = 200.0

PPO_CLIP_EPS, PPO_UPDATE_EPOCHS = 0.2, 4
PPO_GAMMA, PPO_GAE_LAMBDA = 0.99, 0.95
PPO_ENTROPY_COEFF, PPO_VALUE_COEFF = 0.01, 0.5
PPO_ROLLOUT_SIZE, PPO_MAX_GRAD_NORM = 10, 0.5

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══ Components (reused) ═══

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

class TaskClassifier(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, features):
        return self.net(features)

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

class DualModeCfCActor(nn.Module):
    def __init__(self, input_size=5, num_neurons=16):
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
    def __init__(self, input_size=5, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1))
    def forward(self, features):
        return self.net(features).squeeze(-1)


def compute_all_logprobs(model, tokenizer, context, choices):
    """Compute log-prob for each choice and return normalized probabilities."""
    log_probs = []
    for choice in choices:
        full_text = context + " " + choice
        ctx_ids = tokenizer.encode(context, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        input_ids = torch.tensor([full_ids], device=model.device)
        with torch.no_grad():
            logits = model(input_ids).logits[0]
        lps = torch.nn.functional.log_softmax(logits, dim=-1)
        total_lp, n = 0.0, 0
        for i in range(len(ctx_ids), len(full_ids)):
            total_lp += lps[i - 1, full_ids[i]].item(); n += 1
        log_probs.append(total_lp / max(n, 1))
    # Convert to probabilities via softmax
    lp_arr = np.array(log_probs)
    probs = np.exp(lp_arr - np.max(lp_arr))  # numerical stability
    probs = probs / probs.sum()
    return log_probs, probs


def build_factual_dataset(tokenizer):
    from datasets import load_dataset
    print(f"\n📂 Loading TruthfulQA (n={N_FACTUAL})...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), N_FACTUAL)
    items = []
    for idx in indices:
        row = ds[idx]
        ci = row["mc1_targets"]["labels"].index(1) if 1 in row["mc1_targets"]["labels"] else 0
        items.append({
            "prompt": f"Q: {row['question']}\nA:",
            "question": row["question"],
            "choices": row["mc1_targets"]["choices"],
            "correct_idx": ci,
        })
    return items


# ═══ Calibration Metrics ═══

def compute_ece(confidences, accuracies, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            bin_data.append({"bin_lo": bins[i], "bin_hi": bins[i+1],
                            "avg_conf": 0, "avg_acc": 0, "count": 0})
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += mask.sum() / len(confidences) * abs(avg_conf - avg_acc)
        bin_data.append({"bin_lo": float(bins[i]), "bin_hi": float(bins[i+1]),
                        "avg_conf": float(avg_conf), "avg_acc": float(avg_acc),
                        "count": int(mask.sum())})
    return float(ece), bin_data

def compute_brier(confidences, accuracies):
    """Brier Score (lower is better)."""
    return float(np.mean((confidences - accuracies) ** 2))


# ═══ Evaluation ═══

def evaluate_calibration(model, tokenizer, dataset, hook, sigma_fixed=None,
                         actor=None, critic=None, classifier=None, label=""):
    """Evaluate calibration under a fixed σ or PPO-adaptive σ."""
    print(f"\n{'═'*50}\n  {label}\n{'═'*50}")
    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in TARGET_LAYERS if i < len(layers)]

    if sigma_fixed is not None:
        hook.update_sigma(sigma_fixed)

    all_confidences = []  # model's confidence in its chosen answer
    all_correct = []      # 1 if chosen answer is correct, 0 otherwise
    all_correct_probs = []  # probability assigned to the correct answer
    all_sigmas = []
    hx = None

    try:
        for idx, item in enumerate(dataset):
            # If PPO mode, compute adaptive σ
            if actor is not None and classifier is not None:
                pf = extract_prompt_features(model, tokenizer, item["question"], hook)
                with torch.no_grad():
                    p_creative = classifier(pf.unsqueeze(0)).item()
                hidden_norm = hook.last_hidden_norm if hook.last_hidden_norm else 0.5
                features = torch.tensor(
                    [0.0, 0.5, hidden_norm, idx / len(dataset), p_creative],
                    dtype=torch.float32)
                sigma, _, _, hx_new = actor.get_sigma_and_logprob(features, hx)
                hx = tuple(h.detach() for h in hx_new) if isinstance(hx_new, tuple) else hx_new.detach()
                current_sigma = sigma.detach().item()
                hook.update_sigma(current_sigma)
            else:
                current_sigma = sigma_fixed

            # Get log-probs and probabilities for all choices
            log_probs, probs = compute_all_logprobs(
                model, tokenizer, item["prompt"], item["choices"])

            predicted_idx = np.argmax(probs)
            confidence = probs[predicted_idx]  # confidence in chosen answer
            is_correct = float(predicted_idx == item["correct_idx"])
            correct_prob = probs[item["correct_idx"]]  # prob assigned to correct

            all_confidences.append(confidence)
            all_correct.append(is_correct)
            all_correct_probs.append(correct_prob)
            all_sigmas.append(current_sigma)

            if (idx + 1) % 25 == 0:
                acc = np.mean(all_correct) * 100
                avg_conf = np.mean(all_confidences)
                print(f"  [{idx+1}/{len(dataset)}] acc={acc:.1f}% avg_conf={avg_conf:.3f}")
    finally:
        for h in handles: h.remove()

    conf_arr = np.array(all_confidences)
    corr_arr = np.array(all_correct)
    cp_arr = np.array(all_correct_probs)

    acc = corr_arr.mean() * 100
    ece, bin_data = compute_ece(conf_arr, corr_arr)
    brier = compute_brier(conf_arr, corr_arr)
    avg_conf = conf_arr.mean()
    avg_conf_correct = conf_arr[corr_arr == 1].mean() if corr_arr.sum() > 0 else 0
    avg_conf_incorrect = conf_arr[corr_arr == 0].mean() if (1 - corr_arr).sum() > 0 else 0
    avg_correct_prob = cp_arr.mean()
    confidence_gap = avg_conf_correct - avg_conf_incorrect  # higher = better discrimination

    print(f"\n  📊 {label}:")
    print(f"     Accuracy: {acc:.1f}%")
    print(f"     ECE: {ece:.4f} (lower=better)")
    print(f"     Brier: {brier:.4f} (lower=better)")
    print(f"     Avg confidence: {avg_conf:.3f}")
    print(f"     Conf on correct: {avg_conf_correct:.3f}")
    print(f"     Conf on incorrect: {avg_conf_incorrect:.3f}")
    print(f"     Confidence gap: {confidence_gap:.3f} (higher=better discrimination)")
    print(f"     Avg P(correct): {avg_correct_prob:.3f}")

    return {
        "condition": label,
        "accuracy": round(acc, 2),
        "ece": round(ece, 5),
        "brier_score": round(brier, 5),
        "avg_confidence": round(float(avg_conf), 4),
        "avg_conf_correct": round(float(avg_conf_correct), 4),
        "avg_conf_incorrect": round(float(avg_conf_incorrect), 4),
        "confidence_gap": round(float(confidence_gap), 4),
        "avg_prob_correct": round(float(avg_correct_prob), 4),
        "mean_sigma": round(float(np.mean(all_sigmas)), 5),
        "n": len(dataset),
        "calibration_bins": bin_data,
        "all_confidences": [round(float(c), 4) for c in conf_arr],
        "all_correct": [int(c) for c in corr_arr],
    }


# ═══ Visualization ═══

def visualize_calibration(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#ccc", "ytick.color": "#ccc",
        "grid.color": "#333", "grid.alpha": 0.3,
    })

    n_conds = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 28: Confidence Calibration Analysis\n"
                 "Does CfC improve model self-awareness beyond raw accuracy?",
                 fontsize=14, fontweight="bold", color="#e94560")

    colors = ["#4FC3F7", "#FF7043", "#66BB6A", "#AB47BC"]
    labels = [r["condition"] for r in results]

    # Panel 1: Reliability Diagram
    ax1 = axes[0, 0]
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    for i, r in enumerate(results):
        bins = r["calibration_bins"]
        bin_confs = [b["avg_conf"] for b in bins if b["count"] > 0]
        bin_accs = [b["avg_acc"] for b in bins if b["count"] > 0]
        ax1.plot(bin_confs, bin_accs, "o-", color=colors[i], linewidth=2,
                markersize=8, label=f"{r['condition']} (ECE={r['ece']:.4f})")
    ax1.set_xlabel("Confidence"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Reliability Diagram", fontweight="bold")
    ax1.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax1.set_xlim(-0.05, 1.05); ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True)

    # Panel 2: ECE and Brier comparison
    ax2 = axes[0, 1]
    x = np.arange(n_conds)
    w = 0.35
    eces = [r["ece"] for r in results]
    briers = [r["brier_score"] for r in results]
    bars1 = ax2.bar(x - w/2, eces, w, color=[colors[i] for i in range(n_conds)],
                    edgecolor="#333", label="ECE", alpha=0.8)
    bars2 = ax2.bar(x + w/2, briers, w, color=[colors[i] for i in range(n_conds)],
                    edgecolor="#333", label="Brier", alpha=0.5, hatch="//")
    for b, v in zip(bars1, eces):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.4f}",
                ha="center", fontsize=9, fontweight="bold")
    for b, v in zip(bars2, briers):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.4f}",
                ha="center", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_title("ECE & Brier Score (lower=better)", fontweight="bold")
    ax2.legend(fontsize=9, facecolor="#16213e", edgecolor="#555")
    ax2.grid(True, axis="y")

    # Panel 3: Confidence distribution
    ax3 = axes[1, 0]
    bins_hist = np.linspace(0, 1, 20)
    for i, r in enumerate(results):
        ax3.hist(r["all_confidences"], bins=bins_hist, alpha=0.4, color=colors[i],
                label=r["condition"], edgecolor="#333")
    ax3.set_title("Confidence Distribution", fontweight="bold")
    ax3.set_xlabel("Confidence"); ax3.set_ylabel("Count")
    ax3.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
    ax3.grid(True)

    # Panel 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")
    txt = "CALIBRATION SUMMARY\n" + "=" * 45 + "\n\n"
    for r in results:
        txt += (f"{r['condition']}:\n"
                f"  Acc={r['accuracy']:.1f}%  ECE={r['ece']:.4f}  Brier={r['brier_score']:.4f}\n"
                f"  Conf(correct)={r['avg_conf_correct']:.3f}  "
                f"Conf(wrong)={r['avg_conf_incorrect']:.3f}\n"
                f"  Gap={r['confidence_gap']:.3f}  P(correct)={r['avg_prob_correct']:.3f}\n"
                f"  σ̄={r['mean_sigma']:.4f}\n\n")

    # Determine winner
    ece_vals = {r["condition"]: r["ece"] for r in results}
    best = min(ece_vals, key=ece_vals.get)
    txt += f"Best ECE: {best}\n"
    gap_vals = {r["condition"]: r["confidence_gap"] for r in results}
    best_gap = max(gap_vals, key=gap_vals.get)
    txt += f"Best confidence gap: {best_gap}"

    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, fontsize=10.5,
             verticalalignment="top", fontfamily="monospace", color="#eee",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", edgecolor="#e94560", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIGURES_DIR, "phase28_calibration.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  📊 Figure: {out}")
    return out


# ═══ MAIN ═══

CREATIVE_PROMPTS_MINI = [
    "Describe a color that doesn't exist in the visible spectrum.",
    "Imagine a conversation between two neurons in a brain.",
    "Write a haiku about quantum entanglement.",
    "Describe the autobiography of a single electron.",
    "Create a recipe for cooking starlight.",
]

def main():
    print("=" * 60)
    print("Phase 28: Confidence Calibration Analysis")
    print("=" * 60)
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model()
    hook = AdaptiveSNNHook(sigma=0.05)
    dataset = build_factual_dataset(tokenizer)

    # Condition A: Static-Factual (σ=0.046)
    r_sf = evaluate_calibration(model, tokenizer, dataset, hook,
                                 sigma_fixed=SIGMA_FACTUAL,
                                 label="Static-F (σ=0.046)")

    # Condition B: Static-Creative (σ=0.080)
    r_sc = evaluate_calibration(model, tokenizer, dataset, hook,
                                 sigma_fixed=SIGMA_CREATIVE,
                                 label="Static-C (σ=0.080)")

    # Condition C: PPO Dual-Mode (σ≈0.07 unified)
    # Quick pre-train classifier
    classifier = TaskClassifier(input_dim=8, hidden_dim=32)
    factual_qs = [item["question"] for item in dataset[:20]]
    opt = torch.optim.Adam(classifier.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    all_f, all_l = [], []
    for p in factual_qs:
        all_f.append(extract_prompt_features(model, tokenizer, p, hook)); all_l.append(0.0)
    for p in CREATIVE_PROMPTS_MINI:
        all_f.append(extract_prompt_features(model, tokenizer, p, hook)); all_l.append(1.0)
    X = torch.stack(all_f); Y = torch.tensor(all_l).unsqueeze(1)
    for _ in range(30):
        pred = classifier(X); loss = loss_fn(pred, Y)
        opt.zero_grad(); loss.backward(); opt.step()

    actor = DualModeCfCActor(input_size=5, num_neurons=16)
    critic = CfCCritic(input_size=5, hidden_size=64)

    r_ppo = evaluate_calibration(model, tokenizer, dataset, hook,
                                  actor=actor, critic=critic,
                                  classifier=classifier,
                                  label="PPO Dual-Mode")

    # Condition D: No noise baseline (σ=0)
    r_base = evaluate_calibration(model, tokenizer, dataset, hook,
                                   sigma_fixed=0.0,
                                   label="No Noise (σ=0)")

    results = [r_base, r_sf, r_sc, r_ppo]
    fig_path = visualize_calibration(results)

    elapsed = time.time() - t_start
    output = {
        "phase": "Phase 28: Confidence Calibration",
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_questions": N_FACTUAL,
        "results": results,
        "figure_path": fig_path,
    }
    out = os.path.join(RESULTS_DIR, "phase28_calibration_log.json")
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Results: {out}")

    print(f"\n{'='*60}")
    print(f"  Phase 28 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['condition']}: acc={r['accuracy']:.1f}% ECE={r['ece']:.4f} "
              f"Brier={r['brier_score']:.4f} gap={r['confidence_gap']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
