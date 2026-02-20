"""
Phase 30: Linear Probing of CfC Hidden States — Self-Contained
===============================================================
Loads Mistral & Qwen, runs CfC inference on 3-class prompts, collects 16D
hidden states, then performs Linear Probing to create a "Brain Atlas".

Identifies which of the 16 CfC neurons specialize for Factual/Creative/Math.

Usage:
    python experiments/phase30_linear_probing.py
"""

import torch, torch.nn as nn
import os, sys, json, gc, time, datetime, random, math, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ncps.torch import CfC
from ncps.wirings import AutoNCP

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

SEED = 2026
SIGMA_MIN, SIGMA_MAX = 0.001, 0.15

# Same prompts as other experiments
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
    "What is 17 + 28?", "What is 156 - 89?", "What is 12 * 7?",
    "What is 144 / 12?", "What is 23 + 45 + 12?", "What is 15 * 15?",
    "What is 1000 - 573?", "What is 8 * 9?", "What is 256 / 16?",
    "What is 33 + 67?", "If x + 5 = 12, what is x?",
    "What is 2 to the power of 8?", "What is the square root of 144?",
    "What is 99 * 3?", "What is 50% of 240?",
    "If a triangle has sides 3, 4, and 5, what is its perimeter?",
    "What is 7! (7 factorial)?", "What is 1024 / 32?",
    "What is the next prime after 29?", "What is 13 * 13?",
]


class AdaptiveSNNHook:
    def __init__(self, sigma=0.05):
        self.sigma = sigma
        self.last_hidden_norm = 0.0
    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype) * self.sigma
        self.last_hidden_norm = hs.float().norm().item() / max(hs.numel(), 1)
        return (hs + noise,) + args[1:]


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find decoder layers")


class DualModeCfCActor(nn.Module):
    def __init__(self, input_size=4, num_neurons=16):
        super().__init__()
        wiring = AutoNCP(num_neurons, 1)
        self.cfc = CfC(input_size, wiring, batch_first=True)
    def forward(self, x, hx=None):
        output, hx = self.cfc(x, hx=hx)
        return output, hx


def collect_hidden_states(model, tokenizer, target_layers, model_short):
    """Run CfC Actor on 3-class prompts and collect hidden states."""
    from datasets import load_dataset

    print("\n  Collecting CfC hidden states for %s..." % model_short)

    hook = AdaptiveSNNHook(sigma=0.05)
    layers = get_layers(model)
    handles = [layers[i].register_forward_pre_hook(hook) for i in target_layers if i < len(layers)]

    # Load factual prompts
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), 20)
    factual_prompts = [ds[idx]["question"] for idx in indices]

    # Build dataset
    prompts = []
    for p in factual_prompts: prompts.append(("factual", p))
    for p in CREATIVE_PROMPTS: prompts.append(("creative", p))
    for p in MATH_PROMPTS: prompts.append(("math", p))

    # Create CfC actor
    actor = DualModeCfCActor(input_size=4, num_neurons=16)
    hidden_states = []
    hx = None

    for i, (task_type, prompt) in enumerate(prompts):
        text = prompt[:200]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            feat_layer = min(14, len(layers) - 1)
            h = outputs.hidden_states[feat_layer].float().squeeze(0)

        features = torch.tensor([
            h.norm().item() / max(h.numel(), 1),
            h.std().item(),
            hook.last_hidden_norm,
            i / len(prompts),
        ], dtype=torch.float32)

        x = features.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output, hx_new = actor.cfc(x, hx=hx)
            hx = hx_new

        if isinstance(hx, tuple):
            hs_np = hx[0].detach().cpu().numpy().flatten()
        else:
            hs_np = hx.detach().cpu().numpy().flatten()

        hidden_states.append({
            "type": task_type,
            "step": i,
            "hidden": hs_np.tolist()[:16],
        })

        if (i + 1) % 20 == 0:
            print("    [%d/%d] collected" % (i+1, len(prompts)))

    for h in handles:
        h.remove()

    print("  Collected %d hidden states (%dD)" % (len(hidden_states), len(hidden_states[0]["hidden"])))
    return hidden_states


def analyze_model(hidden_states, model_name):
    """Perform Linear Probing analysis on hidden states."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 60)
    print("  Linear Probing: %s" % model_name)
    print("  %d hidden state samples" % len(hidden_states))
    print("=" * 60)

    X = np.array([h["hidden"] for h in hidden_states])
    type_map = {"factual": 0, "creative": 1, "math": 2}
    y = np.array([type_map[h["type"]] for h in hidden_states])

    n_neurons = X.shape[1]
    print("  X shape: %s (samples x neurons)" % str(X.shape))
    counts = dict(zip(*np.unique(y, return_counts=True)))
    print("  Classes: %s" % counts)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Full model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(random_state=SEED, max_iter=1000, multi_class="multinomial")
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
    full_acc = scores.mean() * 100
    print("\n  Full model (all %d neurons): %.1f%% +/- %.1f%%" % (n_neurons, full_acc, scores.std()*100))

    # 2. Per-neuron probing
    neuron_scores = []
    for i in range(n_neurons):
        Xi = X_scaled[:, i:i+1]
        clf_i = LogisticRegression(random_state=SEED, max_iter=1000, multi_class="multinomial")
        try:
            scores_i = cross_val_score(clf_i, Xi, y, cv=cv, scoring="accuracy")
            acc_i = scores_i.mean() * 100
        except:
            acc_i = 33.3
        neuron_scores.append({"neuron": i, "accuracy": round(acc_i, 2)})

    neuron_scores.sort(key=lambda x: x["accuracy"], reverse=True)
    chance = 100.0 / 3
    print("\n  Per-Neuron Classification Accuracy:")
    for ns in neuron_scores:
        marker = " *" if ns["accuracy"] > chance + 5 else ""
        print("    N%2d: %5.1f%%%s" % (ns["neuron"], ns["accuracy"], marker))

    # 3. Feature importance via coefficients
    clf_full = LogisticRegression(random_state=SEED, max_iter=1000, multi_class="multinomial")
    clf_full.fit(X_scaled, y)
    coefs = clf_full.coef_
    class_names = ["Factual", "Creative", "Math"]

    print("\n  Neuron Roles (LR Coefficients):")
    print("  %4s | %8s | %8s | %8s | %15s" % ("N", "Fact", "Creat", "Math", "Role"))
    print("  " + "-"*4 + "-+-" + "-"*8 + "-+-" + "-"*8 + "-+-" + "-"*8 + "-+-" + "-"*15)

    neuron_roles = []
    for i in range(n_neurons):
        c_f, c_c, c_m = coefs[0, i], coefs[1, i], coefs[2, i]
        max_class = np.argmax(np.abs([c_f, c_c, c_m]))
        max_val = [c_f, c_c, c_m][max_class]
        sign = "+" if max_val > 0 else "-"
        role = sign + class_names[max_class]
        if np.abs(max_val) < 0.3:
            role = "Neutral"
        neuron_roles.append({
            "neuron": i, "role": role,
            "factual_coef": round(float(c_f), 4),
            "creative_coef": round(float(c_c), 4),
            "math_coef": round(float(c_m), 4),
            "max_abs_coef": round(float(np.max(np.abs([c_f, c_c, c_m]))), 4),
        })
        print("  N%2d | %+8.3f | %+8.3f | %+8.3f | %15s" % (i, c_f, c_c, c_m, role))

    # 4. Mean activations per task
    activation_patterns = []
    for i in range(n_neurons):
        means = {}
        for c_idx, cn in enumerate(class_names):
            mask = (y == c_idx)
            means[cn.lower()] = round(float(X[:, i][mask].mean()), 4)
        variance = float(np.var(list(means.values())))
        activation_patterns.append({"neuron": i, "means": means, "variance": round(variance, 6)})

    # Top discriminating neurons by variance
    top_var = sorted(activation_patterns, key=lambda x: x["variance"], reverse=True)[:5]
    print("\n  Top 5 Task-Discriminating Neurons:")
    for tv in top_var:
        print("    N%2d: var=%.5f (F=%+.4f C=%+.4f M=%+.4f)" % (
            tv["neuron"], tv["variance"],
            tv["means"]["factual"], tv["means"]["creative"], tv["means"]["math"]))

    return {
        "model": model_name,
        "n_samples": len(hidden_states),
        "n_neurons": n_neurons,
        "full_model_accuracy": round(full_acc, 2),
        "per_neuron_scores": neuron_scores,
        "neuron_roles": neuron_roles,
        "activation_patterns": activation_patterns,
    }


def visualize(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(results)
    fig, axes = plt.subplots(n_models, 3, figsize=(20, 7 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#ccc", "ytick.color": "#ccc",
        "grid.color": "#333", "grid.alpha": 0.3,
    })
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Phase 30: CfC Brain Atlas -- Linear Probing of 16D Hidden States\n"
                 "Which neurons specialize for which task?",
                 fontsize=14, fontweight="bold", color="#e94560")

    class_names = ["Factual", "Creative", "Math"]

    for row, (model_name, result) in enumerate(results.items()):
        n_neurons = result["n_neurons"]

        # Panel 1: Per-neuron accuracy
        ax1 = axes[row, 0]
        neuron_order = list(range(n_neurons))
        accs = []
        for n in neuron_order:
            acc = next((ns["accuracy"] for ns in result["per_neuron_scores"] if ns["neuron"] == n), 33.3)
            accs.append(acc)
        colors = ["#66BB6A" if a > 38.3 else "#EF5350" if a < 28.3 else "#FFA726" for a in accs]
        ax1.barh(range(n_neurons), accs, color=colors, edgecolor="#333")
        ax1.axvline(x=33.3, color="#eee", linestyle="--", alpha=0.5, label="Chance")
        ax1.set_yticks(range(n_neurons))
        ax1.set_yticklabels(["N%d" % n for n in neuron_order], fontsize=8)
        ax1.set_xlabel("Accuracy (%)")
        ax1.set_title("%s\nPer-Neuron Probing" % model_name, fontweight="bold")
        ax1.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
        ax1.invert_yaxis()

        # Panel 2: Coefficient heatmap
        ax2 = axes[row, 1]
        coef_matrix = np.array([[nr["%s_coef" % cn.lower()] for cn in class_names]
                                for nr in sorted(result["neuron_roles"], key=lambda x: x["neuron"])])
        vmax = max(abs(coef_matrix.min()), abs(coef_matrix.max()), 1.0)
        im = ax2.imshow(coef_matrix.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax2.set_xticks(range(n_neurons))
        ax2.set_xticklabels(["N%d" % i for i in range(n_neurons)], fontsize=7, rotation=45)
        ax2.set_yticks(range(3))
        ax2.set_yticklabels(class_names)
        ax2.set_title("LR Coefficients (Red=+, Blue=-)", fontweight="bold")
        plt.colorbar(im, ax=ax2, shrink=0.6)

        # Panel 3: Activation patterns
        ax3 = axes[row, 2]
        x = np.arange(n_neurons)
        w = 0.25
        for c_idx, (cn, color) in enumerate(zip(class_names, ["#4FC3F7", "#FF7043", "#AB47BC"])):
            means = []
            for n in range(n_neurons):
                ap = next((a for a in result["activation_patterns"] if a["neuron"] == n), None)
                means.append(ap["means"][cn.lower()] if ap else 0)
            ax3.bar(x + c_idx * w, means, w, label=cn, color=color, alpha=0.7, edgecolor="#333")
        ax3.set_xticks(x + w)
        ax3.set_xticklabels(["N%d" % i for i in range(n_neurons)], fontsize=7, rotation=45)
        ax3.set_ylabel("Mean Activation")
        ax3.set_title("Activations by Task Type", fontweight="bold")
        ax3.legend(fontsize=8, facecolor="#16213e", edgecolor="#555")
        ax3.grid(True, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIGURES_DIR, "phase30_linear_probing.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print("\n  Figure: %s" % out)
    return out


def main():
    print("=" * 60)
    print("Phase 30: Linear Probing -- CfC Brain Atlas")
    print("  Which of the 16 CfC neurons specialize for which task?")
    print("=" * 60)
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    results = {}
    all_hidden_states = {}

    # -- Model 1: Mistral --
    print("\n" + "-" * 60)
    print("  Loading Mistral-7B...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2", quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    model.eval()
    print("  Mistral-7B loaded: %d layers" % len(get_layers(model)))

    target_layers_m = list(range(15, 21))
    hs_mistral = collect_hidden_states(model, tokenizer, target_layers_m, "Mistral-7B")
    all_hidden_states["Mistral-7B"] = hs_mistral
    results["Mistral-7B"] = analyze_model(hs_mistral, "Mistral-7B")

    del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

    # -- Model 2: Qwen --
    print("\n" + "-" * 60)
    print("  Loading Qwen2.5-7B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    model.eval()
    print("  Qwen2.5-7B loaded: %d layers" % len(get_layers(model)))

    target_layers_q = list(range(12, 18))
    hs_qwen = collect_hidden_states(model, tokenizer, target_layers_q, "Qwen2.5-7B")
    all_hidden_states["Qwen2.5-7B"] = hs_qwen
    results["Qwen2.5-7B"] = analyze_model(hs_qwen, "Qwen2.5-7B")

    del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

    # -- Visualize --
    fig_path = visualize(results)

    # -- Cross-model comparison --
    print("\n" + "=" * 60)
    print("  Cross-Model Brain Atlas Comparison")
    print("=" * 60)
    for model_name, res in results.items():
        print("\n  %s:" % model_name)
        print("    Full probing accuracy: %.1f%%" % res["full_model_accuracy"])
        top3 = res["per_neuron_scores"][:3]
        top3_parts = []
        for n in top3:
            top3_parts.append("N%d=%.1f%%" % (n["neuron"], n["accuracy"]))
        print("    Top 3 neurons: " + ", ".join(top3_parts))
        roles = set()
        for nr in res["neuron_roles"]:
            if nr["role"] != "Neutral":
                roles.add(nr["role"])
        print("    Active roles: %s" % roles)

    elapsed = time.time() - t_start

    output = {
        "phase": "Phase 30: Linear Probing -- CfC Brain Atlas",
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
        "hidden_states": all_hidden_states,
        "figure_path": fig_path,
    }
    out = os.path.join(RESULTS_DIR, "phase30_linear_probing_log.json")
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Results: %s" % out)

    print("\n" + "=" * 60)
    print("  Phase 30 COMPLETE -- %ds (%.1f min)" % (elapsed, elapsed/60))
    print("=" * 60)


if __name__ == "__main__":
    main()
