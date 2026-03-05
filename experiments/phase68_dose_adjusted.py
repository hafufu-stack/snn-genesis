"""
Phase 68: Dose-Adjusted Correlated Noise
=========================================

Phase 67 showed that σ=0.15 on multiple layers = overdose (0% solve rate).
This phase fixes the dosage: σ is scaled by 1/√N to keep total perturbation
comparable to single-layer σ=0.15.

Key insight: If 1 layer with σ=0.15 works, then:
  - 2 layers need σ = 0.15/√2 ≈ 0.106
  - 4 layers need σ = 0.15/√4 = 0.075

Conditions (10, N=50 each):
  1. baseline           — No noise
  2. L18_only_0.15      — Single layer reference (σ=0.15)
  3. L17L18_pos_adj     — ρ=+1, σ=0.106 (√2-adjusted)
  4. L17L18_neg_adj     — ρ=-1, σ=0.106
  5. push_pull_adj      — T-1000 4-layer, σ=0.075 (√4-adjusted)
  6. push_pull_low      — T-1000 4-layer, σ=0.050 (conservative)
  7. push_pull_first10  — T-1000 + flash anneal, σ=0.075
  8. L17L18_pos_low     — ρ=+1, σ=0.075 (well below cliff)
  9. L17L18_pos_first10 — ρ=+1, σ=0.106 + first10 linear decay
 10. single_L17_only    — σ=0.15 on L17 alone (layer comparison)

Total: 500 games. ~10 hours on RTX 5080.

Usage:
    python experiments/phase68_dose_adjusted.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, json, gc, time, random, re, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.stats import fisher_exact

# === Config ===
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
SEED = 2026
MAX_STEPS = 50
N_PER_CONDITION = 50

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================
#  HANOI ENVIRONMENT
# ===================================================

class HanoiEnv:
    def __init__(self, n_disks=3, modified=True):
        self.n_disks = n_disks
        self.modified = modified
        self.reset()

    def reset(self):
        self.pegs = {"A": list(range(self.n_disks, 0, -1)), "B": [], "C": []}
        self.moves = []
        self.illegal_count = 0
        self.total_attempts = 0
        self.self_corrections = 0
        self._prev_illegal = False

    def is_solved(self):
        return len(self.pegs["C"]) == self.n_disks

    def legal_moves(self):
        result = []
        for f in "ABC":
            for t in "ABC":
                if f != t and self.pegs[f]:
                    disk = self.pegs[f][-1]
                    if not self.pegs[t] or \
                       (self.modified and disk > self.pegs[t][-1]) or \
                       (not self.modified and disk < self.pegs[t][-1]):
                        result.append(f"{f}->{t}")
        return result

    def try_move(self, from_p, to_p):
        self.total_attempts += 1
        from_p, to_p = from_p.upper(), to_p.upper()
        if from_p not in "ABC" or to_p not in "ABC" or from_p == to_p:
            self.illegal_count += 1; self._prev_illegal = True
            return False, "Invalid peg"
        if not self.pegs[from_p]:
            self.illegal_count += 1; self._prev_illegal = True
            return False, f"{from_p} is empty"
        disk = self.pegs[from_p][-1]
        if self.pegs[to_p]:
            top = self.pegs[to_p][-1]
            if self.modified and disk <= top:
                self.illegal_count += 1; self._prev_illegal = True
                return False, f"Modified: disk {disk} > {top}"
            if not self.modified and disk >= top:
                self.illegal_count += 1; self._prev_illegal = True
                return False, f"Standard: disk {disk} < {top}"
        if self._prev_illegal:
            self.self_corrections += 1
        self._prev_illegal = False
        self.pegs[from_p].pop()
        self.pegs[to_p].append(disk)
        self.moves.append(f"{from_p}->{to_p} (disk {disk})")
        return True, f"Moved disk {disk}: {from_p}->{to_p}"

    def state_str(self):
        return f"A:{self.pegs['A']} B:{self.pegs['B']} C:{self.pegs['C']}"

    def stats(self):
        return {
            "solved": self.is_solved(),
            "legal_moves": len(self.moves),
            "illegal_moves": self.illegal_count,
            "self_corrections": self.self_corrections,
        }


# ===================================================
#  PROMPT & PARSER
# ===================================================

def build_system_msg(env):
    rules = "MODIFIED RULES: You can ONLY place a LARGER disk onto a SMALLER disk. The opposite of standard."
    return (
        f"You are solving Tower of Hanoi with {env.n_disks} disks. "
        f"{rules} "
        f"Goal: move ALL disks from A to C. "
        f"Respond with EXACTLY one move in format: Move: X->Y (e.g. Move: A->C). "
        f"You may add a brief Think: line before it."
    )

def build_user_msg(env, error=None):
    msg = f"State: {env.state_str()}\n"
    legal = env.legal_moves()
    msg += f"Legal moves: {', '.join(legal)}\n"
    if env.moves:
        recent = env.moves[-3:]
        msg += f"Your last moves: {'; '.join(recent)}\n"
    if error:
        msg += f"ERROR: {error}. Pick from legal moves above.\n"
    msg += "Your move:"
    return msg

def build_chat_prompt(tokenizer, env, error=None):
    messages = [
        {"role": "user", "content": build_system_msg(env) + "\n\n" + build_user_msg(env, error)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_move(response):
    patterns = [
        r'Move:\s*([A-Ca-c])\s*->\s*([A-Ca-c])',
        r'move\s+(?:disk\s+\d+\s+)?(?:from\s+)?([A-Ca-c])\s+to\s+([A-Ca-c])',
        r'([A-Ca-c])\s*->\s*([A-Ca-c])',
    ]
    for p in patterns:
        m = re.search(p, response, re.IGNORECASE)
        if m:
            return m.group(1).upper(), m.group(2).upper()
    return None


# ===================================================
#  CORRELATED NOISE MANAGER (bugfixed from Phase 67)
# ===================================================

class CorrelatedNoiseManager:
    """Manages dose-adjusted correlated noise across multiple layers.
    
    Phase 67 fix: base_noise is reset at the shallowest correlated layer
    on every forward pass, so each autoregressive token gets fresh noise.
    
    Phase 68 key change: sigma is scaled by 1/sqrt(N_layers) to prevent
    cosine similarity collapse when injecting into multiple layers.
    """
    def __init__(self, sigma=0.15):
        self.sigma = sigma
        self.active = False
        self.base_noise = None
        self.layer_configs = {}
        self.handles = []
        self.first_corr_layer = None

    def configure(self, layer_configs):
        self.layer_configs = layer_configs
        corr_layers = [l for l, w in layer_configs.items() if w != 'independent']
        self.first_corr_layer = min(corr_layers) if corr_layers else None

    def register(self, model):
        layers = model.model.layers
        for layer_idx, weight in self.layer_configs.items():
            hook = self._make_hook(layer_idx, weight)
            handle = layers[layer_idx].register_forward_pre_hook(hook)
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make_hook(self, layer_idx, weight):
        manager = self
        def hook_fn(module, args):
            if not manager.active or manager.sigma <= 0:
                return args
            hs = args[0]
            
            if weight == 'independent':
                noise = torch.randn_like(hs) * manager.sigma
            else:
                # Reset base noise at shallowest correlated layer (Phase 67 bugfix)
                if layer_idx == manager.first_corr_layer or manager.base_noise is None or manager.base_noise.shape != hs.shape:
                    manager.base_noise = torch.randn_like(hs) * manager.sigma
                
                if abs(weight) == 1.0:
                    noise = weight * manager.base_noise
                else:
                    rho = weight
                    fresh = torch.randn_like(hs) * manager.sigma
                    noise = rho * manager.base_noise + math.sqrt(1 - rho**2) * fresh
            
            return (hs + noise,) + args[1:]
        return hook_fn


# ===================================================
#  MODEL + GENERATION
# ===================================================

def load_model():
    print(f"\n Loading {MODEL_NAME}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    print(f"  Done: {len(model.model.layers)} layers")
    return model, tok

def generate_with_manager(model, tok, prompt, manager, temperature=0.5, max_tokens=80):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperature, top_p=0.9,
            repetition_penalty=1.2, pad_token_id=tok.pad_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    inp = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(inp):].strip()


# ===================================================
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, env, manager, noise_fn=None):
    env.reset()
    error = None; consec_fail = 0
    legal_move_count = 0

    for step in range(MAX_STEPS):
        if noise_fn:
            noise_fn(step, legal_move_count)

        prompt = build_chat_prompt(tok, env, error)
        resp = generate_with_manager(model, tok, prompt, manager)
        move = parse_move(resp)

        if move is None:
            env.illegal_count += 1; env.total_attempts += 1; env._prev_illegal = True
            error = "Parse fail. Use Move: X->Y"; consec_fail += 1
            if consec_fail >= 10: break
            continue

        ok, msg = env.try_move(move[0], move[1])
        if ok:
            legal_move_count += 1
            error = None; consec_fail = 0
            if env.is_solved(): break
        else:
            error = msg; consec_fail += 1
            if consec_fail >= 10: break

    manager.active = False
    st = env.stats()
    st["total_legal_moves"] = legal_move_count
    return st


# ===================================================
#  EXPERIMENT DEFINITIONS
# ===================================================

SIGMA_1L = 0.15                        # Single-layer optimal
SIGMA_2L = SIGMA_1L / math.sqrt(2)    # ≈ 0.106
SIGMA_4L = SIGMA_1L / math.sqrt(4)    # = 0.075
SIGMA_LOW = 0.050                      # Conservative

EXPERIMENTS = [
    {
        "name": "baseline",
        "sigma": 0,
        "layers": {},
        "first_n": None,
        "desc": "No noise (control)"
    },
    {
        "name": "L18_only_0.15",
        "sigma": SIGMA_1L,
        "layers": {18: 1.0},
        "first_n": None,
        "desc": f"Single-layer reference (σ={SIGMA_1L:.3f})"
    },
    {
        "name": "L17L18_pos_adj",
        "sigma": SIGMA_2L,
        "layers": {17: 1.0, 18: 1.0},
        "first_n": None,
        "desc": f"ρ=+1 on L17,L18, σ={SIGMA_2L:.3f} (√2-adjusted)"
    },
    {
        "name": "L17L18_neg_adj",
        "sigma": SIGMA_2L,
        "layers": {17: 1.0, 18: -1.0},
        "first_n": None,
        "desc": f"ρ=-1 on L17,L18, σ={SIGMA_2L:.3f} (√2-adjusted)"
    },
    {
        "name": "push_pull_adj",
        "sigma": SIGMA_4L,
        "layers": {17: 1.0, 18: 1.0, 15: -1.0, 20: -1.0},
        "first_n": None,
        "desc": f"T-1000 4-layer push-pull, σ={SIGMA_4L:.3f} (√4-adjusted)"
    },
    {
        "name": "push_pull_low",
        "sigma": SIGMA_LOW,
        "layers": {17: 1.0, 18: 1.0, 15: -1.0, 20: -1.0},
        "first_n": None,
        "desc": f"T-1000 4-layer push-pull, σ={SIGMA_LOW:.3f} (conservative)"
    },
    {
        "name": "push_pull_first10",
        "sigma": SIGMA_4L,
        "layers": {17: 1.0, 18: 1.0, 15: -1.0, 20: -1.0},
        "first_n": 10,
        "desc": f"T-1000 + flash anneal (first 10 moves), σ={SIGMA_4L:.3f}"
    },
    {
        "name": "L17L18_pos_low",
        "sigma": SIGMA_4L,
        "layers": {17: 1.0, 18: 1.0},
        "first_n": None,
        "desc": f"ρ=+1 on L17,L18, σ={SIGMA_4L:.3f} (well below cliff)"
    },
    {
        "name": "L17L18_pos_first10",
        "sigma": SIGMA_2L,
        "layers": {17: 1.0, 18: 1.0},
        "first_n": 10,
        "desc": f"ρ=+1 on L17,L18 + first10 linear decay, σ={SIGMA_2L:.3f}"
    },
    {
        "name": "single_L17_only",
        "sigma": SIGMA_1L,
        "layers": {17: 1.0},
        "first_n": None,
        "desc": f"σ={SIGMA_1L:.3f} on L17 alone (layer comparison)"
    },
]


# ===================================================
#  MAIN
# ===================================================

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tok = load_model()

    print("\n" + "="*80)
    print("  PHASE 68: DOSE-ADJUSTED CORRELATED NOISE")
    print("  σ scaled by 1/√N to prevent cosine collapse on multi-layer injection")
    print("="*80)

    all_summaries = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        sigma = exp["sigma"]
        print(f"\n  === {name}: {exp['desc']} | N={N_PER_CONDITION} ===")

        manager = CorrelatedNoiseManager(sigma=sigma)
        manager.configure(exp["layers"])
        manager.register(model)

        # Noise control function
        if not exp["layers"] or sigma == 0:
            def make_fn(mgr):
                def fn(step, lm): mgr.active = False
                return fn
        elif exp["first_n"] is not None:
            limit = exp["first_n"]
            def make_fn(mgr, limit=limit, base_sigma=sigma):
                def fn(step, lm):
                    if lm < limit:
                        mgr.active = True
                        mgr.sigma = base_sigma * (1.0 - lm / limit)  # Linear decay
                    else:
                        mgr.active = False
                return fn
        else:
            def make_fn(mgr, base_sigma=sigma):
                def fn(step, lm):
                    mgr.active = True
                    mgr.sigma = base_sigma
                return fn

        noise_fn = make_fn(manager)
        results = []; solved = 0

        for t in range(N_PER_CONDITION):
            env = HanoiEnv(3, modified=True)
            r = play_game(model, tok, env, manager, noise_fn)
            results.append(r)
            if r["solved"]: solved += 1
            icon = "O" if r["solved"] else "X"
            rate = solved / (t+1) * 100
            print(f"    {t+1:3d}/{N_PER_CONDITION}: {icon} legal={r['legal_moves']:2d} "
                  f"illegal={r['illegal_moves']:2d} [{solved}/{t+1} = {rate:.0f}%]")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if (t+1) % 10 == 0: gc.collect()

        manager.remove()

        sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
        n_solved = sum(1 for r in results if r["solved"])
        all_summaries.append({
            "condition": name,
            "description": exp["desc"],
            "sigma": sigma,
            "layer_config": {str(k): str(v) for k, v in exp["layers"].items()},
            "first_n": exp["first_n"],
            "n_trials": N_PER_CONDITION, "n_solved": n_solved,
            "solve_rate": sr,
            "avg_legal": round(np.mean([r["legal_moves"] for r in results]), 2),
            "avg_illegal": round(np.mean([r["illegal_moves"] for r in results]), 2),
        })
        print(f"    => Solve={sr:.1f}% ({n_solved}/{N_PER_CONDITION})")

    elapsed = time.time() - t0

    # Visualization
    fig_path = visualize(all_summaries)

    # Verdict
    bl = next(s for s in all_summaries if s["condition"] == "baseline")
    ref = next(s for s in all_summaries if s["condition"] == "L18_only_0.15")
    print(f"\n{'='*80}")
    print(f"  PHASE 68 VERDICT: Dose-Adjusted Correlated Noise")
    print(f"{'='*80}")
    for s in all_summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<24} σ={s['sigma']:.3f}  {s['solve_rate']:>6.1f}%   (baseline)")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        # Also compare vs L18_only reference
        table2 = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                   [ref["n_solved"], ref["n_trials"]-ref["n_solved"]]]
        _, p2 = fisher_exact(table2, alternative='two-sided')
        vs_ref = "BETTER" if s["solve_rate"] > ref["solve_rate"] and p2 < 0.1 else \
                 "WORSE" if s["solve_rate"] < ref["solve_rate"] and p2 < 0.1 else "≈same"
        print(f"  {s['condition']:<24} σ={s['sigma']:.3f}  {s['solve_rate']:>6.1f}%   p={p:.4f} {sig}  vs_ref={vs_ref}")
    print(f"\n  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")

    # Save
    out = {
        "experiment": "Phase 68: Dose-Adjusted Correlated Noise",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "purpose": "Test correlated noise with σ scaled by 1/√N to prevent overdose",
        "phase67_lesson": "σ=0.15 on multiple layers causes cosine collapse (0%); need dose adjustment",
        "sigma_formula": "σ_adjusted = σ_optimal / √N_layers",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_summaries, "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase68_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")


def visualize(summaries):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Phase 68: Dose-Adjusted Correlated Noise\n"
                 "σ scaled by 1/√N to prevent cosine collapse",
                 fontsize=14, fontweight="bold", y=1.02)

    bl = next(s for s in summaries if s["condition"] == "baseline")
    names = [s["condition"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]
    sigmas = [s["sigma"] for s in summaries]

    # Panel 1: Bar chart
    ax = axes[0]
    colors = ['#95a5a6', '#3498db',
              '#2ecc71', '#e74c3c',
              '#f39c12', '#9b59b6',
              '#e67e22', '#1abc9c',
              '#d35400', '#2980b9']
    bars = ax.bar(range(len(summaries)), rates, color=colors[:len(summaries)], alpha=0.85,
                  edgecolor='white', linewidth=2)
    for i, (b, s) in enumerate(zip(bars, summaries)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               f'{s["solve_rate"]:.0f}%\nσ={s["sigma"]:.3f}',
               ha='center', fontsize=7, fontweight='bold')

    ax.set_xticks(range(len(summaries)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=5.5)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Performance by Condition", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: σ vs Solve Rate scatter (per layer count)
    ax = axes[1]
    for s in summaries:
        if s["condition"] == "baseline": continue
        n_layers = len([v for v in s.get("layer_config", {}).values()])
        if n_layers == 0: n_layers = 1
        marker = {1: 'o', 2: 's', 4: 'D'}.get(n_layers, '^')
        color = {1: '#3498db', 2: '#2ecc71', 4: '#e74c3c'}.get(n_layers, 'gray')
        ax.scatter(s["sigma"], s["solve_rate"], s=150, marker=marker,
                  c=color, edgecolors='black', zorder=5, label=f'{n_layers}L' if s["condition"].endswith("0.15") or s["condition"].startswith("push_pull_adj") else None)
        ax.annotate(s["condition"].replace("_", "\n"), (s["sigma"], s["solve_rate"]),
                   textcoords="offset points", xytext=(5, 5), fontsize=5)

    ax.set_xlabel("σ per layer", fontsize=12)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("σ vs Performance by Layer Count\n●=1L  ■=2L  ◆=4L", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase68_dose_adjusted.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


if __name__ == "__main__":
    main()

    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n Phase 68 complete.")
