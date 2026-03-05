"""
Phase 67: Inter-Layer Correlated Noise — The T-1000 Protocol
=============================================================

Inspired by Funasaki (2014/2015) master's thesis on correlated inputs
to hippocampal dentate gyrus granule cells, this experiment tests whether
STRUCTURED (correlated) noise across multiple layers outperforms
INDEPENDENT noise at a single layer.

Key insight: Independent noise across layers = "seizure" (random jerks).
Correlated noise = "coordinated locomotion" (T-1000 running).

Conditions (8, N=50 each):
  1. baseline          — No noise
  2. L18_only          — σ=0.15 on L18 only (standard reference)
  3. L17L18_positive   — Same noise (+Z) on L17 and L18 (ρ=+1.0)
  4. L17L18_negative   — Opposite noise (+Z on L17, -Z on L18, ρ=-1.0)
  5. push_pull_4layer  — L17,L18 = +Z; L15,L20 = -Z (T-1000 walk)
  6. push_pull_first10 — Push-pull for first 10 moves only (flash anneal combo)
  7. independent_4layer — Independent noise on L15,L17,L18,L20 (seizure control)
  8. L17L18_half_corr  — ρ=+0.5 partial correlation on L17,L18

Total: 400 games. ~8-10 hours on RTX 5080.

Usage:
    python experiments/phase67_correlated_noise.py
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
#  CORRELATED NOISE HOOK SYSTEM
# ===================================================

class CorrelatedNoiseManager:
    """Manages correlated noise injection across multiple layers.
    
    Generates a single base noise vector Z and distributes it to
    target layers with specified signs/correlations.
    
    This implements the "T-1000 Protocol": nearby layers get positive
    correlation (same direction), distant layers get negative correlation
    (opposite direction), creating coordinated "locomotion" instead of
    random "seizures".
    """
    def __init__(self, sigma=0.15):
        self.sigma = sigma
        self.active = False
        self.base_noise = None  # Shared noise vector Z
        self.layer_configs = {}  # {layer_idx: correlation_weight}
        self.hooks = {}  # {layer_idx: hook_fn}
        self.handles = []
        self.first_corr_layer = None  # Shallowest correlated layer

    def configure(self, layer_configs):
        """Set which layers get noise and their correlation weights.
        
        Args:
            layer_configs: dict of {layer_idx: weight}
              weight = +1.0  -> same direction as base noise
              weight = -1.0  -> opposite direction
              weight = +0.5  -> 50% correlated
              weight = 'independent' -> fresh independent noise
        """
        self.layer_configs = layer_configs
        # Identify shallowest correlated layer for per-token noise reset
        corr_layers = [l for l, w in layer_configs.items() if w != 'independent']
        self.first_corr_layer = min(corr_layers) if corr_layers else None

    def register(self, model):
        """Register hooks on all configured layers."""
        layers = model.model.layers
        for layer_idx, weight in self.layer_configs.items():
            hook = self._make_hook(layer_idx, weight)
            handle = layers[layer_idx].register_forward_pre_hook(hook)
            self.handles.append(handle)

    def remove(self):
        """Remove all hooks."""
        for h in self.handles:
            h.remove()
        self.handles = []

    def new_step(self):
        """Reset base noise for a new generation step.
        Called before each forward pass to ensure correlated layers
        share the same base noise vector.
        """
        self.base_noise = None

    def _make_hook(self, layer_idx, weight):
        manager = self
        def hook_fn(module, args):
            if not manager.active or manager.sigma <= 0:
                return args
            hs = args[0]
            
            if weight == 'independent':
                # Fresh independent noise (seizure control)
                noise = torch.randn_like(hs) * manager.sigma
            else:
                # Reset base noise at shallowest correlated layer (once per forward pass)
                # This ensures each autoregressive token gets FRESH correlated noise
                # instead of reusing the same Z forever ("muscle rigidity bug")
                if layer_idx == manager.first_corr_layer or manager.base_noise is None or manager.base_noise.shape != hs.shape:
                    manager.base_noise = torch.randn_like(hs) * manager.sigma
                
                if abs(weight) == 1.0:
                    # Fully correlated (positive or negative)
                    noise = weight * manager.base_noise
                else:
                    # Partially correlated: weight * Z + sqrt(1-weight²) * fresh
                    rho = weight
                    fresh = torch.randn_like(hs) * manager.sigma
                    noise = rho * manager.base_noise + math.sqrt(1 - rho**2) * fresh
            
            return (hs + noise,) + args[1:]
        return hook_fn


# ===================================================
#  MODEL + GENERATION (with noise step reset)
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
    """Generate with noise step reset before each forward pass."""
    manager.new_step()  # Reset base noise for correlated layers
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
    """Play a game with the correlated noise manager."""
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

EXPERIMENTS = [
    {
        "name": "baseline",
        "layers": {},
        "always_on": True,
        "first_n": None,
        "desc": "No noise (control)"
    },
    {
        "name": "L18_only",
        "layers": {18: 1.0},
        "always_on": True,
        "first_n": None,
        "desc": "Standard single-layer reference"
    },
    {
        "name": "L17L18_positive",
        "layers": {17: 1.0, 18: 1.0},
        "always_on": True,
        "first_n": None,
        "desc": "Same noise on L17 & L18 (ρ=+1.0, push together)"
    },
    {
        "name": "L17L18_negative",
        "layers": {17: 1.0, 18: -1.0},
        "always_on": True,
        "first_n": None,
        "desc": "Opposite noise on L17 vs L18 (ρ=-1.0, edge detection)"
    },
    {
        "name": "push_pull_4layer",
        "layers": {17: 1.0, 18: 1.0, 15: -1.0, 20: -1.0},
        "always_on": True,
        "first_n": None,
        "desc": "T-1000 walk: L17,L18=+Z; L15,L20=-Z"
    },
    {
        "name": "push_pull_first10",
        "layers": {17: 1.0, 18: 1.0, 15: -1.0, 20: -1.0},
        "always_on": False,
        "first_n": 10,
        "desc": "T-1000 walk + flash anneal (first 10 moves only)"
    },
    {
        "name": "independent_4layer",
        "layers": {15: 'independent', 17: 'independent', 18: 'independent', 20: 'independent'},
        "always_on": True,
        "first_n": None,
        "desc": "Independent noise on 4 layers (seizure control)"
    },
    {
        "name": "L17L18_half_corr",
        "layers": {17: 1.0, 18: 0.5},
        "always_on": True,
        "first_n": None,
        "desc": "Partial correlation ρ=0.5 on L17→L18"
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
    print("  PHASE 67: INTER-LAYER CORRELATED NOISE (T-1000 PROTOCOL)")
    print("  Testing structured (correlated) vs independent noise across layers")
    print("="*80)

    all_summaries = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        print(f"\n  === {name}: {exp['desc']} | N={N_PER_CONDITION} ===")

        # Set up manager
        manager = CorrelatedNoiseManager(sigma=0.15)
        manager.configure(exp["layers"])
        manager.register(model)

        # Build noise control function
        if not exp["layers"]:
            # Baseline: no noise
            def make_fn(mgr):
                def fn(step, lm): mgr.active = False
                return fn
        elif exp["first_n"] is not None:
            # Duration-limited
            limit = exp["first_n"]
            def make_fn(mgr, limit=limit):
                def fn(step, lm):
                    if lm < limit:
                        mgr.active = True
                        mgr.sigma = 0.15 * (1.0 - lm / limit)  # Linear decay
                    else:
                        mgr.active = False
                return fn
        else:
            # Always on
            def make_fn(mgr):
                def fn(step, lm):
                    mgr.active = True
                    mgr.sigma = 0.15
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
    print(f"\n{'='*75}")
    print(f"  PHASE 67 VERDICT: Inter-Layer Correlated Noise (T-1000 Protocol)")
    print(f"{'='*75}")
    for s in all_summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<22} {s['solve_rate']:>6.1f}%   (baseline)")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {s['condition']:<22} {s['solve_rate']:>6.1f}%   p={p:.4f} {sig}")
    print(f"\n  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*75}")

    # Save
    out = {
        "experiment": "Phase 67: Inter-Layer Correlated Noise (T-1000 Protocol)",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "purpose": "Test structured (correlated) noise across multiple layers vs independent single-layer noise",
        "inspiration": "Funasaki (2015) 'Mathematical approach for neural information processing using differential equations' — Tamagawa University",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_summaries, "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase67_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")


def visualize(summaries):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Phase 67: Inter-Layer Correlated Noise (T-1000 Protocol)\n"
                 "Structured (correlated) noise vs independent noise across layers",
                 fontsize=14, fontweight="bold", y=1.02)

    bl = next(s for s in summaries if s["condition"] == "baseline")
    names = [s["condition"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]

    # Panel 1: Bar chart
    ax = axes[0]
    colors = ['#95a5a6', '#3498db',  # baseline, L18 reference
              '#2ecc71', '#e74c3c',  # positive, negative
              '#f39c12', '#9b59b6',  # push-pull, push-pull+first10
              '#e67e22', '#1abc9c']  # independent, half-corr
    bars = ax.bar(range(len(summaries)), rates, color=colors[:len(summaries)], alpha=0.85,
                  edgecolor='white', linewidth=2)
    for i, (b, s) in enumerate(zip(bars, summaries)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               f'{s["solve_rate"]:.0f}%', ha='center', fontsize=9, fontweight='bold')
        if i > 0:
            table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                     [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
            _, p = fisher_exact(table, alternative='two-sided')
            sig = "★★★" if p < 0.001 else "★★" if p < 0.01 else "★" if p < 0.05 else "n.s."
            ax.text(i, s["solve_rate"]+4, sig, ha='center', fontsize=8,
                   color='darkred' if p < 0.05 else 'gray')

    ax.set_xticks(range(len(summaries)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=6)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Performance by Noise Structure", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Layer diagram with correlation arrows
    ax = axes[1]
    layer_positions = {15: 0, 17: 2, 18: 3, 20: 5}
    experiment_labels = ["L18_only", "L17L18_positive", "L17L18_negative",
                        "push_pull_4layer", "independent_4layer"]
    y_pos = {name: i for i, name in enumerate(reversed(experiment_labels))}

    for exp_name in experiment_labels:
        exp = next((e for e in EXPERIMENTS if e["name"] == exp_name), None)
        if not exp: continue
        y = y_pos[exp_name]
        s = next((s for s in summaries if s["condition"] == exp_name), None)
        if not s: continue

        for layer_idx, weight in exp["layers"].items():
            x = layer_positions.get(layer_idx, layer_idx)
            if weight == 'independent':
                color = '#e67e22'
                marker = 's'
            elif isinstance(weight, (int, float)) and weight > 0:
                color = '#2ecc71'
                marker = '^'
            else:
                color = '#e74c3c'
                marker = 'v'
            ax.scatter(x, y, c=color, marker=marker, s=200, zorder=5, edgecolors='black')

        # Result label
        ax.text(6.5, y, f'{s["solve_rate"]:.0f}%', fontsize=12, fontweight='bold',
               va='center', color='darkgreen' if s["solve_rate"] > 30 else 'black')

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(reversed(experiment_labels)), fontsize=8)
    ax.set_xticks(list(layer_positions.values()))
    ax.set_xticklabels([f"L{k}" for k in layer_positions.keys()], fontsize=10)
    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_title("Layer Configuration Map\n▲=+Z (push)  ▼=-Z (pull)  ■=independent",
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase67_correlated_noise.png")
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
    print("\n Phase 67 complete.")
