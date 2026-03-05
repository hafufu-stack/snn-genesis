"""
Phase 69: The Master Equation — Adaptive Cosine Annealing
==========================================================

Combines the two best findings from the entire project:
  - Phase 63: First10 linear decay = 46% (timing)
  - Phase 65: Adaptive cos 0.65 = 38% (amplitude)

Key innovation: Instead of decaying σ directly, we decay the TARGET COSINE
from 0.65 (creative chaos) to 1.00 (perfect coherence). The adaptive σ
formula automatically adjusts noise intensity based on hidden state norm,
so the "annealing" is in cosine space — physically meaningful and norm-invariant.

Also includes N=100 validations for the two champions.

Conditions (7, mixed N):
  1. baseline                     (N=50)  — Control
  2. first10_linear_ref          (N=50)  — Phase 63 champion reference
  3. cos_anneal_first10          (N=50)  — Anneal cos 0.65→1.0 over 10 moves  ★
  4. cos_anneal_first20          (N=50)  — Anneal cos 0.65→1.0 over 20 moves
  5. adaptive_cos_0.65_first10   (N=50)  — Fixed cos=0.65 for first 10 only
  6. first10_linear_N100         (N=100) — N=100 validation of champion
  7. adaptive_cos_0.65_N100      (N=100) — N=100 validation

Total: 450 games. ~9 hours on RTX 5080.

Usage:
    python experiments/phase69_master_equation.py
"""

import torch
import torch.nn as nn
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
HIDDEN_DIM = 4096  # Mistral-7B hidden dimension

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
#  ADAPTIVE NOISE HOOK (supports cosine annealing)
# ===================================================

class AdaptiveNoiseHook:
    """Noise injection with adaptive σ based on target cosine similarity.
    
    The key formula from Phase 65:
        σ = ||h|| × sqrt((1/cos²_target - 1) / d)
    
    Phase 69 innovation: target_cos itself is annealed over time:
        target_cos(move) = cos_start + (1.0 - cos_start) × (move / duration)
    
    When target_cos → 1.0, σ → 0 (perfect coherence, no noise).
    When target_cos = 0.65, σ ≈ 0.15 for typical ||h|| values.
    """
    def __init__(self):
        self.active = False
        self.sigma = 0.15  # For fixed-sigma mode
        self.mode = "fixed"  # "fixed", "adaptive", "adaptive_anneal"
        self.target_cos = 0.65
        self.handle = None

    def register(self, model, layer_idx=18):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active:
                return args
            hs = args[0]
            
            if hook_obj.mode == "fixed":
                sigma = hook_obj.sigma
            elif hook_obj.mode in ("adaptive", "adaptive_anneal"):
                # Compute adaptive σ from target cosine
                h_norm = torch.norm(hs, dim=-1, keepdim=True).clamp(min=1e-8)
                d = hs.shape[-1]
                cos_t = max(hook_obj.target_cos, 0.51)  # Safety: never below cliff
                sigma_ratio = math.sqrt((1.0 / cos_t**2 - 1) / d)
                # sigma per element = ||h|| × sigma_ratio / sqrt(d) ... but we apply per-element
                # Actually: noise = randn * σ_elem, where σ_elem = ||h||/sqrt(d) * sqrt(1/cos²-1)
                # Simplified: just scale by norm
                sigma = (h_norm.mean().item() * sigma_ratio)
            else:
                sigma = hook_obj.sigma
            
            if sigma <= 0:
                return args
            
            noise = torch.randn_like(hs) * sigma
            return (hs + noise,) + args[1:]
        
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


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

def generate(model, tok, prompt, temperature=0.5, max_tokens=80):
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

def play_game(model, tok, env, hook, noise_fn=None):
    env.reset()
    error = None; consec_fail = 0
    legal_move_count = 0

    for step in range(MAX_STEPS):
        if noise_fn:
            noise_fn(step, legal_move_count)

        prompt = build_chat_prompt(tok, env, error)
        resp = generate(model, tok, prompt)
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

    hook.active = False
    st = env.stats()
    st["total_legal_moves"] = legal_move_count
    return st


# ===================================================
#  EXPERIMENT DEFINITIONS
# ===================================================

EXPERIMENTS = [
    {
        "name": "baseline",
        "n_trials": 50,
        "noise_setup": lambda hook: None,
        "noise_fn": lambda hook: (lambda step, lm: setattr(hook, 'active', False)),
        "desc": "No noise (control)"
    },
    {
        "name": "first10_linear_ref",
        "n_trials": 50,
        "noise_setup": lambda hook: [setattr(hook, 'mode', 'fixed')],
        "noise_fn": lambda hook: _make_first10_linear(hook, 0.15),
        "desc": "Phase 63 champion: σ=0.15 linear decay over 10 moves"
    },
    {
        "name": "cos_anneal_first10",
        "n_trials": 50,
        "noise_setup": lambda hook: [setattr(hook, 'mode', 'adaptive_anneal')],
        "noise_fn": lambda hook: _make_cos_anneal(hook, 0.65, 10),
        "desc": "★ TARGET COS anneals 0.65→1.0 over 10 moves (adaptive σ)"
    },
    {
        "name": "cos_anneal_first20",
        "n_trials": 50,
        "noise_setup": lambda hook: [setattr(hook, 'mode', 'adaptive_anneal')],
        "noise_fn": lambda hook: _make_cos_anneal(hook, 0.65, 20),
        "desc": "TARGET COS anneals 0.65→1.0 over 20 moves (adaptive σ)"
    },
    {
        "name": "adaptive_cos_0.65_first10",
        "n_trials": 50,
        "noise_setup": lambda hook: [setattr(hook, 'mode', 'adaptive')],
        "noise_fn": lambda hook: _make_adaptive_first_n(hook, 0.65, 10),
        "desc": "Fixed cos=0.65 for first 10 moves only, then OFF"
    },
    {
        "name": "first10_linear_N100",
        "n_trials": 100,
        "noise_setup": lambda hook: [setattr(hook, 'mode', 'fixed')],
        "noise_fn": lambda hook: _make_first10_linear(hook, 0.15),
        "desc": "N=100 validation: first10 linear (Phase 63 champion)"
    },
    {
        "name": "adaptive_cos_0.65_N100",
        "n_trials": 100,
        "noise_setup": lambda hook: [setattr(hook, 'mode', 'adaptive')],
        "noise_fn": lambda hook: _make_adaptive_always(hook, 0.65),
        "desc": "N=100 validation: adaptive cos=0.65 (Phase 65 champion)"
    },
]


def _make_first10_linear(hook, base_sigma):
    """First 10 legal moves: σ decays linearly from base_sigma to 0."""
    def fn(step, legal_moves):
        if legal_moves < 10:
            hook.active = True
            hook.sigma = base_sigma * (1.0 - legal_moves / 10.0)
        else:
            hook.active = False
    return fn

def _make_cos_anneal(hook, cos_start, duration):
    """Anneal target cosine from cos_start to 1.0 over 'duration' legal moves.
    σ is automatically computed from target_cos and ||h||.
    """
    def fn(step, legal_moves):
        if legal_moves < duration:
            progress = legal_moves / duration
            hook.target_cos = cos_start + (1.0 - cos_start) * progress
            hook.active = True
        else:
            hook.active = False
    return fn

def _make_adaptive_first_n(hook, target_cos, n):
    """Fixed target cosine for first n legal moves, then OFF."""
    def fn(step, legal_moves):
        if legal_moves < n:
            hook.target_cos = target_cos
            hook.active = True
        else:
            hook.active = False
    return fn

def _make_adaptive_always(hook, target_cos):
    """Always-on adaptive with fixed target cosine."""
    def fn(step, legal_moves):
        hook.target_cos = target_cos
        hook.active = True
    return fn


# ===================================================
#  MAIN
# ===================================================

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tok = load_model()

    print("\n" + "="*80)
    print("  PHASE 69: THE MASTER EQUATION")
    print("  Adaptive Cosine Annealing: target_cos 0.65→1.0 (timing × amplitude)")
    print("  + N=100 validations of Phase 63 & Phase 65 champions")
    print("="*80)

    all_summaries = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        n_trials = exp["n_trials"]
        print(f"\n  === {name}: {exp['desc']} | N={n_trials} ===")

        hook = AdaptiveNoiseHook()
        hook.register(model, layer_idx=18)
        exp["noise_setup"](hook)
        noise_fn = exp["noise_fn"](hook)

        results = []; solved = 0
        for t in range(n_trials):
            env = HanoiEnv(3, modified=True)
            r = play_game(model, tok, env, hook, noise_fn)
            results.append(r)
            if r["solved"]: solved += 1
            icon = "O" if r["solved"] else "X"
            rate = solved / (t+1) * 100
            print(f"    {t+1:3d}/{n_trials}: {icon} legal={r['legal_moves']:2d} "
                  f"illegal={r['illegal_moves']:2d} [{solved}/{t+1} = {rate:.0f}%]")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if (t+1) % 10 == 0: gc.collect()

        hook.remove()

        sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
        n_solved = sum(1 for r in results if r["solved"])
        all_summaries.append({
            "condition": name,
            "description": exp["desc"],
            "n_trials": n_trials, "n_solved": n_solved,
            "solve_rate": sr,
            "avg_legal": round(np.mean([r["legal_moves"] for r in results]), 2),
            "avg_illegal": round(np.mean([r["illegal_moves"] for r in results]), 2),
        })
        print(f"    => Solve={sr:.1f}% ({n_solved}/{n_trials})")

    elapsed = time.time() - t0

    # Visualization
    fig_path = visualize(all_summaries)

    # Verdict
    bl = next(s for s in all_summaries if s["condition"] == "baseline")
    print(f"\n{'='*80}")
    print(f"  PHASE 69 VERDICT: The Master Equation")
    print(f"{'='*80}")
    for s in all_summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<30} {s['solve_rate']:>6.1f}%  N={s['n_trials']}  (baseline)")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {s['condition']:<30} {s['solve_rate']:>6.1f}%  N={s['n_trials']}  p={p:.4f} {sig}")
    print(f"\n  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")

    # Save
    out = {
        "experiment": "Phase 69: The Master Equation",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "purpose": "Combine adaptive cosine homeostasis with flash annealing via cosine-space annealing",
        "innovation": "Anneal target_cos from 0.65→1.0 instead of σ from 0.15→0. Norm-invariant cooling.",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_summaries, "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase69_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")


def visualize(summaries):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 69: The Master Equation\n"
                 "Adaptive Cosine Annealing + N=100 Validation",
                 fontsize=14, fontweight="bold", y=1.02)

    # Panel 1: All conditions bar chart
    ax = axes[0]
    names = [s["condition"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]
    ns = [s["n_trials"] for s in summaries]
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#e67e22', '#2ecc71', '#9b59b6', '#1abc9c']
    
    bars = ax.bar(range(len(summaries)), rates, color=colors[:len(summaries)],
                  alpha=0.85, edgecolor='white', linewidth=2)
    for i, (b, s) in enumerate(zip(bars, summaries)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               f'{s["solve_rate"]:.0f}%\nN={s["n_trials"]}',
               ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(summaries)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=6)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("All Conditions", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Historical comparison
    ax = axes[1]
    historical = [
        ("P57 always\nσ=0.15", 32, 100),
        ("P60 linear\ndecay", 38, 50),
        ("P62 first20\nflat", 42, 50),
        ("P63 first10\nlinear", 46, 50),
        ("P65 adaptive\ncos=0.65", 38, 50),
    ]
    # Add Phase 69 key results  
    for s in summaries:
        if s["condition"] not in ("baseline",):
            historical.append((f"P69 {s['condition'][:12]}", s["solve_rate"], s["n_trials"]))

    h_names = [h[0] for h in historical]
    h_rates = [h[1] for h in historical]
    h_ns = [h[2] for h in historical]
    h_colors = ['#bdc3c7']*5 + ['#e74c3c']*(len(historical)-5)

    bars2 = ax.barh(range(len(historical)), h_rates, color=h_colors, alpha=0.85, edgecolor='white')
    for i, (b, h) in enumerate(zip(bars2, historical)):
        ax.text(b.get_width()+0.5, b.get_y()+b.get_height()/2,
               f'{h[1]:.0f}% (N={h[2]})', va='center', fontsize=8, fontweight='bold')

    ax.set_yticks(range(len(historical)))
    ax.set_yticklabels(h_names, fontsize=7)
    ax.set_xlabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Historical Comparison", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase69_master_equation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n Phase 69 complete.")
