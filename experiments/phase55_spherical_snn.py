"""
Phase 55: Spherical SNN — Norm-Preserving Noise Injection
==========================================================

Purpose:
  v9.1 discovered the Noise Amplitude Cliff (σ=0.03→σ=0.10 causes collapse).
  Hypothesis: collapse is caused by norm distortion from additive noise.
  
  Spherical noise preserves the original hidden-state norm while changing
  only the DIRECTION. If the cliff disappears, we prove norm distortion
  was the cause.

Conditions (5 × N=50):
  1. baseline          (temp=0.5, no SNN)              — control
  2. additive_0.10     (temp=0.5, additive σ=0.10)     — expect collapse
  3. spherical_0.02    (temp=0.5, spherical σ=0.02)    — match Phase 51
  4. spherical_0.10    (temp=0.5, spherical σ=0.10)    — KEY test
  5. spherical_0.15    (temp=0.5, spherical σ=0.15)    — limit test

Task: Modified-3 Hanoi (same as Phase 51)

Expected runtime: ~12.5 hours on single GPU (Mistral-7B 4-bit)

Usage:
    python experiments/phase55_spherical_snn.py
"""

import torch
import torch.nn as nn
import os, json, gc, time, random, re, copy
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ═══ Prevent Modern Standby (S0 Low Power Idle) — AGGRESSIVE ═══
# Must continuously call it AND simulate input to prevent Idle Timeout.


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ═══ Config ═══
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
TARGET_LAYER = 18
SEED = 2026
N_TRIALS = 50
MAX_STEPS = 50

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
#  HANOI ENVIRONMENT (Modified rules: larger on smaller)
# ═══════════════════════════════════════════════════

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
                return False, f"Modified: disk {disk} must be LARGER than {top}"
            if not self.modified and disk >= top:
                self.illegal_count += 1; self._prev_illegal = True
                return False, f"Standard: disk {disk} must be SMALLER than {top}"

        if self._prev_illegal:
            self.self_corrections += 1
        self._prev_illegal = False

        self.pegs[from_p].pop()
        self.pegs[to_p].append(disk)
        self.moves.append(f"{from_p}->{to_p} (disk {disk})")
        return True, f"Moved disk {disk}: {from_p}->{to_p}"

    def state_str(self):
        return f"A:{self.pegs['A']} B:{self.pegs['B']} C:{self.pegs['C']}"

    def optimal(self):
        return (3**self.n_disks - 1) if self.modified else (2**self.n_disks - 1)

    def stats(self):
        return {
            "solved": self.is_solved(),
            "legal_moves": len(self.moves),
            "illegal_moves": self.illegal_count,
            "self_corrections": self.self_corrections,
            "optimal": self.optimal(),
        }


# ═══════════════════════════════════════════════════
#  PROMPT BUILDER (Chat format)
# ═══════════════════════════════════════════════════

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
        msg += f"⚠️ ERROR: {error}. Pick from legal moves above.\n"

    msg += "Your move:"
    return msg


def build_chat_prompt(tokenizer, env, error=None):
    messages = [
        {"role": "user", "content": build_system_msg(env) + "\n\n" + build_user_msg(env, error)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ═══════════════════════════════════════════════════
#  MOVE PARSER
# ═══════════════════════════════════════════════════

def parse_move(response):
    resp = response.strip().upper()

    m = re.search(r'MOVE:\s*([ABC])\s*->\s*([ABC])', resp)
    if m: return m.group(1), m.group(2)

    m = re.search(r'([ABC])\s*->\s*([ABC])', resp)
    if m: return m.group(1), m.group(2)

    m = re.search(r'MOVE\s+DISK\s+\d+\s+FROM\s+([ABC])\s+TO\s+([ABC])', resp)
    if m: return m.group(1), m.group(2)

    m = re.search(r'FROM\s+([ABC])\s+TO\s+([ABC])', resp)
    if m: return m.group(1), m.group(2)

    m = re.search(r'STEP\s+\d+:\s*MOVE\s+DISK\s+\d+\s+FROM\s+([ABC])\s+TO\s+([ABC])', resp)
    if m: return m.group(1), m.group(2)

    letters = re.findall(r'[ABC]', resp)
    if len(letters) >= 2 and letters[0] != letters[1]:
        return letters[0], letters[1]

    return None


# ═══════════════════════════════════════════════════
#  SNN HOOKS
# ═══════════════════════════════════════════════════

class AdditiveEchoHook:
    """Original additive noise (Phase 51 EchoHook). hs + N(0, σ²)."""
    def __init__(self, sigma=0.02):
        self.sigma = sigma
        self.norm_deltas = []  # track norm change for analysis

    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn_like(hs) * self.sigma
        noisy = hs + noise

        # Record norm change ratio for diagnostics
        orig_norm = hs.norm(p=2, dim=-1).mean().item()
        new_norm = noisy.norm(p=2, dim=-1).mean().item()
        if orig_norm > 0:
            self.norm_deltas.append(new_norm / orig_norm)

        return (noisy,) + args[1:]


class SphericalEchoHook:
    """
    Norm-preserving (spherical) noise injection.
    
    Adds Gaussian noise then re-normalizes to the original L2 norm.
    Only the DIRECTION changes; the magnitude stays constant.
    
    Inspired by Spherical Steering (arxiv 2602.08169).
    """
    def __init__(self, sigma=0.02):
        self.sigma = sigma
        self.norm_deltas = []  # should be ~1.0 always
        self.angle_deltas = []  # cosine similarity: how much direction changed

    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn_like(hs) * self.sigma
        noisy = hs + noise

        # Preserve original norm
        original_norm = hs.norm(p=2, dim=-1, keepdim=True)
        noisy_norm = noisy.norm(p=2, dim=-1, keepdim=True)
        normalized = noisy * (original_norm / (noisy_norm + 1e-8))

        # Diagnostics
        final_norm = normalized.norm(p=2, dim=-1).mean().item()
        orig_norm = hs.norm(p=2, dim=-1).mean().item()
        if orig_norm > 0:
            self.norm_deltas.append(final_norm / orig_norm)

        # Cosine similarity (direction change)
        cos_sim = torch.nn.functional.cosine_similarity(
            hs.float().flatten(), normalized.float().flatten(), dim=0
        ).item()
        self.angle_deltas.append(cos_sim)

        return (normalized,) + args[1:]


# ═══════════════════════════════════════════════════
#  MODEL + GENERATION
# ═══════════════════════════════════════════════════

def load_model():
    print(f"\n📦 Loading {MODEL_NAME}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    print(f"  ✅ {len(model.model.layers)} layers")
    return model, tok


def generate(model, tok, prompt, temperature=0.5, max_tokens=100):
    """Generate with specified temperature."""
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperature, top_p=0.9, top_k=40,
            repetition_penalty=1.2, pad_token_id=tok.pad_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    inp = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(inp):].strip()


# ═══════════════════════════════════════════════════
#  GAME LOOP
# ═══════════════════════════════════════════════════

def play_game(model, tok, env, temperature=0.5, hook=None):
    env.reset()
    error = None
    consec_fail = 0
    thoughts = []

    for step in range(MAX_STEPS):
        prompt = build_chat_prompt(tok, env, error)

        resp = generate(model, tok, prompt, temperature=temperature, max_tokens=80)

        # Extract think
        th = re.search(r'Think:\s*(.+?)(?:\n|Move:|$)', resp, re.IGNORECASE)
        if th: thoughts.append(th.group(1).strip()[:80])

        move = parse_move(resp)
        if move is None:
            env.illegal_count += 1
            env.total_attempts += 1
            env._prev_illegal = True
            error = f"Couldn't parse: '{resp[:50]}'. Use format Move: X->Y"
            consec_fail += 1
            if consec_fail >= 10: break
            continue

        ok, msg = env.try_move(move[0], move[1])
        if ok:
            error = None; consec_fail = 0
            if env.is_solved(): break
        else:
            error = msg; consec_fail += 1
            if consec_fail >= 10: break

    st = env.stats()
    st["thoughts"] = thoughts[:3]
    return st


def run_condition(model, tok, cond_name, sigma, hook_type, n_trials=N_TRIALS,
                  existing_results=None):
    """
    Run one experimental condition with per-game checkpointing.
    
    existing_results: list of already-completed game results (for resume)
    hook_type: None, "additive", or "spherical"
    """
    start_from = len(existing_results) if existing_results else 0
    if start_from > 0:
        print(f"\n  🔄 Resuming {cond_name} from game {start_from+1}/{n_trials}")
    print(f"\n  🎮 Mod-3 | {cond_name} | σ={sigma} | {hook_type or 'none'} | {n_trials} trials")

    layers = model.model.layers
    hook, handle = None, None

    if hook_type == "additive":
        hook = AdditiveEchoHook(sigma)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)
    elif hook_type == "spherical":
        hook = SphericalEchoHook(sigma)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    results = list(existing_results) if existing_results else []
    solved = sum(1 for r in results if r.get("solved"))
    
    ckpt_path = os.path.join(RESULTS_DIR, "phase55_checkpoint.json")
    
    for t in range(start_from, n_trials):
        try:
            env = HanoiEnv(3, modified=True)
            r = play_game(model, tok, env, temperature=0.5, hook=hook)
            results.append(r)
            if r["solved"]: solved += 1
            icon = "✅" if r["solved"] else "❌"
            rate = solved / (t+1) * 100
            print(f"    {t+1:2d}/{n_trials}: {icon} legal={r['legal_moves']:2d} "
                  f"illegal={r['illegal_moves']:2d} self_corr={r['self_corrections']:2d}  "
                  f"[running: {solved}/{t+1} = {rate:.0f}%]")
        except Exception as e:
            print(f"    {t+1:2d}/{n_trials}: ⚠️ GAME ERROR: {e}")
            results.append({"solved": False, "legal_moves": 0, "illegal_moves": 0,
                           "self_corrections": 0, "optimal": 26, "thoughts": []})

        # Save mini-checkpoint after EVERY game
        try:
            with open(ckpt_path, "r") as f:
                ckpt = json.load(f)
        except:
            ckpt = {"experiment": "Phase 55: Spherical SNN (checkpoint)",
                    "completed_conditions": [], "summaries": [],
                    "in_progress": {}}
        ckpt["in_progress"] = {
            "condition": cond_name, "sigma": sigma, "hook_type": hook_type or "none",
            "completed_games": t + 1, "total_games": n_trials,
            "results": results
        }
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2, default=str)

        # Prevent CUDA memory leak
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (t + 1) % 10 == 0:
            gc.collect()

    if handle: handle.remove()

    s = lambda key: round(np.mean([r[key] for r in results]), 2)
    sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
    n_solved = sum(1 for r in results if r["solved"])

    # Hook diagnostics
    hook_diagnostics = {}
    if hook and hasattr(hook, "norm_deltas") and hook.norm_deltas:
        hook_diagnostics["avg_norm_ratio"] = round(np.mean(hook.norm_deltas), 6)
        hook_diagnostics["std_norm_ratio"] = round(np.std(hook.norm_deltas), 6)
    if hook and hasattr(hook, "angle_deltas") and hook.angle_deltas:
        hook_diagnostics["avg_cos_similarity"] = round(np.mean(hook.angle_deltas), 6)
        hook_diagnostics["std_cos_similarity"] = round(np.std(hook.angle_deltas), 6)

    summary = {
        "condition": cond_name, "sigma": sigma, "hook_type": hook_type or "none",
        "n_trials": n_trials, "n_solved": n_solved,
        "solve_rate": sr,
        "avg_legal": s("legal_moves"), "avg_illegal": s("illegal_moves"),
        "avg_self_corr": s("self_corrections"),
        "optimal": results[0]["optimal"],
        "hook_diagnostics": hook_diagnostics,
        "thoughts": results[0].get("thoughts", []),
    }
    print(f"    📊 Solve={sr:.1f}% ({n_solved}/{n_trials}) "
          f"Legal={summary['avg_legal']:.1f} Illegal={summary['avg_illegal']:.1f}")
    if hook_diagnostics:
        nr = hook_diagnostics.get('avg_norm_ratio', 'N/A')
        cs = hook_diagnostics.get('avg_cos_similarity', 'N/A')
        nr_str = f"{nr:.4f}" if isinstance(nr, (int, float)) else str(nr)
        cs_str = f"{cs:.4f}" if isinstance(cs, (int, float)) else str(cs)
        print(f"    🔬 Norm ratio={nr_str} Cos-sim={cs_str}")
    return summary, results


# ═══════════════════════════════════════════════════
#  FISHER'S EXACT TEST
# ═══════════════════════════════════════════════════

def fisher_exact_test(a, b, c, d):
    """
    2x2 contingency table:
                Solved  Not-solved
    GroupA:      a        b
    GroupB:      c        d

    Returns one-sided p-value (P(X >= a) under H0).
    """
    try:
        from scipy.stats import fisher_exact
        table = [[a, b], [c, d]]
        _, p_two = fisher_exact(table)
        _, p_one = fisher_exact(table, alternative='greater')
        return {"p_two_sided": round(p_two, 6), "p_one_sided": round(p_one, 6)}
    except ImportError:
        from math import comb
        n = a + b + c + d
        r1 = a + b; c1 = a + c; c2 = b + d
        def hypergeom_pmf(x):
            return comb(c1, x) * comb(c2, r1 - x) / comb(n, r1)
        p_val = sum(hypergeom_pmf(x) for x in range(a, min(r1, c1) + 1))
        return {"p_one_sided": round(p_val, 6), "p_two_sided": None,
                "note": "scipy not available, manual one-sided only"}


def compute_all_fisher_tests(summaries):
    """Compare all interesting pairs."""
    tests = {}
    
    def get(name):
        return next((s for s in summaries if s["condition"] == name), None)
    
    pairs = [
        # Key comparisons
        ("spherical_0.10", "additive_0.10",  "Core: Does norm preservation save σ=0.10?"),
        ("spherical_0.10", "baseline",       "Is spherical σ=0.10 different from baseline?"),
        ("spherical_0.02", "baseline",       "Reproduce Phase 51: spherical σ=0.02 vs baseline"),
        ("spherical_0.15", "baseline",       "Limit test: spherical σ=0.15 vs baseline"),
        ("additive_0.10",  "baseline",       "Confirm collapse: additive σ=0.10 vs baseline"),
        ("spherical_0.10", "spherical_0.02", "Dose-response: σ=0.10 vs σ=0.02"),
    ]
    
    for name_a, name_b, description in pairs:
        sa, sb = get(name_a), get(name_b)
        if sa and sb:
            a = sa["n_solved"]; b = sa["n_trials"] - a
            c = sb["n_solved"]; d = sb["n_trials"] - c
            tests[f"{name_a}_vs_{name_b}"] = {
                "description": description,
                f"{name_a}_solved": a, f"{name_a}_total": sa["n_trials"],
                f"{name_b}_solved": c, f"{name_b}_total": sb["n_trials"],
                **fisher_exact_test(a, b, c, d)
            }
    
    return tests


# ═══════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════

def visualize(summaries, fisher_tests):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle("Phase 55: Spherical SNN — Norm-Preserving Noise Injection\n"
                 "Does norm preservation eliminate the Noise Amplitude Cliff? (N=50 per condition)",
                 fontsize=14, fontweight="bold", y=1.02)

    conds = ["baseline", "additive_0.10", "spherical_0.02", "spherical_0.10", "spherical_0.15"]
    labels = ["Baseline\n(no SNN)", "Additive\nσ=0.10", "Spherical\nσ=0.02",
              "Spherical\nσ=0.10", "Spherical\nσ=0.15"]
    colors = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    # Panel 1: Solve Rate
    ax = axes[0]
    solve_rates = [next((s["solve_rate"] for s in summaries if s["condition"]==c), 0) for c in conds]
    n_solved = [next((s["n_solved"] for s in summaries if s["condition"]==c), 0) for c in conds]
    bars = ax.bar(range(5), solve_rates, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    for i, (b, sr, ns) in enumerate(zip(bars, solve_rates, n_solved)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
               f"{sr:.1f}%\n({ns}/50)", ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("① Solve Rate", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Add key Fisher p-value
    key_test = fisher_tests.get("spherical_0.10_vs_additive_0.10", {})
    if key_test:
        p = key_test.get("p_two_sided", "N/A")
        sig = "★★★" if isinstance(p, float) and p < 0.001 else \
              ("★★" if isinstance(p, float) and p < 0.01 else \
              ("★" if isinstance(p, float) and p < 0.05 else "n.s."))
        ax.text(2, max(solve_rates)*0.6,
                f"Spherical vs Additive (σ=0.10):\np₂={p:.4f} {sig}" if isinstance(p, float) else f"p={p}",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel 2: Norm Ratio (the key diagnostic)
    ax = axes[1]
    norm_ratios = []
    norm_labels_used = []
    norm_colors_used = []
    for i, c in enumerate(conds):
        s = next((s for s in summaries if s["condition"]==c), None)
        if s and s.get("hook_diagnostics", {}).get("avg_norm_ratio"):
            norm_ratios.append(s["hook_diagnostics"]["avg_norm_ratio"])
            norm_labels_used.append(labels[i])
            norm_colors_used.append(colors[i])
    
    if norm_ratios:
        bars = ax.bar(range(len(norm_ratios)), norm_ratios,
                      color=norm_colors_used, alpha=0.85, edgecolor='white', linewidth=2)
        for b, nr in zip(bars, norm_ratios):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
                   f"{nr:.4f}", ha='center', fontsize=10, fontweight='bold')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label="Perfect preservation (1.0)")
        ax.set_xticks(range(len(norm_ratios))); ax.set_xticklabels(norm_labels_used, fontsize=9)
        ax.set_ylabel("Norm Ratio (output/input)", fontsize=12)
        ax.legend(fontsize=9)
    ax.set_title("② Norm Preservation\n(Spherical should be ≈1.0)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 3: Legal vs Illegal moves
    ax = axes[2]
    legal = [next((s["avg_legal"] for s in summaries if s["condition"]==c), 0) for c in conds]
    illegal = [next((s["avg_illegal"] for s in summaries if s["condition"]==c), 0) for c in conds]
    x = np.arange(5)
    w = 0.35
    ax.bar(x - w/2, legal, w, label="Legal", color="#2ecc71", alpha=0.85)
    ax.bar(x + w/2, illegal, w, label="Illegal", color="#e74c3c", alpha=0.85)
    for i, (l, il) in enumerate(zip(legal, illegal)):
        ax.text(i-w/2, l+0.3, f"{l:.1f}", ha='center', fontsize=8)
        ax.text(i+w/2, il+0.3, f"{il:.1f}", ha='center', fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Average Moves", fontsize=12)
    ax.set_title("③ Legal vs Illegal Moves", fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase55_spherical_snn.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure: {path}")
    return path


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    # ═══ Checkpoint Resume ═══
    ckpt_path = os.path.join(RESULTS_DIR, "phase55_checkpoint.json")
    all_sum = []
    all_det = {}
    completed_conditions = set()
    in_progress_data = {}  # partial results for interrupted condition

    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r") as f:
                ckpt = json.load(f)
            all_sum = ckpt.get("summaries", [])
            completed_conditions = set(ckpt.get("completed_conditions", []))
            in_progress_data = ckpt.get("in_progress", {})
            print(f"\n🔄 RESUMING from checkpoint!")
            print(f"   Completed: {', '.join(completed_conditions) or 'none'}")
            if in_progress_data:
                ip = in_progress_data
                print(f"   In-progress: {ip['condition']} ({ip['completed_games']}/{ip['total_games']} games)")
            print(f"   Remaining: {5 - len(completed_conditions)} conditions")
        except Exception as e:
            print(f"⚠️ Checkpoint load failed ({e}), starting fresh")
            all_sum = []
            completed_conditions = set()

    model, tok = load_model()

    # 5 conditions
    conditions = [
        # (name, sigma, hook_type)
        ("baseline",       0.0,   None),
        ("additive_0.10",  0.10, "additive"),
        ("spherical_0.02", 0.02, "spherical"),
        ("spherical_0.10", 0.10, "spherical"),
        ("spherical_0.15", 0.15, "spherical"),
    ]

    for cond_name, sigma, hook_type in conditions:
        if cond_name in completed_conditions:
            print(f"\n⏭️ Skipping {cond_name} (already completed)")
            continue

        # Check for in-progress partial results
        existing_results = None
        if in_progress_data and in_progress_data.get("condition") == cond_name:
            existing_results = in_progress_data.get("results", [])
            print(f"\n📂 Found {len(existing_results)} saved games for {cond_name}")

        print(f"\n{'═'*60}")
        print(f"  🏰 Modified-3 Hanoi | {cond_name}")
        print(f"{'═'*60}")
        s, d = run_condition(model, tok, cond_name, sigma, hook_type,
                            existing_results=existing_results)
        all_sum.append(s)
        all_det[cond_name] = d

        # Checkpoint after each condition
        checkpoint = {
            "experiment": "Phase 55: Spherical SNN (checkpoint)",
            "completed_conditions": [s["condition"] for s in all_sum],
            "summaries": all_sum,
        }
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        print(f"  💾 Checkpoint saved: {ckpt_path}")

    # Fisher's exact tests
    fisher_tests = compute_all_fisher_tests(all_sum)

    elapsed = time.time() - t0
    fig = visualize(all_sum, fisher_tests)

    # Save results
    out = {
        "experiment": "Phase 55: Spherical SNN — Norm-Preserving Noise Injection",
        "model": MODEL_SHORT,
        "task": "Modified-3 Hanoi (reversed rules)",
        "purpose": "Test if norm-preserving noise eliminates the Noise Amplitude Cliff",
        "hypothesis": "Additive σ=0.10 collapses (norm distortion) but spherical σ=0.10 survives (norm preserved)",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_sum,
        "fisher_tests": fisher_tests,
        "figure": fig,
    }
    log = os.path.join(RESULTS_DIR, "phase55_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # ═══ VERDICT ═══
    print(f"\n{'═'*75}")
    print(f"  ⚡ PHASE 55: SPHERICAL SNN — VERDICT")
    print(f"{'═'*75}")
    print(f"  Task: Modified-3 Hanoi | N={N_TRIALS} per condition | 5 conditions")
    print(f"{'─'*75}")
    print(f"  {'Condition':<18} {'σ':>5} {'Type':<10} {'Solved':>8} {'Solve%':>8} "
          f"{'Legal':>8} {'Illegal':>8} {'NormΔ':>8}")
    print(f"{'─'*75}")
    for s in all_sum:
        nr = s.get("hook_diagnostics", {}).get("avg_norm_ratio", "—")
        nr_str = f"{nr:.4f}" if isinstance(nr, float) else nr
        print(f"  {s['condition']:<18} {s['sigma']:>5.2f} {s['hook_type']:<10} "
              f"{s['n_solved']:>4}/{s['n_trials']:<3} {s['solve_rate']:>7.1f}% "
              f"{s['avg_legal']:>8.1f} {s['avg_illegal']:>8.1f} {nr_str:>8}")
    print(f"{'─'*75}")

    print(f"\n  🧪 Fisher's Exact Tests:")
    for name, test in fisher_tests.items():
        p = test.get("p_two_sided", test.get("p_one_sided", "N/A"))
        sig = ""
        if isinstance(p, float):
            if p < 0.001: sig = " ★★★ (p<0.001)"
            elif p < 0.01: sig = " ★★ (p<0.01)"
            elif p < 0.05: sig = " ★ (p<0.05)"
            else: sig = " n.s."
        desc = test.get("description", "")
        print(f"    {name}:")
        print(f"      {desc}")
        print(f"      p₂={p}{sig}")

    # Key verdict
    sph10 = next((s for s in all_sum if s["condition"] == "spherical_0.10"), None)
    add10 = next((s for s in all_sum if s["condition"] == "additive_0.10"), None)
    base = next((s for s in all_sum if s["condition"] == "baseline"), None)
    
    print(f"\n  {'═'*60}")
    print(f"  🎯 KEY QUESTION: Does norm preservation save σ=0.10?")
    if sph10 and add10:
        print(f"    Additive σ=0.10:  {add10['solve_rate']:.1f}% solve, "
              f"{add10['avg_illegal']:.1f} illegal moves")
        print(f"    Spherical σ=0.10: {sph10['solve_rate']:.1f}% solve, "
              f"{sph10['avg_illegal']:.1f} illegal moves")
        if sph10['solve_rate'] > add10['solve_rate'] + 5:
            print(f"    ✅ YES — Norm preservation significantly helps!")
        elif abs(sph10['solve_rate'] - add10['solve_rate']) <= 5:
            print(f"    ❌ NO — Both similar. Collapse is NOT about norm distortion.")
        else:
            print(f"    🤔 UNCLEAR — Spherical is worse? Unexpected.")
    print(f"  {'═'*60}")

    print(f"\n  ⏱ {elapsed/60:.1f} min ({elapsed/3600:.1f} hours) | 💾 {log}")
    print(f"{'═'*75}")


if __name__ == "__main__":
    main()
    # Auto-hibernate after completion
    time.sleep(10)
