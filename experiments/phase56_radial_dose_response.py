"""
Phase 56: Radial Noise + Additive Dose-Response
================================================

Purpose:
  Phase 55 discovered: Additive σ=0.10 boosts Hanoi solve rate 16%→38% (p=0.023).
  Spherical (norm-preserving) noise had NO effect.
  
  This experiment decomposes the noise into orthogonal components:
    - Additive = Direction change + Magnitude change
    - Spherical = Direction change ONLY (Phase 55: no effect)
    - Radial = Magnitude change ONLY (Phase 56: the key test!)
    
  Also maps the full dose-response curve for Additive noise.

Conditions (9 × N=50):
  1. baseline          (no SNN)                 — control
  2. additive_0.05     (additive σ=0.05)        — dose-response: low
  3. additive_0.08     (additive σ=0.08)        — dose-response: mid-low
  4. additive_0.10     (additive σ=0.10)        — Phase 55 peak (38%)
  5. additive_0.12     (additive σ=0.12)        — dose-response: mid-high
  6. additive_0.15     (additive σ=0.15)        — dose-response: high
  7. radial_0.05       (radial σ=0.05)          — magnitude only: low
  8. radial_0.10       (radial σ=0.10)          — KEY: matches additive?
  9. radial_0.15       (radial σ=0.15)          — magnitude only: high

Task: Modified-3 Hanoi (same as Phase 51/55)
Expected runtime: ~16 hours on single GPU (Mistral-7B 4-bit)

Usage:
    python experiments/phase56_radial_dose_response.py
"""

import torch
import torch.nn as nn
import os, json, gc, time, random, re, copy
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ═══ Prevent Modern Standby (S0 Low Power Idle) — AGGRESSIVE ═══


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
        msg += f"ERROR: {error}. Pick from legal moves above.\n"

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
    patterns = [
        r'Move:\s*([A-Ca-c])\s*->\s*([A-Ca-c])',
        r'Move:\s*([A-Ca-c])\s*→\s*([A-Ca-c])',
        r'move\s+(?:disk\s+\d+\s+)?(?:from\s+)?([A-Ca-c])\s+to\s+([A-Ca-c])',
        r'([A-Ca-c])\s*->\s*([A-Ca-c])',
    ]
    for p in patterns:
        m = re.search(p, response, re.IGNORECASE)
        if m:
            return m.group(1).upper(), m.group(2).upper()
    return None


# ═══════════════════════════════════════════════════
#  SNN HOOKS
# ═══════════════════════════════════════════════════

class AdditiveEchoHook:
    """Original additive noise (Phase 51 EchoHook). hs + N(0, σ²)."""
    def __init__(self, sigma=0.02):
        self.sigma = sigma
        self.norm_deltas = []

    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn_like(hs) * self.sigma
        noisy = hs + noise

        orig_norm = hs.norm(p=2, dim=-1).mean().item()
        new_norm = noisy.norm(p=2, dim=-1).mean().item()
        if orig_norm > 0:
            self.norm_deltas.append(new_norm / orig_norm)

        return (noisy,) + args[1:]


class RadialEchoHook:
    """Magnitude-only noise: scales vector length without changing direction.
    
    h' = h * (1 + N(0, σ²))
    
    Direction (meaning) is UNCHANGED; only the norm (attention confidence) fluctuates.
    This isolates the "magnitude jitter" component of additive noise.
    """
    def __init__(self, sigma=0.02):
        self.sigma = sigma
        self.norm_deltas = []   # track how much norm changed
        self.angle_deltas = []  # should be ~1.0 always (direction unchanged)

    def __call__(self, module, args):
        hs = args[0]
        # Per-token scalar scale factor: (1 + N(0, σ²))
        scale = 1.0 + torch.randn(hs.shape[0], hs.shape[1], 1, device=hs.device, dtype=hs.dtype) * self.sigma
        scaled = hs * scale

        # Diagnostics
        orig_norm = hs.norm(p=2, dim=-1).mean().item()
        new_norm = scaled.norm(p=2, dim=-1).mean().item()
        if orig_norm > 0:
            self.norm_deltas.append(new_norm / orig_norm)

        # Cosine similarity should be ~1.0 (direction unchanged)
        cos_sim = torch.nn.functional.cosine_similarity(
            hs.float().flatten(), scaled.float().flatten(), dim=0
        ).item()
        self.angle_deltas.append(cos_sim)

        return (scaled,) + args[1:]


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
    elif hook_type == "radial":
        hook = RadialEchoHook(sigma)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    results = list(existing_results) if existing_results else []
    solved = sum(1 for r in results if r.get("solved"))

    ckpt_path = os.path.join(RESULTS_DIR, "phase56_checkpoint.json")

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
            ckpt = {"experiment": "Phase 56 (checkpoint)",
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
        return {"p_one_sided": round(p_val, 6), "p_two_sided": None}


def compute_all_fisher_tests(summaries):
    """Compare all interesting pairs for Phase 56."""
    tests = {}

    def get(name):
        return next((s for s in summaries if s["condition"] == name), None)

    # All conditions vs baseline
    baseline = get("baseline")
    if not baseline:
        return tests

    for s in summaries:
        if s["condition"] == "baseline":
            continue
        name = f"{s['condition']}_vs_baseline"
        a, b = s["n_solved"], s["n_trials"] - s["n_solved"]
        c, d = baseline["n_solved"], baseline["n_trials"] - baseline["n_solved"]
        result = fisher_exact_test(a, b, c, d)
        result["description"] = f"{s['condition']} vs baseline"
        result[f"{s['condition']}_solved"] = s["n_solved"]
        result[f"{s['condition']}_total"] = s["n_trials"]
        result["baseline_solved"] = baseline["n_solved"]
        result["baseline_total"] = baseline["n_trials"]
        tests[name] = result

    # Key comparison: radial_0.10 vs additive_0.10
    r10 = get("radial_0.10")
    a10 = get("additive_0.10")
    if r10 and a10:
        a, b = r10["n_solved"], r10["n_trials"] - r10["n_solved"]
        c, d = a10["n_solved"], a10["n_trials"] - a10["n_solved"]
        result = fisher_exact_test(a, b, c, d)
        result["description"] = "Core: Is magnitude-only noise as good as additive?"
        tests["radial_0.10_vs_additive_0.10"] = result

    return tests


# ═══════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════

def visualize(summaries, fisher_tests):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle("Phase 56: Radial Noise + Additive Dose-Response\n"
                 "Does magnitude jitter alone replicate additive noise benefits? (N=50 per condition)",
                 fontsize=14, fontweight="bold", y=1.02)

    # Sort conditions for display
    conds = ["baseline",
             "additive_0.05", "additive_0.08", "additive_0.10", "additive_0.12", "additive_0.15",
             "radial_0.05", "radial_0.10", "radial_0.15"]
    labels = ["Base\n(none)", "Add\nσ=.05", "Add\nσ=.08", "Add\nσ=.10", "Add\nσ=.12", "Add\nσ=.15",
              "Rad\nσ=.05", "Rad\nσ=.10", "Rad\nσ=.15"]
    colors = ["#95a5a6",
              "#e74c3c", "#c0392b", "#a93226", "#922b21", "#7b241c",
              "#3498db", "#2980b9", "#2471a3"]

    # Panel 1: Solve Rate (all conditions)
    ax = axes[0]
    solve_rates = [next((s["solve_rate"] for s in summaries if s["condition"]==c), 0) for c in conds]
    n_solved = [next((s["n_solved"] for s in summaries if s["condition"]==c), 0) for c in conds]
    bars = ax.bar(range(len(conds)), solve_rates, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    for i, (b, sr, ns) in enumerate(zip(bars, solve_rates, n_solved)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
               f"{sr:.0f}%\n({ns}/50)", ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("① Solve Rate: Additive vs Radial", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Dose-Response Curve (line plot)
    ax = axes[1]
    add_sigmas = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15]
    add_names = ["baseline", "additive_0.05", "additive_0.08", "additive_0.10", "additive_0.12", "additive_0.15"]
    add_rates = [next((s["solve_rate"] for s in summaries if s["condition"]==n), 0) for n in add_names]

    rad_sigmas = [0.05, 0.10, 0.15]
    rad_names = ["radial_0.05", "radial_0.10", "radial_0.15"]
    rad_rates = [next((s["solve_rate"] for s in summaries if s["condition"]==n), 0) for n in rad_names]

    ax.plot(add_sigmas, add_rates, 'o-', color='#e74c3c', linewidth=2.5, markersize=8, label='Additive (dir+mag)')
    ax.plot(rad_sigmas, rad_rates, 's--', color='#3498db', linewidth=2.5, markersize=8, label='Radial (mag only)')
    ax.axhline(y=add_rates[0], color='gray', linestyle=':', alpha=0.5, label=f'Baseline ({add_rates[0]:.0f}%)')
    ax.set_xlabel("σ (noise amplitude)", fontsize=12)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("② Dose-Response Curve\n(Stochastic Resonance?)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 3: Norm Ratio comparison
    ax = axes[2]
    norm_data = []
    for s in summaries:
        nr = s.get("hook_diagnostics", {}).get("avg_norm_ratio")
        if nr is not None:
            norm_data.append((s["condition"], nr))
    if norm_data:
        names, ratios = zip(*norm_data)
        short_names = [n.replace("additive_", "A").replace("radial_", "R") for n in names]
        c = ['#e74c3c' if 'A' in n else '#3498db' for n in short_names]
        bars = ax.bar(range(len(ratios)), ratios, color=c, alpha=0.85, edgecolor='white')
        for b, r in zip(bars, ratios):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
                   f"{r:.4f}", ha='center', fontsize=8, fontweight='bold')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label="No change (1.0)")
        ax.set_xticks(range(len(ratios))); ax.set_xticklabels(short_names, fontsize=8, rotation=45)
        ax.legend(fontsize=9)
    ax.set_ylabel("Norm Ratio", fontsize=12)
    ax.set_title("③ Norm Change\n(Additive distorts, Radial varies)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase56_radial_dose_response.png")
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
    ckpt_path = os.path.join(RESULTS_DIR, "phase56_checkpoint.json")
    all_sum = []
    all_det = {}
    completed_conditions = set()
    in_progress_data = {}

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
            print(f"   Remaining: {9 - len(completed_conditions)} conditions")
        except Exception as e:
            print(f"⚠️ Checkpoint load failed ({e}), starting fresh")
            all_sum = []
            completed_conditions = set()

    model, tok = load_model()

    # 9 conditions
    conditions = [
        # (name, sigma, hook_type)
        ("baseline",       0.0,   None),
        ("additive_0.05",  0.05, "additive"),
        ("additive_0.08",  0.08, "additive"),
        ("additive_0.10",  0.10, "additive"),
        ("additive_0.12",  0.12, "additive"),
        ("additive_0.15",  0.15, "additive"),
        ("radial_0.05",    0.05, "radial"),
        ("radial_0.10",    0.10, "radial"),
        ("radial_0.15",    0.15, "radial"),
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

        # Clear in_progress after completing the condition
        in_progress_data = {}

        # Checkpoint after each completed condition
        checkpoint = {
            "experiment": "Phase 56 (checkpoint)",
            "completed_conditions": [s["condition"] for s in all_sum],
            "summaries": all_sum,
            "in_progress": {},
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
        "experiment": "Phase 56: Radial Noise + Additive Dose-Response",
        "model": MODEL_SHORT,
        "task": "Modified-3 Hanoi (reversed rules)",
        "purpose": "Decompose noise into magnitude-only vs direction+magnitude, map dose-response curve",
        "hypothesis": "If radial (magnitude-only) noise matches additive, then norm jitter is the active ingredient",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_sum,
        "fisher_tests": fisher_tests,
        "figure": fig,
    }
    log = os.path.join(RESULTS_DIR, "phase56_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # ═══ VERDICT ═══
    print(f"\n{'═'*75}")
    print(f"  ⚡ PHASE 56: RADIAL + DOSE-RESPONSE — VERDICT")
    print(f"{'═'*75}")
    print(f"  Task: Modified-3 Hanoi | N={N_TRIALS} per condition | 9 conditions")
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
        print(f"    {name}: {desc} → p₂={p}{sig}")

    # Key verdict
    r10 = next((s for s in all_sum if s["condition"] == "radial_0.10"), None)
    a10 = next((s for s in all_sum if s["condition"] == "additive_0.10"), None)
    base = next((s for s in all_sum if s["condition"] == "baseline"), None)

    print(f"\n  {'═'*60}")
    print(f"  🎯 KEY QUESTION: Is magnitude jitter the active ingredient?")
    if r10 and a10:
        print(f"    Radial σ=0.10:   {r10['solve_rate']:.1f}% (magnitude only)")
        print(f"    Additive σ=0.10: {a10['solve_rate']:.1f}% (direction + magnitude)")
        if abs(r10['solve_rate'] - a10['solve_rate']) <= 5:
            print(f"    ✅ YES — Magnitude jitter IS the active ingredient!")
        elif r10['solve_rate'] > a10['solve_rate'] + 5:
            print(f"    🤯 RADIAL IS BETTER — Pure magnitude jitter is even more effective!")
        else:
            print(f"    ❌ NO — Direction change also matters (or interaction effect)")
    print(f"  {'═'*60}")

    print(f"\n  ⏱ {elapsed/60:.1f} min ({elapsed/3600:.1f} hours) | 💾 {log}")
    print(f"{'═'*75}")


if __name__ == "__main__":
    main()
    # Auto-hibernate after completion
    time.sleep(10)
