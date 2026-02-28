"""
Phase 59: Smart Defibrillator — Conditional Noise Injection
============================================================

Purpose:
  Phase 58 revealed the cliff anatomy: cosine similarity ≈ 0.50 is the critical
  threshold for reasoning collapse. Instead of ALWAYS applying noise (open-loop),
  this experiment tests CONDITIONAL injection (closed-loop):
    - Run baseline (no noise) until the model makes a mistake
    - At the moment of error detection, fire σ=0.15 additive noise for ONLY
      the correction move(s)
    - Compare: constant noise vs smart defibrillation

  Hypothesis: By only applying noise when the model is "stuck" (after illegal
  move or parse failure), we can achieve the benefits of stochastic resonance
  (escaping local minima) without the costs (reasoning collapse at high σ).

Conditions (5, N=50 each):
  1. baseline           — No noise ever (control)
  2. always_0.15        — Constant σ=0.15 at L18 (v10 peak, replication)
  3. defib_on_error     — σ=0.15 only on the next move after illegal/parse error
  4. defib_2_moves      — σ=0.15 for 2 moves after error (longer pulse)
  5. defib_on_error_0.10 — σ=0.10 on error (lower-dose defibrillation)

Total: 250 games. ~5-6 hours on RTX 5080.

Key idea: The defibrillator fires a targeted "shock" to dislodge the model
from a reasoning dead-end, then lets it run naturally. Like a cardiac AED:
don't shock the healthy heart, only shock during arrhythmia.

Usage:
    python experiments/phase59_smart_defibrillator.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, json, gc, time, random, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.stats import fisher_exact

# === Config ===
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
TARGET_LAYER = 18
SEED = 2026
MAX_STEPS = 50
N_PER_CONDITION = 50

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================
#  HANOI ENVIRONMENT (Phase 55 compatible)
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
#  SWITCHABLE NOISE HOOK
# ===================================================

class DefibrillatorHook:
    """Additive noise hook with dynamic on/off control."""
    def __init__(self, sigma=0.15):
        self.sigma = sigma
        self.active = False  # Start OFF — activate only when needed
        self.fire_count = 0  # How many times the defib has been fired
        self.total_forwards = 0

    def __call__(self, module, args):
        self.total_forwards += 1
        if not self.active or self.sigma <= 0:
            return args

        hs = args[0]
        noise = torch.randn_like(hs) * self.sigma
        noisy = hs + noise
        self.fire_count += 1
        return (noisy,) + args[1:]

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def stats(self):
        return {
            "fire_count": self.fire_count,
            "total_forwards": self.total_forwards,
            "fire_ratio": round(self.fire_count / max(1, self.total_forwards), 4),
        }


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
#  GAME STRATEGIES
# ===================================================

def play_baseline(model, tok, env):
    """No noise at all."""
    env.reset()
    error = None; consec_fail = 0

    for step in range(MAX_STEPS):
        prompt = build_chat_prompt(tok, env, error)
        resp = generate(model, tok, prompt)
        move = parse_move(resp)
        if move is None:
            env.illegal_count += 1; env.total_attempts += 1; env._prev_illegal = True
            error = f"Parse fail. Use Move: X->Y"; consec_fail += 1
            if consec_fail >= 10: break
            continue
        ok, msg = env.try_move(move[0], move[1])
        if ok:
            error = None; consec_fail = 0
            if env.is_solved(): break
        else:
            error = msg; consec_fail += 1
            if consec_fail >= 10: break

    return env.stats()


def play_always_on(model, tok, env, hook):
    """Always inject noise (replicate v10 peak)."""
    env.reset()
    hook.active = True
    hook.fire_count = 0; hook.total_forwards = 0
    error = None; consec_fail = 0

    for step in range(MAX_STEPS):
        prompt = build_chat_prompt(tok, env, error)
        resp = generate(model, tok, prompt)
        move = parse_move(resp)
        if move is None:
            env.illegal_count += 1; env.total_attempts += 1; env._prev_illegal = True
            error = f"Parse fail. Use Move: X->Y"; consec_fail += 1
            if consec_fail >= 10: break
            continue
        ok, msg = env.try_move(move[0], move[1])
        if ok:
            error = None; consec_fail = 0
            if env.is_solved(): break
        else:
            error = msg; consec_fail += 1
            if consec_fail >= 10: break

    hook.active = False
    st = env.stats()
    st["hook_stats"] = hook.stats()
    return st


def play_defibrillator(model, tok, env, hook, pulse_duration=1):
    """Smart defibrillation: noise only after errors.

    Args:
        pulse_duration: how many moves to keep noise active after error
    """
    env.reset()
    hook.active = False
    hook.fire_count = 0; hook.total_forwards = 0
    error = None; consec_fail = 0
    moves_until_deactivate = 0  # countdown for pulse duration
    defib_events = 0  # how many times we triggered defib

    for step in range(MAX_STEPS):
        # Manage defib state
        if moves_until_deactivate > 0:
            hook.activate()
            moves_until_deactivate -= 1
            if moves_until_deactivate == 0:
                hook.deactivate()
        else:
            hook.deactivate()

        prompt = build_chat_prompt(tok, env, error)
        resp = generate(model, tok, prompt)
        move = parse_move(resp)

        if move is None:
            # PARSE ERROR -> FIRE DEFIBRILLATOR
            env.illegal_count += 1; env.total_attempts += 1; env._prev_illegal = True
            error = f"Parse fail. Use Move: X->Y"; consec_fail += 1
            moves_until_deactivate = pulse_duration
            defib_events += 1
            if consec_fail >= 10: break
            continue

        ok, msg = env.try_move(move[0], move[1])
        if ok:
            error = None; consec_fail = 0
            if env.is_solved(): break
        else:
            # ILLEGAL MOVE -> FIRE DEFIBRILLATOR
            error = msg; consec_fail += 1
            moves_until_deactivate = pulse_duration
            defib_events += 1
            if consec_fail >= 10: break

    hook.deactivate()
    st = env.stats()
    st["hook_stats"] = hook.stats()
    st["defib_events"] = defib_events
    return st


# ===================================================
#  CONDITION RUNNER
# ===================================================

def run_condition(model, tok, cond_name, play_fn, n_trials, hook=None):
    print(f"\n  === {cond_name} | N={n_trials} ===")
    results = []
    solved = 0

    for t in range(n_trials):
        env = HanoiEnv(3, modified=True)
        r = play_fn(model, tok, env) if hook is None else play_fn(model, tok, env, hook)
        results.append(r)
        if r["solved"]: solved += 1

        icon = "O" if r["solved"] else "X"
        rate = solved / (t+1) * 100
        defib = r.get("defib_events", "-")
        fire = r.get("hook_stats", {}).get("fire_ratio", "-")
        print(f"    {t+1:3d}/{n_trials}: {icon} legal={r['legal_moves']:2d} "
              f"illegal={r['illegal_moves']:2d} defib={defib} "
              f"[{solved}/{t+1} = {rate:.0f}%]")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (t + 1) % 10 == 0:
            gc.collect()

    sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
    n_solved = sum(1 for r in results if r["solved"])

    # Fisher exact vs baseline
    summary = {
        "condition": cond_name,
        "n_trials": n_trials,
        "n_solved": n_solved,
        "solve_rate": sr,
        "avg_legal": round(np.mean([r["legal_moves"] for r in results]), 2),
        "avg_illegal": round(np.mean([r["illegal_moves"] for r in results]), 2),
        "avg_defib_events": round(np.mean([r.get("defib_events", 0) for r in results]), 2),
    }
    print(f"    => Solve={sr:.1f}% ({n_solved}/{n_trials})")
    return summary, results


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(summaries, bl_solved, bl_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Phase 59: Smart Defibrillator — Conditional Noise Injection\n"
                 "Does targeted noise (only after errors) outperform constant noise?",
                 fontsize=14, fontweight="bold", y=1.02)

    names = [s["condition"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]

    # Panel 1: Solve rates
    ax = axes[0]
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#27ae60', '#e67e22']
    bars = ax.bar(range(len(names)), rates, color=colors[:len(names)],
                  alpha=0.85, edgecolor='white', linewidth=2)
    for b, r in zip(bars, rates):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               f'{r:.0f}%', ha='center', fontsize=11, fontweight='bold')

    # Significance markers
    for i, s in enumerate(summaries):
        if i == 0: continue  # skip baseline
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl_solved, bl_n - bl_solved]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "★★★" if p < 0.001 else "★★" if p < 0.01 else "★" if p < 0.05 else "n.s."
        ax.text(i, rates[i]+4, f'p={p:.3f}\n{sig}', ha='center', fontsize=8,
               color='darkred' if p < 0.05 else 'gray')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Solve Rate by Strategy", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Defib events vs solve rate
    ax = axes[1]
    defib_events = [s["avg_defib_events"] for s in summaries]
    for i, s in enumerate(summaries):
        color = '#2ecc71' if s["solve_rate"] > 20 else '#e74c3c' if s["solve_rate"] < 5 else '#f39c12'
        ax.scatter(defib_events[i], s["solve_rate"], s=200, c=color,
                  edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(s["condition"].replace("_", "\n"), (defib_events[i], s["solve_rate"]),
                   textcoords="offset points", xytext=(10, 5), fontsize=8)

    ax.set_xlabel("Avg Defibrillation Events per Game", fontsize=12)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Noise Dosage vs Performance\n(less noise = better?)",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 3: Illegal moves comparison
    ax = axes[2]
    illegals = [s["avg_illegal"] for s in summaries]
    ax.bar(range(len(names)), illegals, color=colors[:len(names)], alpha=0.7,
           edgecolor='white', linewidth=2)
    for i, (b, il) in enumerate(zip(ax.patches, illegals)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
               f'{il:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Avg Illegal Moves per Game", fontsize=12)
    ax.set_title("Error Rate by Strategy\n(does defib reduce errors?)",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase59_smart_defibrillator.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tok = load_model()
    layers = model.model.layers

    # Register hooks (shared across conditions, toggled on/off)
    hook_015 = DefibrillatorHook(sigma=0.15)
    handle_015 = layers[TARGET_LAYER].register_forward_pre_hook(hook_015)

    hook_010 = DefibrillatorHook(sigma=0.10)
    handle_010 = layers[TARGET_LAYER].register_forward_pre_hook(hook_010)

    # Initially both OFF
    hook_015.active = False
    hook_010.active = False

    all_summaries = []
    all_details = {}

    # --- Condition 1: Baseline (no noise) ---
    # Remove all hooks for clean baseline
    handle_015.remove()
    handle_010.remove()
    s, d = run_condition(model, tok, "baseline", play_baseline, N_PER_CONDITION)
    all_summaries.append(s)
    all_details["baseline"] = d
    bl_solved = s["n_solved"]; bl_n = s["n_trials"]
    gc.collect(); torch.cuda.empty_cache()

    # Re-register hooks for remaining conditions
    handle_015 = layers[TARGET_LAYER].register_forward_pre_hook(hook_015)

    # --- Condition 2: Always-on σ=0.15 ---
    s, d = run_condition(model, tok, "always_0.15",
                         lambda m, t, e, h=hook_015: play_always_on(m, t, e, h),
                         N_PER_CONDITION, hook=hook_015)
    all_summaries.append(s)
    all_details["always_0.15"] = d
    hook_015.active = False
    gc.collect(); torch.cuda.empty_cache()

    # --- Condition 3: Defibrillator on error (1-move pulse, σ=0.15) ---
    s, d = run_condition(model, tok, "defib_1move_0.15",
                         lambda m, t, e, h=hook_015: play_defibrillator(m, t, e, h, pulse_duration=1),
                         N_PER_CONDITION, hook=hook_015)
    all_summaries.append(s)
    all_details["defib_1move_0.15"] = d
    hook_015.active = False
    gc.collect(); torch.cuda.empty_cache()

    # --- Condition 4: Defibrillator on error (2-move pulse, σ=0.15) ---
    s, d = run_condition(model, tok, "defib_2move_0.15",
                         lambda m, t, e, h=hook_015: play_defibrillator(m, t, e, h, pulse_duration=2),
                         N_PER_CONDITION, hook=hook_015)
    all_summaries.append(s)
    all_details["defib_2move_0.15"] = d
    hook_015.active = False
    gc.collect(); torch.cuda.empty_cache()

    # Switch to σ=0.10 hook
    handle_015.remove()
    handle_010 = layers[TARGET_LAYER].register_forward_pre_hook(hook_010)

    # --- Condition 5: Defibrillator on error (1-move pulse, σ=0.10) ---
    s, d = run_condition(model, tok, "defib_1move_0.10",
                         lambda m, t, e, h=hook_010: play_defibrillator(m, t, e, h, pulse_duration=1),
                         N_PER_CONDITION, hook=hook_010)
    all_summaries.append(s)
    all_details["defib_1move_0.10"] = d
    handle_010.remove()
    gc.collect(); torch.cuda.empty_cache()

    elapsed = time.time() - t0
    fig = visualize(all_summaries, bl_solved, bl_n)

    # Fisher exact tests
    print(f"\n{'='*80}")
    print(f"  PHASE 59: SMART DEFIBRILLATOR — VERDICT")
    print(f"{'='*80}")
    print(f"  {'Condition':<22} {'Solve%':>8} {'N':>4} {'pFisher':>10} {'Sig':>6} {'AvgDefib':>10}")
    print(f"{'-'*80}")
    for s in all_summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<22} {s['solve_rate']:>7.1f}% {s['n_trials']:>4} {'---':>10} {'---':>6} {'-':>10}")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl_solved, bl_n - bl_solved]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {s['condition']:<22} {s['solve_rate']:>7.1f}% {s['n_trials']:>4} {p:>10.4f} {sig:>6} "
              f"{s['avg_defib_events']:>10.1f}")
    print(f"{'-'*80}")
    print(f"\n  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")

    # Save
    out = {
        "experiment": "Phase 59: Smart Defibrillator — Conditional Noise Injection",
        "model": MODEL_SHORT,
        "task": "Modified-3 Hanoi (reversed rules)",
        "purpose": "Test conditional noise injection (only after errors) vs constant noise",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_summaries,
        "figure": fig,
    }
    log = os.path.join(RESULTS_DIR, "phase59_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
    # Restore power settings before hibernating
