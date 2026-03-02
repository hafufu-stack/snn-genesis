"""
Phase 60-61 Batch Runner: Annealing + KV-Cache Rollback
========================================================

Runs TWO experiments sequentially with shared model loading:

Phase 60: Simulated Annealing Schedule (σ decay over time)
  - baseline (no noise)
  - always σ=0.15 (replication)
  - linear decay: σ = 0.15 * (1 - step/MAX_STEPS)
  - cosine decay: σ = 0.15 * 0.5 * (1 + cos(π * step/MAX_STEPS))
  - step function: σ=0.15 first half, 0 second half

Phase 61: KV-Cache Rollback + Stochastic Resonance
  - baseline (no noise, no rollback)
  - always σ=0.15 (no rollback) — replication
  - rollback only (rollback on error, regenerate without noise)
  - rollback + σ=0.15 (rollback on error, regenerate WITH noise)
  - rollback + σ=0.15 always (rollback + constant noise throughout)

Total: 10 conditions × N=50 = 500 games. ~12-15 hours.

Usage:
    python experiments/phase60_61_batch.py
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
TARGET_LAYER = 18
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
#  DYNAMIC NOISE HOOK
# ===================================================

class DynamicNoiseHook:
    """Noise hook with dynamically adjustable sigma."""
    def __init__(self, sigma=0.15):
        self.sigma = sigma
        self.active = False

    def __call__(self, module, args):
        if not self.active or self.sigma <= 0:
            return args
        hs = args[0]
        noise = torch.randn_like(hs) * self.sigma
        return (hs + noise,) + args[1:]


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
#  PHASE 60: SIMULATED ANNEALING SCHEDULE
# ===================================================

def play_annealing(model, tok, env, hook, schedule_fn, max_steps=MAX_STEPS):
    """Play with time-varying σ based on schedule_fn(step, max_steps) -> sigma."""
    env.reset()
    error = None; consec_fail = 0

    for step in range(max_steps):
        # Set sigma based on schedule
        current_sigma = schedule_fn(step, max_steps)
        hook.sigma = current_sigma
        hook.active = current_sigma > 0

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
    return env.stats()


def run_phase60(model, tok):
    """Phase 60: Simulated Annealing Schedule."""
    print(f"\n{'#'*80}")
    print(f"  PHASE 60: SIMULATED ANNEALING SCHEDULE")
    print(f"{'#'*80}")
    t0 = time.time()

    layers = model.model.layers
    hook = DynamicNoiseHook(sigma=0.15)
    handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    # Define schedules
    schedules = {
        "baseline": lambda step, ms: 0.0,
        "always_0.15": lambda step, ms: 0.15,
        "linear_decay": lambda step, ms: 0.15 * (1.0 - step / ms),
        "cosine_decay": lambda step, ms: 0.15 * 0.5 * (1.0 + math.cos(math.pi * step / ms)),
        "step_half": lambda step, ms: 0.15 if step < ms // 2 else 0.0,
    }

    summaries = []
    for name, sched in schedules.items():
        print(f"\n  --- {name} | N={N_PER_CONDITION} ---")
        results = []; solved = 0

        for t in range(N_PER_CONDITION):
            env = HanoiEnv(3, modified=True)
            r = play_annealing(model, tok, env, hook, sched)
            results.append(r)
            if r["solved"]: solved += 1
            icon = "O" if r["solved"] else "X"
            rate = solved / (t+1) * 100
            print(f"    {t+1:3d}/{N_PER_CONDITION}: {icon} legal={r['legal_moves']:2d} "
                  f"illegal={r['illegal_moves']:2d} [{solved}/{t+1} = {rate:.0f}%]")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if (t+1) % 10 == 0: gc.collect()

        sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
        n_solved = sum(1 for r in results if r["solved"])
        summaries.append({
            "condition": name, "n_trials": N_PER_CONDITION,
            "n_solved": n_solved, "solve_rate": sr,
            "avg_legal": round(np.mean([r["legal_moves"] for r in results]), 2),
            "avg_illegal": round(np.mean([r["illegal_moves"] for r in results]), 2),
        })
        print(f"    => Solve={sr:.1f}% ({n_solved}/{N_PER_CONDITION})")

    handle.remove()
    elapsed = time.time() - t0

    # Visualization
    fig_path = visualize_phase60(summaries)

    # Baseline for Fisher test
    bl = next(s for s in summaries if s["condition"] == "baseline")

    # Print verdict
    print(f"\n{'='*80}")
    print(f"  PHASE 60: SIMULATED ANNEALING — VERDICT")
    print(f"{'='*80}")
    print(f"  {'Condition':<18} {'Solve%':>8} {'N':>4} {'pFisher':>10} {'Sig':>6}")
    print(f"{'-'*60}")
    for s in summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<18} {s['solve_rate']:>7.1f}% {s['n_trials']:>4} {'---':>10} {'---':>6}")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {s['condition']:<18} {s['solve_rate']:>7.1f}% {s['n_trials']:>4} {p:>10.4f} {sig:>6}")
    print(f"{'-'*60}")
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")

    # Save
    out = {
        "experiment": "Phase 60: Simulated Annealing Schedule",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": summaries, "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase60_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")
    return summaries


def visualize_phase60(summaries):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 60: Simulated Annealing Schedule\n"
                 "Does cooling the noise over time beat constant noise?",
                 fontsize=14, fontweight="bold", y=1.02)

    names = [s["condition"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]

    # Bar chart
    ax = axes[0]
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(range(len(names)), rates, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    for b, r in zip(bars, rates):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               f'{r:.0f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Solve Rate by Annealing Schedule", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Schedule visualization
    ax = axes[1]
    steps = np.arange(MAX_STEPS)
    schedules_vis = {
        "always_0.15": [0.15] * MAX_STEPS,
        "linear_decay": [0.15 * (1.0 - s/MAX_STEPS) for s in steps],
        "cosine_decay": [0.15 * 0.5 * (1.0 + math.cos(math.pi * s/MAX_STEPS)) for s in steps],
        "step_half": [0.15 if s < MAX_STEPS//2 else 0.0 for s in steps],
    }
    vis_colors = {'always_0.15': '#3498db', 'linear_decay': '#e74c3c',
                  'cosine_decay': '#2ecc71', 'step_half': '#f39c12'}
    for name, vals in schedules_vis.items():
        sr = next(s["solve_rate"] for s in summaries if s["condition"] == name)
        ax.plot(steps, vals, label=f'{name} ({sr:.0f}%)', linewidth=2.5, color=vis_colors[name])
    ax.set_xlabel("Game Step", fontsize=12)
    ax.set_ylabel("σ (noise level)", fontsize=12)
    ax.set_title("Annealing Schedules\n(σ over time)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase60_annealing.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


# ===================================================
#  PHASE 61: KV-CACHE ROLLBACK + STOCHASTIC RESONANCE
# ===================================================

def play_rollback(model, tok, env, hook, use_noise_on_retry=False,
                  always_noise=False, max_retries=3):
    """Play with KV-Cache rollback on errors.

    When the model makes an illegal move or parse failure:
    1. DON'T feed error message back (avoids context contamination)
    2. Instead, re-prompt from the SAME state (rollback)
    3. Optionally inject noise on the retry attempt

    Args:
        use_noise_on_retry: If True, activate noise only on retry attempts
        always_noise: If True, noise is always on (combined with rollback)
        max_retries: Max retries per move before giving up
    """
    env.reset()
    consec_fail = 0
    rollback_count = 0

    for step in range(MAX_STEPS):
        # Set noise mode
        if always_noise:
            hook.active = True
        else:
            hook.active = False

        prompt = build_chat_prompt(tok, env, error=None)  # NEVER pass error
        resp = generate(model, tok, prompt)
        move = parse_move(resp)

        if move is None:
            # Parse failure -> try rollback (retry same state)
            retried = False
            for retry in range(max_retries):
                if use_noise_on_retry or always_noise:
                    hook.active = True
                resp2 = generate(model, tok, prompt)  # Same prompt, no error context
                move2 = parse_move(resp2)
                if move2 is not None:
                    # Try this move
                    ok, msg = env.try_move(move2[0], move2[1])
                    if ok:
                        retried = True
                        rollback_count += 1
                        consec_fail = 0
                        if env.is_solved():
                            break
                        break
                    else:
                        continue  # Try another retry
                # Still can't parse, try again
            if not retried:
                env.illegal_count += 1; env.total_attempts += 1
                env._prev_illegal = True
                consec_fail += 1
                if consec_fail >= 10: break
            if env.is_solved(): break
            continue

        ok, msg = env.try_move(move[0], move[1])
        if ok:
            consec_fail = 0
            if env.is_solved(): break
        else:
            # Illegal move -> rollback: retry same state with noise
            retried = False
            for retry in range(max_retries):
                if use_noise_on_retry or always_noise:
                    hook.active = True
                resp2 = generate(model, tok, prompt)  # Same prompt, clean context
                move2 = parse_move(resp2)
                if move2 is not None:
                    ok2, msg2 = env.try_move(move2[0], move2[1])
                    if ok2:
                        retried = True
                        rollback_count += 1
                        consec_fail = 0
                        if env.is_solved():
                            break
                        break
            if not retried:
                consec_fail += 1
                if consec_fail >= 10: break
            if env.is_solved(): break

    hook.active = False
    st = env.stats()
    st["rollback_count"] = rollback_count
    return st


def run_phase61(model, tok):
    """Phase 61: KV-Cache Rollback + Stochastic Resonance."""
    print(f"\n{'#'*80}")
    print(f"  PHASE 61: KV-CACHE ROLLBACK + STOCHASTIC RESONANCE")
    print(f"{'#'*80}")
    t0 = time.time()

    layers = model.model.layers
    hook = DynamicNoiseHook(sigma=0.15)
    handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    conditions = [
        ("baseline_no_rb", False, False),          # No noise, no rollback
        ("always_0.15_no_rb", False, True),         # Constant noise, no rollback (replication)
        ("rollback_only", False, False),            # Rollback but no noise on retry
        ("rollback_noise_retry", True, False),      # Rollback + noise only on retry
        ("rollback_always_noise", False, True),     # Rollback + always noise
    ]

    summaries = []
    for name, noise_retry, always_noise in conditions:
        print(f"\n  --- {name} | N={N_PER_CONDITION} ---")
        results = []; solved = 0

        for t in range(N_PER_CONDITION):
            env = HanoiEnv(3, modified=True)

            if "no_rb" in name:
                # No rollback: use standard play
                r = play_annealing(model, tok, env, hook,
                                   lambda s, ms: 0.15 if always_noise else 0.0)
            else:
                r = play_rollback(model, tok, env, hook,
                                  use_noise_on_retry=noise_retry,
                                  always_noise=always_noise)
            results.append(r)
            if r["solved"]: solved += 1
            icon = "O" if r["solved"] else "X"
            rate = solved / (t+1) * 100
            rb = r.get("rollback_count", "-")
            print(f"    {t+1:3d}/{N_PER_CONDITION}: {icon} legal={r['legal_moves']:2d} "
                  f"illegal={r['illegal_moves']:2d} rb={rb} [{solved}/{t+1} = {rate:.0f}%]")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if (t+1) % 10 == 0: gc.collect()

        sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
        n_solved = sum(1 for r in results if r["solved"])
        avg_rb = round(np.mean([r.get("rollback_count", 0) for r in results]), 2)
        summaries.append({
            "condition": name, "n_trials": N_PER_CONDITION,
            "n_solved": n_solved, "solve_rate": sr,
            "avg_legal": round(np.mean([r["legal_moves"] for r in results]), 2),
            "avg_illegal": round(np.mean([r["illegal_moves"] for r in results]), 2),
            "avg_rollbacks": avg_rb,
        })
        print(f"    => Solve={sr:.1f}% ({n_solved}/{N_PER_CONDITION}) avg_rb={avg_rb}")

    handle.remove()
    elapsed = time.time() - t0

    fig_path = visualize_phase61(summaries)

    bl = next(s for s in summaries if s["condition"] == "baseline_no_rb")
    print(f"\n{'='*80}")
    print(f"  PHASE 61: KV-CACHE ROLLBACK — VERDICT")
    print(f"{'='*80}")
    print(f"  {'Condition':<25} {'Solve%':>8} {'N':>4} {'pFisher':>10} {'Sig':>6} {'AvgRB':>8}")
    print(f"{'-'*75}")
    for s in summaries:
        if s["condition"] == "baseline_no_rb":
            print(f"  {s['condition']:<25} {s['solve_rate']:>7.1f}% {s['n_trials']:>4} "
                  f"{'---':>10} {'---':>6} {'-':>8}")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {s['condition']:<25} {s['solve_rate']:>7.1f}% {s['n_trials']:>4} "
              f"{p:>10.4f} {sig:>6} {s['avg_rollbacks']:>8.1f}")
    print(f"{'-'*75}")
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")

    out = {
        "experiment": "Phase 61: KV-Cache Rollback + Stochastic Resonance",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": summaries, "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase61_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")
    return summaries


def visualize_phase61(summaries):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Phase 61: KV-Cache Rollback + Stochastic Resonance\n"
                 "Does erasing error context + noise rescue reasoning?",
                 fontsize=14, fontweight="bold", y=1.02)

    names = [s["condition"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]

    bl = next(s for s in summaries if s["condition"] == "baseline_no_rb")

    # Panel 1: Solve rates
    ax = axes[0]
    colors = ['#95a5a6', '#3498db', '#f39c12', '#2ecc71', '#e74c3c']
    bars = ax.bar(range(len(names)), rates, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    for i, (b, r) in enumerate(zip(bars, rates)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               f'{r:.0f}%', ha='center', fontsize=10, fontweight='bold')
        if i > 0:
            s = summaries[i]
            table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                     [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
            _, p = fisher_exact(table, alternative='two-sided')
            sig = "★★★" if p < 0.001 else "★★" if p < 0.01 else "★" if p < 0.05 else "n.s."
            ax.text(i, r+4, f'p={p:.3f}\n{sig}', ha='center', fontsize=7, color='darkred' if p < 0.05 else 'gray')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Solve Rate by Strategy", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Rollback count vs solve rate
    ax = axes[1]
    for i, s in enumerate(summaries):
        rb = s.get("avg_rollbacks", 0)
        color = colors[i]
        ax.scatter(rb, s["solve_rate"], s=200, c=color, edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(s["condition"].replace("_", "\n"), (rb, s["solve_rate"]),
                   textcoords="offset points", xytext=(10, 5), fontsize=7)
    ax.set_xlabel("Avg Rollbacks per Game", fontsize=12)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Rollback Usage vs Performance", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 3: Error rates
    ax = axes[2]
    illegals = [s["avg_illegal"] for s in summaries]
    ax.bar(range(len(names)), illegals, color=colors, alpha=0.7, edgecolor='white', linewidth=2)
    for i, il in enumerate(illegals):
        ax.text(i, il+0.3, f'{il:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Avg Illegal Moves", fontsize=12)
    ax.set_title("Error Rates by Strategy", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase61_rollback.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


# ===================================================
#  MAIN: BATCH RUNNER
# ===================================================

def main():
    t0_total = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tok = load_model()

    # === PHASE 60 ===
    print("\n" + "="*80)
    print("  STARTING PHASE 60: SIMULATED ANNEALING")
    print("="*80)
    p60 = run_phase60(model, tok)
    gc.collect(); torch.cuda.empty_cache()

    # === PHASE 61 ===
    print("\n" + "="*80)
    print("  STARTING PHASE 61: KV-CACHE ROLLBACK")
    print("="*80)
    p61 = run_phase61(model, tok)

    total = time.time() - t0_total
    print(f"\n{'='*80}")
    print(f"  BATCH COMPLETE: Phase 60 + 61")
    print(f"  Total time: {total/60:.1f} min ({total/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
    # Restore power settings before hibernating
