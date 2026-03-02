"""
Phase 62: Noise Half-Life — How Long Must Prevention Last?
==========================================================

Phase 60 showed that step_half (noise first 25 moves only) achieves 32%
— nearly as good as always-on (24%) and even better!
This proves the prophylactic effect PERSISTS after noise is removed.

Phase 62 precisely measures the "half-life" of this preventive effect
by scanning the noise duration: inject σ=0.15 for the first K moves only.

Conditions (7, N=50 each):
  1. baseline        — No noise (control)
  2. noise_first_5   — σ=0.15 for moves 1-5 only
  3. noise_first_10  — σ=0.15 for moves 1-10 only
  4. noise_first_15  — σ=0.15 for moves 1-15 only
  5. noise_first_20  — σ=0.15 for moves 1-20 only
  6. noise_first_25  — σ=0.15 for moves 1-25 only (= step_half)
  7. always_0.15     — σ=0.15 for all moves (positive control)

Total: 350 games. ~7-8 hours on RTX 5080.

Usage:
    python experiments/phase62_noise_halflife.py
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
#  GAME WITH DURATION-LIMITED NOISE
# ===================================================

def play_limited_noise(model, tok, env, hook, noise_duration):
    """Play with noise active only for the first 'noise_duration' moves.

    Args:
        noise_duration: Number of LEGAL moves to keep noise on.
                        0 = no noise (baseline), MAX_STEPS = always on.
    """
    env.reset()
    error = None; consec_fail = 0
    legal_move_count = 0  # Track legal moves for duration

    for step in range(MAX_STEPS):
        # Noise on if we haven't exceeded duration
        hook.active = (legal_move_count < noise_duration) and (noise_duration > 0)
        hook.sigma = 0.15

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
            legal_move_count += 1
            error = None; consec_fail = 0
            if env.is_solved(): break
        else:
            error = msg; consec_fail += 1
            if consec_fail >= 10: break

    hook.active = False
    st = env.stats()
    st["noise_moves"] = min(legal_move_count, noise_duration)
    st["total_legal_moves"] = legal_move_count
    return st


# ===================================================
#  MAIN
# ===================================================

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tok = load_model()
    layers = model.model.layers
    hook = DynamicNoiseHook(sigma=0.15)
    handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    # Conditions: noise for first K legal moves
    durations = {
        "baseline": 0,
        "noise_first_5": 5,
        "noise_first_10": 10,
        "noise_first_15": 15,
        "noise_first_20": 20,
        "noise_first_25": 25,
        "always_0.15": MAX_STEPS,  # effectively always-on
    }

    all_summaries = []
    for name, duration in durations.items():
        print(f"\n  === {name} (noise for {duration} legal moves) | N={N_PER_CONDITION} ===")
        results = []; solved = 0

        for t in range(N_PER_CONDITION):
            env = HanoiEnv(3, modified=True)
            r = play_limited_noise(model, tok, env, hook, duration)
            results.append(r)
            if r["solved"]: solved += 1
            icon = "O" if r["solved"] else "X"
            rate = solved / (t+1) * 100
            nm = r.get("noise_moves", 0)
            print(f"    {t+1:3d}/{N_PER_CONDITION}: {icon} legal={r['legal_moves']:2d} "
                  f"illegal={r['illegal_moves']:2d} noise_moves={nm} "
                  f"[{solved}/{t+1} = {rate:.0f}%]")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if (t+1) % 10 == 0: gc.collect()

        sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
        n_solved = sum(1 for r in results if r["solved"])
        avg_nm = round(np.mean([r.get("noise_moves", 0) for r in results]), 2)
        all_summaries.append({
            "condition": name, "noise_duration": duration,
            "n_trials": N_PER_CONDITION, "n_solved": n_solved, "solve_rate": sr,
            "avg_legal": round(np.mean([r["legal_moves"] for r in results]), 2),
            "avg_illegal": round(np.mean([r["illegal_moves"] for r in results]), 2),
            "avg_noise_moves": avg_nm,
        })
        print(f"    => Solve={sr:.1f}% ({n_solved}/{N_PER_CONDITION}) avg_noise_moves={avg_nm}")

    handle.remove()
    elapsed = time.time() - t0

    # Visualization
    fig_path = visualize(all_summaries)

    # Baseline for Fisher test
    bl = next(s for s in all_summaries if s["condition"] == "baseline")
    always = next(s for s in all_summaries if s["condition"] == "always_0.15")

    # Verdict
    print(f"\n{'='*80}")
    print(f"  PHASE 62: NOISE HALF-LIFE — VERDICT")
    print(f"{'='*80}")
    print(f"  {'Condition':<18} {'Duration':>8} {'Solve%':>8} {'pVsBL':>10} {'Sig':>6}")
    print(f"{'-'*65}")
    for s in all_summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<18} {s['noise_duration']:>8} {s['solve_rate']:>7.1f}% "
                  f"{'---':>10} {'---':>6}")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {s['condition']:<18} {s['noise_duration']:>8} {s['solve_rate']:>7.1f}% "
              f"{p:>10.4f} {sig:>6}")
    print(f"{'-'*65}")

    # Half-life analysis
    always_rate = always["solve_rate"]
    half_target = always_rate / 2
    print(f"\n  HALF-LIFE ANALYSIS:")
    print(f"    Always-on rate: {always_rate:.1f}%")
    print(f"    Half target: {half_target:.1f}%")
    for s in all_summaries:
        if s["noise_duration"] > 0 and s["solve_rate"] >= half_target:
            min_duration = s["noise_duration"]
    print(f"    Minimum duration for ≥ half effect: {min_duration} moves")
    print(f"\n  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")

    # Save
    out = {
        "experiment": "Phase 62: Noise Half-Life — Duration Scan",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "purpose": "Measure how long prophylactic noise must be applied for the effect to persist",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_summaries, "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase62_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")
    print(f"{'='*80}")


def visualize(summaries):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 62: Noise Half-Life\n"
                 "How long must prophylactic noise be applied for the effect to persist?",
                 fontsize=14, fontweight="bold", y=1.02)

    durations = [s["noise_duration"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]

    # Panel 1: Duration vs solve rate (line plot)
    ax = axes[0]
    ax.plot(durations, rates, 'o-', color='#e74c3c', linewidth=2.5, markersize=10,
            markerfacecolor='white', markeredgewidth=2, markeredgecolor='#e74c3c')
    for d, r in zip(durations, rates):
        ax.annotate(f'{r:.0f}%', (d, r), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

    # Mark half-life
    always_rate = rates[-1]
    ax.axhline(y=always_rate/2, color='gray', linestyle='--', alpha=0.5,
              label=f'50% of always-on ({always_rate/2:.0f}%)')
    ax.axhline(y=always_rate, color='#3498db', linestyle='--', alpha=0.5,
              label=f'Always-on ({always_rate:.0f}%)')
    ax.fill_between([0, MAX_STEPS], always_rate/2, always_rate, alpha=0.1, color='green',
                    label='Effective zone')

    ax.set_xlabel("Noise Duration (legal moves)", fontsize=12)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Prophylactic Effect vs Noise Duration",
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(durations)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Bar chart with significance
    ax = axes[1]
    bl = next(s for s in summaries if s["condition"] == "baseline")
    colors_map = {0: '#95a5a6', 5: '#e74c3c', 10: '#f39c12', 15: '#f1c40f',
                  20: '#2ecc71', 25: '#27ae60', 50: '#3498db'}
    colors = [colors_map.get(d, '#95a5a6') for d in durations]
    bars = ax.bar(range(len(summaries)), rates, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=2)
    for i, (b, s) in enumerate(zip(bars, summaries)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               f'{s["solve_rate"]:.0f}%', ha='center', fontsize=10, fontweight='bold')
        if i > 0:
            table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                     [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
            _, p = fisher_exact(table, alternative='two-sided')
            sig = "★★★" if p < 0.001 else "★★" if p < 0.01 else "★" if p < 0.05 else "n.s."
            ax.text(i, s["solve_rate"]+4, f'{sig}', ha='center', fontsize=8,
                   color='darkred' if p < 0.05 else 'gray')

    ax.set_xticks(range(len(summaries)))
    labels = [f'{s["condition"]}\n({s["noise_duration"]}m)' for s in summaries]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Performance by Noise Duration\n(with Fisher significance vs baseline)",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase62_noise_halflife.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


if __name__ == "__main__":
    main()
    # Restore power settings before hibernating
