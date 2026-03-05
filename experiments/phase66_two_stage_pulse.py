"""
Phase 66: Two-Stage Pulse — Vaccine Booster Shot
=================================================

Inspired by the analogy: if first 5 moves yield 30% and first 20 yields 42%,
what happens with "pulse" dosing? Like a vaccine booster shot.

Conditions (6, N=50 each):
  1. baseline        — No noise (control)
  2. first20_flat    — σ=0.15 for moves 1-20 (Phase 62 reference)
  3. pulse_5_off5_5  — Noise moves 1-5, off 6-10, noise 11-15
  4. pulse_5_off10_5 — Noise moves 1-5, off 6-15, noise 16-20
  5. pulse_5_off5_10 — Noise moves 1-5, off 6-10, noise 11-20
  6. pulse_10_off5_5 — Noise moves 1-10, off 11-15, noise 16-20

Total: 300 games. ~6-7 hours on RTX 5080.

Usage:
    python experiments/phase66_two_stage_pulse.py
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
#  NOISE HOOK
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
#  GAME WITH PULSE NOISE
# ===================================================

def play_pulse_noise(model, tok, env, hook, pulse_schedule):
    """Play with pulse noise schedule.
    
    Args:
        pulse_schedule: list of (start, end) tuples for noise-on windows.
                       e.g. [(0,5), (10,15)] = noise on moves 0-4, off 5-9, on 10-14
    """
    env.reset()
    error = None; consec_fail = 0
    legal_move_count = 0

    for step in range(MAX_STEPS):
        # Check if current legal_move_count falls within any noise window
        noise_on = any(start <= legal_move_count < end for start, end in pulse_schedule)
        hook.active = noise_on
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

    # Pulse schedules: list of (start_move, end_move) noise-on windows
    conditions = {
        "baseline":         [],                          # No noise
        "first20_flat":     [(0, 20)],                   # Reference: continuous noise moves 0-19
        "pulse_5+off5+5":   [(0, 5), (10, 15)],          # 5 on, 5 off, 5 on (total 10 noisy moves)
        "pulse_5+off10+5":  [(0, 5), (15, 20)],          # 5 on, 10 off, 5 on (spaced booster)
        "pulse_5+off5+10":  [(0, 5), (10, 20)],          # 5 on, 5 off, 10 on (delayed main dose)
        "pulse_10+off5+5":  [(0, 10), (15, 20)],         # 10 on, 5 off, 5 on (front-loaded + booster)
    }

    all_summaries = []
    for name, schedule in conditions.items():
        noise_desc = " + ".join([f"moves {s}-{e-1}" for s, e in schedule]) if schedule else "none"
        total_noisy = sum(e - s for s, e in schedule)
        print(f"\n  === {name} (noise: {noise_desc}, {total_noisy} total noisy moves) | N={N_PER_CONDITION} ===")
        results = []; solved = 0

        for t in range(N_PER_CONDITION):
            env = HanoiEnv(3, modified=True)
            r = play_pulse_noise(model, tok, env, hook, schedule)
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
        total_noisy = sum(e - s for s, e in schedule)
        all_summaries.append({
            "condition": name, "pulse_schedule": [(s,e) for s,e in schedule],
            "total_noisy_moves": total_noisy,
            "n_trials": N_PER_CONDITION, "n_solved": n_solved, "solve_rate": sr,
            "avg_legal": round(np.mean([r["legal_moves"] for r in results]), 2),
            "avg_illegal": round(np.mean([r["illegal_moves"] for r in results]), 2),
        })
        print(f"    => Solve={sr:.1f}% ({n_solved}/{N_PER_CONDITION})")

    handle.remove()
    elapsed = time.time() - t0

    # Visualization
    fig_path = visualize(all_summaries)

    # Verdict
    bl = next(s for s in all_summaries if s["condition"] == "baseline")
    print(f"\n{'='*70}")
    print(f"  PHASE 66 VERDICT: Two-Stage Pulse (Vaccine Booster)")
    print(f"{'='*70}")
    for s in all_summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<22} {s['solve_rate']:>6.1f}%   (baseline)")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {s['condition']:<22} {s['solve_rate']:>6.1f}%   total_noisy={s['total_noisy_moves']:2d}   p={p:.4f} {sig}")
    print(f"\n  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*70}")

    # Save
    out = {
        "experiment": "Phase 66: Two-Stage Pulse (Vaccine Booster)",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "purpose": "Test multi-pulse noise dosing: initial dose + booster after a gap",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_summaries, "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase66_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")


def visualize(summaries):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 66: Two-Stage Pulse (Vaccine Booster)\n"
                 "Can pulsed noise dosing ('booster shots') outperform continuous noise?",
                 fontsize=14, fontweight="bold", y=1.02)

    bl = next(s for s in summaries if s["condition"] == "baseline")
    names = [s["condition"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]
    noisy_counts = [s["total_noisy_moves"] for s in summaries]

    # Panel 1: Bar chart
    ax = axes[0]
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#2ecc71']
    bars = ax.bar(range(len(summaries)), rates, color=colors[:len(summaries)], alpha=0.85,
                  edgecolor='white', linewidth=2)
    for i, (b, s) in enumerate(zip(bars, summaries)):
        label = f'{s["solve_rate"]:.0f}%'
        if s["total_noisy_moves"] > 0:
            label += f'\n({s["total_noisy_moves"]}m)'
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               label, ha='center', fontsize=9, fontweight='bold')
        if i > 0:
            table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                     [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
            _, p = fisher_exact(table, alternative='two-sided')
            sig = "★★★" if p < 0.001 else "★★" if p < 0.01 else "★" if p < 0.05 else "n.s."
            ax.text(i, s["solve_rate"]+6, sig, ha='center', fontsize=8,
                   color='darkred' if p < 0.05 else 'gray')

    ax.set_xticks(range(len(summaries)))
    ax.set_xticklabels([n.replace("_", "\n").replace("+", "+\n") for n in names], fontsize=7)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Performance by Pulse Schedule", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Timeline visualization of pulse schedules
    ax = axes[1]
    for i, s in enumerate(summaries):
        y = len(summaries) - 1 - i
        # Draw background (no noise = gray)
        ax.barh(y, MAX_STEPS, height=0.6, color='#ecf0f1', edgecolor='white')
        # Draw noise windows
        for start, end in s.get("pulse_schedule", []):
            ax.barh(y, end - start, left=start, height=0.6,
                   color=colors[i], alpha=0.85)
        # Label
        ax.text(MAX_STEPS + 1, y, f'{s["solve_rate"]:.0f}%',
               va='center', fontsize=11, fontweight='bold',
               color='darkgreen' if s["solve_rate"] > 30 else 'black')

    ax.set_yticks(range(len(summaries)))
    ax.set_yticklabels([names[len(summaries)-1-i] for i in range(len(summaries))], fontsize=8)
    ax.set_xlabel("Legal Move Number", fontsize=12)
    ax.set_title("Noise Schedule Timeline\n(colored = noise ON, gray = noise OFF)",
                fontsize=12, fontweight='bold')
    ax.set_xlim(-1, MAX_STEPS + 8)
    ax.grid(True, alpha=0.2, axis='x')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase66_two_stage_pulse.png")
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
    print("\n Phase 66 complete.")

