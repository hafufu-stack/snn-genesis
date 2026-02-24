"""
Phase 51: Statistical Validation — The True Noise Showdown
==========================================================

Purpose:
  Phase 50c showed SNN-only solve (10%) on Modified-3 Hanoi.
  But N=10 → Fisher p≈0.53 (not significant).

  Phase 51 scales to N=50 per condition and adds a
  HIGH-TEMPERATURE CONTROL to prove SNN ≠ mere randomness.

Conditions:
  1. Baseline        (temp=0.5, no SNN)        × 50 trials
  2. High-Temp       (temp=1.2, no SNN)        × 50 trials
  3. SNN L18-Echo    (temp=0.5, σ=0.02)        × 50 trials

Task: Modified-3 only (the condition where SNN succeeded)

Expected runtime: ~5.5 hours on single GPU (Mistral-7B 4-bit)

Usage:
    python experiments/phase51_stats_validation.py
"""

import torch
import torch.nn as nn
import os, json, gc, time, random, re, copy
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ═══ Config ═══
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
SIGMA_ECHO = 0.02
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
#  SNN HOOK
# ═══════════════════════════════════════════════════

class EchoHook:
    def __init__(self, sigma=0.02):
        self.sigma = sigma
    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn_like(hs) * self.sigma
        return (hs + noise,) + args[1:]


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


def run_condition(model, tok, cond_name, temperature, n_trials=N_TRIALS):
    print(f"\n  🎮 Mod-3 | {cond_name} | temp={temperature} | {n_trials} trials")

    layers = model.model.layers
    hook, handle = None, None
    if cond_name == "snn_echo":
        hook = EchoHook(SIGMA_ECHO)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    results = []
    solved = 0
    for t in range(n_trials):
        env = HanoiEnv(3, modified=True)
        r = play_game(model, tok, env, temperature=temperature, hook=hook)
        results.append(r)
        if r["solved"]: solved += 1
        icon = "✅" if r["solved"] else "❌"
        rate = solved / (t+1) * 100
        print(f"    {t+1:2d}/{n_trials}: {icon} legal={r['legal_moves']:2d} "
              f"illegal={r['illegal_moves']:2d} self_corr={r['self_corrections']:2d}  "
              f"[running: {solved}/{t+1} = {rate:.0f}%]")

    if handle: handle.remove()

    s = lambda key: round(np.mean([r[key] for r in results]), 2)
    sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
    n_solved = sum(1 for r in results if r["solved"])
    summary = {
        "condition": cond_name, "temperature": temperature,
        "n_trials": n_trials, "n_solved": n_solved,
        "solve_rate": sr,
        "avg_legal": s("legal_moves"), "avg_illegal": s("illegal_moves"),
        "avg_self_corr": s("self_corrections"),
        "optimal": results[0]["optimal"],
        "thoughts": results[0].get("thoughts", []),
    }
    print(f"    📊 Solve={sr:.1f}% ({n_solved}/{n_trials}) "
          f"Legal={summary['avg_legal']:.1f} Illegal={summary['avg_illegal']:.1f}")
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
    Uses scipy if available, otherwise manual computation.
    """
    try:
        from scipy.stats import fisher_exact
        table = [[a, b], [c, d]]
        _, p_two = fisher_exact(table)
        # one-sided: is group A better than B?
        _, p_one = fisher_exact(table, alternative='greater')
        return {"p_two_sided": round(p_two, 6), "p_one_sided": round(p_one, 6)}
    except ImportError:
        # Manual Fisher exact (fallback)
        from math import comb, factorial
        n = a + b + c + d
        r1 = a + b
        r2 = c + d
        c1 = a + c
        c2 = b + d
        def hypergeom_pmf(x):
            return comb(c1, x) * comb(c2, r1 - x) / comb(n, r1)
        p_val = sum(hypergeom_pmf(x) for x in range(a, min(r1, c1) + 1))
        return {"p_one_sided": round(p_val, 6), "p_two_sided": None, "note": "scipy not available, manual one-sided only"}


def compute_fisher_tests(summaries):
    """Compare SNN vs Baseline and SNN vs High-Temp."""
    tests = {}
    baseline = next((s for s in summaries if s["condition"] == "baseline"), None)
    high_temp = next((s for s in summaries if s["condition"] == "high_temp"), None)
    snn = next((s for s in summaries if s["condition"] == "snn_echo"), None)

    if snn and baseline:
        # SNN > Baseline?
        a = snn["n_solved"]; b = snn["n_trials"] - a
        c = baseline["n_solved"]; d = baseline["n_trials"] - c
        tests["snn_vs_baseline"] = {
            "snn_solved": a, "snn_total": snn["n_trials"],
            "baseline_solved": c, "baseline_total": baseline["n_trials"],
            **fisher_exact_test(a, b, c, d)
        }

    if snn and high_temp:
        # SNN > High-Temp?
        a = snn["n_solved"]; b = snn["n_trials"] - a
        c = high_temp["n_solved"]; d = high_temp["n_trials"] - c
        tests["snn_vs_high_temp"] = {
            "snn_solved": a, "snn_total": snn["n_trials"],
            "high_temp_solved": c, "high_temp_total": high_temp["n_trials"],
            **fisher_exact_test(a, b, c, d)
        }

    if baseline and high_temp:
        # High-Temp > Baseline?
        a = high_temp["n_solved"]; b = high_temp["n_trials"] - a
        c = baseline["n_solved"]; d = baseline["n_trials"] - c
        tests["high_temp_vs_baseline"] = {
            "high_temp_solved": a, "high_temp_total": high_temp["n_trials"],
            "baseline_solved": c, "baseline_total": baseline["n_trials"],
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Phase 51: Statistical Validation — Modified-3 Hanoi\n"
                 "Is SNN-induced solving real? (N=50 per condition)",
                 fontsize=14, fontweight="bold", y=1.02)

    conds = ["baseline", "high_temp", "snn_echo"]
    labels = ["Baseline\n(temp=0.5)", "High-Temp\n(temp=1.2)", "SNN L18-Echo\n(σ=0.02, temp=0.5)"]
    colors = ["#95a5a6", "#f39c12", "#3498db"]

    # Panel 1: Solve Rate (the key metric)
    ax = axes[0]
    solve_rates = [next((s["solve_rate"] for s in summaries if s["condition"]==c), 0) for c in conds]
    n_solved = [next((s["n_solved"] for s in summaries if s["condition"]==c), 0) for c in conds]
    bars = ax.bar(range(3), solve_rates, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    for i, (b, sr, ns) in enumerate(zip(bars, solve_rates, n_solved)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
               f"{sr:.1f}%\n({ns}/50)", ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("① Solve Rate (Fisher's Exact Test)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Add Fisher p-values
    if "snn_vs_baseline" in fisher_tests:
        p = fisher_tests["snn_vs_baseline"]["p_one_sided"]
        sig = "★★★" if p < 0.001 else ("★★" if p < 0.01 else ("★" if p < 0.05 else "n.s."))
        ax.text(1, max(solve_rates)*0.7,
               f"SNN vs Baseline: p={p:.4f} {sig}\n"
               f"SNN vs High-Temp: p={fisher_tests.get('snn_vs_high_temp',{}).get('p_one_sided','N/A'):.4f}",
               ha='center', fontsize=9, style='italic',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel 2: Legal vs Illegal moves
    ax = axes[1]
    legal = [next((s["avg_legal"] for s in summaries if s["condition"]==c), 0) for c in conds]
    illegal = [next((s["avg_illegal"] for s in summaries if s["condition"]==c), 0) for c in conds]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, legal, w, label="Legal", color="#2ecc71", alpha=0.85)
    ax.bar(x + w/2, illegal, w, label="Illegal", color="#e74c3c", alpha=0.85)
    for i, (l, il) in enumerate(zip(legal, illegal)):
        ax.text(i-w/2, l+0.3, f"{l:.1f}", ha='center', fontsize=9)
        ax.text(i+w/2, il+0.3, f"{il:.1f}", ha='center', fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Average Moves", fontsize=12)
    ax.set_title("② Legal vs Illegal Moves", fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 3: Self-corrections
    ax = axes[2]
    sc = [next((s["avg_self_corr"] for s in summaries if s["condition"]==c), 0) for c in conds]
    bars = ax.bar(range(3), sc, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    for b, v in zip(bars, sc):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
               f"{v:.1f}", ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Avg Self-Corrections", fontsize=12)
    ax.set_title("③ Self-Corrections (Apple: 'cannot self-correct')", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase51_stats_validation.png")
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

    model, tok = load_model()

    # 3 conditions, all on Modified-3
    conditions = [
        ("baseline",  0.5),    # Standard temperature, no SNN
        ("high_temp", 1.2),    # High temperature, no SNN (control for "just randomness")
        ("snn_echo",  0.5),    # SNN L18-Echo, standard temperature
    ]

    all_sum = []
    all_det = {}

    for cond_name, temp in conditions:
        print(f"\n{'═'*60}")
        print(f"  🏰 Modified-3 Hanoi | {cond_name} | temp={temp}")
        print(f"{'═'*60}")
        s, d = run_condition(model, tok, cond_name, temperature=temp)
        all_sum.append(s)
        all_det[cond_name] = d

    # Fisher's exact tests
    fisher_tests = compute_fisher_tests(all_sum)

    elapsed = time.time() - t0
    fig = visualize(all_sum, fisher_tests)

    # Save results
    out = {
        "experiment": "Phase 51: Statistical Validation",
        "model": MODEL_SHORT,
        "task": "Modified-3 Hanoi (reversed rules)",
        "purpose": "Scale N=10→50 to test statistical significance of SNN-only solve",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_sum,
        "fisher_tests": fisher_tests,
        "figure": fig,
    }
    log = os.path.join(RESULTS_DIR, "phase51_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # ═══ VERDICT ═══
    print(f"\n{'═'*70}")
    print(f"  ⚡ PHASE 51: STATISTICAL VALIDATION — VERDICT")
    print(f"{'═'*70}")
    print(f"  Task: Modified-3 Hanoi | N={N_TRIALS} per condition")
    print(f"{'─'*70}")
    print(f"  {'Condition':<15} {'Temp':>6} {'Solved':>8} {'Solve%':>8} "
          f"{'Legal':>8} {'Illegal':>8} {'SelfCorr':>8}")
    print(f"{'─'*70}")
    for s in all_sum:
        print(f"  {s['condition']:<15} {s['temperature']:>5.1f} "
              f"{s['n_solved']:>4}/{s['n_trials']:<3} {s['solve_rate']:>7.1f}% "
              f"{s['avg_legal']:>8.1f} {s['avg_illegal']:>8.1f} "
              f"{s['avg_self_corr']:>8.1f}")
    print(f"{'─'*70}")

    print(f"\n  🧪 Fisher's Exact Tests:")
    for name, test in fisher_tests.items():
        p = test.get("p_one_sided", "N/A")
        sig = ""
        if isinstance(p, float):
            if p < 0.001: sig = " ★★★ (p<0.001)"
            elif p < 0.01: sig = " ★★ (p<0.01)"
            elif p < 0.05: sig = " ★ (p<0.05)"
            else: sig = " n.s."
        print(f"    {name}: p={p}{sig}")

    print(f"\n  ⏱ {elapsed/60:.1f} min | 💾 {log}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
