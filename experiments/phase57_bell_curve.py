"""
Phase 57: Bell Curve Completion + Large-N Replication
=====================================================

Purpose:
  Phase 56 confirmed:
    - Additive sigma=0.15 -> 38% (p=0.0006, ***)
    - Radial (magnitude-only) -> no effect
    - Direction x Magnitude interaction required

  This experiment completes the stochastic resonance bell curve:
    - Large-N replication of the peak (sigma=0.15, N=100)
    - Right-side cliff detection (sigma=0.20, 0.25)
    - Baseline stability check (N=100)

Conditions (4):
  1. baseline        (N=100) -- stability check
  2. additive_0.15   (N=100) -- peak replication
  3. additive_0.20   (N=100) -- right-side cliff?
  4. additive_0.25   (N=50)  -- collapse confirmation

Task: Modified-3 Hanoi (same as Phase 51/55/56)
Expected runtime: ~10-11 hours on single GPU (Mistral-7B 4-bit)

Usage:
    python experiments/phase57_bell_curve.py
"""

import torch
import torch.nn as nn
import os, json, gc, time, random, re, copy
import numpy as np
import warnings
warnings.filterwarnings("ignore")


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# === Config ===
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
TARGET_LAYER = 18
SEED = 2026
MAX_STEPS = 50

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
#  PROMPT BUILDER (Phase 55 compatible)
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


# ===================================================
#  MOVE PARSER
# ===================================================

def parse_move(response):
    patterns = [
        r'Move:\s*([A-Ca-c])\s*->\s*([A-Ca-c])',
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
#  SNN HOOK (Additive only for Phase 57)
# ===================================================

class AdditiveEchoHook:
    """Original additive noise. hs + N(0, sigma^2)."""
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


def generate(model, tok, prompt, temperature=0.5, max_tokens=100):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperature, top_p=0.9, top_k=40,
            repetition_penalty=1.2, pad_token_id=tok.pad_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    inp = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(inp):].strip()


# ===================================================
#  GAME LOOP
# ===================================================

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


def run_condition(model, tok, cond_name, sigma, n_trials,
                  existing_results=None):
    """Run one experimental condition with per-game checkpointing."""
    start_from = len(existing_results) if existing_results else 0
    if start_from > 0:
        print(f"\n  Resuming {cond_name} from game {start_from+1}/{n_trials}")
    print(f"\n  Mod-3 | {cond_name} | sigma={sigma} | {n_trials} trials")

    layers = model.model.layers
    hook, handle = None, None

    if sigma > 0:
        hook = AdditiveEchoHook(sigma)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    results = list(existing_results) if existing_results else []
    solved = sum(1 for r in results if r.get("solved"))

    ckpt_path = os.path.join(RESULTS_DIR, "phase57_checkpoint.json")

    for t in range(start_from, n_trials):
        try:
            env = HanoiEnv(3, modified=True)
            r = play_game(model, tok, env, temperature=0.5, hook=hook)
            results.append(r)
            if r["solved"]: solved += 1
            icon = "O" if r["solved"] else "X"
            rate = solved / (t+1) * 100
            print(f"    {t+1:3d}/{n_trials}: {icon} legal={r['legal_moves']:2d} "
                  f"illegal={r['illegal_moves']:2d} self_corr={r['self_corrections']:2d}  "
                  f"[running: {solved}/{t+1} = {rate:.0f}%]")
        except Exception as e:
            print(f"    {t+1:3d}/{n_trials}: ERROR: {e}")
            results.append({"solved": False, "legal_moves": 0, "illegal_moves": 0,
                           "self_corrections": 0, "optimal": 26, "thoughts": []})

        # Save mini-checkpoint after EVERY game
        try:
            with open(ckpt_path, "r") as f:
                ckpt = json.load(f)
        except:
            ckpt = {"experiment": "Phase 57 (checkpoint)",
                    "completed_conditions": [], "summaries": [],
                    "in_progress": {}}
        ckpt["in_progress"] = {
            "condition": cond_name, "sigma": sigma,
            "completed_games": t + 1, "total_games": n_trials,
            "results": results
        }
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2, default=str)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (t + 1) % 10 == 0:
            gc.collect()

    if handle: handle.remove()

    s = lambda key: round(np.mean([r[key] for r in results]), 2)
    sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
    n_solved = sum(1 for r in results if r["solved"])

    hook_diagnostics = {}
    if hook and hook.norm_deltas:
        hook_diagnostics["avg_norm_ratio"] = round(np.mean(hook.norm_deltas), 6)
        hook_diagnostics["std_norm_ratio"] = round(np.std(hook.norm_deltas), 6)

    summary = {
        "condition": cond_name, "sigma": sigma,
        "n_trials": n_trials, "n_solved": n_solved,
        "solve_rate": sr,
        "avg_legal": s("legal_moves"), "avg_illegal": s("illegal_moves"),
        "avg_self_corr": s("self_corrections"),
        "optimal": results[0]["optimal"],
        "hook_diagnostics": hook_diagnostics,
    }
    print(f"    Result: Solve={sr:.1f}% ({n_solved}/{n_trials}) "
          f"Legal={summary['avg_legal']:.1f} Illegal={summary['avg_illegal']:.1f}")
    return summary, results


# ===================================================
#  FISHER'S EXACT TEST
# ===================================================

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


def compute_fisher_tests(summaries):
    tests = {}
    baseline = next((s for s in summaries if s["condition"] == "baseline"), None)
    if not baseline:
        return tests

    for s in summaries:
        if s["condition"] == "baseline":
            continue
        name = f"{s['condition']}_vs_baseline"
        a, b = s["n_solved"], s["n_trials"] - s["n_solved"]
        c, d = baseline["n_solved"], baseline["n_trials"] - baseline["n_solved"]
        result = fisher_exact_test(a, b, c, d)
        result["description"] = f"{s['condition']} ({s['solve_rate']}%) vs baseline ({baseline['solve_rate']}%)"
        tests[name] = result

    # Compare 0.15 vs 0.20 (is there a cliff?)
    a15 = next((s for s in summaries if s["condition"] == "additive_0.15"), None)
    a20 = next((s for s in summaries if s["condition"] == "additive_0.20"), None)
    if a15 and a20:
        a, b = a15["n_solved"], a15["n_trials"] - a15["n_solved"]
        c, d = a20["n_solved"], a20["n_trials"] - a20["n_solved"]
        result = fisher_exact_test(a, b, c, d)
        result["description"] = f"Cliff test: 0.15 ({a15['solve_rate']}%) vs 0.20 ({a20['solve_rate']}%)"
        tests["additive_0.15_vs_0.20"] = result

    return tests


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(summaries, fisher_tests, phase56_data=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Phase 57: Bell Curve Completion + Large-N Replication\n"
                 "Mapping the full stochastic resonance curve (N=100 per condition)",
                 fontsize=14, fontweight="bold", y=1.02)

    # Panel 1: Combined dose-response with Phase 56 data
    ax = axes[0]

    # Phase 56 data points (small, transparent)
    if phase56_data:
        p56_sigmas = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15]
        p56_rates = [phase56_data.get(n, 0) for n in
                    ["baseline", "additive_0.05", "additive_0.08",
                     "additive_0.10", "additive_0.12", "additive_0.15"]]
        ax.plot(p56_sigmas, p56_rates, 'o--', color='#e74c3c', alpha=0.3,
                linewidth=1, markersize=5, label='Phase 56 (N=50)')

    # Phase 57 data (main)
    p57_sigmas = []
    p57_rates = []
    p57_ns = []
    for s in sorted(summaries, key=lambda x: x["sigma"]):
        p57_sigmas.append(s["sigma"])
        p57_rates.append(s["solve_rate"])
        p57_ns.append(s["n_trials"])

    ax.plot(p57_sigmas, p57_rates, 'o-', color='#e74c3c', linewidth=3,
            markersize=10, label='Phase 57 (N=100)', zorder=5)

    for i, (sigma, rate, n) in enumerate(zip(p57_sigmas, p57_rates, p57_ns)):
        ax.annotate(f'{rate:.0f}%\n(N={n})', (sigma, rate),
                   textcoords="offset points", xytext=(0, 15),
                   ha='center', fontsize=9, fontweight='bold')

    ax.axhline(y=p57_rates[0] if p57_rates else 0, color='gray',
              linestyle=':', alpha=0.5, label=f'Baseline')
    ax.set_xlabel("sigma (noise amplitude)", fontsize=13)
    ax.set_ylabel("Solve Rate (%)", fontsize=13)
    ax.set_title("Dose-Response Curve (Stochastic Resonance)\n"
                 "Does sigma=0.15 remain the peak? Does sigma=0.20 crash?",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: Bar chart with significance
    ax = axes[1]
    colors = ['#95a5a6', '#a93226', '#7b241c', '#5b1c14']
    conds = [s["condition"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]
    ns = [f"N={s['n_trials']}" for s in summaries]

    bars = ax.bar(range(len(conds)), rates, color=colors[:len(conds)],
                 alpha=0.85, edgecolor='white', linewidth=2)

    for i, (b, r, n_str) in enumerate(zip(bars, rates, ns)):
        s = summaries[i]
        txt = f"{r:.0f}%\n({s['n_solved']}/{s['n_trials']})"
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               txt, ha='center', fontsize=9, fontweight='bold')

        # Add p-value annotation
        test_key = f"{s['condition']}_vs_baseline"
        if test_key in fisher_tests:
            p = fisher_tests[test_key].get("p_two_sided", 1.0)
            if p < 0.001:
                sig_txt = f"p={p:.4f} ***"
            elif p < 0.01:
                sig_txt = f"p={p:.3f} **"
            elif p < 0.05:
                sig_txt = f"p={p:.3f} *"
            else:
                sig_txt = f"p={p:.3f} n.s."
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+6,
                   sig_txt, ha='center', fontsize=7, color='#555')

    labels = [c.replace("additive_", "Add ").replace("baseline", "Baseline") for c in conds]
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Solve Rate (%)", fontsize=13)
    ax.set_title("Phase 57: Large-N Results\nFisher's Exact Test vs Baseline",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase57_bell_curve.png")
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

    # === Checkpoint Resume ===
    ckpt_path = os.path.join(RESULTS_DIR, "phase57_checkpoint.json")
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
            print(f"\n RESUMING from checkpoint!")
            print(f"   Completed: {', '.join(completed_conditions) or 'none'}")
            if in_progress_data:
                ip = in_progress_data
                print(f"   In-progress: {ip['condition']} ({ip['completed_games']}/{ip['total_games']} games)")
            print(f"   Remaining: {4 - len(completed_conditions)} conditions")
        except Exception as e:
            print(f"Checkpoint load failed ({e}), starting fresh")

    model, tok = load_model()

    # 4 conditions: (name, sigma, n_trials)
    conditions = [
        ("baseline",       0.0,  100),
        ("additive_0.15",  0.15, 100),
        ("additive_0.20",  0.20, 100),
        ("additive_0.25",  0.25,  50),
    ]

    for cond_name, sigma, n_trials in conditions:
        if cond_name in completed_conditions:
            print(f"\n Skipping {cond_name} (already completed)")
            continue

        existing_results = None
        if in_progress_data and in_progress_data.get("condition") == cond_name:
            existing_results = in_progress_data.get("results", [])
            print(f"\n Found {len(existing_results)} saved games for {cond_name}")

        print(f"\n{'='*60}")
        print(f"  Modified-3 Hanoi | {cond_name} | N={n_trials}")
        print(f"{'='*60}")
        s, d = run_condition(model, tok, cond_name, sigma, n_trials,
                            existing_results=existing_results)
        all_sum.append(s)
        all_det[cond_name] = d

        in_progress_data = {}

        checkpoint = {
            "experiment": "Phase 57 (checkpoint)",
            "completed_conditions": [s["condition"] for s in all_sum],
            "summaries": all_sum,
            "in_progress": {},
        }
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        print(f"  Checkpoint saved: {ckpt_path}")

    # Fisher's exact tests
    fisher_tests = compute_fisher_tests(all_sum)

    # Load Phase 56 data for combined graph
    phase56_data = None
    p56_path = os.path.join(RESULTS_DIR, "phase56_log.json")
    if os.path.exists(p56_path):
        try:
            p56 = json.load(open(p56_path))
            phase56_data = {c["condition"]: c["solve_rate"] for c in p56.get("conditions", [])}
            print("\n  Loaded Phase 56 data for combined visualization")
        except:
            pass

    elapsed = time.time() - t0
    fig = visualize(all_sum, fisher_tests, phase56_data)

    # Save results
    out = {
        "experiment": "Phase 57: Bell Curve Completion + Large-N Replication",
        "model": MODEL_SHORT,
        "task": "Modified-3 Hanoi (reversed rules)",
        "purpose": "Complete the stochastic resonance bell curve and replicate peak at N=100",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_sum,
        "fisher_tests": fisher_tests,
        "figure": fig,
    }
    log = os.path.join(RESULTS_DIR, "phase57_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # === VERDICT ===
    print(f"\n{'='*75}")
    print(f"  PHASE 57: BELL CURVE + LARGE-N -- VERDICT")
    print(f"{'='*75}")
    print(f"  Task: Modified-3 Hanoi | 4 conditions")
    print(f"{'-'*75}")
    print(f"  {'Condition':<18} {'sigma':>5} {'N':>5} {'Solved':>8} {'Solve%':>8} "
          f"{'Legal':>8} {'Illegal':>8}")
    print(f"{'-'*75}")
    for s in all_sum:
        print(f"  {s['condition']:<18} {s['sigma']:>5.2f} {s['n_trials']:>5} "
              f"{s['n_solved']:>4}/{s['n_trials']:<3} {s['solve_rate']:>7.1f}% "
              f"{s['avg_legal']:>8.1f} {s['avg_illegal']:>8.1f}")
    print(f"{'-'*75}")

    print(f"\n  Fisher's Exact Tests:")
    for name, test in fisher_tests.items():
        p = test.get("p_two_sided", test.get("p_one_sided", "N/A"))
        sig = ""
        if isinstance(p, float):
            if p < 0.001: sig = " *** (p<0.001)"
            elif p < 0.01: sig = " ** (p<0.01)"
            elif p < 0.05: sig = " * (p<0.05)"
            else: sig = " n.s."
        desc = test.get("description", "")
        print(f"    {name}: {desc} -> p={p}{sig}")

    # Bell curve verdict
    a15 = next((s for s in all_sum if s["condition"] == "additive_0.15"), None)
    a20 = next((s for s in all_sum if s["condition"] == "additive_0.20"), None)
    a25 = next((s for s in all_sum if s["condition"] == "additive_0.25"), None)
    base = next((s for s in all_sum if s["condition"] == "baseline"), None)

    print(f"\n  {'='*60}")
    print(f"  BELL CURVE ANALYSIS")
    if a15 and a20 and base:
        print(f"    Baseline:   {base['solve_rate']:.1f}%")
        print(f"    sigma=0.15: {a15['solve_rate']:.1f}% (peak candidate)")
        print(f"    sigma=0.20: {a20['solve_rate']:.1f}% (post-peak)")
        if a25:
            print(f"    sigma=0.25: {a25['solve_rate']:.1f}% (collapse zone)")

        if a20['solve_rate'] < a15['solve_rate']:
            print(f"\n    BELL CURVE CONFIRMED!")
            print(f"    Peak at sigma=0.15, drops at sigma=0.20")
            print(f"    Classic inverted-U stochastic resonance pattern!")
        elif a20['solve_rate'] >= a15['solve_rate']:
            print(f"\n    Peak NOT at sigma=0.15 -- resonance peak is higher!")
            print(f"    Need to test sigma=0.25, 0.30 to find the true peak")
    print(f"  {'='*60}")

    print(f"\n  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours) | Log: {log}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
    # Auto-hibernate after completion
    time.sleep(10)
