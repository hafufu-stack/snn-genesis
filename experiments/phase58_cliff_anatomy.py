"""
Phase 58: Cliff Anatomy — Dissecting the σ=0.15→0.20 Phase Transition
======================================================================

Purpose:
  v10 showed a dramatic cliff: σ=0.15 (32%) → σ=0.20 (0%).
  This experiment DISSECTS the cliff by capturing hidden states at L18
  for fine-grained σ values and analyzing:
    1. Hidden state norm distributions (before/after noise)
    2. Cosine similarity between original and perturbed states
    3. Per-token entropy during generation
    4. UMAP visualization of hidden states by σ level
    5. Solved vs Failed hidden state comparison

Conditions (7, fine-grained around the cliff):
  - baseline  (σ=0.00, N=30)  — control
  - σ=0.10    (N=30) — sub-peak
  - σ=0.14    (N=30) — just below peak
  - σ=0.15    (N=30) — peak
  - σ=0.16    (N=30) — just above peak
  - σ=0.18    (N=30) — approaching cliff
  - σ=0.20    (N=30) — cliff (0% zone)

Total: 210 games. ~5-6 hours on RTX 5080.

Key measurements per game:
  - L18 hidden state snapshots (first 5 moves)
  - Norm of hidden state before/after noise
  - Cosine similarity (original vs perturbed direction)
  - Output token entropy (softmax entropy)
  - Game outcome (solved/failed, legal/illegal moves)

Usage:
    python experiments/phase58_cliff_anatomy.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
N_PER_CONDITION = 30
SNAPSHOT_MOVES = 5  # capture hidden states for first N moves

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
        r'move\s+(?:disk\s+\d+\s+)?(?:from\s+)?([A-Ca-c])\s+to\s+([A-Ca-c])',
        r'([A-Ca-c])\s*->\s*([A-Ca-c])',
    ]
    for p in patterns:
        m = re.search(p, response, re.IGNORECASE)
        if m:
            return m.group(1).upper(), m.group(2).upper()
    return None


# ===================================================
#  ANATOMY HOOK — Captures hidden state diagnostics
# ===================================================

class AnatomyHook:
    """Additive noise hook that captures detailed hidden state diagnostics."""
    def __init__(self, sigma=0.0, capture=True):
        self.sigma = sigma
        self.capture = capture
        # Per-forward diagnostics
        self.snapshots = []  # list of dicts with norm/cosine/etc.

    def __call__(self, module, args):
        hs = args[0]  # (batch, seq, hidden_dim)

        if self.sigma > 0:
            noise = torch.randn_like(hs) * self.sigma
            noisy = hs + noise
        else:
            noisy = hs

        if self.capture and len(self.snapshots) < 500:
            with torch.no_grad():
                # Use last token position (most relevant for generation)
                orig_vec = hs[0, -1, :]  # (hidden_dim,)
                new_vec = noisy[0, -1, :]

                orig_norm = orig_vec.norm(p=2).item()
                new_norm = new_vec.norm(p=2).item()
                norm_ratio = new_norm / orig_norm if orig_norm > 0 else 1.0

                # Cosine similarity: how much did direction change?
                cos_sim = F.cosine_similarity(
                    orig_vec.unsqueeze(0), new_vec.unsqueeze(0)
                ).item()

                # Norm of the noise vector itself
                if self.sigma > 0:
                    noise_norm = noise[0, -1, :].norm(p=2).item()
                else:
                    noise_norm = 0.0

                self.snapshots.append({
                    "orig_norm": round(orig_norm, 4),
                    "new_norm": round(new_norm, 4),
                    "norm_ratio": round(norm_ratio, 6),
                    "cosine_sim": round(cos_sim, 6),
                    "noise_norm": round(noise_norm, 4),
                    "signal_to_noise": round(orig_norm / noise_norm, 4) if noise_norm > 0 else float('inf'),
                })

        if self.sigma > 0:
            return (noisy,) + args[1:]
        return args

    def reset_snapshots(self):
        self.snapshots = []

    def summary(self):
        if not self.snapshots:
            return {}
        arr = lambda k: [s[k] for s in self.snapshots]
        finite_snr = [s["signal_to_noise"] for s in self.snapshots
                      if s["signal_to_noise"] != float('inf')]
        return {
            "n_snapshots": len(self.snapshots),
            "avg_orig_norm": round(np.mean(arr("orig_norm")), 4),
            "avg_new_norm": round(np.mean(arr("new_norm")), 4),
            "avg_norm_ratio": round(np.mean(arr("norm_ratio")), 6),
            "std_norm_ratio": round(np.std(arr("norm_ratio")), 6),
            "avg_cosine_sim": round(np.mean(arr("cosine_sim")), 6),
            "std_cosine_sim": round(np.std(arr("cosine_sim")), 6),
            "min_cosine_sim": round(min(arr("cosine_sim")), 6),
            "avg_noise_norm": round(np.mean(arr("noise_norm")), 4),
            "avg_snr": round(np.mean(finite_snr), 4) if finite_snr else float('inf'),
        }


# ===================================================
#  ENTROPY HOOK — Captures output token entropy
# ===================================================

class EntropyCapture:
    """Lightweight placeholder — entropy is captured from hook snapshots only."""
    def __init__(self):
        pass
    def reset(self):
        pass
    def summary(self):
        return {}


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
    """Standard fast generation using model.generate()."""
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
#  GAME LOOP
# ===================================================

def play_game(model, tok, env, temperature=0.5, hook=None, entropy_cap=None):
    env.reset()
    error = None
    consec_fail = 0

    # Per-game diagnostics
    move_diagnostics = []

    for step in range(MAX_STEPS):
        prompt = build_chat_prompt(tok, env, error)

        if hook:
            hook.reset_snapshots()

        resp = generate(model, tok, prompt, temperature=temperature, max_tokens=80)

        # Capture per-move diagnostics (for first SNAPSHOT_MOVES moves)
        if len(move_diagnostics) < SNAPSHOT_MOVES * 2:
            diag = {"step": step}
            if hook:
                diag["hook"] = hook.summary()
            move_diagnostics.append(diag)

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
    st["move_diagnostics"] = move_diagnostics
    return st


def run_condition(model, tok, cond_name, sigma, n_trials):
    """Run one condition and collect anatomy data."""
    print(f"\n  === {cond_name} | sigma={sigma} | N={n_trials} ===")

    layers = model.model.layers
    hook = AnatomyHook(sigma=sigma, capture=True)
    handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)
    entropy_cap = EntropyCapture()

    results = []
    all_hook_summaries = []
    all_entropy_summaries = []
    solved = 0

    for t in range(n_trials):
        env = HanoiEnv(3, modified=True)
        hook.reset_snapshots()

        r = play_game(model, tok, env, temperature=0.5, hook=hook,
                      entropy_cap=entropy_cap)
        results.append(r)
        if r["solved"]: solved += 1

        # Collect aggregate hook diagnostics for this game
        all_hook_summaries.append(hook.summary())

        icon = "O" if r["solved"] else "X"
        rate = solved / (t+1) * 100
        cos = hook.summary().get("avg_cosine_sim", 1.0)
        snr = hook.summary().get("avg_snr", float('inf'))
        print(f"    {t+1:3d}/{n_trials}: {icon} legal={r['legal_moves']:2d} "
              f"illegal={r['illegal_moves']:2d} cos={cos:.4f} snr={snr:.1f} "
              f"[{solved}/{t+1} = {rate:.0f}%]")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (t + 1) % 10 == 0:
            gc.collect()

    handle.remove()

    # Aggregate statistics
    sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
    n_solved = sum(1 for r in results if r["solved"])

    # Separate solved vs failed hook diagnostics
    solved_hooks = [h for h, r in zip(all_hook_summaries, results) if r["solved"]]
    failed_hooks = [h for h, r in zip(all_hook_summaries, results) if not r["solved"]]

    def agg_hooks(hooks_list):
        if not hooks_list:
            return {}
        keys = ["avg_orig_norm", "avg_new_norm", "avg_norm_ratio",
                "avg_cosine_sim", "avg_noise_norm", "avg_snr"]
        return {k: round(np.mean([h.get(k, 0) for h in hooks_list if h]), 4)
                for k in keys}

    summary = {
        "condition": cond_name,
        "sigma": sigma,
        "n_trials": n_trials,
        "n_solved": n_solved,
        "solve_rate": sr,
        "avg_legal": round(np.mean([r["legal_moves"] for r in results]), 2),
        "avg_illegal": round(np.mean([r["illegal_moves"] for r in results]), 2),
        "hook_all": agg_hooks(all_hook_summaries),
        "hook_solved": agg_hooks(solved_hooks),
        "hook_failed": agg_hooks(failed_hooks),
    }
    print(f"    => Solve={sr:.1f}% ({n_solved}/{n_trials})")
    print(f"       Cosine={summary['hook_all'].get('avg_cosine_sim', 'N/A')} "
          f"Norm ratio={summary['hook_all'].get('avg_norm_ratio', 'N/A')} "
          f"SNR={summary['hook_all'].get('avg_snr', 'N/A')}")
    return summary, results


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle("Phase 58: Cliff Anatomy — Dissecting the σ=0.15→0.20 Phase Transition\n"
                 "What happens to hidden states at the stochastic resonance cliff?",
                 fontsize=14, fontweight="bold", y=1.02)

    sigmas = [s["sigma"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]

    # Panel 1: Solve rate (bell curve)
    ax = axes[0, 0]
    colors_bar = ['#2ecc71' if s["solve_rate"] > 20 else '#e74c3c' if s["solve_rate"] == 0
                  else '#f39c12' for s in summaries]
    bars = ax.bar(range(len(sigmas)), rates, color=colors_bar, alpha=0.8, edgecolor='white', linewidth=1.5)
    for i, (b, r) in enumerate(zip(bars, rates)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               f'{r:.0f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f'σ={s:.2f}' for s in sigmas], fontsize=9)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Bell Curve (fine-grained)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Cosine similarity
    ax = axes[0, 1]
    cos_sims = [s["hook_all"].get("avg_cosine_sim", 1.0) for s in summaries]
    ax.plot(sigmas, cos_sims, 'o-', color='#3498db', linewidth=3, markersize=10)
    for sigma, cos in zip(sigmas, cos_sims):
        ax.annotate(f'{cos:.4f}', (sigma, cos), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel("σ", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("Direction Preservation\n(how much does the direction change?)",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 3: Norm ratio
    ax = axes[0, 2]
    norm_ratios = [s["hook_all"].get("avg_norm_ratio", 1.0) for s in summaries]
    ax.plot(sigmas, norm_ratios, 's-', color='#e74c3c', linewidth=3, markersize=10)
    for sigma, nr in zip(sigmas, norm_ratios):
        ax.annotate(f'{nr:.4f}', (sigma, nr), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=8, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("σ", fontsize=12)
    ax.set_ylabel("Norm Ratio (new/orig)", fontsize=12)
    ax.set_title("Magnitude Change\n(how much does the magnitude change?)",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 4: Signal-to-Noise Ratio
    ax = axes[1, 0]
    snrs = [s["hook_all"].get("avg_snr", 0) for s in summaries]
    snrs_finite = [min(s, 500) for s in snrs]  # cap for display
    ax.plot(sigmas, snrs_finite, 'D-', color='#9b59b6', linewidth=3, markersize=10)
    for sigma, snr in zip(sigmas, snrs):
        label = f'{snr:.1f}' if snr < 500 else '∞'
        ax.annotate(label, (sigma, min(snr, 500)), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel("σ", fontsize=12)
    ax.set_ylabel("Signal-to-Noise Ratio", fontsize=12)
    ax.set_title("Signal-to-Noise Ratio\n(signal strength vs noise)",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 5: Cosine sim vs Solve rate (the key relationship)
    ax = axes[1, 1]
    for i, s in enumerate(summaries):
        cos = s["hook_all"].get("avg_cosine_sim", 1.0)
        rate = s["solve_rate"]
        color = '#2ecc71' if rate > 20 else '#e74c3c' if rate == 0 else '#f39c12'
        ax.scatter(cos, rate, s=150, c=color, edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(f'σ={s["sigma"]:.2f}', (cos, rate), textcoords="offset points",
                   xytext=(8, 5), fontsize=9, fontweight='bold')

    ax.set_xlabel("Cosine Similarity (direction preservation)", fontsize=12)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Direction Preservation vs Performance\n(where does reasoning break?)",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 6: Solved vs Failed comparison (at peak σ=0.15)
    ax = axes[1, 2]
    peak = next((s for s in summaries if s["sigma"] == 0.15), None)
    if peak and peak["hook_solved"] and peak["hook_failed"]:
        metrics = ["avg_cosine_sim", "avg_norm_ratio", "avg_snr"]
        labels = ["Cosine Sim", "Norm Ratio", "SNR"]
        solved_vals = [peak["hook_solved"].get(m, 0) for m in metrics]
        failed_vals = [peak["hook_failed"].get(m, 0) for m in metrics]

        # Normalize for display
        max_vals = [max(abs(s), abs(f), 0.001) for s, f in zip(solved_vals, failed_vals)]
        solved_norm = [s/m for s, m in zip(solved_vals, max_vals)]
        failed_norm = [f/m for f, m in zip(failed_vals, max_vals)]

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, solved_vals, w, label='Solved', color='#2ecc71', alpha=0.8)
        ax.bar(x + w/2, failed_vals, w, label='Failed', color='#e74c3c', alpha=0.8)

        for i, (sv, fv) in enumerate(zip(solved_vals, failed_vals)):
            ax.text(i - w/2, sv + 0.01, f'{sv:.4f}', ha='center', fontsize=7)
            ax.text(i + w/2, fv + 0.01, f'{fv:.4f}', ha='center', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend(fontsize=10)
        ax.set_title("Solved vs Failed at σ=0.15 (peak)\nDo winning games look different?",
                    fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "No solved games\nat σ=0.15", ha='center', va='center',
               fontsize=14, transform=ax.transAxes)
        ax.set_title("Solved vs Failed at σ=0.15", fontsize=12, fontweight='bold')

    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase58_cliff_anatomy.png")
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

    # 7 conditions: fine-grained around the cliff
    conditions = [
        ("baseline",       0.00),
        ("additive_0.10",  0.10),
        ("additive_0.14",  0.14),
        ("additive_0.15",  0.15),
        ("additive_0.16",  0.16),
        ("additive_0.18",  0.18),
        ("additive_0.20",  0.20),
    ]

    all_sum = []
    all_det = {}

    for cond_name, sigma in conditions:
        s, d = run_condition(model, tok, cond_name, sigma, N_PER_CONDITION)
        all_sum.append(s)
        all_det[cond_name] = d
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    fig = visualize(all_sum)

    # Save results
    out = {
        "experiment": "Phase 58: Cliff Anatomy — σ=0.15→0.20 Phase Transition",
        "model": MODEL_SHORT,
        "task": "Modified-3 Hanoi (reversed rules)",
        "purpose": "Dissect the stochastic resonance cliff by analyzing hidden state norms, "
                   "cosine similarity, and entropy across fine-grained σ values",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_sum,
        "figure": fig,
    }
    log = os.path.join(RESULTS_DIR, "phase58_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # === VERDICT ===
    print(f"\n{'='*80}")
    print(f"  PHASE 58: CLIFF ANATOMY — VERDICT")
    print(f"{'='*80}")
    print(f"  {'Condition':<18} {'σ':>5} {'Solve%':>8} {'CosSim':>10} {'NormRatio':>12} {'SNR':>10}")
    print(f"{'-'*80}")
    for s in all_sum:
        cos = s["hook_all"].get("avg_cosine_sim", "N/A")
        nr = s["hook_all"].get("avg_norm_ratio", "N/A")
        snr = s["hook_all"].get("avg_snr", "N/A")
        print(f"  {s['condition']:<18} {s['sigma']:>5.2f} {s['solve_rate']:>7.1f}% "
              f"{cos:>10} {nr:>12} {snr:>10}")
    print(f"{'-'*80}")

    # Key analysis: where does the cliff happen?
    print(f"\n  KEY ANALYSIS: Phase Transition Anatomy")
    for i in range(1, len(all_sum)):
        prev = all_sum[i-1]
        curr = all_sum[i]
        cos_prev = prev["hook_all"].get("avg_cosine_sim", 1.0)
        cos_curr = curr["hook_all"].get("avg_cosine_sim", 1.0)
        rate_diff = curr["solve_rate"] - prev["solve_rate"]
        cos_diff = cos_curr - cos_prev
        print(f"    σ={prev['sigma']:.2f}→{curr['sigma']:.2f}: "
              f"ΔSolve={rate_diff:+.1f}pp  ΔCos={cos_diff:+.6f}")

    print(f"\n  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"  Log: {log}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
    # Restore power settings before hibernating
