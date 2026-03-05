"""
Phase 63-65 Batch Runner: Flash Annealing, N=100 Validation, Adaptive Cosine Homeostasis
========================================================================================

Runs THREE experiments sequentially with shared model loading:

Phase 63: Flash Annealing — combine Phase 60's linear decay with Phase 62's first-20 window
Phase 64: N=100 Validation — large-sample validation of key conditions
Phase 65: Adaptive Cosine Homeostasis — norm-adaptive σ to maintain target cosine similarity

Total: ~600 games. Estimated ~12-15 hours on RTX 5080.

Usage:
    python experiments/phase63_65_batch.py
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
#  NOISE HOOKS
# ===================================================

class DynamicNoiseHook:
    """Standard noise hook with adjustable sigma."""
    def __init__(self, sigma=0.15):
        self.sigma = sigma
        self.active = False

    def __call__(self, module, args):
        if not self.active or self.sigma <= 0:
            return args
        hs = args[0]
        noise = torch.randn_like(hs) * self.sigma
        return (hs + noise,) + args[1:]


class AdaptiveCosineHook:
    """Norm-adaptive noise hook that maintains target cosine similarity.
    
    Instead of fixed σ, computes the optimal σ for each token based on
    the hidden state norm, so that cos(h, h+noise) ≈ target_cos.
    
    Physics: For h ∈ R^d and noise ~ N(0, σ²I):
        E[cos(h, h+n)] ≈ 1 / sqrt(1 + d·σ²/||h||²)
    
    Solving for σ:
        σ = ||h|| · sqrt((1/target² - 1) / d)
    """
    def __init__(self, target_cos=0.60):
        self.target_cos = target_cos
        self.active = False
        self.last_cos = None  # For logging
        self.last_sigma = None

    def __call__(self, module, args):
        if not self.active:
            return args
        hs = args[0]  # shape: (batch, seq_len, hidden_dim)
        d = hs.shape[-1]
        
        # Compute norm per token, then average
        h_norm = hs.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (batch, seq, 1)
        
        # Compute adaptive σ per token
        cos_target = self.target_cos
        sigma_adaptive = h_norm * math.sqrt(max(1.0 / cos_target**2 - 1.0, 0) / d)
        
        # Add noise with adaptive σ
        noise = torch.randn_like(hs) * sigma_adaptive
        hs_noised = hs + noise
        
        # Log actual cosine similarity (averaged over batch/seq)
        with torch.no_grad():
            cos_actual = F.cosine_similarity(
                hs.reshape(-1, d), hs_noised.reshape(-1, d), dim=-1
            ).mean().item()
            self.last_cos = cos_actual
            self.last_sigma = sigma_adaptive.mean().item()
        
        return (hs_noised,) + args[1:]


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
#  GAME FUNCTIONS
# ===================================================

def play_with_hook(model, tok, env, hook, noise_fn=None):
    """General game player. noise_fn(step, legal_move_count) controls hook state."""
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


def run_condition(model, tok, hook, name, n_trials, noise_fn_factory, extra_info=""):
    """Run a single condition and return summary."""
    print(f"\n  === {name} {extra_info} | N={n_trials} ===")
    results = []; solved = 0

    for t in range(n_trials):
        env = HanoiEnv(3, modified=True)
        noise_fn = noise_fn_factory(hook)
        r = play_with_hook(model, tok, env, hook, noise_fn)
        results.append(r)
        if r["solved"]: solved += 1
        icon = "O" if r["solved"] else "X"
        rate = solved / (t+1) * 100
        print(f"    {t+1:3d}/{n_trials}: {icon} legal={r['legal_moves']:2d} "
              f"illegal={r['illegal_moves']:2d} [{solved}/{t+1} = {rate:.0f}%]")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if (t+1) % 10 == 0: gc.collect()

    sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
    n_solved = sum(1 for r in results if r["solved"])
    print(f"    => Solve={sr:.1f}% ({n_solved}/{n_trials})")
    return {
        "condition": name, "n_trials": n_trials, "n_solved": n_solved,
        "solve_rate": sr,
        "avg_legal": round(np.mean([r["legal_moves"] for r in results]), 2),
        "avg_illegal": round(np.mean([r["illegal_moves"] for r in results]), 2),
    }


# ===================================================
#  PHASE 63: FLASH ANNEALING
# ===================================================

def run_phase63(model, tok):
    """Phase 63: Combine Phase 60's decay schedules with Phase 62's first-20 window."""
    print("\n" + "="*80)
    print("  PHASE 63: FLASH ANNEALING (Duration × Schedule)")
    print("  Combining Phase 60 (decay) with Phase 62 (first-20 window)")
    print("="*80)
    t0 = time.time()
    N = 50

    layers = model.model.layers
    hook = DynamicNoiseHook(sigma=0.15)
    handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    # Noise function factories
    def make_baseline(hook_ref):
        def fn(step, lm):
            hook_ref.active = False
        return fn

    def make_first20_flat(hook_ref):
        def fn(step, lm):
            hook_ref.active = (lm < 20)
            hook_ref.sigma = 0.15
        return fn

    def make_first20_linear(hook_ref):
        def fn(step, lm):
            if lm < 20:
                hook_ref.active = True
                hook_ref.sigma = 0.15 * (1.0 - lm / 20.0)  # 0.15 -> 0
            else:
                hook_ref.active = False
        return fn

    def make_first20_cosine(hook_ref):
        def fn(step, lm):
            if lm < 20:
                hook_ref.active = True
                hook_ref.sigma = 0.15 * (1 + math.cos(math.pi * lm / 20)) / 2
            else:
                hook_ref.active = False
        return fn

    def make_first20_high_start(hook_ref):
        """Start higher (σ=0.20) and decay to 0 over first 20 moves."""
        def fn(step, lm):
            if lm < 20:
                hook_ref.active = True
                hook_ref.sigma = 0.20 * (1.0 - lm / 20.0)
            else:
                hook_ref.active = False
        return fn

    def make_first10_linear(hook_ref):
        """Fast flash: linear decay over first 10 moves only."""
        def fn(step, lm):
            if lm < 10:
                hook_ref.active = True
                hook_ref.sigma = 0.15 * (1.0 - lm / 10.0)
            else:
                hook_ref.active = False
        return fn

    def make_always_linear(hook_ref):
        """Full-game linear decay (Phase 60 control)."""
        def fn(step, lm):
            hook_ref.active = True
            hook_ref.sigma = 0.15 * (1.0 - lm / MAX_STEPS)
        return fn

    conditions = [
        ("baseline",            make_baseline),
        ("first20_flat",        make_first20_flat),
        ("first20_linear",      make_first20_linear),
        ("first20_cosine",      make_first20_cosine),
        ("first20_high_σ0.20",  make_first20_high_start),
        ("first10_linear",      make_first10_linear),
        ("full_linear_decay",   make_always_linear),
    ]

    summaries = []
    for name, factory in conditions:
        s = run_condition(model, tok, hook, name, N, factory)
        summaries.append(s)

    handle.remove()
    elapsed = time.time() - t0

    # Visualization
    fig_path = visualize_phase63(summaries)

    # Print verdict
    bl = next(s for s in summaries if s["condition"] == "baseline")
    print(f"\n{'='*70}")
    print(f"  PHASE 63 VERDICT: Flash Annealing")
    print(f"{'='*70}")
    for s in summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<22} {s['solve_rate']:>6.1f}%   (baseline)")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {s['condition']:<22} {s['solve_rate']:>6.1f}%   p={p:.4f} {sig}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"{'='*70}")

    # Save
    out = {
        "experiment": "Phase 63: Flash Annealing (Duration × Schedule)",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "purpose": "Combine Phase 60 (decay schedules) with Phase 62 (first-20 window)",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": summaries, "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase63_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")
    return summaries


# ===================================================
#  PHASE 64: N=100 VALIDATION
# ===================================================

def run_phase64(model, tok):
    """Phase 64: Large-sample validation of key conditions."""
    print("\n" + "="*80)
    print("  PHASE 64: N=100 VALIDATION")
    print("  Large-sample validation of best conditions")
    print("="*80)
    t0 = time.time()
    N = 100

    layers = model.model.layers
    hook = DynamicNoiseHook(sigma=0.15)
    handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    def make_baseline(hook_ref):
        def fn(step, lm):
            hook_ref.active = False
        return fn

    def make_first20_flat(hook_ref):
        def fn(step, lm):
            hook_ref.active = (lm < 20)
            hook_ref.sigma = 0.15
        return fn

    def make_first20_linear(hook_ref):
        def fn(step, lm):
            if lm < 20:
                hook_ref.active = True
                hook_ref.sigma = 0.15 * (1.0 - lm / 20.0)
            else:
                hook_ref.active = False
        return fn

    conditions = [
        ("baseline",        make_baseline),
        ("first20_flat",    make_first20_flat),
        ("first20_linear",  make_first20_linear),
    ]

    summaries = []
    for name, factory in conditions:
        s = run_condition(model, tok, hook, name, N, factory)
        summaries.append(s)

    handle.remove()
    elapsed = time.time() - t0

    # Visualization
    fig_path = visualize_phase64(summaries)

    # Print verdict
    bl = next(s for s in summaries if s["condition"] == "baseline")
    print(f"\n{'='*70}")
    print(f"  PHASE 64 VERDICT: N=100 Validation")
    print(f"{'='*70}")
    for s in summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<22} {s['solve_rate']:>6.1f}%   (N={N} baseline)")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {s['condition']:<22} {s['solve_rate']:>6.1f}%   p={p:.6f} {sig}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"{'='*70}")

    # Save
    out = {
        "experiment": "Phase 64: N=100 Validation",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "purpose": "Large-sample validation of best conditions from Phase 62-63",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": summaries, "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase64_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")
    return summaries


# ===================================================
#  PHASE 65: ADAPTIVE COSINE HOMEOSTASIS
# ===================================================

def run_phase65(model, tok):
    """Phase 65: Norm-adaptive σ to maintain target cosine similarity."""
    print("\n" + "="*80)
    print("  PHASE 65: ADAPTIVE COSINE HOMEOSTASIS")
    print("  Norm-adaptive σ maintains target cos(h, h+noise)")
    print("="*80)
    t0 = time.time()
    N = 50

    layers = model.model.layers

    # --- Fixed σ conditions use DynamicNoiseHook ---
    hook_fixed = DynamicNoiseHook(sigma=0.15)

    def make_baseline(hook_ref):
        def fn(step, lm):
            hook_ref.active = False
        return fn

    def make_always_fixed(hook_ref):
        def fn(step, lm):
            hook_ref.active = True
            hook_ref.sigma = 0.15
        return fn

    # Run fixed conditions first
    handle_fixed = layers[TARGET_LAYER].register_forward_pre_hook(hook_fixed)

    summaries = []
    for name, factory in [("baseline", make_baseline), ("always_fixed_0.15", make_always_fixed)]:
        s = run_condition(model, tok, hook_fixed, name, N, factory)
        summaries.append(s)

    handle_fixed.remove()

    # --- Adaptive conditions use AdaptiveCosineHook ---
    for target_cos in [0.55, 0.60, 0.65, 0.70]:
        hook_adaptive = AdaptiveCosineHook(target_cos=target_cos)
        handle_adaptive = layers[TARGET_LAYER].register_forward_pre_hook(hook_adaptive)

        def make_adaptive_always(hook_ref):
            def fn(step, lm):
                hook_ref.active = True
            return fn

        name = f"adaptive_cos_{target_cos:.2f}"
        s = run_condition(model, tok, hook_adaptive, name, N, make_adaptive_always)
        summaries.append(s)

        handle_adaptive.remove()

    # Adaptive + first 20 window (best target from above decided at cos=0.60)
    hook_adaptive20 = AdaptiveCosineHook(target_cos=0.60)
    handle_adaptive20 = layers[TARGET_LAYER].register_forward_pre_hook(hook_adaptive20)

    def make_adaptive_first20(hook_ref):
        def fn(step, lm):
            hook_ref.active = (lm < 20)
        return fn

    s = run_condition(model, tok, hook_adaptive20, "adaptive_cos0.60_first20", N, make_adaptive_first20)
    summaries.append(s)

    handle_adaptive20.remove()

    elapsed = time.time() - t0

    # Visualization
    fig_path = visualize_phase65(summaries)

    # Print verdict
    bl = next(s for s in summaries if s["condition"] == "baseline")
    print(f"\n{'='*70}")
    print(f"  PHASE 65 VERDICT: Adaptive Cosine Homeostasis")
    print(f"{'='*70}")
    for s in summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<28} {s['solve_rate']:>6.1f}%   (baseline)")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {s['condition']:<28} {s['solve_rate']:>6.1f}%   p={p:.4f} {sig}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"{'='*70}")

    # Save
    out = {
        "experiment": "Phase 65: Adaptive Cosine Homeostasis",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "purpose": "Norm-adaptive σ to maintain target cosine similarity between original and noised hidden states",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": summaries, "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase65_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")
    return summaries


# ===================================================
#  VISUALIZATIONS
# ===================================================

def visualize_phase63(summaries):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 63: Flash Annealing (Duration × Schedule)\n"
                 "Combining Phase 60's decay with Phase 62's first-20 window",
                 fontsize=14, fontweight="bold", y=1.02)

    names = [s["condition"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]
    bl = next(s for s in summaries if s["condition"] == "baseline")

    # Panel 1: Bar chart
    ax = axes[0]
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#2ecc71', '#1abc9c']
    bars = ax.bar(range(len(summaries)), rates, color=colors[:len(summaries)], alpha=0.85,
                  edgecolor='white', linewidth=2)
    for i, (b, s) in enumerate(zip(bars, summaries)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               f'{s["solve_rate"]:.0f}%', ha='center', fontsize=10, fontweight='bold')
        if i > 0:
            table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                     [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
            _, p = fisher_exact(table, alternative='two-sided')
            sig = "★★★" if p < 0.001 else "★★" if p < 0.01 else "★" if p < 0.05 else "n.s."
            ax.text(i, s["solve_rate"]+4, sig, ha='center', fontsize=8,
                   color='darkred' if p < 0.05 else 'gray')

    ax.set_xticks(range(len(summaries)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Performance by Schedule", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Phase 62 reference comparison
    ax = axes[1]
    ref = {"Phase 62\nfirst20_flat\n(N=50)": 42.0, "Phase 60\nfull_linear\n(N=50)": 38.0}
    compare_names = list(ref.keys())
    compare_rates = list(ref.values())
    # Add key Phase 63 results
    for s in summaries:
        if s["condition"] in ["first20_linear", "first20_cosine", "first20_high_σ0.20"]:
            compare_names.append(f"Phase 63\n{s['condition']}\n(N=50)")
            compare_rates.append(s["solve_rate"])

    bars2 = ax.bar(range(len(compare_names)), compare_rates, alpha=0.85,
                  edgecolor='white', linewidth=2,
                  color=['#3498db','#2ecc71'] + ['#e74c3c']*len(compare_names))
    for i, (b, r) in enumerate(zip(bars2, compare_rates)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
               f'{r:.0f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(compare_names)))
    ax.set_xticklabels(compare_names, fontsize=7)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Cross-Phase Comparison", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase63_flash_annealing.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


def visualize_phase64(summaries):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle("Phase 64: N=100 Validation\nLarge-sample validation of key conditions",
                 fontsize=14, fontweight="bold", y=1.02)

    bl = next(s for s in summaries if s["condition"] == "baseline")
    names = [s["condition"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]
    colors = ['#95a5a6', '#3498db', '#e74c3c']

    bars = ax.bar(range(len(summaries)), rates, color=colors[:len(summaries)], alpha=0.85,
                  edgecolor='white', linewidth=2, width=0.6)
    for i, (b, s) in enumerate(zip(bars, summaries)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               f'{s["solve_rate"]:.0f}%\n(N={s["n_trials"]})',
               ha='center', fontsize=11, fontweight='bold')
        if i > 0:
            table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                     [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
            _, p = fisher_exact(table, alternative='two-sided')
            sig = f"p={p:.2e}" if p < 0.001 else f"p={p:.4f}"
            ax.text(i, s["solve_rate"]+5, sig, ha='center', fontsize=9, color='darkred')

    ax.set_xticks(range(len(summaries)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase64_validation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


def visualize_phase65(summaries):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 65: Adaptive Cosine Homeostasis\n"
                 "Norm-adaptive σ maintains target cos(h, h+noise)",
                 fontsize=14, fontweight="bold", y=1.02)

    bl = next(s for s in summaries if s["condition"] == "baseline")
    names = [s["condition"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]

    # Panel 1: All conditions
    ax = axes[0]
    colors = ['#95a5a6', '#3498db',  # baseline, fixed
              '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6',  # adaptive cos targets
              '#1abc9c']  # adaptive + first20
    bars = ax.bar(range(len(summaries)), rates, color=colors[:len(summaries)], alpha=0.85,
                  edgecolor='white', linewidth=2)
    for i, (b, s) in enumerate(zip(bars, summaries)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               f'{s["solve_rate"]:.0f}%', ha='center', fontsize=9, fontweight='bold')
        if i > 0:
            table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                     [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
            _, p = fisher_exact(table, alternative='two-sided')
            sig = "★★★" if p < 0.001 else "★★" if p < 0.01 else "★" if p < 0.05 else "n.s."
            ax.text(i, s["solve_rate"]+4, sig, ha='center', fontsize=8,
                   color='darkred' if p < 0.05 else 'gray')

    ax.set_xticks(range(len(summaries)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=6, rotation=0)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Performance by Condition", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Adaptive conditions only (target cos vs solve rate)
    ax = axes[1]
    adaptive = [s for s in summaries if s["condition"].startswith("adaptive_cos_0")]
    if adaptive:
        targets = [float(s["condition"].split("_")[-1]) for s in adaptive]
        ad_rates = [s["solve_rate"] for s in adaptive]
        ax.plot(targets, ad_rates, 'o-', color='#e74c3c', linewidth=2.5, markersize=12,
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#e74c3c')
        for t, r in zip(targets, ad_rates):
            ax.annotate(f'{r:.0f}%', (t, r), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=11, fontweight='bold')
        # Add reference lines
        fixed_rate = next((s["solve_rate"] for s in summaries if s["condition"] == "always_fixed_0.15"), None)
        if fixed_rate:
            ax.axhline(y=fixed_rate, color='#3498db', linestyle='--', alpha=0.7,
                      label=f'Fixed σ=0.15 ({fixed_rate:.0f}%)')
        ax.axvline(x=0.50, color='gray', linestyle=':', alpha=0.5, label='Phase 58 cliff (cos=0.50)')
        ax.set_xlabel("Target Cosine Similarity", fontsize=12)
        ax.set_ylabel("Solve Rate (%)", fontsize=12)
        ax.set_title("Adaptive σ: Target Cosine vs Performance", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase65_adaptive_cosine.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


# ===================================================
#  MAIN: BATCH RUNNER
# ===================================================

def main():
    t_total = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tok = load_model()

    print("\n" + "#"*80)
    print("  PHASE 63-65 BATCH RUNNER")
    print("  1. Phase 63: Flash Annealing (Duration × Schedule)")
    print("  2. Phase 64: N=100 Validation")
    print("  3. Phase 65: Adaptive Cosine Homeostasis")
    print("#"*80)

    # Run all three phases
    p63 = run_phase63(model, tok)
    gc.collect(); torch.cuda.empty_cache()

    p64 = run_phase64(model, tok)
    gc.collect(); torch.cuda.empty_cache()

    p65 = run_phase65(model, tok)
    gc.collect(); torch.cuda.empty_cache()

    elapsed_total = time.time() - t_total
    print("\n" + "#"*80)
    print(f"  ALL PHASES COMPLETE!")
    print(f"  Total time: {elapsed_total/60:.1f} min ({elapsed_total/3600:.1f} hours)")
    print("#"*80)

    # Clean up model
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    print("\n All phases complete.")

