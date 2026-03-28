"""
Phase 70: Rank-k Noise Dimension Sweep — How Many Dimensions Does Thought Need?
================================================================================

The stochastic resonance discovered in Phase 57 uses 4096-dimensional noise
(full hidden dimension). But does an LLM really need noise in ALL dimensions
to benefit from stochastic resonance? Or is there a lower-dimensional
"thought manifold" where noise is most effective?

Key innovation: Project noise into a k-dimensional random subspace of R^4096.
A fixed orthogonal basis Q∈R^{4096×k} is pre-generated, and noise z∈R^k is
mapped to 4096D via Q@z. Total noise norm is kept constant (σ × sqrt(k/4096)
in each basis direction gives ||noise|| = σ in expectation).

Conditions (8, N=30 each, first10 linear decay, L18):
  1. baseline       — No noise
  2. rank_1         — k=1:    1 direction
  3. rank_4         — k=4:    4 directions
  4. rank_16        — k=16:   16 directions
  5. rank_64        — k=64:   64 directions
  6. rank_256       — k=256:  256 directions
  7. rank_1024      — k=1024: 1024 directions
  8. rank_4096      — k=4096: full rank (= Phase 63 champion)

Total: 240 games. ~4 hours on RTX 5080.

Usage:
    python experiments/phase70_rank_k_sweep.py
"""

import torch
import torch.nn as nn
import os, json, gc, time, random, re, math, csv
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.stats import fisher_exact

# === Config ===
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
SEED = 2026
MAX_STEPS = 50
HIDDEN_DIM = 4096  # Mistral-7B hidden dimension
BASE_SIGMA = 0.15  # Phase 63 champion sigma
N_PER_CONDITION = 30

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Hibernate control
CSV_PATH = r"C:\tmp\experiment_control.csv"

def should_hibernate(phase_num):
    """Check CSV to determine if we should hibernate after this phase."""
    if not os.path.exists(CSV_PATH):
        return True
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['phase']) == phase_num:
                return int(row['hibernate']) == 1
    return True


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
#  RANK-k NOISE HOOK
# ===================================================

class RankKNoiseHook:
    """Noise injection constrained to a k-dimensional subspace of R^4096.
    
    Key math:
      1. Pre-generate orthogonal basis Q ∈ R^{d×k} via QR decomposition
      2. Sample z ~ N(0, σ²I_k) in k dimensions
      3. Map to full space: noise = Q @ z ∈ R^d
      4. By construction, E[||noise||²] = k × σ² = E[||full_noise||²] when
         σ_k = σ_full × sqrt(d/k) × sqrt(k/d) = σ_full
         
    Actually: for ||noise|| normalization, we want E[||noise||²] to match
    the full-rank case E[||noise_full||²] = d × σ².
    Rank-k: E[||Q@z||²] = k × σ_k². Set k × σ_k² = d × σ_full²:
        σ_k = σ_full × sqrt(d/k)
    
    This ensures the total noise energy is the same regardless of k.
    """
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA  # Base sigma (adjusted per-k)
        self.rank_k = HIDDEN_DIM  # Default: full rank
        self.basis = None  # Orthogonal basis Q ∈ R^{d×k} (on GPU)
        self.handle = None
        self._sigma_k = BASE_SIGMA  # Adjusted sigma for rank-k

    def set_rank(self, k, device='cuda'):
        """Pre-generate k-dimensional orthogonal basis."""
        self.rank_k = k
        if k >= HIDDEN_DIM:
            # Full rank: no projection needed
            self.basis = None
            self._sigma_k = BASE_SIGMA
        else:
            # Generate random orthogonal basis via QR
            # Create random matrix and orthogonalize
            # Note: QR decomposition requires float32 on CUDA (geqrf_cuda)
            torch.manual_seed(42)  # Fixed basis across all trials for fairness
            A = torch.randn(HIDDEN_DIM, k, dtype=torch.float32, device=device)
            Q, _ = torch.linalg.qr(A)
            self.basis = Q.to(torch.float16)  # d×k orthonormal columns, cast to fp16
            # Adjust sigma so total noise energy matches full-rank case
            self._sigma_k = BASE_SIGMA * math.sqrt(HIDDEN_DIM / k)
        
        print(f"    Rank-{k}: σ_k = {self._sigma_k:.4f}, basis shape = "
              f"{'None (full)' if self.basis is None else str(tuple(self.basis.shape))}")

    def register(self, model, layer_idx=18):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active:
                return args
            hs = args[0]  # (batch, seq, hidden_dim)
            
            if hook_obj.basis is None:
                # Full rank: standard noise
                noise = torch.randn_like(hs) * hook_obj.sigma
            else:
                # Low-rank projection
                batch, seq, d = hs.shape
                # Sample in k dimensions
                z = torch.randn(batch, seq, hook_obj.rank_k, 
                               dtype=hs.dtype, device=hs.device) * hook_obj.sigma
                # Project to d dimensions via basis
                noise = z @ hook_obj.basis.T  # (batch, seq, k) @ (k, d) = (batch, seq, d)
            
            return (hs + noise,) + args[1:]
        
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


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
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, env, hook, noise_fn=None):
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
            error = "Parse fail. Use Move: X->Y"; consec_fail += 1
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
#  EXPERIMENT DEFINITIONS
# ===================================================

RANK_VALUES = [1, 4, 16, 64, 256, 1024, 4096]

EXPERIMENTS = [
    {"name": "baseline", "rank": None, "desc": "No noise (control)"},
] + [
    {"name": f"rank_{k}", "rank": k, "desc": f"k={k} dimensions, first10 linear decay"}
    for k in RANK_VALUES
]


def _make_first10_linear(hook, base_sigma):
    """First 10 legal moves: σ decays linearly from base_sigma to 0."""
    def fn(step, legal_moves):
        if legal_moves < 10:
            hook.active = True
            hook.sigma = base_sigma * (1.0 - legal_moves / 10.0)
        else:
            hook.active = False
    return fn


# ===================================================
#  MAIN
# ===================================================

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tok = load_model()
    device = next(model.parameters()).device

    print("\n" + "="*80)
    print("  PHASE 70: RANK-k NOISE DIMENSION SWEEP")
    print("  How many dimensions does LLM thought need?")
    print(f"  k ∈ {RANK_VALUES}, N={N_PER_CONDITION}, first10 linear decay, L18")
    print(f"  σ_base = {BASE_SIGMA}, norm-equalized: σ_k = σ × sqrt(d/k)")
    print("="*80)

    all_summaries = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        rank = exp["rank"]
        print(f"\n  === {name}: {exp['desc']} | N={N_PER_CONDITION} ===")

        hook = RankKNoiseHook()
        hook.register(model, layer_idx=18)

        if rank is None:
            # Baseline: no noise
            noise_fn = lambda step, lm: setattr(hook, 'active', False)
        else:
            hook.set_rank(rank, device=device)
            noise_fn = _make_first10_linear(hook, hook._sigma_k)

        results = []; solved = 0
        for t in range(N_PER_CONDITION):
            env = HanoiEnv(3, modified=True)
            r = play_game(model, tok, env, hook, noise_fn)
            results.append(r)
            if r["solved"]: solved += 1
            icon = "O" if r["solved"] else "X"
            rate = solved / (t+1) * 100
            print(f"    {t+1:3d}/{N_PER_CONDITION}: {icon} legal={r['legal_moves']:2d} "
                  f"illegal={r['illegal_moves']:2d} [{solved}/{t+1} = {rate:.0f}%]")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if (t+1) % 10 == 0: gc.collect()

        hook.remove()

        sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
        n_solved = sum(1 for r in results if r["solved"])
        all_summaries.append({
            "condition": name,
            "description": exp["desc"],
            "rank_k": rank,
            "sigma_k": round(hook._sigma_k, 4) if rank else 0,
            "n_trials": N_PER_CONDITION, "n_solved": n_solved,
            "solve_rate": sr,
            "avg_legal": round(np.mean([r["legal_moves"] for r in results]), 2),
            "avg_illegal": round(np.mean([r["illegal_moves"] for r in results]), 2),
        })
        print(f"    => Solve={sr:.1f}% ({n_solved}/{N_PER_CONDITION})")

    elapsed = time.time() - t0

    # Visualization
    fig_path = visualize(all_summaries)

    # Verdict
    bl = next(s for s in all_summaries if s["condition"] == "baseline")
    full = next(s for s in all_summaries if s["condition"] == "rank_4096")
    print(f"\n{'='*75}")
    print(f"  PHASE 70 VERDICT: Rank-k Noise Dimension Sweep")
    print(f"{'='*75}")
    for s in all_summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<18} {s['solve_rate']:>6.1f}%  k={str(s['rank_k']):>5s}  σ_k={s['sigma_k']:.4f}  (baseline)")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        vs_full = ""
        if s["condition"] != "rank_4096":
            ft = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                  [full["n_solved"], full["n_trials"]-full["n_solved"]]]
            _, pf = fisher_exact(ft, alternative='two-sided')
            vs_full = f"  vs_full: p={pf:.3f}"
        print(f"  {s['condition']:<18} {s['solve_rate']:>6.1f}%  k={str(s['rank_k']):>5s}  "
              f"σ_k={s['sigma_k']:.4f}  p={p:.4f} {sig}{vs_full}")

    # Find sweet spot
    noise_conds = [s for s in all_summaries if s["rank_k"] is not None]
    best = max(noise_conds, key=lambda s: s["solve_rate"])
    print(f"\n  BEST: {best['condition']} at {best['solve_rate']}% "
          f"(k={best['rank_k']}, σ_k={best['sigma_k']:.4f})")
    print(f"  vs baseline: {best['solve_rate'] - bl['solve_rate']:+.1f}pp")
    print(f"  vs full rank: {best['solve_rate'] - full['solve_rate']:+.1f}pp")
    print(f"\n  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*75}")

    # Save
    out = {
        "experiment": "Phase 70: Rank-k Noise Dimension Sweep",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "hidden_dim": HIDDEN_DIM,
        "base_sigma": BASE_SIGMA,
        "schedule": "first10_linear_decay",
        "layer": 18,
        "rank_values": RANK_VALUES,
        "purpose": "Discover the effective dimensionality of beneficial noise in LLM reasoning",
        "norm_equalization": "σ_k = σ_base × sqrt(d/k) so that E[||noise||²] = d × σ_base² for all k",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_summaries, "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase70_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")
    print(f"  Figure: {fig_path}")


def visualize(summaries):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Phase 70: Rank-k Noise Dimension Sweep\n"
                 "How Many Dimensions Does LLM Thought Need?",
                 fontsize=14, fontweight="bold", y=1.02)

    bl = next(s for s in summaries if s["condition"] == "baseline")
    noise_conds = [s for s in summaries if s["rank_k"] is not None]

    # Panel 1: Bar chart (all conditions)
    ax = axes[0]
    names = [s["condition"] for s in summaries]
    rates = [s["solve_rate"] for s in summaries]
    colors = ['#95a5a6'] + ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', 
              '#3498db', '#9b59b6', '#1abc9c']
    
    bars = ax.bar(range(len(summaries)), rates, color=colors[:len(summaries)],
                  alpha=0.85, edgecolor='white', linewidth=2)
    for i, (b, s) in enumerate(zip(bars, summaries)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
               f'{s["solve_rate"]:.0f}%', ha='center', fontsize=9, fontweight='bold')
        if i > 0:
            table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                     [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
            _, p = fisher_exact(table, alternative='two-sided')
            sig = "★★★" if p < 0.001 else "★★" if p < 0.01 else "★" if p < 0.05 else ""
            if sig:
                ax.text(i, s["solve_rate"]+4, sig, ha='center', fontsize=8, color='darkred')
    
    ax.set_xticks(range(len(summaries)))
    ax.set_xticklabels([f"k={s['rank_k']}" if s['rank_k'] else "base" for s in summaries], 
                       fontsize=8, rotation=45)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Performance by Noise Rank", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Log-scale curve (k vs solve rate) — the "sweet spot" curve
    ax = axes[1]
    ks = [s["rank_k"] for s in noise_conds]
    rs = [s["solve_rate"] for s in noise_conds]
    
    ax.semilogx(ks, rs, 'o-', color='#e74c3c', markersize=10, linewidth=2, 
                markerfacecolor='white', markeredgewidth=2)
    ax.axhline(y=bl["solve_rate"], color='#95a5a6', linestyle='--', linewidth=1.5, 
               label=f'Baseline ({bl["solve_rate"]:.0f}%)')
    
    for k, r in zip(ks, rs):
        ax.annotate(f'{r:.0f}%', (k, r), textcoords="offset points", 
                   xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Noise Rank k (log scale)", fontsize=12)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Dimension Sweet Spot Curve", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 3: σ_k vs solve rate (showing energy equalization)
    ax = axes[2]
    sigmas_k = [s["sigma_k"] for s in noise_conds]
    
    ax.plot(sigmas_k, rs, 'o-', color='#3498db', markersize=10, linewidth=2,
            markerfacecolor='white', markeredgewidth=2)
    for sig, r, k in zip(sigmas_k, rs, ks):
        ax.annotate(f'k={k}', (sig, r), textcoords="offset points", 
                   xytext=(8, 5), ha='left', fontsize=8)
    
    ax.axhline(y=bl["solve_rate"], color='#95a5a6', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Per-dimension σ_k", fontsize=12)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("σ_k After Norm Equalization", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase70_rank_k_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time()))
    print(f"\n Phase 70 complete.")

    print("  Phase 70 done.")
