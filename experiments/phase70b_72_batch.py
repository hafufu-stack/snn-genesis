"""
Phase 70b + 72 Batch: Per-Dim σ Sweep + Rank × Multi-Layer Cross
=================================================================

Phase 70b: Per-Dimension σ Equalized Rank Sweep
  Phase 70 used norm equalization (total ||noise|| constant across k).
  Phase 70b uses PER-DIMENSION σ equalization: σ=0.15 for each perturbed
  dimension, regardless of k. This means total energy SCALES with k.
  
  Answers: "At the optimal per-dimension intensity (σ=0.15), how many
  dimensions of noise are needed for stochastic resonance?"

  If k<4096 matches k=4096, then some noise dimensions are redundant.

  Conditions (7, N=30): baseline + k ∈ {4, 16, 64, 256, 1024, 4096}
  Total: 210 games, ~3.5 hours

Phase 72: Rank × Multi-Layer Cross Experiment
  Combines Phase 68 (1/√N multi-layer dose law) with Phase 70 (rank-k efficiency).
  Tests whether low-rank noise "stacks" differently across layers than full-rank.

  Conditions (6, N=30): k ∈ {4, 64, 4096} × layers ∈ {1 (L18), 2 (L17+L18 ρ=+1)}
  2-layer uses dose adjustment: σ_adj = σ/√2 (Phase 68 law)
  Total: 180 games, ~3 hours

Grand total: 390 games, ~6.5 hours

Usage:
    python experiments/phase70b_72_batch.py
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
HIDDEN_DIM = 4096
BASE_SIGMA = 0.15  # Phase 63 champion
N_PER_CONDITION = 30

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CSV_PATH = r"C:\tmp\experiment_control.csv"

def should_hibernate(phase_num):
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
#  NOISE HOOKS
# ===================================================

def generate_basis(k, d=HIDDEN_DIM, device='cuda'):
    """Generate k-dimensional orthogonal basis via QR (float32 → float16)."""
    if k >= d:
        return None
    torch.manual_seed(42)
    A = torch.randn(d, k, dtype=torch.float32, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q.to(torch.float16)


class RankKHook:
    """Single-layer rank-k noise hook."""
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.rank_k = HIDDEN_DIM
        self.basis = None
        self.handle = None

    def setup(self, k, basis):
        self.rank_k = k
        self.basis = basis

    def register(self, model, layer_idx=18):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active:
                return args
            hs = args[0]
            if hook_obj.basis is None:
                noise = torch.randn_like(hs) * hook_obj.sigma
            else:
                b, s, d = hs.shape
                z = torch.randn(b, s, hook_obj.rank_k,
                               dtype=hs.dtype, device=hs.device) * hook_obj.sigma
                noise = z @ hook_obj.basis.T
            return (hs + noise,) + args[1:]
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


class CorrelatedRankKManager:
    """Multi-layer rank-k noise with positive correlation (shared base noise Z)."""
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.rank_k = HIDDEN_DIM
        self.basis = None
        self.layer_indices = []
        self.base_noise = None
        self.first_layer = None
        self.handles = []

    def setup(self, k, basis, layer_indices):
        self.rank_k = k
        self.basis = basis
        self.layer_indices = sorted(layer_indices)
        self.first_layer = self.layer_indices[0]

    def register(self, model):
        for lidx in self.layer_indices:
            hook = self._make_hook(lidx)
            handle = model.model.layers[lidx].register_forward_pre_hook(hook)
            self.handles.append(handle)

    def _make_hook(self, layer_idx):
        mgr = self
        def hook_fn(module, args):
            if not mgr.active:
                return args
            hs = args[0]
            # Reset base noise at first layer (per forward pass)
            if layer_idx == mgr.first_layer or mgr.base_noise is None or mgr.base_noise.shape != hs.shape:
                if mgr.basis is None:
                    mgr.base_noise = torch.randn_like(hs) * mgr.sigma
                else:
                    b, s, d = hs.shape
                    z = torch.randn(b, s, mgr.rank_k,
                                   dtype=hs.dtype, device=hs.device) * mgr.sigma
                    mgr.base_noise = z @ mgr.basis.T
            return (hs + mgr.base_noise,) + args[1:]
        return hook_fn

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


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

def play_game_single(model, tok, env, hook, noise_fn):
    env.reset()
    error = None; consec_fail = 0; legal_move_count = 0
    for step in range(MAX_STEPS):
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
            legal_move_count += 1; error = None; consec_fail = 0
            if env.is_solved(): break
        else:
            error = msg; consec_fail += 1
            if consec_fail >= 10: break
    hook.active = False
    st = env.stats(); st["total_legal_moves"] = legal_move_count
    return st

def play_game_multi(model, tok, env, mgr, noise_fn):
    env.reset()
    error = None; consec_fail = 0; legal_move_count = 0
    for step in range(MAX_STEPS):
        noise_fn(step, legal_move_count)
        mgr.base_noise = None  # Reset for new generation
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
            legal_move_count += 1; error = None; consec_fail = 0
            if env.is_solved(): break
        else:
            error = msg; consec_fail += 1
            if consec_fail >= 10: break
    mgr.active = False
    st = env.stats(); st["total_legal_moves"] = legal_move_count
    return st


def run_condition(model, tok, name, desc, n_trials, play_fn, hook_obj, noise_fn):
    """Run one experimental condition."""
    print(f"\n  === {name}: {desc} | N={n_trials} ===")
    results = []; solved = 0
    for t in range(n_trials):
        env = HanoiEnv(3, modified=True)
        r = play_fn(model, tok, env, hook_obj, noise_fn)
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
        "condition": name, "description": desc,
        "n_trials": n_trials, "n_solved": n_solved, "solve_rate": sr,
        "avg_legal": round(np.mean([r["legal_moves"] for r in results]), 2),
        "avg_illegal": round(np.mean([r["illegal_moves"] for r in results]), 2),
    }


# ===================================================
#  PHASE 70b: Per-Dimension σ Equalized Rank Sweep
# ===================================================

def run_phase70b(model, tok, device):
    print("\n" + "="*80)
    print("  PHASE 70b: PER-DIMENSION σ EQUALIZED RANK SWEEP")
    print("  σ=0.15 per perturbed dimension (total energy scales with k)")
    print("="*80)
    t0 = time.time()

    RANK_VALUES = [4, 16, 64, 256, 1024, 4096]
    all_summaries = []

    # Baseline
    hook = RankKHook()
    hook.register(model, layer_idx=18)
    nf = lambda step, lm: setattr(hook, 'active', False)
    s = run_condition(model, tok, "baseline", "No noise", N_PER_CONDITION,
                      play_game_single, hook, nf)
    all_summaries.append(s)
    hook.remove()

    # Rank conditions
    for k in RANK_VALUES:
        basis = generate_basis(k, device=device)
        hook = RankKHook()
        hook.setup(k, basis)
        hook.register(model, layer_idx=18)

        # Per-dim σ = 0.15 (constant, NOT norm-equalized)
        sigma = BASE_SIGMA
        print(f"    Rank-{k}: σ_per_dim = {sigma:.4f}, total_energy = k×σ² = {k*sigma**2:.4f}")

        def make_fn(h, sig):
            def fn(step, lm):
                if lm < 10:
                    h.active = True
                    h.sigma = sig * (1.0 - lm / 10.0)
                else:
                    h.active = False
            return fn

        nf = make_fn(hook, sigma)
        s = run_condition(model, tok, f"perdim_k{k}", 
                         f"k={k}, σ_per_dim=0.15, first10 linear",
                         N_PER_CONDITION, play_game_single, hook, nf)
        s["rank_k"] = k
        s["sigma_per_dim"] = sigma
        s["total_energy"] = round(k * sigma**2, 4)
        all_summaries.append(s)
        hook.remove()

    elapsed = time.time() - t0

    # Verdict
    bl = next(s for s in all_summaries if s["condition"] == "baseline")
    print(f"\n{'='*75}")
    print(f"  PHASE 70b VERDICT: Per-Dimension σ Equalized")
    print(f"{'='*75}")
    for s in all_summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<18} {s['solve_rate']:>6.1f}%  (baseline)")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {s['condition']:<18} {s['solve_rate']:>6.1f}%  k={s['rank_k']:>5}  "
              f"E={s['total_energy']:.4f}  p={p:.4f} {sig}")
    print(f"  Time: {elapsed/60:.1f} min\n{'='*75}")

    # Save
    out = {
        "experiment": "Phase 70b: Per-Dimension σ Equalized Rank Sweep",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "sigma_per_dim": BASE_SIGMA, "schedule": "first10_linear_decay",
        "design": "σ=0.15 per perturbed dim (constant); total energy scales with k",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_summaries,
    }
    log = os.path.join(RESULTS_DIR, "phase70b_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")
    return all_summaries, elapsed


# ===================================================
#  PHASE 72: Rank × Multi-Layer Cross Experiment
# ===================================================

def run_phase72(model, tok, device):
    print("\n" + "="*80)
    print("  PHASE 72: RANK × MULTI-LAYER CROSS EXPERIMENT")
    print("  k ∈ {4, 64, 4096} × layers ∈ {1 (L18), 2 (L17+L18 ρ=+1)}")
    print("  2-layer: σ_adj = σ/√2 (Phase 68 1/√N law)")
    print("="*80)
    t0 = time.time()

    RANK_VALUES = [4, 64, 4096]
    all_summaries = []

    # Baseline
    hook = RankKHook()
    hook.register(model, layer_idx=18)
    nf = lambda step, lm: setattr(hook, 'active', False)
    s = run_condition(model, tok, "baseline", "No noise", N_PER_CONDITION,
                      play_game_single, hook, nf)
    all_summaries.append(s)
    hook.remove()

    for k in RANK_VALUES:
        basis = generate_basis(k, device=device)
        # Norm-equalized sigma (same as Phase 70)
        sigma_1layer = BASE_SIGMA * math.sqrt(HIDDEN_DIM / k) if k < HIDDEN_DIM else BASE_SIGMA
        sigma_2layer = sigma_1layer / math.sqrt(2)  # 1/√N dose adjustment

        # --- 1-layer (L18 only) ---
        hook = RankKHook()
        hook.setup(k, basis)
        hook.register(model, layer_idx=18)
        
        def make_fn1(h, sig):
            def fn(step, lm):
                if lm < 10:
                    h.active = True
                    h.sigma = sig * (1.0 - lm / 10.0)
                else:
                    h.active = False
            return fn

        nf = make_fn1(hook, sigma_1layer)
        s = run_condition(model, tok, f"k{k}_1layer",
                         f"k={k}, 1-layer L18, σ={sigma_1layer:.4f}",
                         N_PER_CONDITION, play_game_single, hook, nf)
        s["rank_k"] = k
        s["n_layers"] = 1
        s["sigma"] = round(sigma_1layer, 4)
        all_summaries.append(s)
        hook.remove()

        # --- 2-layer (L17+L18, positive correlation) ---
        mgr = CorrelatedRankKManager()
        mgr.setup(k, basis, [17, 18])
        mgr.register(model)

        def make_fn2(m, sig):
            def fn(step, lm):
                if lm < 10:
                    m.active = True
                    m.sigma = sig * (1.0 - lm / 10.0)
                else:
                    m.active = False
            return fn

        nf = make_fn2(mgr, sigma_2layer)
        print(f"    k={k}, 2-layer: σ_adj = {sigma_2layer:.4f} (1/√2 of {sigma_1layer:.4f})")
        s = run_condition(model, tok, f"k{k}_2layer",
                         f"k={k}, 2-layer L17+L18 ρ=+1, σ_adj={sigma_2layer:.4f}",
                         N_PER_CONDITION, play_game_multi, mgr, nf)
        s["rank_k"] = k
        s["n_layers"] = 2
        s["sigma"] = round(sigma_2layer, 4)
        all_summaries.append(s)
        mgr.remove()

    elapsed = time.time() - t0

    # Verdict
    bl = next(s for s in all_summaries if s["condition"] == "baseline")
    print(f"\n{'='*75}")
    print(f"  PHASE 72 VERDICT: Rank × Multi-Layer Cross")
    print(f"{'='*75}")
    for s in all_summaries:
        if s["condition"] == "baseline":
            print(f"  {s['condition']:<18} {s['solve_rate']:>6.1f}%  (baseline)")
            continue
        table = [[s["n_solved"], s["n_trials"]-s["n_solved"]],
                 [bl["n_solved"], bl["n_trials"]-bl["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        layers = s.get("n_layers", "?")
        print(f"  {s['condition']:<18} {s['solve_rate']:>6.1f}%  k={s['rank_k']:>5}  "
              f"L={layers}  σ={s['sigma']:.4f}  p={p:.4f} {sig_str}")

    # Cross-comparison: 1-layer vs 2-layer for each k
    print(f"\n  --- 1-layer vs 2-layer comparison ---")
    for k in RANK_VALUES:
        s1 = next(s for s in all_summaries if s.get("rank_k") == k and s.get("n_layers") == 1)
        s2 = next(s for s in all_summaries if s.get("rank_k") == k and s.get("n_layers") == 2)
        table = [[s2["n_solved"], s2["n_trials"]-s2["n_solved"]],
                 [s1["n_solved"], s1["n_trials"]-s1["n_solved"]]]
        _, p = fisher_exact(table, alternative='two-sided')
        delta = s2["solve_rate"] - s1["solve_rate"]
        print(f"  k={k:>5}: 1L={s1['solve_rate']:.0f}% → 2L={s2['solve_rate']:.0f}% "
              f"(Δ={delta:+.1f}pp, p={p:.3f})")

    print(f"\n  Time: {elapsed/60:.1f} min\n{'='*75}")

    # Save
    out = {
        "experiment": "Phase 72: Rank × Multi-Layer Cross",
        "model": MODEL_SHORT, "task": "Modified-3 Hanoi",
        "design": "k ∈ {4,64,4096} × layers ∈ {1,2}. 2-layer uses 1/√N dose adj + positive correlation",
        "elapsed_min": round(elapsed/60, 1),
        "conditions": all_summaries,
    }
    log = os.path.join(RESULTS_DIR, "phase72_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Log: {log}")
    return all_summaries, elapsed


# ===================================================
#  MAIN
# ===================================================

def main():
    t0_total = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tok = load_model()
    device = next(model.parameters()).device

    # Phase 70b
    summaries_70b, elapsed_70b = run_phase70b(model, tok, device)

    # Phase 72
    summaries_72, elapsed_72 = run_phase72(model, tok, device)

    total_elapsed = time.time() - t0_total
    print(f"\n{'='*80}")
    print(f"  BATCH COMPLETE: Phase 70b + Phase 72")
    print(f"  Phase 70b: {elapsed_70b/60:.1f} min")
    print(f"  Phase 72:  {elapsed_72/60:.1f} min")
    print(f"  Total:     {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 70b + 72 batch complete.")

    print("  Phase 70b + 72 batch done.")

