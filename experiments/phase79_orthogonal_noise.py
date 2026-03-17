"""
Phase 79: Orthogonal Complement Noise — Surgical Avoidance of Reasoning Axis
==============================================================================

Phase 71 discovered that:
  - PCA-top noise (perturbing the reasoning axis) HURTS performance
  - Random noise and PCA-bottom noise HELP
  
This experiment tests the logical conclusion: if we MATHEMATICALLY REMOVE
the top-k PC components from random noise (Gram-Schmidt orthogonalization),
we get "safe" random noise that is GUARANTEED to never touch the reasoning axis.

Procedure:
  Step 1 — COLLECT: Run 50 baseline games, collect L18 hidden states (same as Phase 71)
  Step 2 — PCA: Compute principal components
  Step 3 — EXPERIMENT: Compare noise types:
    (a) random          — Full random noise (Phase 70b replication)
    (b) orth_top4       — Random noise with top-4 PCs projected out
    (c) orth_top16      — Random noise with top-16 PCs projected out
    (d) orth_top64      — Random noise with top-64 PCs projected out
    (e) orth_top256     — Random noise with top-256 PCs projected out
    (f) pca_top4        — Noise ONLY along top-4 PCs (negative control)

  All conditions use k=4096 (full-rank) with Flash Annealing (first-10 decay).
  N=30 per condition, 7 conditions total = 210 games + 50 collection = 260 games

Key hypothesis:
  If orth_topK > random > pca_top, then the "avoid reasoning axis" principle
  is confirmed and we've found the OPTIMAL noise strategy.
  If orth_topK ≈ random, then accidental top-PC hits are too rare to matter
  in 4096-D space (geometric argument: P(hit) ≈ k/d).
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
BASE_SIGMA = 0.15
N_PER_CONDITION = 30
N_COLLECTION_GAMES = 50
LAYER_IDX = 18

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


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
#  HIDDEN STATE COLLECTION
# ===================================================

class CollectorHook:
    def __init__(self):
        self.collected = []
        self.recording = False
        self.handle = None

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args, output):
            if hook_obj.recording:
                hs = output[0]
                # Handle both 2D (seq, hidden) and 3D (batch, seq, hidden)
                if hs.dim() == 3:
                    last_hs = hs[0, -1, :].detach().cpu().float().numpy()
                else:
                    last_hs = hs[-1, :].detach().cpu().float().numpy()
                hook_obj.collected.append(last_hs)
        self.handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


def collect_hidden_states(model, tok, n_games=N_COLLECTION_GAMES):
    print(f"\n  Step 1: Collecting hidden states from {n_games} baseline games...")
    collector = CollectorHook()
    collector.register(model, LAYER_IDX)

    solved_count = 0
    for game_idx in range(n_games):
        env = HanoiEnv(n_disks=3, modified=True)
        error = None; consec_fail = 0
        collector.recording = True

        for step in range(MAX_STEPS):
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
                error = None; consec_fail = 0
                if env.is_solved(): break
            else:
                error = msg; consec_fail += 1
                if consec_fail >= 10: break

        collector.recording = False
        if env.is_solved(): solved_count += 1
        if (game_idx + 1) % 10 == 0:
            print(f"    [{game_idx+1}/{n_games}] Collected {len(collector.collected)} vectors. "
                  f"Solved: {solved_count}/{game_idx+1}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    collector.remove()
    states = np.array(collector.collected)
    print(f"  Collected {states.shape[0]} vectors of dim {states.shape[1]}")
    return states


# ===================================================
#  ORTHOGONAL COMPLEMENT NOISE HOOK
# ===================================================

class OrthComplementHook:
    """Inject noise with top-k PCs projected out (orthogonal complement).
    
    For each forward pass:
      1. Generate random noise z ~ N(0, sigma^2 * I)
      2. Project out top-k PCs: z_orth = z - V_top @ V_top^T @ z
      3. Re-scale to maintain original noise magnitude
      4. Inject z_orth into hidden states
    """
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.projector = None  # (d, d) projection matrix or None for full random
        self.mode = "random"   # "random", "orth_topK", "pca_topK"
        self.pca_basis = None  # (k, d) for pca_top mode
        self.handle = None

    def setup_random(self):
        """Full random noise (no projection)."""
        self.mode = "random"
        self.projector = None
        self.pca_basis = None

    def setup_orthogonal(self, Vt_top, device='cuda'):
        """Random noise with top-k PCs projected out.
        
        Vt_top: (k, d) — top-k principal component vectors
        Projection: z_orth = z - Vt_top^T @ Vt_top @ z = (I - Vt_top^T @ Vt_top) @ z
        
        Instead of storing the full (d,d) projection matrix, we store Vt_top
        and compute the projection on-the-fly for memory efficiency.
        """
        self.mode = "orthogonal"
        self.pca_basis = Vt_top.to(device)  # (k, d)
        self.projector = None

    def setup_pca_top(self, Vt_top, device='cuda'):
        """Noise ONLY along top-k PCs (negative control)."""
        self.mode = "pca_top"
        self.pca_basis = Vt_top.to(device)  # (k, d)
        self.projector = None

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            hs = args[0]

            if hook_obj.mode == "random":
                noise = torch.randn_like(hs) * hook_obj.sigma

            elif hook_obj.mode == "orthogonal":
                # Generate random noise
                z = torch.randn_like(hs) * hook_obj.sigma
                # Project out top-k PCs: z_orth = z - V^T @ V @ z
                # V is (k, d), so V @ z is (k, b, s) ... need per-token projection
                V = hook_obj.pca_basis  # (k, d)
                # z: (b, s, d) -> project each (d,) vector
                proj = z @ V.T  # (b, s, k) — coefficients along top PCs
                z_parallel = proj @ V  # (b, s, d) — component along top PCs
                z_orth = z - z_parallel
                # Re-scale to maintain original noise magnitude
                # ||z_orth|| < ||z|| because we removed components, so scale up
                scale = z.norm(dim=-1, keepdim=True) / (z_orth.norm(dim=-1, keepdim=True) + 1e-8)
                noise = z_orth * scale

            elif hook_obj.mode == "pca_top":
                # Noise ONLY along top-k PCs
                V = hook_obj.pca_basis  # (k, d)
                k = V.shape[0]
                b, s, d = hs.shape
                coeffs = torch.randn(b, s, k, dtype=hs.dtype, device=hs.device) * hook_obj.sigma
                noise = coeffs @ V  # (b, s, d)

            return (hs + noise,) + args[1:]

        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, env, hook, use_flash=True):
    env.reset()
    error = None; consec_fail = 0; legal_move_count = 0

    for step in range(MAX_STEPS):
        if use_flash:
            if legal_move_count < 10:
                hook.active = True
                hook.sigma = BASE_SIGMA * (1.0 - legal_move_count / 10.0)
            else:
                hook.active = False; hook.sigma = 0.0
        else:
            hook.active = False

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
    stats = env.stats()
    stats["steps_taken"] = step + 1
    return stats


# ===================================================
#  MAIN EXPERIMENT
# ===================================================

CONDITIONS = [
    {"name": "baseline",      "mode": "baseline",    "orth_k": 0},
    {"name": "random",        "mode": "random",      "orth_k": 0},
    {"name": "orth_top4",     "mode": "orthogonal",  "orth_k": 4},
    {"name": "orth_top16",    "mode": "orthogonal",  "orth_k": 16},
    {"name": "orth_top64",    "mode": "orthogonal",  "orth_k": 64},
    {"name": "orth_top256",   "mode": "orthogonal",  "orth_k": 256},
    {"name": "pca_top4",      "mode": "pca_top",     "orth_k": 4},
]


def run_phase79(model, tok, device):
    print(f"\n{'='*80}")
    print(f"  Phase 79: Orthogonal Complement Noise")
    print(f"  {len(CONDITIONS)} conditions x N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    # Step 1: Collect hidden states
    states = collect_hidden_states(model, tok)

    # Step 2: PCA
    print(f"\n  Step 2: Computing PCA on {states.shape[0]} x {states.shape[1]} matrix...")
    mean = states.mean(axis=0)
    centered = states - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = S**2 / (S**2).sum()
    cumvar = np.cumsum(explained_var)

    print(f"  Variance explained:")
    for k in [4, 16, 64, 256]:
        print(f"    Top-{k}: {cumvar[k-1]*100:.1f}%")

    Vt_torch = torch.tensor(Vt, dtype=torch.float16, device=device)

    pca_info = {
        "n_samples": states.shape[0],
        "explained_variance_top4": round(cumvar[3] * 100, 2),
        "explained_variance_top16": round(cumvar[15] * 100, 2),
        "explained_variance_top64": round(cumvar[63] * 100, 2),
        "explained_variance_top256": round(cumvar[255] * 100, 2) if len(cumvar) > 255 else None,
    }

    all_results = {
        "experiment": "Phase 79: Orthogonal Complement Noise",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "n_per_condition": N_PER_CONDITION,
        "schedule": "first10_linear_decay (full-rank)",
        "pca_info": pca_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    # Step 3: Run conditions
    hook = OrthComplementHook()
    hook.register(model, LAYER_IDX)

    for cond_idx, cond in enumerate(CONDITIONS):
        cond_name = cond["name"]
        print(f"\n  [{cond_idx+1}/{len(CONDITIONS)}] Condition: {cond_name}")

        # Configure hook
        if cond["mode"] == "baseline":
            use_flash = False
            hook.setup_random()  # won't be used
        elif cond["mode"] == "random":
            use_flash = True
            hook.setup_random()
        elif cond["mode"] == "orthogonal":
            use_flash = True
            k = cond["orth_k"]
            hook.setup_orthogonal(Vt_torch[:k], device)
            print(f"    Projecting out top-{k} PCs ({cumvar[k-1]*100:.1f}% variance)")
        elif cond["mode"] == "pca_top":
            use_flash = True
            k = cond["orth_k"]
            hook.setup_pca_top(Vt_torch[:k], device)
            print(f"    Noise ONLY along top-{k} PCs")

        games = []
        for trial in range(N_PER_CONDITION):
            env = HanoiEnv(n_disks=3, modified=True)
            stats = play_game(model, tok, env, hook, use_flash=use_flash)
            games.append(stats)

            if (trial + 1) % 10 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                elapsed = time.time() - t0
                print(f"    [{trial+1}/{N_PER_CONDITION}] Solve rate: {sr:.1f}% | {elapsed/60:.1f}min")

        solved = sum(1 for g in games if g["solved"])
        summary = {
            "condition": cond_name,
            "mode": cond["mode"],
            "orth_k": cond["orth_k"],
            "solve_rate": solved / len(games),
            "n_solved": solved,
            "n_total": len(games),
            "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
            "avg_illegal_moves": round(np.mean([g["illegal_moves"] for g in games]), 2),
            "games": games
        }
        all_results["conditions"].append(summary)
        print(f"    Solve rate: {solved/len(games)*100:.1f}% ({solved}/{len(games)})")

    hook.remove()

    # Statistical analysis
    bl = all_results["conditions"][0]
    rand = all_results["conditions"][1]
    
    print(f"\n  === Fisher Exact Tests vs Baseline ===")
    for cond in all_results["conditions"][1:]:
        table = [[cond["n_solved"], cond["n_total"]-cond["n_solved"]],
                 [bl["n_solved"], bl["n_total"]-bl["n_solved"]]]
        _, p = fisher_exact(table)
        delta = cond["solve_rate"] - bl["solve_rate"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    {cond['condition']:20s}: {cond['solve_rate']*100:5.1f}% "
              f"(delta={delta*100:+5.1f}pp, p={p:.4f}) {sig}")

    # Key comparison: orthogonal vs random
    print(f"\n  === Orthogonal Complement vs Random ===")
    for cond in all_results["conditions"]:
        if cond["mode"] == "orthogonal":
            table = [[cond["n_solved"], cond["n_total"]-cond["n_solved"]],
                     [rand["n_solved"], rand["n_total"]-rand["n_solved"]]]
            _, p = fisher_exact(table)
            delta = cond["solve_rate"] - rand["solve_rate"]
            result = "BETTER" if delta > 5 else "SAME" if abs(delta) <= 5 else "WORSE"
            print(f"    {cond['condition']:20s}: {cond['solve_rate']*100:.1f}% vs random {rand['solve_rate']*100:.1f}% "
                  f"(delta={delta*100:+.1f}pp, p={p:.4f}) -> {result}")

    # Geometric argument
    print(f"\n  === Geometric Interpretation ===")
    for cond in all_results["conditions"]:
        if cond["mode"] == "orthogonal":
            k = cond["orth_k"]
            excluded_frac = k / HIDDEN_DIM * 100
            var_excluded = cumvar[k-1] * 100
            print(f"    orth_top{k}: excluded {k}/{HIDDEN_DIM} dims ({excluded_frac:.2f}%), "
                  f"covering {var_excluded:.1f}% variance")

    # Save
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)

    results_path = os.path.join(RESULTS_DIR, "phase79_log.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  Total elapsed: {elapsed_total/60:.1f} min ({elapsed_total/3600:.1f} hours)")

    return all_results, elapsed_total


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    model, tok = load_model()
    device = next(model.parameters()).device
    results, elapsed = run_phase79(model, tok, device)

    print(f"\n{'='*80}")
    print(f"  Phase 79 COMPLETE: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 79 complete. All v14 experiments done!")

    if should_hibernate(79):
        print("  Hibernating...")
    else:
        import winsound
        for _ in range(3):
            time.sleep(0.2)
        print("  All done!")
