"""
Phase 71: PCA-Aligned Noise — Reasoning Manifold Perturbation
==============================================================

Phase 70b showed random k=4 noise achieves 16.7% (5x baseline).
Question: "What if noise is aligned to the MEANINGFUL dimensions?"

Procedure:
  Step 1 — COLLECT: Run 50 baseline games (no noise), recording L18
           hidden states at every generation step. Collect ~1000+ vectors.
  Step 2 — PCA: Fit PCA on the collected 4096-D hidden states.
           Extract top-k and bottom-k principal components.
  Step 3 — EXPERIMENT: Compare noise injected along:
           (a) Random k-dim (Phase 70b replication)
           (b) PCA top-k (perturb the "reasoning axis")
           (c) PCA bottom-k (perturb "irrelevant" dimensions)

  k = {4, 16, 64} × basis = {random, pca_top, pca_bottom}
  Total: 1 baseline + 9 conditions = 10 × N=30 = 300 games

  Flash Annealing (first-10 linear decay) schedule for all noise conditions.

Key hypothesis: PCA-top-k should outperform random-k (surgical perturbation).
If PCA-bottom-k also works, then "where in the manifold" doesn't matter.
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
N_COLLECTION_GAMES = 50  # games for hidden state collection
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
#  STEP 1: HIDDEN STATE COLLECTION (no noise)
# ===================================================

class CollectorHook:
    """Record L18 hidden states during generation."""
    def __init__(self):
        self.collected = []
        self.recording = False
        self.handle = None

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args, output):
            if hook_obj.recording:
                # output is a tuple; output[0] is hidden states
                hs = output[0]
                # Handle both 2D (seq, hidden) and 3D (batch, seq, hidden) shapes
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
    """Play baseline games and collect L18 hidden states."""
    print(f"\n  Step 1: Collecting hidden states from {n_games} baseline games...")
    collector = CollectorHook()
    collector.register(model, LAYER_IDX)

    solved_count = 0
    for game_idx in range(n_games):
        env = HanoiEnv(n_disks=3, modified=True)
        error = None
        consec_fail = 0

        collector.recording = True
        for step in range(MAX_STEPS):
            prompt = build_chat_prompt(tok, env, error)
            resp = generate(model, tok, prompt)
            move = parse_move(resp)

            if move is None:
                env.illegal_count += 1; env.total_attempts += 1; env._prev_illegal = True
                error = "Parse fail. Use Move: X->Y"
                consec_fail += 1
                if consec_fail >= 10: break
                continue

            ok, msg = env.try_move(move[0], move[1])
            if ok:
                error = None; consec_fail = 0
                if env.is_solved(): break
            else:
                error = msg
                consec_fail += 1
                if consec_fail >= 10: break

        collector.recording = False
        if env.is_solved():
            solved_count += 1

        if (game_idx + 1) % 10 == 0:
            print(f"    [{game_idx+1}/{n_games}] Collected {len(collector.collected)} vectors. "
                  f"Solved: {solved_count}/{game_idx+1}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    collector.remove()

    states = np.array(collector.collected)
    print(f"  Collected {states.shape[0]} hidden state vectors of dim {states.shape[1]}")
    print(f"  Baseline solve rate: {solved_count}/{n_games} = {solved_count/n_games*100:.1f}%")
    return states


# ===================================================
#  STEP 2: PCA DECOMPOSITION
# ===================================================

def compute_pca_basis(states, device='cuda'):
    """Compute PCA on collected hidden states. Returns principal components."""
    print(f"\n  Step 2: Computing PCA on {states.shape[0]} x {states.shape[1]} matrix...")

    # Center the data
    mean = states.mean(axis=0)
    centered = states - mean

    # SVD (use numpy for numerical stability)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Vt rows are principal components (sorted by variance explained)
    explained_var = S**2 / (S**2).sum()
    cumvar = np.cumsum(explained_var)

    print(f"  Variance explained by top components:")
    for k in [4, 16, 64, 256]:
        print(f"    Top-{k}: {cumvar[k-1]*100:.1f}%")

    # Convert to torch basis matrices on GPU
    Vt_torch = torch.tensor(Vt, dtype=torch.float16, device=device)

    pca_info = {
        "n_samples": states.shape[0],
        "explained_variance_top4": round(cumvar[3] * 100, 2),
        "explained_variance_top16": round(cumvar[15] * 100, 2),
        "explained_variance_top64": round(cumvar[63] * 100, 2),
        "explained_variance_top256": round(cumvar[255] * 100, 2) if len(cumvar) > 255 else None,
    }

    return Vt_torch, pca_info


# ===================================================
#  STEP 3: NOISE HOOKS (PCA-aligned or random)
# ===================================================

class PCANoiseHook:
    """Inject noise along a specific k-dimensional subspace defined by basis vectors."""
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.basis = None  # (k, d) basis vectors
        self.handle = None

    def setup(self, basis_vectors):
        """basis_vectors: (k, d) tensor — rows are basis directions."""
        self.basis = basis_vectors  # (k, d)

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            hs = args[0]  # (batch, seq, d)
            b, s, d = hs.shape
            # Generate k-dimensional noise, project to d-dimensional space
            k = hook_obj.basis.shape[0]
            z = torch.randn(b, s, k, dtype=hs.dtype, device=hs.device) * hook_obj.sigma
            noise = z @ hook_obj.basis  # (b, s, k) @ (k, d) -> (b, s, d)
            return (hs + noise,) + args[1:]
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


def generate_random_basis(k, d=HIDDEN_DIM, device='cuda'):
    """Random orthogonal basis via QR (for comparison)."""
    torch.manual_seed(42)
    A = torch.randn(d, k, dtype=torch.float32, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q.T.to(torch.float16)  # (k, d)


# ===================================================
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, env, hook, use_flash=True):
    """Play game with flash annealing (first-10 linear decay)."""
    env.reset()
    error = None
    consec_fail = 0
    legal_move_count = 0

    for step in range(MAX_STEPS):
        if use_flash:
            if legal_move_count < 10:
                hook.active = True
                hook.sigma = BASE_SIGMA * (1.0 - legal_move_count / 10.0)
            else:
                hook.active = False
                hook.sigma = 0.0
        else:
            hook.active = False

        prompt = build_chat_prompt(tok, env, error)
        resp = generate(model, tok, prompt)
        move = parse_move(resp)

        if move is None:
            env.illegal_count += 1; env.total_attempts += 1; env._prev_illegal = True
            error = "Parse fail. Use Move: X->Y"
            consec_fail += 1
            if consec_fail >= 10: break
            continue

        ok, msg = env.try_move(move[0], move[1])
        if ok:
            legal_move_count += 1
            error = None; consec_fail = 0
            if env.is_solved(): break
        else:
            error = msg
            consec_fail += 1
            if consec_fail >= 10: break

    hook.active = False
    stats = env.stats()
    stats["steps_taken"] = step + 1
    return stats


# ===================================================
#  MAIN EXPERIMENT
# ===================================================

def run_phase71(model, tok, device):
    print(f"\n{'='*80}")
    print(f"  Phase 71: PCA-Aligned Noise — Reasoning Manifold Perturbation")
    print(f"{'='*80}")

    t0 = time.time()

    # Step 1: Collect hidden states
    states = collect_hidden_states(model, tok)

    # Step 2: PCA
    Vt, pca_info = compute_pca_basis(states, device)

    all_results = {
        "experiment": "Phase 71: PCA-Aligned Noise",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "n_per_condition": N_PER_CONDITION,
        "schedule": "first10_linear_decay",
        "pca_info": pca_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    # Step 3: Experiment conditions
    RANK_VALUES = [4, 16, 64]
    BASIS_TYPES = ["random", "pca_top", "pca_bottom"]

    # Baseline (no noise)
    print(f"\n  [0/10] Condition: baseline")
    hook = PCANoiseHook()
    hook.setup(Vt[:4])  # dummy, won't be used
    hook.register(model, LAYER_IDX)
    games = []
    for trial in range(N_PER_CONDITION):
        env = HanoiEnv(n_disks=3, modified=True)
        stats = play_game(model, tok, env, hook, use_flash=False)
        games.append(stats)
        if (trial + 1) % 10 == 0:
            sr = sum(1 for g in games if g["solved"]) / len(games) * 100
            print(f"    [{trial+1}/{N_PER_CONDITION}] Solve rate: {sr:.1f}%")
    hook.remove()

    solved = sum(1 for g in games if g["solved"])
    all_results["conditions"].append({
        "condition": "baseline", "rank_k": 0, "basis_type": "none",
        "solve_rate": solved / len(games), "n_solved": solved,
        "n_total": len(games),
        "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
        "games": games
    })
    print(f"    Solve rate: {solved/len(games)*100:.1f}%")

    # Noise conditions
    cond_idx = 1
    total_conds = len(RANK_VALUES) * len(BASIS_TYPES) + 1
    for k in RANK_VALUES:
        for basis_type in BASIS_TYPES:
            cond_name = f"{basis_type}_k{k}"
            print(f"\n  [{cond_idx}/{total_conds-1}] Condition: {cond_name}")

            if basis_type == "random":
                basis = generate_random_basis(k, HIDDEN_DIM, device)
            elif basis_type == "pca_top":
                basis = Vt[:k]  # Top-k principal components
            elif basis_type == "pca_bottom":
                basis = Vt[-k:]  # Bottom-k principal components

            hook = PCANoiseHook()
            hook.setup(basis)
            hook.register(model, LAYER_IDX)

            games = []
            for trial in range(N_PER_CONDITION):
                env = HanoiEnv(n_disks=3, modified=True)
                stats = play_game(model, tok, env, hook, use_flash=True)
                games.append(stats)
                if (trial + 1) % 10 == 0:
                    sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                    elapsed = time.time() - t0
                    print(f"    [{trial+1}/{N_PER_CONDITION}] Solve rate: {sr:.1f}% | {elapsed/60:.1f}min")

            hook.remove()

            solved = sum(1 for g in games if g["solved"])
            all_results["conditions"].append({
                "condition": cond_name,
                "rank_k": k,
                "basis_type": basis_type,
                "solve_rate": solved / len(games),
                "n_solved": solved,
                "n_total": len(games),
                "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
                "games": games
            })
            print(f"    Solve rate: {solved/len(games)*100:.1f}%")
            cond_idx += 1

    # Statistical analysis
    bl = all_results["conditions"][0]
    print(f"\n  === Fisher Exact Tests vs Baseline ===")
    for cond in all_results["conditions"][1:]:
        table = [[cond["n_solved"], cond["n_total"]-cond["n_solved"]],
                 [bl["n_solved"], bl["n_total"]-bl["n_solved"]]]
        _, p = fisher_exact(table)
        delta = cond["solve_rate"] - bl["solve_rate"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    {cond['condition']:20s}: {cond['solve_rate']*100:5.1f}% "
              f"(delta={delta*100:+5.1f}pp, p={p:.4f}) {sig}")

    # Cross-comparison: PCA-top vs random for each k
    print(f"\n  === PCA-top vs Random (same k) ===")
    for k in RANK_VALUES:
        r = next(c for c in all_results["conditions"] if c.get("rank_k") == k and c.get("basis_type") == "random")
        p_top = next(c for c in all_results["conditions"] if c.get("rank_k") == k and c.get("basis_type") == "pca_top")
        table = [[p_top["n_solved"], p_top["n_total"]-p_top["n_solved"]],
                 [r["n_solved"], r["n_total"]-r["n_solved"]]]
        _, p = fisher_exact(table)
        delta = p_top["solve_rate"] - r["solve_rate"]
        print(f"    k={k:3d}: random={r['solve_rate']*100:.1f}% -> pca_top={p_top['solve_rate']*100:.1f}% "
              f"(delta={delta*100:+.1f}pp, p={p:.4f})")

    # Save
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)

    results_path = os.path.join(RESULTS_DIR, "phase71_log.json")
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
    results, elapsed = run_phase71(model, tok, device)

    print(f"\n{'='*80}")
    print(f"  Phase 71 COMPLETE: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 71 complete.")

    if should_hibernate(71):
        print("  Hibernating...")
    else:
        import winsound, subprocess
        for _ in range(3):
            time.sleep(0.2)
        print("  Phase 71 done. Chaining to Phase 78...")
        phase78_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase78_dim_annealing.py")
        if os.path.exists(phase78_script):
            subprocess.Popen(["python", phase78_script],
                           cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            print(f"  Started: {phase78_script}")
        else:
            print(f"  WARNING: Phase 78 script not found: {phase78_script}")
