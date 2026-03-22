"""
Phase 87: Differential PCA — Finding Qwen's True Reasoning Direction
=====================================================================

Phase 84 showed: Qwen's PCA-top-4 noise = baseline (46.7%).
Phase 85 showed: ALL noise harms Qwen at N=100.
But standard PCA finds directions of maximum VARIANCE, not maximum
REASONING. Qwen's reasoning may be encoded in a direction that doesn't
dominate variance.

This experiment uses Differential PCA:
  1. Play 100 baseline games, recording hidden states WITH outcome labels
  2. Compute mean hidden state for SOLVED vs FAILED games
  3. The difference vector (solved_mean - failed_mean) is the
     "reasoning direction" — the axis that best separates success/failure
  4. Perform PCA on success-failure centered data to find
     orthogonal components of this discriminant
  5. Test noise injection along these discriminant directions

Conditions (N=30 each):
  1. baseline:          No noise
  2. random:            Full random noise (reference)
  3. diff_pca_top10:    Noise in top-10 discriminant PCs
  4. diff_pca_top50:    Noise in top-50 discriminant PCs
  5. diff_pca_bottom:   Noise in bottom discriminant PCs
  6. anti_reasoning:    Noise OPPOSITE to reasoning direction

Qwen2.5-7B-Instruct, Layer 16, Flash Annealing, σ=0.15

Total: 100 collection + 6x30=180 test = 280 games
"""

import torch
import torch.nn as nn
import os, json, gc, time, random, re, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.stats import fisher_exact

# === Config ===
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "Qwen2.5-7B"
SEED = 2026
MAX_STEPS = 50
HIDDEN_DIM = 3584
BASE_SIGMA = 0.15
N_PER_CONDITION = 30
N_BASELINE_GAMES = 100
LAYER_IDX = 16

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
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    print(f"  Done: {len(model.model.layers)} layers, hidden_dim={model.config.hidden_size}")
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
#  LABELED HIDDEN STATE COLLECTION
# ===================================================

class LabeledCollectorHook:
    """Collects hidden states with per-game labels (solved/failed)."""
    def __init__(self):
        self.current_game_states = []
        self.all_games = []  # list of (states_array, solved_bool)
        self.recording = False
        self.handle = None

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args, output):
            if hook_obj.recording:
                hs = output[0]
                if hs.dim() == 3:
                    last_hs = hs[0, -1, :].detach().cpu().float().numpy()
                else:
                    last_hs = hs[-1, :].detach().cpu().float().numpy()
                hook_obj.current_game_states.append(last_hs)
        self.handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    def start_game(self):
        self.current_game_states = []
        self.recording = True

    def end_game(self, solved):
        self.recording = False
        if self.current_game_states:
            states = np.array(self.current_game_states)
            self.all_games.append((states, solved))
        self.current_game_states = []

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


def collect_labeled_states(model, tok, n_games=N_BASELINE_GAMES):
    print(f"\n  Step 1: Collecting LABELED hidden states from {n_games} games...")
    collector = LabeledCollectorHook()
    collector.register(model, LAYER_IDX)

    for game_idx in range(n_games):
        env = HanoiEnv(n_disks=3, modified=True)
        error = None; consec_fail = 0
        collector.start_game()

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

        collector.end_game(env.is_solved())

        if (game_idx + 1) % 10 == 0:
            n_solved = sum(1 for _, s in collector.all_games if s)
            n_total = len(collector.all_games)
            print(f"    [{game_idx+1}/{n_games}] Games: {n_total}, "
                  f"Solved: {n_solved}, Failed: {n_total - n_solved}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    collector.remove()

    n_solved = sum(1 for _, s in collector.all_games if s)
    n_failed = len(collector.all_games) - n_solved
    print(f"  Collection complete: {n_solved} solved, {n_failed} failed out of {len(collector.all_games)}")

    return collector.all_games


def compute_differential_pca(labeled_games):
    """Compute differential PCA: directions that separate solved from failed games.

    Returns:
        diff_Vt: (k, d) matrix of discriminant PC directions
        diff_S: singular values
        info: dict with analysis results
    """
    print(f"\n  Step 2: Computing Differential PCA...")

    # Separate solved and failed game states
    solved_states = []
    failed_states = []
    for states, solved in labeled_games:
        # Use the mean hidden state per game as the game-level representation
        game_mean = states.mean(axis=0)
        if solved:
            solved_states.append(game_mean)
        else:
            failed_states.append(game_mean)

    solved_states = np.array(solved_states)
    failed_states = np.array(failed_states)
    print(f"    Solved games: {len(solved_states)}, Failed games: {len(failed_states)}")

    # Compute class means
    solved_mean = solved_states.mean(axis=0)
    failed_mean = failed_states.mean(axis=0)

    # The discriminant direction: solved_mean - failed_mean
    diff_vector = solved_mean - failed_mean
    diff_norm = np.linalg.norm(diff_vector)
    diff_unit = diff_vector / (diff_norm + 1e-8)
    print(f"    Discriminant vector L2 norm: {diff_norm:.4f}")

    # Cosine similarity between discriminant and standard PCA
    # (load Phase 84 PCA if available)
    cos_with_pca = None
    pca_path = os.path.join(RESULTS_DIR, "phase84_qwen_pca.npz")
    if os.path.exists(pca_path):
        pca_data = np.load(pca_path)
        Vt_std = pca_data["Vt"]
        cos_sims = np.abs(Vt_std @ diff_unit)
        top_k = np.argsort(cos_sims)[::-1][:10]
        cos_with_pca = {f"PC{k+1}": round(float(cos_sims[k]), 4) for k in top_k}
        print(f"    Top cosine similarities with standard PCA:")
        for k in top_k[:5]:
            print(f"      PC{k+1}: {cos_sims[k]:.4f}")

    # Within-class PCA: PCA of the deviation from class means
    # This finds the DIMENSIONS that vary most between solved/failed
    all_game_means = np.vstack([solved_states, failed_states])
    labels = np.array([1]*len(solved_states) + [0]*len(failed_states))
    global_mean = all_game_means.mean(axis=0)
    centered = all_game_means - global_mean

    # Between-class scatter direction is diff_unit
    # Within-class: PCA on centered data (Fisher-like)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Check how much each PC separates solved vs failed
    projections = centered @ Vt.T  # (n_games, k)
    separation_scores = []
    for pc_idx in range(min(100, Vt.shape[0])):
        proj_solved = projections[labels == 1, pc_idx]
        proj_failed = projections[labels == 0, pc_idx]
        # Cohen's d as separation metric
        pooled_std = np.sqrt((proj_solved.std()**2 + proj_failed.std()**2) / 2 + 1e-8)
        cohens_d = abs(proj_solved.mean() - proj_failed.mean()) / pooled_std
        separation_scores.append(cohens_d)

    separation_scores = np.array(separation_scores)
    best_pcs = np.argsort(separation_scores)[::-1][:20]
    print(f"\n    Top separating PCs (by Cohen's d):")
    for rank, pc_idx in enumerate(best_pcs[:10]):
        print(f"      Rank {rank+1}: PC{pc_idx+1} (d={separation_scores[pc_idx]:.3f})")

    info = {
        "n_solved": int(len(solved_states)),
        "n_failed": int(len(failed_states)),
        "diff_vector_norm": round(float(diff_norm), 4),
        "cosine_with_standard_pca": cos_with_pca,
        "top_separating_pcs": [
            {"pc": int(best_pcs[i]+1), "cohens_d": round(float(separation_scores[best_pcs[i]]), 4)}
            for i in range(min(20, len(best_pcs)))
        ],
        "separation_scores_top50_mean": round(float(separation_scores[:50].mean()), 4),
    }

    return Vt, S, diff_unit, info


# ===================================================
#  DIFFERENTIAL NOISE HOOK
# ===================================================

class DiffNoiseHook:
    """Inject noise in differential PCA directions."""
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.mode = "random"
        self.band_basis = None  # (k, d) discriminant PCs
        self.handle = None

    def setup_random(self):
        self.mode = "random"
        self.band_basis = None

    def setup_diff_band(self, Vt_band, device='cuda'):
        self.mode = "diff_band"
        self.band_basis = Vt_band.to(device)

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            hs = args[0]

            if hook_obj.mode == "random":
                noise = torch.randn_like(hs) * hook_obj.sigma
            elif hook_obj.mode == "diff_band":
                V = hook_obj.band_basis  # (k, d)
                band_k = V.shape[0]
                d = hs.shape[-1]
                if hs.dim() == 3:
                    b, s, _ = hs.shape
                    coeffs = torch.randn(b, s, band_k, dtype=hs.dtype, device=hs.device) * hook_obj.sigma
                else:
                    s, _ = hs.shape
                    coeffs = torch.randn(s, band_k, dtype=hs.dtype, device=hs.device) * hook_obj.sigma
                noise = coeffs @ V
                noise = noise * math.sqrt(d / band_k)
            else:
                return args

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
#  VISUALIZATION
# ===================================================

def visualize(all_results, diff_info):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Phase 87: Differential PCA — Where Is Qwen's Reasoning?",
                 fontsize=14, fontweight="bold")

    # Panel 1: Solve rates
    ax = axes[0]
    conds = all_results["conditions"]
    names = [c["condition"] for c in conds]
    rates = [c["solve_rate"] * 100 for c in conds]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#8BC34A", "#FF9800", "#F44336"]
    bars = ax.bar(range(len(conds)), rates, color=colors[:len(conds)], alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Solve Rate (%)", fontsize=11)
    ax.set_title("Solve Rate by Condition", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(rates) + 10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2: Top separating PCs (Cohen's d)
    ax = axes[1]
    top_pcs = diff_info["top_separating_pcs"][:15]
    pc_labels = [f"PC{p['pc']}" for p in top_pcs]
    d_vals = [p["cohens_d"] for p in top_pcs]
    ax.barh(range(len(top_pcs)-1, -1, -1), d_vals, color="#9C27B0", alpha=0.7)
    ax.set_yticks(range(len(top_pcs)-1, -1, -1))
    ax.set_yticklabels(pc_labels, fontsize=8)
    ax.set_xlabel("Cohen's d (separation strength)", fontsize=11)
    ax.set_title("Top PCs Separating Solved vs Failed", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase87_differential_pca.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

CONDITIONS = [
    {"name": "baseline",         "mode": "baseline"},
    {"name": "random",           "mode": "random"},
    {"name": "diff_pca_top10",   "mode": "diff_band", "pc_start": 0,  "pc_end": 10},
    {"name": "diff_pca_top50",   "mode": "diff_band", "pc_start": 0,  "pc_end": 50},
    {"name": "diff_pca_bottom",  "mode": "diff_band", "pc_start": 50, "pc_end": None},
    {"name": "anti_reasoning",   "mode": "diff_band", "pc_start": 0,  "pc_end": 10},
]


def run_phase87(model, tok, device):
    print(f"\n{'='*80}")
    print(f"  Phase 87: Differential PCA — Qwen Reasoning Direction")
    print(f"  {len(CONDITIONS)} conditions x N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    # Step 1: Collect labeled states
    labeled_games = collect_labeled_states(model, tok, N_BASELINE_GAMES)

    # Step 2: Differential PCA
    diff_Vt, diff_S, diff_unit, diff_info = compute_differential_pca(labeled_games)
    diff_Vt_torch = torch.tensor(diff_Vt, dtype=torch.float16, device=device)

    # Save differential PCA for reuse
    pca_save = os.path.join(RESULTS_DIR, "phase87_diff_pca.npz")
    np.savez_compressed(pca_save, Vt=diff_Vt, S=diff_S, diff_unit=diff_unit)
    print(f"  Differential PCA saved: {pca_save}")

    all_results = {
        "experiment": "Phase 87: Differential PCA — Qwen Reasoning Direction",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "layer": LAYER_IDX,
        "hidden_dim": HIDDEN_DIM,
        "sigma": BASE_SIGMA,
        "n_per_condition": N_PER_CONDITION,
        "n_baseline_games": N_BASELINE_GAMES,
        "schedule": "first10_linear_decay",
        "differential_pca_info": diff_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    # Step 3: Run conditions
    hook = DiffNoiseHook()
    hook.register(model, LAYER_IDX)

    for cond_idx, cond in enumerate(CONDITIONS):
        cond_name = cond["name"]
        print(f"\n  [{cond_idx+1}/{len(CONDITIONS)}] Condition: {cond_name}")

        if cond["mode"] == "baseline":
            use_flash = False
            hook.setup_random()
        elif cond["mode"] == "random":
            use_flash = True
            hook.setup_random()
        elif cond["mode"] == "diff_band":
            use_flash = True
            start = cond["pc_start"]
            end = cond["pc_end"] if cond["pc_end"] is not None else diff_Vt_torch.shape[0]
            end = min(end, diff_Vt_torch.shape[0])
            start = min(start, end)  # Guard: ensure start <= end
            if end - start < 1:
                print(f"    SKIP: Diff-PCA band PC {start+1}-{end} is empty (not enough PCs)")
                use_flash = False
                hook.setup_random()
            else:
                band_Vt = diff_Vt_torch[start:end]
                hook.setup_diff_band(band_Vt, device)
                print(f"    Diff-PCA band: PC {start+1}-{end} ({end-start} dims)")

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
            "solve_rate": solved / len(games),
            "n_solved": solved,
            "n_total": len(games),
            "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
            "avg_illegal_moves": round(np.mean([g["illegal_moves"] for g in games]), 2),
            "games": games
        }
        if "pc_start" in cond:
            summary["diff_pc_start"] = cond["pc_start"]
            summary["diff_pc_end"] = cond.get("pc_end")
        all_results["conditions"].append(summary)
        print(f"    Solve rate: {solved/len(games)*100:.1f}% ({solved}/{len(games)})")

        # Intermediate save
        results_path = os.path.join(RESULTS_DIR, "phase87_log.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    hook.remove()

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

    # Visualization
    fig_path = visualize(all_results, diff_info)

    # Save final
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)
    all_results["figure"] = fig_path

    results_path = os.path.join(RESULTS_DIR, "phase87_log.json")
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
    results, elapsed = run_phase87(model, tok, device)

    print(f"\n{'='*80}")
    print(f"  Phase 87 COMPLETE: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 87 complete.")
