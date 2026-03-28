"""
Phase 107: Prior Conflict Gradient — 5-Level Spectrum
=====================================================

Phase 102 showed noise helps more with Modified Hanoi (high prior conflict)
than Standard Hanoi (no conflict). This experiment fills in the gradient:
what happens at INTERMEDIATE levels of prior conflict?

5 conflict levels × 2 conditions (baseline vs aha+noise) × N=50 = 500 games

Conflict Levels (Modified Hanoi variants):
  0. standard:       Standard rules (no conflict)
  1. mild:           Only moves TO peg C use modified rules
  2. moderate:       Full modified rules (= v18 standard)
  3. heavy:          Modified + start from peg C (extra confusion)
  4. novel:          Novel rules: disks can only move clockwise (A→B→C→A)

Mistral-7B-Instruct-v0.3, Layer 18, σ=0.15
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
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
SEED = 2026
MAX_STEPS = 50
HIDDEN_DIM = 4096
BASE_SIGMA = 0.15
N_PER_CONDITION = 50
LAYER_IDX = 18

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================
#  HANOI ENVIRONMENTS (5 conflict levels)
# ===================================================

class HanoiEnvStandard:
    """Level 0: Standard Hanoi — no prior conflict."""
    CONFLICT_LEVEL = 0
    CONFLICT_NAME = "standard"

    def __init__(self, n_disks=3):
        self.n_disks = n_disks
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

    def is_legal(self, from_p, to_p):
        if not self.pegs[from_p]: return False
        disk = self.pegs[from_p][-1]
        if not self.pegs[to_p]: return True
        return disk < self.pegs[to_p][-1]  # Standard: smaller on larger

    def legal_moves(self):
        result = []
        for f in "ABC":
            for t in "ABC":
                if f != t and self.is_legal(f, t):
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
        if not self.is_legal(from_p, to_p):
            self.illegal_count += 1; self._prev_illegal = True
            return False, "Illegal move"
        if self._prev_illegal:
            self.self_corrections += 1
        self._prev_illegal = False
        disk = self.pegs[from_p].pop()
        self.pegs[to_p].append(disk)
        self.moves.append(f"{from_p}->{to_p} (disk {disk})")
        return True, f"Moved disk {disk}"

    def state_str(self):
        return f"A:{self.pegs['A']} B:{self.pegs['B']} C:{self.pegs['C']}"

    def stats(self):
        return {"solved": self.is_solved(), "legal_moves": len(self.moves),
                "illegal_moves": self.illegal_count, "self_corrections": self.self_corrections}

    def rules_text(self):
        return "STANDARD RULES: You can ONLY place a SMALLER disk onto a LARGER disk."


class HanoiEnvMild(HanoiEnvStandard):
    """Level 1: Mild conflict — only moves TO peg C use modified (larger-on-smaller)."""
    CONFLICT_LEVEL = 1
    CONFLICT_NAME = "mild"

    def is_legal(self, from_p, to_p):
        if not self.pegs[from_p]: return False
        disk = self.pegs[from_p][-1]
        if not self.pegs[to_p]: return True
        if to_p == "C":
            return disk > self.pegs[to_p][-1]  # Modified for peg C
        else:
            return disk < self.pegs[to_p][-1]  # Standard for A, B

    def rules_text(self):
        return ("MIXED RULES: For pegs A and B, place SMALLER on LARGER (standard). "
                "For peg C ONLY, place LARGER on SMALLER (reversed).")


class HanoiEnvModerate(HanoiEnvStandard):
    """Level 2: Moderate conflict — full modified rules (= v18 standard)."""
    CONFLICT_LEVEL = 2
    CONFLICT_NAME = "moderate"

    def is_legal(self, from_p, to_p):
        if not self.pegs[from_p]: return False
        disk = self.pegs[from_p][-1]
        if not self.pegs[to_p]: return True
        return disk > self.pegs[to_p][-1]  # Modified: larger on smaller

    def rules_text(self):
        return "MODIFIED RULES: You can ONLY place a LARGER disk onto a SMALLER disk. The opposite of standard."


class HanoiEnvHeavy(HanoiEnvModerate):
    """Level 3: Heavy conflict — modified rules + start from peg C."""
    CONFLICT_LEVEL = 3
    CONFLICT_NAME = "heavy"

    def reset(self):
        # Start from C instead of A — extra confusion
        self.pegs = {"A": [], "B": [], "C": list(range(self.n_disks, 0, -1))}
        self.moves = []
        self.illegal_count = 0
        self.total_attempts = 0
        self.self_corrections = 0
        self._prev_illegal = False

    def is_solved(self):
        return len(self.pegs["A"]) == self.n_disks  # Goal: move to A

    def rules_text(self):
        return ("MODIFIED RULES: You can ONLY place a LARGER disk onto a SMALLER disk. "
                "ALSO: Disks START on peg C. Goal: move ALL to peg A.")


class HanoiEnvNovel(HanoiEnvStandard):
    """Level 4: Novel rules — clockwise only (A→B, B→C, C→A) + modified stacking."""
    CONFLICT_LEVEL = 4
    CONFLICT_NAME = "novel"

    ALLOWED_DIRECTIONS = {("A", "B"), ("B", "C"), ("C", "A")}

    def is_legal(self, from_p, to_p):
        if (from_p, to_p) not in self.ALLOWED_DIRECTIONS:
            return False  # Must move clockwise
        if not self.pegs[from_p]: return False
        disk = self.pegs[from_p][-1]
        if not self.pegs[to_p]: return True
        return disk > self.pegs[to_p][-1]  # Modified stacking

    def rules_text(self):
        return ("NOVEL RULES: 1) Disks can ONLY move CLOCKWISE: A→B, B→C, C→A. "
                "No other directions allowed! 2) You can ONLY place a LARGER disk "
                "onto a SMALLER disk.")


ENV_CLASSES = [HanoiEnvStandard, HanoiEnvMild, HanoiEnvModerate, HanoiEnvHeavy, HanoiEnvNovel]


# ===================================================
#  PROMPT & PARSER
# ===================================================

def build_prompt(tokenizer, env, error=None):
    rules = env.rules_text()
    goal = "move ALL disks from A to C" if env.CONFLICT_NAME != "heavy" else "move ALL disks from C to A"
    system = (
        f"You are solving Tower of Hanoi with {env.n_disks} disks. "
        f"{rules} "
        f"Goal: {goal}. "
        f"Respond with EXACTLY one move in format: Move: X->Y (e.g. Move: A->C). "
        f"You may add a brief Think: line before it."
    )
    msg = f"State: {env.state_str()}\n"
    legal = env.legal_moves()
    msg += f"Legal moves: {', '.join(legal)}\n"
    if env.moves:
        recent = env.moves[-3:]
        msg += f"Your last moves: {'; '.join(recent)}\n"
    if error:
        msg += f"ERROR: {error}. Pick from legal moves above.\n"
    msg += "Your move:"
    messages = [{"role": "user", "content": system + "\n\n" + msg}]
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
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
        local_files_only=True)
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
#  AHA + NOISE HOOK
# ===================================================

class AhaNoiseHook:
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.mode = "baseline"
        self.fixed_offset = None
        self.handle = None

    def setup_baseline(self):
        self.mode = "baseline"

    def setup_aha_noise(self, diff_unit_vec, device='cuda'):
        self.mode = "aha_noise"
        du = torch.tensor(diff_unit_vec, dtype=torch.float16, device=device)
        self.fixed_offset = du

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            if hook_obj.mode == "baseline":
                return args
            hs = args[0]
            d = hs.shape[-1]
            if hook_obj.mode == "aha_noise":
                offset = hook_obj.fixed_offset
                det_scale = hook_obj.sigma * math.sqrt(d) * 0.5
                det_noise = offset * det_scale
                if hs.dim() == 3:
                    det_noise = det_noise.unsqueeze(0).unsqueeze(0).expand_as(hs)
                else:
                    det_noise = det_noise.unsqueeze(0).expand_as(hs)
                stoch_noise = torch.randn_like(hs) * (hook_obj.sigma * 0.5)
                return (hs + det_noise + stoch_noise,) + args[1:]
            return args
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, hook, env_class, use_aha=False):
    env = env_class(n_disks=3)
    error = None; consec_fail = 0; legal_move_count = 0
    for step in range(MAX_STEPS):
        if use_aha:
            if legal_move_count < 10:
                hook.active = True
                hook.sigma = BASE_SIGMA * (1.0 - legal_move_count / 10.0)
            else:
                hook.active = False; hook.sigma = 0.0
        else:
            hook.active = False
        prompt = build_prompt(tok, env, error)
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

def visualize(all_results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Phase 107: Prior Conflict Gradient — Does Noise Gain Scale with Conflict?",
                 fontsize=12, fontweight="bold")

    levels = all_results["conflict_levels"]
    x = range(len(levels))
    bl_rates = [l["baseline_rate"] * 100 for l in levels]
    aha_rates = [l["aha_rate"] * 100 for l in levels]
    gains = [l["gain"] * 100 for l in levels]

    w = 0.35
    ax.bar([i-w/2 for i in x], bl_rates, w, label="Baseline", color="#9E9E9E", alpha=0.85)
    ax.bar([i+w/2 for i in x], aha_rates, w, label="Aha!+Noise", color="#9C27B0", alpha=0.85)

    for i, (bl, aha, g) in enumerate(zip(bl_rates, aha_rates, gains)):
        ax.text(i-w/2, bl+1, f"{bl:.0f}%", ha="center", fontsize=9)
        ax.text(i+w/2, aha+1, f"{aha:.0f}%", ha="center", fontsize=9)
        ax.text(i+w/2, aha+6, f"Δ={g:+.0f}pp", ha="center", fontsize=8,
                color="green" if g > 0 else "red", fontweight="bold")

    names = [l["name"] for l in levels]
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"L{l['level']}: {l['name']}" for l in levels], fontsize=9)
    ax.set_ylabel("Solve Rate (%)", fontsize=11)
    ax.set_xlabel("Prior Conflict Level", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase107_prior_conflict_gradient.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*80}")
    print(f"  Phase 107: Prior Conflict Gradient (5 levels)")
    print(f"  5 levels x 2 conditions x N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load Diff-PCA
    diff_pca_path = os.path.join(RESULTS_DIR, "phase91_diff_pca.npz")
    if not os.path.exists(diff_pca_path):
        raise FileNotFoundError(f"Phase 91 Diff-PCA not found: {diff_pca_path}")
    data = np.load(diff_pca_path)
    diff_unit = data["diff_unit"]

    model, tok = load_model()
    device = next(model.parameters()).device
    hook = AhaNoiseHook()
    hook.register(model, LAYER_IDX)
    hook.setup_aha_noise(diff_unit, device)

    all_results = {
        "experiment": "Phase 107: Prior Conflict Gradient",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "n_per_condition": N_PER_CONDITION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conflict_levels": [],
    }

    results_path = os.path.join(RESULTS_DIR, "phase107_log.json")

    for env_class in ENV_CLASSES:
        level = env_class.CONFLICT_LEVEL
        name = env_class.CONFLICT_NAME
        print(f"\n  === Level {level}: {name} ===")

        # Baseline
        print(f"    Baseline...")
        hook.setup_baseline()
        bl_games = []
        for trial in range(N_PER_CONDITION):
            stats = play_game(model, tok, hook, env_class, use_aha=False)
            bl_games.append(stats)
            if (trial + 1) % 25 == 0:
                sr = sum(1 for g in bl_games if g["solved"]) / len(bl_games) * 100
                print(f"      [{trial+1}/{N_PER_CONDITION}] {sr:.1f}%")

        # Aha!+Noise
        print(f"    Aha!+Noise...")
        hook.setup_aha_noise(diff_unit, device)
        aha_games = []
        for trial in range(N_PER_CONDITION):
            stats = play_game(model, tok, hook, env_class, use_aha=True)
            aha_games.append(stats)
            if (trial + 1) % 25 == 0:
                sr = sum(1 for g in aha_games if g["solved"]) / len(aha_games) * 100
                print(f"      [{trial+1}/{N_PER_CONDITION}] {sr:.1f}%")

        bl_solved = sum(1 for g in bl_games if g["solved"])
        aha_solved = sum(1 for g in aha_games if g["solved"])
        bl_rate = bl_solved / len(bl_games)
        aha_rate = aha_solved / len(aha_games)
        gain = aha_rate - bl_rate

        # Fisher exact
        table = [[aha_solved, len(aha_games)-aha_solved],
                 [bl_solved, len(bl_games)-bl_solved]]
        _, p_val = fisher_exact(table)

        level_result = {
            "level": level,
            "name": name,
            "baseline_rate": round(bl_rate, 4),
            "aha_rate": round(aha_rate, 4),
            "gain": round(gain, 4),
            "baseline_solved": bl_solved,
            "aha_solved": aha_solved,
            "n_total": N_PER_CONDITION,
            "fisher_p": round(p_val, 6),
        }
        all_results["conflict_levels"].append(level_result)
        print(f"    BL: {bl_rate*100:.1f}%, Aha: {aha_rate*100:.1f}%, Gain: {gain*100:+.1f}pp, p={p_val:.4f}")

        # Intermediate save
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    hook.remove()

    # === Analysis: correlation between conflict level and gain ===
    print(f"\n  === Gain vs Conflict Level ===")
    levels_arr = np.array([l["level"] for l in all_results["conflict_levels"]])
    gains_arr = np.array([l["gain"] for l in all_results["conflict_levels"]])
    if len(levels_arr) > 2:
        from scipy.stats import pearsonr, spearmanr
        r_pearson, p_pearson = pearsonr(levels_arr, gains_arr)
        r_spearman, p_spearman = spearmanr(levels_arr, gains_arr)
        print(f"  Pearson: r={r_pearson:.4f}, p={p_pearson:.4f}")
        print(f"  Spearman: ρ={r_spearman:.4f}, p={p_spearman:.4f}")
        all_results["correlation"] = {
            "pearson_r": round(r_pearson, 4), "pearson_p": round(p_pearson, 4),
            "spearman_rho": round(r_spearman, 4), "spearman_p": round(p_spearman, 4),
        }

    # Verdict
    if np.all(gains_arr[1:] > gains_arr[:-1]):
        verdict = "MONOTONIC_SCALING"
        print(f"\n  VERDICT: {verdict} — Noise gain scales monotonically with conflict!")
    elif gains_arr[-1] > gains_arr[0] + 0.1:
        verdict = "POSITIVE_TREND"
        print(f"\n  VERDICT: {verdict} — Noise gain increases with conflict (non-monotonic)")
    elif np.std(gains_arr) < 0.03:
        verdict = "CONFLICT_INDEPENDENT"
        print(f"\n  VERDICT: {verdict} — Noise gain is independent of conflict level")
    else:
        verdict = "COMPLEX_PATTERN"
        print(f"\n  VERDICT: {verdict} — Non-linear relationship")

    all_results["verdict"] = verdict

    fig_path = visualize(all_results)
    all_results["figure"] = fig_path

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed, 1)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  Total elapsed: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")

    return all_results, elapsed


if __name__ == "__main__":
    main()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n Phase 107 complete.")
