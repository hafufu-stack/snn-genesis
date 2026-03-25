"""
Phase 99: Aha! + Flash Annealing — Integrated Reasoning Enhancement
====================================================================

Phase 86 introduced Flash Annealing (linear σ decay over first 10 moves).
Phase 92/95 added Aha! steering (directional offset in discriminant space).
This experiment systematically tests the INTERACTION between these two
mechanisms.

Design:
  σ_base = 0.15 (validated in v12-v18)

Conditions (N=30 each, Qwen baseline model):
  1. baseline:             No intervention
  2. flash_only:           Flash Annealing σ decay, uniform noise
  3. aha_only:             Constant σ, directional offset in Diff-PCA
  4. aha_flash_sequential: Aha! for moves 1-5, Flash uniform for 6-10
  5. aha_flash_parallel:   Aha! + Flash combined (current best from Phase 92)
  6. double_aha:           2× Aha! energy (σ=0.30 for Aha! component)

Qwen2.5-7B-Instruct, Layer 16, σ=0.15

Total: 6 x 30 = 180 games
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
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True,
                                         trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
        trust_remote_code=True, local_files_only=True)
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
#  INTEGRATED AHA! + FLASH HOOK
# ===================================================

class IntegratedHook:
    """Flexible hook supporting multiple interaction modes between Aha! and Flash."""

    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.mode = "baseline"
        self.diff_unit = None    # (d,) unit vector
        self.band_basis = None   # (k, d) Diff-PCA top-K components
        self.handle = None
        # Per-game state
        self.legal_move_count = 0

    def setup(self, mode, diff_unit=None, band_basis=None, device='cuda'):
        self.mode = mode
        if diff_unit is not None:
            du = torch.tensor(diff_unit, dtype=torch.float32, device=device)
            # Project onto Diff-PCA subspace
            if band_basis is not None:
                V = torch.tensor(band_basis, dtype=torch.float32, device=device)
                proj_coeffs = V @ du
                proj = proj_coeffs @ V
                proj_norm = proj.norm()
                if proj_norm > 1e-8:
                    proj = proj / proj_norm
                self.diff_unit = proj.half()
                self.band_basis = V.half()
            else:
                self.diff_unit = du.half() / (du.norm() + 1e-8)

    def set_move_count(self, count):
        self.legal_move_count = count

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            if hook_obj.mode == "baseline":
                return args
            hs = args[0]
            d = hs.shape[-1]
            mc = hook_obj.legal_move_count

            if hook_obj.mode == "flash_only":
                # Uniform noise with flash annealing (Phase 86-style)
                noise = torch.randn_like(hs) * hook_obj.sigma
                return (hs + noise,) + args[1:]

            elif hook_obj.mode == "aha_only":
                # Aha! offset, NO annealing (constant sigma)
                offset = hook_obj.diff_unit
                scaled = offset * hook_obj.sigma * math.sqrt(d)
                if hs.dim() == 3:
                    scaled = scaled.unsqueeze(0).unsqueeze(0).expand_as(hs)
                else:
                    scaled = scaled.unsqueeze(0).expand_as(hs)
                return (hs + scaled,) + args[1:]

            elif hook_obj.mode == "aha_flash_sequential":
                # Aha! for first 5 moves, then Flash uniform for moves 6-10
                if mc < 5:
                    # Aha! mode
                    offset = hook_obj.diff_unit
                    scaled = offset * hook_obj.sigma * math.sqrt(d)
                    if hs.dim() == 3:
                        scaled = scaled.unsqueeze(0).unsqueeze(0).expand_as(hs)
                    else:
                        scaled = scaled.unsqueeze(0).expand_as(hs)
                    return (hs + scaled,) + args[1:]
                else:
                    # Flash uniform
                    noise = torch.randn_like(hs) * hook_obj.sigma
                    return (hs + noise,) + args[1:]

            elif hook_obj.mode == "aha_flash_parallel":
                # Combined: Aha! offset + stochastic noise (Phase 92's best)
                V = hook_obj.band_basis
                if V is not None:
                    k = V.shape[0]
                    # Half energy deterministic Aha!
                    offset = hook_obj.diff_unit
                    det_scale = hook_obj.sigma * math.sqrt(d) * 0.5
                    det_noise = offset * det_scale
                    if hs.dim() == 3:
                        det_noise = det_noise.unsqueeze(0).unsqueeze(0).expand_as(hs)
                    else:
                        det_noise = det_noise.unsqueeze(0).expand_as(hs)
                    # Half energy stochastic in Diff-PCA subspace
                    stoch_sigma = hook_obj.sigma * 0.5
                    if hs.dim() == 3:
                        b, s, _ = hs.shape
                        coeffs = torch.randn(b, s, k, dtype=hs.dtype, device=hs.device) * stoch_sigma
                    else:
                        s, _ = hs.shape
                        coeffs = torch.randn(s, k, dtype=hs.dtype, device=hs.device) * stoch_sigma
                    stoch_noise = coeffs @ V * math.sqrt(d / k)
                    return (hs + det_noise + stoch_noise,) + args[1:]
                else:
                    # Fallback: Aha! + full-space noise
                    offset = hook_obj.diff_unit
                    det_scale = hook_obj.sigma * math.sqrt(d) * 0.5
                    det_noise = offset * det_scale
                    if hs.dim() == 3:
                        det_noise = det_noise.unsqueeze(0).unsqueeze(0).expand_as(hs)
                    else:
                        det_noise = det_noise.unsqueeze(0).expand_as(hs)
                    stoch_noise = torch.randn_like(hs) * (hook_obj.sigma * 0.5)
                    return (hs + det_noise + stoch_noise,) + args[1:]

            elif hook_obj.mode == "double_aha":
                # 2× Aha! energy
                offset = hook_obj.diff_unit
                scaled = offset * (hook_obj.sigma * 2) * math.sqrt(d)
                if hs.dim() == 3:
                    scaled = scaled.unsqueeze(0).unsqueeze(0).expand_as(hs)
                else:
                    scaled = scaled.unsqueeze(0).expand_as(hs)
                return (hs + scaled,) + args[1:]

            return args
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, hook, use_flash_schedule=True):
    """Play one game. If use_flash_schedule, apply σ annealing over first 10 moves."""
    env = HanoiEnv(n_disks=3, modified=True)
    error = None; consec_fail = 0; legal_move_count = 0

    for step in range(MAX_STEPS):
        hook.set_move_count(legal_move_count)

        if use_flash_schedule:
            if legal_move_count < 10:
                hook.active = True
                hook.sigma = BASE_SIGMA * (1.0 - legal_move_count / 10.0)
            else:
                hook.active = False; hook.sigma = 0.0
        else:
            # Constant sigma (no annealing)
            hook.active = True

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

def visualize(all_results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Phase 99: Aha! + Flash Annealing Interaction Study on Qwen\n"
                 "How do directional steering and noise annealing interact?",
                 fontsize=12, fontweight="bold")

    conds = all_results["conditions"]
    names = [c["condition"] for c in conds]
    rates = [c["solve_rate"] * 100 for c in conds]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    bars = ax.bar(range(len(conds)), rates, color=colors[:len(conds)], alpha=0.85,
                  edgecolor="white", linewidth=2)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=50, color="gold", linestyle="--", alpha=0.6, label="Phase 87 record (50%)")
    ax.axhline(y=40, color="red", linestyle=":", alpha=0.5, label="40% ceiling")

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_ylim(0, max(rates) + 12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase99_aha_flash_interaction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

CONDITIONS = [
    {"name": "baseline",             "mode": "baseline",              "flash": False},
    {"name": "flash_only",           "mode": "flash_only",            "flash": True},
    {"name": "aha_only",             "mode": "aha_only",              "flash": False},
    {"name": "aha_flash_sequential", "mode": "aha_flash_sequential",  "flash": True},
    {"name": "aha_flash_parallel",   "mode": "aha_flash_parallel",    "flash": True},
    {"name": "double_aha",           "mode": "double_aha",            "flash": True},
]


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*80}")
    print(f"  Phase 99: Aha! + Flash Annealing Interaction Study")
    print(f"  {len(CONDITIONS)} conditions x N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load Phase 87's Diff-PCA
    diff_pca_path = os.path.join(RESULTS_DIR, "phase87_diff_pca.npz")
    if not os.path.exists(diff_pca_path):
        raise FileNotFoundError(f"Phase 87 Diff-PCA not found: {diff_pca_path}")
    data = np.load(diff_pca_path)
    diff_Vt = data["Vt"]
    diff_unit = data["diff_unit"]
    TOP_K = 10
    diff_Vt_top = diff_Vt[:TOP_K]
    print(f"  Loaded Diff-PCA: Vt={diff_Vt.shape}, diff_unit norm={np.linalg.norm(diff_unit):.4f}")

    model, tok = load_model()
    device = next(model.parameters()).device

    hook = IntegratedHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 99: Aha! + Flash Annealing Interaction Study",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "layer": LAYER_IDX,
        "hidden_dim": HIDDEN_DIM,
        "sigma": BASE_SIGMA,
        "diff_pca_source": "phase87_diff_pca.npz",
        "top_k_pcs": TOP_K,
        "n_per_condition": N_PER_CONDITION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    for cond_idx, cond in enumerate(CONDITIONS):
        cond_name = cond["name"]
        print(f"\n  [{cond_idx+1}/{len(CONDITIONS)}] Condition: {cond_name}")

        if cond["mode"] == "baseline":
            hook.setup("baseline")
        else:
            hook.setup(cond["mode"], diff_unit=diff_unit, band_basis=diff_Vt_top, device=device)

        use_flash = cond["flash"]
        games = []
        for trial in range(N_PER_CONDITION):
            stats = play_game(model, tok, hook, use_flash_schedule=use_flash)
            games.append(stats)
            if (trial + 1) % 10 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                elapsed = time.time() - t0
                print(f"    [{trial+1}/{N_PER_CONDITION}] Solve rate: {sr:.1f}% | {elapsed/60:.1f}min")

        solved = sum(1 for g in games if g["solved"])
        summary = {
            "condition": cond_name,
            "mode": cond["mode"],
            "flash_schedule": use_flash,
            "solve_rate": round(solved / len(games), 4),
            "n_solved": solved,
            "n_total": len(games),
            "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
            "avg_illegal_moves": round(np.mean([g["illegal_moves"] for g in games]), 2),
            "games": games
        }
        all_results["conditions"].append(summary)
        print(f"    Solve rate: {solved/len(games)*100:.1f}% ({solved}/{len(games)})")

        # Intermediate save
        results_path = os.path.join(RESULTS_DIR, "phase99_log.json")
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
        print(f"    {cond['condition']:25s}: {cond['solve_rate']*100:5.1f}% "
              f"(delta={delta*100:+5.1f}pp, p={p:.4f}) {sig}")

    # Find best
    best = max(all_results["conditions"], key=lambda c: c["solve_rate"])
    print(f"\n  BEST condition: {best['condition']} = {best['solve_rate']*100:.1f}%")

    # Interaction analysis
    flash_rate = next((c["solve_rate"] for c in all_results["conditions"]
                       if c["condition"] == "flash_only"), 0)
    aha_rate = next((c["solve_rate"] for c in all_results["conditions"]
                     if c["condition"] == "aha_only"), 0)
    combined_rate = next((c["solve_rate"] for c in all_results["conditions"]
                          if c["condition"] == "aha_flash_parallel"), 0)

    additive = flash_rate + aha_rate - bl["solve_rate"]
    interaction = combined_rate - additive
    print(f"\n  === Interaction Analysis ===")
    print(f"  Flash alone: {flash_rate*100:.1f}%")
    print(f"  Aha! alone:  {aha_rate*100:.1f}%")
    print(f"  Combined:    {combined_rate*100:.1f}%")
    print(f"  Additive prediction: {additive*100:.1f}%")
    print(f"  Interaction effect:  {interaction*100:+.1f}pp")
    if interaction > 0.05:
        print(f"  → SYNERGISTIC: Combined > sum of parts")
    elif interaction < -0.05:
        print(f"  → ANTAGONISTIC: Combined < sum of parts")
    else:
        print(f"  → ADDITIVE: No significant interaction")

    all_results["interaction_analysis"] = {
        "flash_rate": round(flash_rate, 4),
        "aha_rate": round(aha_rate, 4),
        "combined_rate": round(combined_rate, 4),
        "additive_prediction": round(additive, 4),
        "interaction_effect": round(interaction, 4),
    }

    # Visualization
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
    print(f"\n Phase 99 complete.")
