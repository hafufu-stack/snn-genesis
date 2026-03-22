"""
Phase 89: Deterministic/Stochastic Mixing Ratio Sweep
=====================================================

Phase 86 showed:
  - Deterministic offset in PC 257+: 30% (direction alone)
  - Stochastic noise in PC 257+:    40% (direction + randomness)
  - Baseline:                        10%

The "2/3 direction, 1/3 randomness" is a rough 2-point estimate.
This experiment sweeps the mixing ratio α from 0.0 to 1.0:

  h' = h + [(1-α) · d_fixed + α · N(0, σ²)] projected onto PC 257+

Where:
  α=0.0 → 100% deterministic (Phase 86 reproduction)
  α=0.5 → 50/50 mix
  α=1.0 → 100% stochastic (Phase 83/86 reproduction)

Conditions (N=30 each):
  1. baseline:   No noise
  2. α=0.00:     100% deterministic offset
  3. α=0.25:     75% deterministic + 25% stochastic
  4. α=0.50:     50% deterministic + 50% stochastic
  5. α=0.75:     25% deterministic + 75% stochastic
  6. α=1.00:     100% stochastic noise

Mistral-7B, Layer 18, PC 257+, Flash Annealing, σ=0.15
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
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
SEED = 2026
MAX_STEPS = 50
HIDDEN_DIM = 4096
BASE_SIGMA = 0.15
N_PER_CONDITION = 30
LAYER_IDX = 18

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


def collect_hidden_states(model, tok, n_games=50):
    print(f"\n  Collecting hidden states from {n_games} baseline games...")
    collector = CollectorHook()
    collector.register(model, LAYER_IDX)
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
                error = "Parse fail"; consec_fail += 1
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
        if (game_idx + 1) % 10 == 0:
            print(f"    [{game_idx+1}/{n_games}] vectors: {len(collector.collected)}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    collector.remove()
    states = np.array(collector.collected)
    print(f"  Collected {states.shape[0]} vectors of dim {states.shape[1]}")
    return states


# ===================================================
#  MIXING RATIO NOISE HOOK
# ===================================================

class MixingNoiseHook:
    """Inject a mix of deterministic offset + stochastic noise in PCA subspace.

    noise = (1-α) * fixed_offset + α * random_noise
    Both components are restricted to the PC 257+ subspace.
    """
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.alpha = 1.0       # mixing ratio: 0=all deterministic, 1=all stochastic
        self.mode = "baseline"
        self.fixed_offset = None     # (d,) deterministic offset vector
        self.band_basis = None       # (k, d) PCA basis for subspace
        self.handle = None

    def setup_baseline(self):
        self.mode = "baseline"

    def setup_mixing(self, alpha, Vt_band, device='cuda'):
        self.mode = "mixing"
        self.alpha = alpha
        self.band_basis = Vt_band.to(device)

        # Compute fixed offset direction: mean of all band PC directions
        V = Vt_band  # (k, d)
        fixed_dir = V.mean(dim=0)  # (d,)
        fixed_dir = fixed_dir / fixed_dir.norm()
        self.fixed_offset = fixed_dir.to(device)

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            if hook_obj.mode == "baseline":
                return args
            hs = args[0]

            V = hook_obj.band_basis  # (k, d)
            k = V.shape[0]
            d = hs.shape[-1]
            alpha = hook_obj.alpha
            sigma = hook_obj.sigma

            # Deterministic component: (1-α) * fixed_offset * σ * √d
            det_noise = hook_obj.fixed_offset * sigma * math.sqrt(d) * (1.0 - alpha)
            det_noise = det_noise.unsqueeze(0).unsqueeze(0).expand_as(hs)

            # Stochastic component: α * random_noise in PC 257+ subspace
            if alpha > 0:
                if hs.dim() == 3:
                    b, s, _ = hs.shape
                    coeffs = torch.randn(b, s, k, dtype=hs.dtype, device=hs.device) * sigma * alpha
                else:
                    s, _ = hs.shape
                    coeffs = torch.randn(s, k, dtype=hs.dtype, device=hs.device) * sigma * alpha
                stoch_noise = coeffs @ V
                stoch_noise = stoch_noise * math.sqrt(d / k)
            else:
                stoch_noise = 0.0

            noise = det_noise + stoch_noise
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

def visualize(all_results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Phase 89: Deterministic/Stochastic Mixing Ratio Sweep\n"
                 "h' = h + [(1-α)·d + α·N(0,σ²)] in PC 257+",
                 fontsize=13, fontweight="bold")

    conds = all_results["conditions"]
    names = [c["condition"] for c in conds]
    rates = [c["solve_rate"] * 100 for c in conds]

    # Panel 1: Bar chart
    ax = axes[0]
    colors = ["#9E9E9E", "#E91E63", "#FF5722", "#FF9800", "#4CAF50", "#2196F3"]
    bars = ax.bar(range(len(conds)), rates, color=colors[:len(conds)], alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Solve Rate (%)", fontsize=11)
    ax.set_title("Solve Rate by Mixing Ratio", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(rates) + 12)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2: α vs Solve Rate line plot (exclude baseline)
    ax = axes[1]
    mixing_conds = [c for c in conds if "alpha" in c]
    if mixing_conds:
        alphas = [c["alpha"] for c in mixing_conds]
        mix_rates = [c["solve_rate"] * 100 for c in mixing_conds]
        ax.plot(alphas, mix_rates, "o-", color="#9C27B0", linewidth=2, markersize=10)
        for a, r in zip(alphas, mix_rates):
            ax.annotate(f"{r:.0f}%", (a, r), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")

        # Theoretical lines
        bl_rate = conds[0]["solve_rate"] * 100
        ax.axhline(y=bl_rate, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({bl_rate:.0f}%)")
        ax.axhline(y=40, color="red", linestyle=":", alpha=0.5, label="40% ceiling")

    ax.set_xlabel("α (Stochastic fraction)", fontsize=12)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Continuous Mixing Curve", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase89_mixing_ratio.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]

CONDITIONS = [{"name": "baseline", "alpha": None}] + \
             [{"name": f"alpha_{a:.2f}", "alpha": a} for a in ALPHAS]


def run_phase89(model, tok, device):
    print(f"\n{'='*80}")
    print(f"  Phase 89: Deterministic/Stochastic Mixing Ratio Sweep")
    print(f"  {len(CONDITIONS)} conditions x N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load or compute PCA
    pca_candidates = [
        os.path.join(RESULTS_DIR, "phase83_mistral_pca.npz"),
        os.path.join(RESULTS_DIR, "phase86_mistral_pca.npz"),
        os.path.join(RESULTS_DIR, "phase88_mistral_pca.npz"),
    ]
    found_pca = False
    for candidate in pca_candidates:
        if os.path.exists(candidate):
            print(f"  Loading PCA from {candidate}...")
            data = np.load(candidate)
            Vt = data["Vt"]
            found_pca = True
            break

    if not found_pca:
        print(f"  No saved Mistral PCA found. Collecting states...")
        states = collect_hidden_states(model, tok)
        mean = states.mean(axis=0)
        centered = states - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pca_save = os.path.join(RESULTS_DIR, "phase89_mistral_pca.npz")
        np.savez_compressed(pca_save, Vt=Vt, S=S, mean=mean)
        print(f"  PCA saved: {pca_save}")

    Vt_torch = torch.tensor(Vt, dtype=torch.float16, device=device)
    Vt_bottom = Vt_torch[256:]  # PC 257+
    print(f"  PCA loaded. Bottom subspace: {Vt_bottom.shape[0]} dims")

    hook = MixingNoiseHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 89: Deterministic/Stochastic Mixing Ratio Sweep",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "subspace": "PC 257+",
        "n_per_condition": N_PER_CONDITION,
        "schedule": "first10_linear_decay",
        "mixing_formula": "h' = h + [(1-alpha)*d_fixed + alpha*N(0,sigma^2)] in PC 257+",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    for cond_idx, cond in enumerate(CONDITIONS):
        cond_name = cond["name"]
        alpha = cond["alpha"]
        print(f"\n  [{cond_idx+1}/{len(CONDITIONS)}] Condition: {cond_name}")

        if alpha is None:
            use_flash = False
            hook.setup_baseline()
        else:
            use_flash = True
            hook.setup_mixing(alpha, Vt_bottom, device)
            print(f"    α={alpha:.2f}: {(1-alpha)*100:.0f}% deterministic + {alpha*100:.0f}% stochastic")

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
            "solve_rate": solved / len(games),
            "n_solved": solved,
            "n_total": len(games),
            "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
            "avg_illegal_moves": round(np.mean([g["illegal_moves"] for g in games]), 2),
            "games": games
        }
        if alpha is not None:
            summary["alpha"] = alpha
        all_results["conditions"].append(summary)
        print(f"    Solve rate: {solved/len(games)*100:.1f}% ({solved}/{len(games)})")

        # Intermediate save
        results_path = os.path.join(RESULTS_DIR, "phase89_log.json")
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
        print(f"    {cond['condition']:15s}: {cond['solve_rate']*100:5.1f}% "
              f"(delta={delta*100:+5.1f}pp, p={p:.4f}) {sig}")

    # Mixing analysis
    print(f"\n  === Mixing Ratio Analysis ===")
    mixing_conds = [c for c in all_results["conditions"] if "alpha" in c]
    if len(mixing_conds) >= 2:
        best = max(mixing_conds, key=lambda c: c["solve_rate"])
        print(f"    Optimal α: {best['alpha']:.2f} ({best['solve_rate']*100:.1f}%)")
        # Direction contribution vs randomness
        det_only = next((c for c in mixing_conds if c["alpha"] == 0.0), None)
        stoch_only = next((c for c in mixing_conds if c["alpha"] == 1.0), None)
        if det_only and stoch_only:
            dir_effect = det_only["solve_rate"] - bl["solve_rate"]
            total_effect = stoch_only["solve_rate"] - bl["solve_rate"]
            if total_effect > 0:
                dir_frac = dir_effect / total_effect
                rand_frac = 1 - dir_frac
                print(f"    Direction contribution: {dir_frac*100:.0f}%")
                print(f"    Randomness contribution: {rand_frac*100:.0f}%")
                all_results["direction_fraction"] = round(dir_frac, 3)
                all_results["randomness_fraction"] = round(rand_frac, 3)

    # Visualization
    fig_path = visualize(all_results)

    # Save final
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)
    all_results["figure"] = fig_path

    results_path = os.path.join(RESULTS_DIR, "phase89_log.json")
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
    results, elapsed = run_phase89(model, tok, device)

    print(f"\n{'='*80}")
    print(f"  Phase 89 COMPLETE: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 89 complete.")
