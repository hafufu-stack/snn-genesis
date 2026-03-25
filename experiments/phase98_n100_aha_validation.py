"""
Phase 98: N=100 Validation of Mistral aha+noise 46.7%
======================================================

Phase 80 showed Qwen's 50% (N=30) → 40% (N=100) honest correction.
Phase 95's aha+noise = 46.7% is v18's headline result, but at N=30 only.

This experiment validates the Mistral aha+noise result at N=100.

Conditions:
  1. baseline:      No noise, N=100
  2. aha_plus_noise: Aha! + stochastic noise, N=100

Mistral-7B-Instruct-v0.3, Layer 18, Flash Annealing, σ=0.15, vAha from Phase 91

Total: 2 x 100 = 200 games
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
N_PER_CONDITION = 100
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
#  AHA+ NOISE HOOK
# ===================================================

class AhaSteeringHook:
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.mode = "baseline"
        self.fixed_offset = None
        self.handle = None

    def setup_baseline(self):
        self.mode = "baseline"

    def setup_aha_plus_noise(self, diff_unit_vec, device='cuda'):
        self.mode = "aha_plus_noise"
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
            if hook_obj.mode == "aha_plus_noise":
                d = hs.shape[-1]
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

def play_game(model, tok, hook, use_flash=True):
    env = HanoiEnv(n_disks=3, modified=True)
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
    fig.suptitle("Phase 98: N=100 Validation of Mistral aha+noise\n"
                 "Does the 46.7% (N=30) hold at N=100?",
                 fontsize=13, fontweight="bold")

    # Panel 1: Solve rates
    ax = axes[0]
    conds = all_results["conditions"]
    names = [c["condition"] for c in conds]
    rates = [c["solve_rate"] * 100 for c in conds]
    colors = ["#9E9E9E", "#9C27B0"]
    bars = ax.bar(range(len(conds)), rates, color=colors, alpha=0.85, width=0.5,
                  edgecolor="white", linewidth=2)
    for bar, val, cond in zip(bars, rates, conds):
        ci_lo = cond.get("ci95_low", 0) * 100
        ci_hi = cond.get("ci95_high", 0) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.1f}%\n[{ci_lo:.1f}-{ci_hi:.1f}]",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=46.7, color="gold", linestyle="--", alpha=0.6, label="Phase 95 (N=30): 46.7%")
    ax.axhline(y=40, color="red", linestyle=":", alpha=0.5, label="40% ceiling")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title(f"N=100 Validation", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(rates) + 20)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2: Running solve rate
    ax = axes[1]
    for cond in conds:
        if "running_solve_rate" in cond:
            ax.plot(range(1, len(cond["running_solve_rate"])+1),
                    [r*100 for r in cond["running_solve_rate"]],
                    label=cond["condition"], linewidth=2)
    ax.axhline(y=46.7, color="gold", linestyle="--", alpha=0.6, label="Phase 95 (N=30)")
    ax.set_xlabel("Game Number", fontsize=11)
    ax.set_ylabel("Running Solve Rate (%)", fontsize=11)
    ax.set_title("Convergence to True Rate", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase98_n100_aha_validation.png")
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
    print(f"  Phase 98: N=100 Validation of Mistral aha+noise 46.7%")
    print(f"  Scientific integrity check — does N=30 extrapolate to N=100?")
    print(f"{'='*80}")

    t0 = time.time()

    # Load Phase 91's Diff-PCA
    diff_pca_path = os.path.join(RESULTS_DIR, "phase91_diff_pca.npz")
    if not os.path.exists(diff_pca_path):
        raise FileNotFoundError(f"Phase 91 Diff-PCA not found: {diff_pca_path}")
    data = np.load(diff_pca_path)
    diff_unit = data["diff_unit"]
    print(f"  Loaded diff_unit: norm={np.linalg.norm(diff_unit):.4f}")

    model, tok = load_model()
    device = next(model.parameters()).device
    hook = AhaSteeringHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 98: N=100 Validation of Mistral aha+noise",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "layer": LAYER_IDX,
        "hidden_dim": HIDDEN_DIM,
        "sigma": BASE_SIGMA,
        "diff_pca_source": "phase91_diff_pca.npz",
        "n_per_condition": N_PER_CONDITION,
        "schedule": "first10_linear_decay",
        "reference_n30": 0.467,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    conditions = [
        {"name": "baseline", "vector": None, "use_flash": False},
        {"name": "aha_plus_noise", "vector": diff_unit, "use_flash": True},
    ]

    for cond in conditions:
        print(f"\n  === Condition: {cond['name']} (N={N_PER_CONDITION}) ===")
        if cond["vector"] is None:
            hook.setup_baseline()
        else:
            hook.setup_aha_plus_noise(cond["vector"], device)

        games = []
        running_solved = 0
        running_rates = []

        for trial in range(N_PER_CONDITION):
            stats = play_game(model, tok, hook, use_flash=cond["use_flash"])
            games.append(stats)
            if stats["solved"]:
                running_solved += 1
            running_rates.append(running_solved / (trial + 1))

            if (trial + 1) % 10 == 0:
                sr = running_solved / (trial + 1) * 100
                elapsed = time.time() - t0
                print(f"    [{trial+1}/{N_PER_CONDITION}] Running rate: {sr:.1f}% | {elapsed/60:.1f}min")

        solved = sum(1 for g in games if g["solved"])
        rate = solved / len(games)

        # Wilson confidence interval
        from math import sqrt
        n = len(games)
        z = 1.96
        p_hat = rate
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denom
        margin = z * sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
        ci_lo = max(0, center - margin)
        ci_hi = min(1, center + margin)

        summary = {
            "condition": cond["name"],
            "solve_rate": round(rate, 4),
            "n_solved": solved,
            "n_total": n,
            "ci95_low": round(ci_lo, 4),
            "ci95_high": round(ci_hi, 4),
            "running_solve_rate": [round(r, 4) for r in running_rates],
            "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
            "avg_illegal_moves": round(np.mean([g["illegal_moves"] for g in games]), 2),
            "games": games
        }
        all_results["conditions"].append(summary)
        print(f"\n  {cond['name']}: {solved}/{n} = {rate*100:.1f}% "
              f"[95% CI: {ci_lo*100:.1f}-{ci_hi*100:.1f}%]")

        # Intermediate save
        results_path = os.path.join(RESULTS_DIR, "phase98_log.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    hook.remove()

    # Statistical comparison
    bl = all_results["conditions"][0]
    aha = all_results["conditions"][1]
    table = [[aha["n_solved"], aha["n_total"]-aha["n_solved"]],
             [bl["n_solved"], bl["n_total"]-bl["n_solved"]]]
    _, p = fisher_exact(table)

    print(f"\n  === Statistical Summary ===")
    print(f"  baseline:      {bl['solve_rate']*100:.1f}% (N={bl['n_total']})")
    print(f"  aha+noise:     {aha['solve_rate']*100:.1f}% (N={aha['n_total']})")
    print(f"  Fisher exact:  p={p:.6f}")
    print(f"  95% CI:        [{aha['ci95_low']*100:.1f}, {aha['ci95_high']*100:.1f}]%")
    print(f"  Phase 95 (N=30): 46.7%")

    # Verdict
    n30_rate = 0.467
    n100_rate = aha["solve_rate"]
    if n100_rate >= 0.40:
        verdict = "VALIDATED"
        print(f"\n  VERDICT: {verdict} — N=100 rate ({n100_rate*100:.1f}%) >= 40%")
    elif n100_rate >= 0.30:
        verdict = "HONEST_CORRECTION"
        print(f"\n  VERDICT: {verdict} — N=100 rate ({n100_rate*100:.1f}%) lower but still beneficial")
    else:
        verdict = "REGRESSION"
        print(f"\n  VERDICT: {verdict} — N=100 rate ({n100_rate*100:.1f}%) dropped significantly")

    all_results["verdict"] = verdict
    all_results["fisher_p"] = round(p, 6)
    all_results["n30_vs_n100_delta"] = round((n100_rate - n30_rate) * 100, 1)

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
    print(f"\n Phase 98 complete.")
