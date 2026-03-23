"""
Phase 95: Aha! Steering on Mistral — Deterministic Success Vector
=================================================================

Completes the 2x2 experimental matrix:

             Qwen          Mistral
Noise:     Phase 87(50%) | Phase 91(?)
Steering:  Phase 92(?)   | Phase 95(?)

Uses Phase 91's Differential PCA vectors (Mistral's discriminant axis)
for deterministic steering on Mistral-7B. Same design as Phase 92.

Conditions (N=30 each):
  1. baseline:           No noise
  2. diff_pca_noise:     Stochastic noise in Mistral's top-10 Diff-PCs
  3. aha_steering_top10: Deterministic Δ offset in top-10 Diff-PCs
  4. aha_plus_noise:     Δ offset + stochastic noise (50/50 energy split)
  5. anti_aha:           Deterministic offset in OPPOSITE direction (-Δ)

Mistral-7B-Instruct-v0.3, Layer 18, Flash Annealing, σ=0.15

Total: 5 x 30 = 150 games
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
#  AHA! STEERING HOOK (Mistral version)
# ===================================================

class AhaSteeringHook:
    """Inject deterministic success vector offset and/or stochastic noise
    in the Differential PCA discriminant direction for Mistral.
    """
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.mode = "baseline"
        self.band_basis = None
        self.fixed_offset = None
        self.handle = None

    def setup_baseline(self):
        self.mode = "baseline"

    def setup_diff_noise(self, Vt_band, device='cuda'):
        self.mode = "diff_noise"
        self.band_basis = Vt_band.to(device)

    def setup_aha_steering(self, Vt_band, diff_unit, device='cuda', anti=False):
        self.mode = "anti_aha" if anti else "aha_steering"
        V = Vt_band.to(device)

        du = torch.tensor(diff_unit, dtype=torch.float32, device=device)
        proj_coeffs = V.float() @ du
        proj = proj_coeffs @ V.float()
        proj_norm = proj.norm()
        if proj_norm > 1e-8:
            proj = proj / proj_norm

        if anti:
            proj = -proj

        self.fixed_offset = proj.half()
        self.band_basis = V

    def setup_aha_plus_noise(self, Vt_band, diff_unit, device='cuda'):
        self.mode = "aha_plus_noise"
        V = Vt_band.to(device)

        du = torch.tensor(diff_unit, dtype=torch.float32, device=device)
        proj_coeffs = V.float() @ du
        proj = proj_coeffs @ V.float()
        proj_norm = proj.norm()
        if proj_norm > 1e-8:
            proj = proj / proj_norm

        self.fixed_offset = proj.half()
        self.band_basis = V

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            if hook_obj.mode == "baseline":
                return args
            hs = args[0]

            if hook_obj.mode == "diff_noise":
                V = hook_obj.band_basis
                k = V.shape[0]
                d = hs.shape[-1]
                if hs.dim() == 3:
                    b, s, _ = hs.shape
                    coeffs = torch.randn(b, s, k, dtype=hs.dtype, device=hs.device) * hook_obj.sigma
                else:
                    s, _ = hs.shape
                    coeffs = torch.randn(s, k, dtype=hs.dtype, device=hs.device) * hook_obj.sigma
                noise = coeffs @ V
                noise = noise * math.sqrt(d / k)
                return (hs + noise,) + args[1:]

            elif hook_obj.mode in ("aha_steering", "anti_aha"):
                offset = hook_obj.fixed_offset
                scaled = offset * hook_obj.sigma * math.sqrt(hs.shape[-1])
                if hs.dim() == 3:
                    noise = scaled.unsqueeze(0).unsqueeze(0).expand_as(hs)
                else:
                    noise = scaled.unsqueeze(0).expand_as(hs)
                return (hs + noise,) + args[1:]

            elif hook_obj.mode == "aha_plus_noise":
                V = hook_obj.band_basis
                k = V.shape[0]
                d = hs.shape[-1]
                # Half energy deterministic
                offset = hook_obj.fixed_offset
                det_scale = hook_obj.sigma * math.sqrt(d) * 0.5
                det_noise = offset * det_scale
                if hs.dim() == 3:
                    det_noise = det_noise.unsqueeze(0).unsqueeze(0).expand_as(hs)
                else:
                    det_noise = det_noise.unsqueeze(0).expand_as(hs)
                # Half energy stochastic
                stoch_sigma = hook_obj.sigma * 0.5
                if hs.dim() == 3:
                    b, s, _ = hs.shape
                    coeffs = torch.randn(b, s, k, dtype=hs.dtype, device=hs.device) * stoch_sigma
                else:
                    s, _ = hs.shape
                    coeffs = torch.randn(s, k, dtype=hs.dtype, device=hs.device) * stoch_sigma
                stoch_noise = coeffs @ V * math.sqrt(d / k)
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

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle("Phase 95: Aha! Steering — Deterministic Success Vector on Mistral\n"
                 "Completing the 2×2 matrix: Does steering work on Mistral too?",
                 fontsize=12, fontweight="bold")

    conds = all_results["conditions"]
    names = [c["condition"] for c in conds]
    rates = [c["solve_rate"] * 100 for c in conds]
    colors = ["#9E9E9E", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    bars = ax.bar(range(len(conds)), rates, color=colors[:len(conds)], alpha=0.85,
                  edgecolor="white", linewidth=2)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=40, color="red", linestyle=":", alpha=0.5, label="40% ceiling")
    ax.axhline(y=43.3, color="green", linestyle="--", alpha=0.5, label="Phase 83 record (43.3%)")

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_ylim(0, max(rates) + 12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase95_mistral_aha_steering.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

CONDITIONS = [
    {"name": "baseline",           "mode": "baseline"},
    {"name": "diff_pca_noise",     "mode": "diff_noise"},
    {"name": "aha_steering_top10", "mode": "aha_steering"},
    {"name": "aha_plus_noise",     "mode": "aha_plus_noise"},
    {"name": "anti_aha",           "mode": "anti_aha"},
]


def run_phase95(model, tok, device):
    print(f"\n{'='*80}")
    print(f"  Phase 95: Aha! Steering — Deterministic Success Vector on Mistral")
    print(f"  {len(CONDITIONS)} conditions x N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load Phase 91's Differential PCA
    diff_pca_path = os.path.join(RESULTS_DIR, "phase91_diff_pca.npz")
    if not os.path.exists(diff_pca_path):
        raise FileNotFoundError(f"Phase 91 Diff-PCA not found: {diff_pca_path}\n"
                                f"Run Phase 91 first to generate Mistral discriminant vectors.")

    print(f"  Loading Phase 91 Diff-PCA: {diff_pca_path}")
    data = np.load(diff_pca_path)
    diff_Vt = data["Vt"]
    diff_unit = data["diff_unit"]
    print(f"  Loaded: Vt shape {diff_Vt.shape}, diff_unit norm {np.linalg.norm(diff_unit):.4f}")

    TOP_K = 10
    diff_Vt_top = torch.tensor(diff_Vt[:TOP_K], dtype=torch.float16, device=device)
    print(f"  Using top-{TOP_K} discriminant PCs")

    hook = AhaSteeringHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 95: Aha! Steering — Deterministic Success Vector on Mistral",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "layer": LAYER_IDX,
        "hidden_dim": HIDDEN_DIM,
        "sigma": BASE_SIGMA,
        "diff_pca_source": "phase91_diff_pca.npz",
        "top_k_pcs": TOP_K,
        "n_per_condition": N_PER_CONDITION,
        "schedule": "first10_linear_decay",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    for cond_idx, cond in enumerate(CONDITIONS):
        cond_name = cond["name"]
        print(f"\n  [{cond_idx+1}/{len(CONDITIONS)}] Condition: {cond_name}")

        if cond["mode"] == "baseline":
            use_flash = False
            hook.setup_baseline()
        elif cond["mode"] == "diff_noise":
            use_flash = True
            hook.setup_diff_noise(diff_Vt_top, device)
        elif cond["mode"] == "aha_steering":
            use_flash = True
            hook.setup_aha_steering(diff_Vt_top, diff_unit, device, anti=False)
        elif cond["mode"] == "aha_plus_noise":
            use_flash = True
            hook.setup_aha_plus_noise(diff_Vt_top, diff_unit, device)
        elif cond["mode"] == "anti_aha":
            use_flash = True
            hook.setup_aha_steering(diff_Vt_top, diff_unit, device, anti=True)

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
        all_results["conditions"].append(summary)
        print(f"    Solve rate: {solved/len(games)*100:.1f}% ({solved}/{len(games)})")

        # Intermediate save
        results_path = os.path.join(RESULTS_DIR, "phase95_log.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    hook.remove()

    # Statistical analysis
    bl = all_results["conditions"][0]
    noise_cond = next((c for c in all_results["conditions"] if c["condition"] == "diff_pca_noise"), None)

    print(f"\n  === Fisher Exact Tests vs Baseline ===")
    for cond in all_results["conditions"][1:]:
        table = [[cond["n_solved"], cond["n_total"]-cond["n_solved"]],
                 [bl["n_solved"], bl["n_total"]-bl["n_solved"]]]
        _, p = fisher_exact(table)
        delta = cond["solve_rate"] - bl["solve_rate"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    {cond['condition']:20s}: {cond['solve_rate']*100:5.1f}% "
              f"(delta={delta*100:+5.1f}pp, p={p:.4f}) {sig}")

    if noise_cond:
        print(f"\n  === Aha! Steering vs Stochastic Noise ===")
        for cond in all_results["conditions"]:
            if cond["condition"] in ("aha_steering_top10", "aha_plus_noise"):
                table = [[cond["n_solved"], cond["n_total"]-cond["n_solved"]],
                         [noise_cond["n_solved"], noise_cond["n_total"]-noise_cond["n_solved"]]]
                _, p = fisher_exact(table)
                delta = cond["solve_rate"] - noise_cond["solve_rate"]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                print(f"    {cond['condition']:20s}: {cond['solve_rate']*100:5.1f}% "
                      f"(vs noise: {delta*100:+5.1f}pp, p={p:.4f}) {sig}")

    # 2x2 Matrix summary
    print(f"\n  === 2x2 Matrix Summary ===")
    print(f"               Qwen          Mistral")
    print(f"  Noise:     Phase 87(50%) | Phase 91(?)")
    aha_r = next((c["solve_rate"]*100 for c in all_results["conditions"]
                  if c["condition"] == "aha_steering_top10"), 0)
    print(f"  Steering:  Phase 92(?)   | Phase 95({aha_r:.1f}%)")

    # Visualization
    fig_path = visualize(all_results)

    # Save final
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)
    all_results["figure"] = fig_path

    results_path = os.path.join(RESULTS_DIR, "phase95_log.json")
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
    results, elapsed = run_phase95(model, tok, device)

    print(f"\n{'='*80}")
    print(f"  Phase 95 COMPLETE: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 95 complete.")
