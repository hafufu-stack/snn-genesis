"""
Phase 90: Temporal Noise Correlation — Thought Inertia via AR(1)
================================================================

All previous SNN noise was IID: each token generation step received
completely independent random noise. But biological neural fluctuations
have TEMPORAL structure — oscillations, drift, and inertia.

This experiment adds temporal correlation to the noise process:
  z_t = ρ · z_{t-1} + √(1-ρ²) · ε_t    (AR(1) / Ornstein-Uhlenbeck)

Where:
  ρ = temporal correlation strength
  ρ=0.0 → IID noise (standard, previous experiments)
  ρ=0.5 → moderate temporal correlation
  ρ=0.8 → strong temporal correlation (slow drift)
  ρ=0.95 → very strong inertia (persistent exploration direction)

All noise restricted to PC 257+ subspace (safe band).
The hypothesis: temporal correlation creates "exploration momentum" —
instead of random twitching, the model explores one direction for
several steps before drifting elsewhere. This mimics deliberate
"what-if" reasoning.

Conditions (N=30 each):
  1. baseline:     No noise
  2. rho_0.00:     IID noise (ρ=0, standard reference)
  3. rho_0.50:     Moderate temporal correlation
  4. rho_0.80:     Strong temporal correlation
  5. rho_0.95:     Very strong inertia

Mistral-7B, Layer 18, PC 257+, Flash Annealing, σ=0.15
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
#  HIDDEN STATE COLLECTION (for PCA)
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
#  TEMPORAL CORRELATED NOISE HOOK (AR(1) Process)
# ===================================================

class TemporalNoiseHook:
    """Inject temporally correlated noise via AR(1) / Ornstein-Uhlenbeck process.

    z_t = rho * z_{t-1} + sqrt(1 - rho^2) * epsilon_t

    The noise vector persists across token generation steps within a game,
    creating "exploration momentum" — the model consistently explores in
    one direction before gradually drifting elsewhere.

    All noise is restricted to the PC 257+ subspace.
    """
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.rho = 0.0          # temporal correlation: 0=IID, 1=frozen
        self.mode = "baseline"  # baseline / temporal
        self.band_basis = None  # (k, d) PCA basis for safe subspace
        self.z_prev = None      # (d,) previous noise vector (temporal state)
        self.handle = None

    def setup_baseline(self):
        self.mode = "baseline"
        self.z_prev = None

    def setup_temporal(self, rho, Vt_band, device='cuda'):
        self.mode = "temporal"
        self.rho = rho
        self.band_basis = Vt_band.to(device)
        self.z_prev = None  # Reset temporal state for new game

    def reset_temporal_state(self):
        """Call at the start of each game to reset AR(1) state."""
        self.z_prev = None

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

            # Generate fresh innovation noise in PC 257+ subspace
            if hs.dim() == 3:
                # Use last token's shape for the noise (1, 1, d)
                coeffs = torch.randn(1, 1, k, dtype=hs.dtype, device=hs.device) * hook_obj.sigma
            else:
                coeffs = torch.randn(1, k, dtype=hs.dtype, device=hs.device) * hook_obj.sigma
            epsilon = coeffs @ V  # innovation noise
            epsilon = epsilon * math.sqrt(d / k)

            # Flatten to (d,) for AR(1) state tracking
            eps_flat = epsilon.view(-1)  # (d,)

            # AR(1) update: z_t = rho * z_{t-1} + sqrt(1-rho^2) * epsilon_t
            rho = hook_obj.rho
            if hook_obj.z_prev is None or hook_obj.z_prev.shape != eps_flat.shape:
                # First step: z_0 = epsilon_0 (no history)
                z_t = eps_flat
            else:
                z_t = rho * hook_obj.z_prev + math.sqrt(1 - rho**2) * eps_flat

            hook_obj.z_prev = z_t.detach()  # Save for next step

            # Expand back to match hs shape and add
            noise = z_t.view(epsilon.shape).expand_as(hs)
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
    hook.reset_temporal_state()  # Fresh AR(1) state per game
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
    fig.suptitle("Phase 90: Temporal Noise Correlation — Thought Inertia\n"
                 r"AR(1): $z_t = \rho \cdot z_{t-1} + \sqrt{1-\rho^2} \cdot \epsilon_t$ in PC 257+",
                 fontsize=13, fontweight="bold")

    conds = all_results["conditions"]
    names = [c["condition"] for c in conds]
    rates = [c["solve_rate"] * 100 for c in conds]

    # Panel 1: Bar chart
    ax = axes[0]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    bars = ax.bar(range(len(conds)), rates, color=colors[:len(conds)], alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val, cond in zip(bars, rates, conds):
        label = f"{val:.1f}%"
        if "rho" in cond:
            label += f"\nρ={cond['rho']:.2f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Solve Rate (%)", fontsize=11)
    ax.set_title("Solve Rate by Temporal Correlation", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(rates) + 12)
    ax.axhline(y=40, color="red", linestyle=":", alpha=0.5, label="40% ceiling")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2: ρ vs Solve Rate line plot
    ax = axes[1]
    temporal_conds = [c for c in conds if "rho" in c]
    if temporal_conds:
        rhos = [c["rho"] for c in temporal_conds]
        t_rates = [c["solve_rate"] * 100 for c in temporal_conds]
        ax.plot(rhos, t_rates, "o-", color="#9C27B0", linewidth=2.5, markersize=12,
                markerfacecolor="white", markeredgewidth=2)
        for r, rate in zip(rhos, t_rates):
            ax.annotate(f"{rate:.0f}%", (r, rate), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=11, fontweight="bold")

        bl_rate = conds[0]["solve_rate"] * 100
        ax.axhline(y=bl_rate, color="gray", linestyle="--", alpha=0.5,
                   label=f"Baseline ({bl_rate:.0f}%)")
        ax.axhline(y=40, color="red", linestyle=":", alpha=0.5, label="40% ceiling")

    ax.set_xlabel("ρ (Temporal Correlation Strength)", fontsize=12)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Effect of Noise Persistence", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.05, 1.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase90_temporal_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

RHOS = [0.0, 0.50, 0.80, 0.95]

CONDITIONS = [{"name": "baseline", "rho": None}] + \
             [{"name": f"rho_{r:.2f}", "rho": r} for r in RHOS]


def run_phase90(model, tok, device):
    print(f"\n{'='*80}")
    print(f"  Phase 90: Temporal Noise Correlation (AR(1) Process)")
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
        pca_save = os.path.join(RESULTS_DIR, "phase90_mistral_pca.npz")
        np.savez_compressed(pca_save, Vt=Vt, S=S, mean=mean)
        print(f"  PCA saved: {pca_save}")

    Vt_torch = torch.tensor(Vt, dtype=torch.float16, device=device)
    Vt_bottom = Vt_torch[256:]  # PC 257+
    print(f"  PCA loaded. Bottom subspace: {Vt_bottom.shape[0]} dims")

    hook = TemporalNoiseHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 90: Temporal Noise Correlation (AR(1))",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "subspace": "PC 257+",
        "n_per_condition": N_PER_CONDITION,
        "schedule": "first10_linear_decay",
        "ar1_formula": "z_t = rho * z_{t-1} + sqrt(1-rho^2) * epsilon_t",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    for cond_idx, cond in enumerate(CONDITIONS):
        cond_name = cond["name"]
        rho = cond["rho"]
        print(f"\n  [{cond_idx+1}/{len(CONDITIONS)}] Condition: {cond_name}")

        if rho is None:
            use_flash = False
            hook.setup_baseline()
        else:
            use_flash = True
            hook.setup_temporal(rho, Vt_bottom, device)
            autocorr_halflife = -1 / math.log(rho + 1e-8) if rho > 0 else 0
            print(f"    ρ={rho:.2f} | autocorrelation half-life: {autocorr_halflife:.1f} steps")

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
        if rho is not None:
            summary["rho"] = rho
        all_results["conditions"].append(summary)
        print(f"    Solve rate: {solved/len(games)*100:.1f}% ({solved}/{len(games)})")

        # Intermediate save
        results_path = os.path.join(RESULTS_DIR, "phase90_log.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    hook.remove()

    # Statistical analysis
    bl = all_results["conditions"][0]
    iid = next((c for c in all_results["conditions"] if c.get("rho") == 0.0), None)

    print(f"\n  === Fisher Exact Tests vs Baseline ===")
    for cond in all_results["conditions"][1:]:
        table = [[cond["n_solved"], cond["n_total"]-cond["n_solved"]],
                 [bl["n_solved"], bl["n_total"]-bl["n_solved"]]]
        _, p = fisher_exact(table)
        delta = cond["solve_rate"] - bl["solve_rate"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    {cond['condition']:15s}: {cond['solve_rate']*100:5.1f}% "
              f"(delta={delta*100:+5.1f}pp, p={p:.4f}) {sig}")

    if iid:
        print(f"\n  === Fisher Exact Tests vs IID (ρ=0) ===")
        for cond in all_results["conditions"][1:]:
            if cond.get("rho", -1) == 0.0:
                continue
            if "rho" not in cond:
                continue
            table = [[cond["n_solved"], cond["n_total"]-cond["n_solved"]],
                     [iid["n_solved"], iid["n_total"]-iid["n_solved"]]]
            _, p = fisher_exact(table)
            delta = cond["solve_rate"] - iid["solve_rate"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"    ρ={cond['rho']:.2f}: {cond['solve_rate']*100:5.1f}% "
                  f"(vs IID: {delta*100:+5.1f}pp, p={p:.4f}) {sig}")

    # Visualization
    fig_path = visualize(all_results)

    # Save final
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)
    all_results["figure"] = fig_path

    results_path = os.path.join(RESULTS_DIR, "phase90_log.json")
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
    results, elapsed = run_phase90(model, tok, device)

    print(f"\n{'='*80}")
    print(f"  Phase 90 COMPLETE: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 90 complete.")
