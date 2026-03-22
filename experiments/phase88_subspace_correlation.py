"""
Phase 88: Subspace-Targeted Inter-Layer Correlation Noise
=========================================================

The ultimate fusion experiment:
  Phase 68: L17+L18 correlation (ρ=+1) at σ=0.075 achieved 40% — but used
            full-rank noise that risks destroying reasoning PCs.
  Phase 83: PC 257+ noise is the safe subspace for Mistral (43.3%).
  Phase 86: Direction provides ~2/3, randomness ~1/3 of the SR benefit.

This experiment COMBINES them: correlated noise (L17+L18, ρ=+1)
RESTRICTED to the PC 257+ subspace. The reasoning manifold (top-64 PCs)
is protected by a "titanium shield" while the safe subspace resonates
across two layers simultaneously.

Conditions (N=30 each):
  1. baseline:           No noise
  2. random_full:        Random noise full-rank (σ=0.15, Flash 10) — Phase 83 ref
  3. random_bottom:      PC 257+ random noise (σ=0.15, Flash 10) — Phase 83 ref
  4. corr_bottom_pos:    PC 257+ corr noise L17+L18 ρ=+1 (σ=0.106, Flash 10)
  5. corr_bottom_low:    PC 257+ corr noise L17+L18 ρ=+1 (σ=0.075, Flash 10)

Mistral-7B, Flash Annealing
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
SIGMA_2L = BASE_SIGMA / math.sqrt(2)  # ≈ 0.106 for 2 layers
SIGMA_LOW = 0.075                      # Conservative 2-layer
N_PER_CONDITION = 30
LAYER_17 = 17
LAYER_18 = 18

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

    def register(self, model, layer_idx):
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
    collector.register(model, LAYER_18)
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
#  SUBSPACE CORRELATED NOISE MANAGER
# ===================================================

class SubspaceCorrelatedNoiseManager:
    """Inject correlated noise (ρ=+1) across L17+L18, restricted to PCA subspace.

    Combines Phase 68's inter-layer correlation with Phase 83's PCA band-limiting.
    Key innovation: noise is projected onto the PC 257+ subspace BEFORE injection,
    so the reasoning manifold (top-64 PCs) is completely protected.
    """
    def __init__(self, sigma=SIGMA_2L):
        self.sigma = sigma
        self.active = False
        self.mode = "none"       # none / random_full / random_bottom / corr_bottom
        self.band_basis = None   # (k, d) PCA basis for safe subspace
        self.base_noise = None   # shared noise for correlation
        self.handles = []

    def setup_none(self):
        self.mode = "none"

    def setup_random_full(self):
        self.mode = "random_full"

    def setup_random_bottom(self, Vt_band, device='cuda'):
        self.mode = "random_bottom"
        self.band_basis = Vt_band.to(device)

    def setup_corr_bottom(self, Vt_band, device='cuda'):
        self.mode = "corr_bottom"
        self.band_basis = Vt_band.to(device)

    def register(self, model):
        for layer_idx in [LAYER_17, LAYER_18]:
            hook = self._make_hook(layer_idx)
            handle = model.model.layers[layer_idx].register_forward_pre_hook(hook)
            self.handles.append(handle)

    def _make_hook(self, layer_idx):
        mgr = self
        def hook_fn(module, args):
            if not mgr.active or mgr.sigma <= 0:
                return args
            hs = args[0]

            if mgr.mode == "none":
                return args

            elif mgr.mode == "random_full":
                # Only inject on L18 for single-layer random (Phase 83 reference)
                if layer_idx != LAYER_18:
                    return args
                noise = torch.randn_like(hs) * mgr.sigma
                return (hs + noise,) + args[1:]

            elif mgr.mode == "random_bottom":
                # Only inject on L18 for single-layer band noise
                if layer_idx != LAYER_18:
                    return args
                V = mgr.band_basis  # (k, d)
                k = V.shape[0]
                d = hs.shape[-1]
                if hs.dim() == 3:
                    b, s, _ = hs.shape
                    coeffs = torch.randn(b, s, k, dtype=hs.dtype, device=hs.device) * mgr.sigma
                else:
                    s, _ = hs.shape
                    coeffs = torch.randn(s, k, dtype=hs.dtype, device=hs.device) * mgr.sigma
                noise = coeffs @ V
                noise = noise * math.sqrt(d / k)
                return (hs + noise,) + args[1:]

            elif mgr.mode == "corr_bottom":
                # Correlated noise: L17 generates base noise, L18 reuses it (ρ=+1)
                V = mgr.band_basis  # (k, d)
                k = V.shape[0]
                d = hs.shape[-1]

                if layer_idx == LAYER_17:
                    # First layer: generate new base noise in PC 257+ subspace
                    if hs.dim() == 3:
                        b, s, _ = hs.shape
                        coeffs = torch.randn(b, s, k, dtype=hs.dtype, device=hs.device) * mgr.sigma
                    else:
                        s, _ = hs.shape
                        coeffs = torch.randn(s, k, dtype=hs.dtype, device=hs.device) * mgr.sigma
                    noise = coeffs @ V
                    noise = noise * math.sqrt(d / k)
                    mgr.base_noise = noise  # Save for L18
                    return (hs + noise,) + args[1:]

                elif layer_idx == LAYER_18:
                    # Second layer: reuse SAME noise (ρ=+1 correlation)
                    if mgr.base_noise is not None and mgr.base_noise.shape == hs.shape:
                        noise = mgr.base_noise  # Perfect positive correlation
                    else:
                        # Fallback: generate fresh (should not happen normally)
                        if hs.dim() == 3:
                            b, s, _ = hs.shape
                            coeffs = torch.randn(b, s, k, dtype=hs.dtype, device=hs.device) * mgr.sigma
                        else:
                            s, _ = hs.shape
                            coeffs = torch.randn(s, k, dtype=hs.dtype, device=hs.device) * mgr.sigma
                        noise = coeffs @ V
                        noise = noise * math.sqrt(d / k)
                    return (hs + noise,) + args[1:]

            return args
        return hook_fn

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ===================================================
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, env, manager, use_flash=True, base_sigma=None):
    env.reset()
    error = None; consec_fail = 0; legal_move_count = 0
    if base_sigma is None:
        base_sigma = manager.sigma

    for step in range(MAX_STEPS):
        if use_flash:
            if legal_move_count < 10:
                manager.active = True
                manager.sigma = base_sigma * (1.0 - legal_move_count / 10.0)
            else:
                manager.active = False; manager.sigma = 0.0
        else:
            manager.active = False

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

    manager.active = False
    stats = env.stats()
    stats["steps_taken"] = step + 1
    return stats


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(all_results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Phase 88: Subspace-Targeted Inter-Layer Correlation\n"
                 "PC 257+ Protected Resonance (L17+L18, ρ=+1)",
                 fontsize=13, fontweight="bold")

    conds = all_results["conditions"]
    names = [c["condition"] for c in conds]
    rates = [c["solve_rate"] * 100 for c in conds]
    sigmas = [c.get("sigma_used", 0) for c in conds]

    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    bars = ax.bar(range(len(conds)), rates, color=colors[:len(conds)], alpha=0.85,
                  edgecolor="white", linewidth=2)

    bl_rate = rates[0]
    for i, (bar, val) in enumerate(zip(bars, rates)):
        label = f"{val:.1f}%"
        if i > 0:
            label += f"\nσ={sigmas[i]:.3f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Reference lines
    ax.axhline(y=bl_rate, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({bl_rate:.0f}%)")
    ax.axhline(y=40, color="red", linestyle=":", alpha=0.5, label="40% ceiling (Phase 68/86)")

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_ylim(0, max(rates) + 15)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase88_subspace_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

CONDITIONS = [
    {"name": "baseline",         "mode": "none",          "sigma": 0},
    {"name": "random_full",      "mode": "random_full",   "sigma": BASE_SIGMA},
    {"name": "random_bottom",    "mode": "random_bottom", "sigma": BASE_SIGMA},
    {"name": "corr_bottom_pos",  "mode": "corr_bottom",   "sigma": SIGMA_2L},
    {"name": "corr_bottom_low",  "mode": "corr_bottom",   "sigma": SIGMA_LOW},
]


def run_phase88(model, tok, device):
    print(f"\n{'='*80}")
    print(f"  Phase 88: Subspace-Targeted Inter-Layer Correlation")
    print(f"  {len(CONDITIONS)} conditions x N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load or compute PCA
    pca_path_83 = os.path.join(RESULTS_DIR, "phase83_mistral_pca.npz")
    pca_path_86 = os.path.join(RESULTS_DIR, "phase86_mistral_pca.npz")
    found_pca = False

    for candidate in [pca_path_83, pca_path_86]:
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
        pca_save = os.path.join(RESULTS_DIR, "phase88_mistral_pca.npz")
        np.savez_compressed(pca_save, Vt=Vt, S=S, mean=mean)
        print(f"  PCA saved: {pca_save}")

    Vt_torch = torch.tensor(Vt, dtype=torch.float16, device=device)
    # PC 257+ subspace (index 256 onwards)
    Vt_bottom = Vt_torch[256:]
    print(f"  PCA loaded. Bottom subspace: PC 257-{Vt_torch.shape[0]} ({Vt_bottom.shape[0]} dims)")

    manager = SubspaceCorrelatedNoiseManager()
    manager.register(model)

    all_results = {
        "experiment": "Phase 88: Subspace-Targeted Inter-Layer Correlation",
        "model": MODEL_SHORT,
        "layers": [LAYER_17, LAYER_18],
        "correlation": "+1",
        "subspace": "PC 257+",
        "n_per_condition": N_PER_CONDITION,
        "schedule": "first10_linear_decay",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    for cond_idx, cond in enumerate(CONDITIONS):
        cond_name = cond["name"]
        cond_sigma = cond["sigma"]
        print(f"\n  [{cond_idx+1}/{len(CONDITIONS)}] Condition: {cond_name} (σ={cond_sigma:.3f})")

        if cond["mode"] == "none":
            use_flash = False
            manager.setup_none()
        elif cond["mode"] == "random_full":
            use_flash = True
            manager.setup_random_full()
        elif cond["mode"] == "random_bottom":
            use_flash = True
            manager.setup_random_bottom(Vt_bottom, device)
        elif cond["mode"] == "corr_bottom":
            use_flash = True
            manager.setup_corr_bottom(Vt_bottom, device)

        games = []
        for trial in range(N_PER_CONDITION):
            env = HanoiEnv(n_disks=3, modified=True)
            stats = play_game(model, tok, env, manager, use_flash=use_flash,
                              base_sigma=cond_sigma)
            games.append(stats)

            if (trial + 1) % 10 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                elapsed = time.time() - t0
                print(f"    [{trial+1}/{N_PER_CONDITION}] Solve rate: {sr:.1f}% | {elapsed/60:.1f}min")

        solved = sum(1 for g in games if g["solved"])
        summary = {
            "condition": cond_name,
            "mode": cond["mode"],
            "sigma_used": cond_sigma,
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
        results_path = os.path.join(RESULTS_DIR, "phase88_log.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    manager.remove()

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

    # 40% ceiling check
    print(f"\n  === 40% Ceiling Check ===")
    for cond in all_results["conditions"][1:]:
        rate = cond["solve_rate"] * 100
        if rate > 40:
            print(f"    *** {cond['condition']}: {rate:.1f}% — CEILING BROKEN! ***")
        else:
            print(f"    {cond['condition']}: {rate:.1f}%")

    # Visualization
    fig_path = visualize(all_results)

    # Save final
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)
    all_results["figure"] = fig_path

    results_path = os.path.join(RESULTS_DIR, "phase88_log.json")
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
    results, elapsed = run_phase88(model, tok, device)

    print(f"\n{'='*80}")
    print(f"  Phase 88 COMPLETE: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 88 complete.")
