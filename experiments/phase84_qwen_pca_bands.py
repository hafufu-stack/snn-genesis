"""
Phase 84: Qwen 7-Band PCA Decomposition — Reasoning Manifold Fine Structure
============================================================================

Phase 83 (Mistral-7B) showed:
  - Reasoning manifold occupies ~64 dimensions (PC 1-64 harmful)
  - Transition at PC ~65 (13.3% > baseline 6.7%)
  - PCA-bottom (43.3%) ≈ random (46.7%)

Phase 82 (Qwen2.5-7B) showed:
  - Random noise HARMS reasoning (30% < baseline 46.7%) — opposite of Mistral
  - PCA-bottom slightly helps (53.3% > baseline 46.7%)

This experiment asks: Does Qwen have the same ~64-dim reasoning manifold,
or is its representation fundamentally different?

Conditions (7 bands, N=30 each):
  1. baseline:        No noise
  2. random:          Full random (reference)
  3. pca_top4:        PC 1-4 only
  4. pca_mid_5_16:    PC 5-16 only
  5. pca_mid_17_64:   PC 17-64 only
  6. pca_mid_65_256:  PC 65-256 only
  7. pca_bottom:      PC 257+ only

All use Flash Annealing (first-10 linear σ decay), σ=0.15, Layer 16.
Step 1: Collect hidden states (50 baseline games).
Step 2: PCA decomposition.
Step 3: Run conditions with band-limited noise.
Step 4: Save PCA vectors to NPZ for Phase 85 reuse.

Total: 7 conditions × N=30 = 210 games + 50 collection = ~260 games
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
N_COLLECTION_GAMES = 50
LAYER_IDX = 16  # 28 layers total, ~57% depth

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
    print(f"  Baseline solve rate during collection: {solved_count/n_games*100:.1f}%")
    return states


# ===================================================
#  BAND-LIMITED PCA NOISE HOOK
# ===================================================

class BandNoiseHook:
    """Inject noise limited to a specific PCA band."""
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.mode = "random"
        self.band_basis = None
        self.handle = None

    def setup_random(self):
        self.mode = "random"
        self.band_basis = None

    def setup_band(self, Vt_band, device='cuda'):
        self.mode = "pca_band"
        self.band_basis = Vt_band.to(device)

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            hs = args[0]

            if hook_obj.mode == "random":
                noise = torch.randn_like(hs) * hook_obj.sigma
            elif hook_obj.mode == "pca_band":
                V = hook_obj.band_basis
                band_k = V.shape[0]
                b, s, d = hs.shape
                coeffs = torch.randn(b, s, band_k, dtype=hs.dtype, device=hs.device) * hook_obj.sigma
                noise = coeffs @ V
                noise = noise * math.sqrt(d / band_k)

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

# Bands adjusted for Qwen's 3584-dim hidden space
CONDITIONS = [
    {"name": "baseline",         "mode": "baseline",   "pc_start": 0,   "pc_end": 0},
    {"name": "random",           "mode": "random",     "pc_start": 0,   "pc_end": HIDDEN_DIM},
    {"name": "pca_top4",         "mode": "pca_band",   "pc_start": 0,   "pc_end": 4},
    {"name": "pca_mid_5_16",     "mode": "pca_band",   "pc_start": 4,   "pc_end": 16},
    {"name": "pca_mid_17_64",    "mode": "pca_band",   "pc_start": 16,  "pc_end": 64},
    {"name": "pca_mid_65_256",   "mode": "pca_band",   "pc_start": 64,  "pc_end": 256},
    {"name": "pca_bottom",       "mode": "pca_band",   "pc_start": 256, "pc_end": HIDDEN_DIM},
]


def run_phase84(model, tok, device):
    print(f"\n{'='*80}")
    print(f"  Phase 84: Qwen 7-Band PCA Decomposition")
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

    print(f"  Variance explained by bands:")
    bands = [(0, 4), (4, 16), (16, 64), (64, 256), (256, min(len(S), HIDDEN_DIM))]
    for start, end in bands:
        band_var = explained_var[start:end].sum() * 100
        print(f"    PC {start+1}-{end}: {band_var:.1f}%")

    print(f"  Cumulative variance:")
    for k in [4, 16, 64, 256]:
        if k <= len(cumvar):
            print(f"    Top-{k}: {cumvar[k-1]*100:.1f}%")

    Vt_torch = torch.tensor(Vt, dtype=torch.float16, device=device)

    # Save PCA vectors for Phase 85 reuse
    pca_save_path = os.path.join(RESULTS_DIR, "phase84_qwen_pca.npz")
    np.savez_compressed(pca_save_path, Vt=Vt, S=S, mean=mean,
                        explained_var=explained_var)
    print(f"  PCA vectors saved: {pca_save_path}")

    pca_info = {
        "n_samples": states.shape[0],
        "hidden_dim": states.shape[1],
        "band_variances": {},
        "cumulative_variances": {},
    }
    for start, end in bands:
        band_var = explained_var[start:end].sum() * 100
        pca_info["band_variances"][f"PC{start+1}-{end}"] = round(band_var, 2)
    for k in [4, 16, 64, 256]:
        if k <= len(cumvar):
            pca_info["cumulative_variances"][f"top{k}"] = round(cumvar[k-1] * 100, 2)

    all_results = {
        "experiment": "Phase 84: Qwen 7-Band PCA Decomposition",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "layer": LAYER_IDX,
        "hidden_dim": HIDDEN_DIM,
        "sigma": BASE_SIGMA,
        "n_per_condition": N_PER_CONDITION,
        "schedule": "first10_linear_decay",
        "pca_info": pca_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    # Step 3: Run conditions
    hook = BandNoiseHook()
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
        elif cond["mode"] == "pca_band":
            use_flash = True
            start, end = cond["pc_start"], cond["pc_end"]
            end = min(end, Vt_torch.shape[0])
            band_Vt = Vt_torch[start:end]
            hook.setup_band(band_Vt, device)
            band_var = explained_var[start:end].sum() * 100
            print(f"    Band PC {start+1}-{end}: {band_var:.1f}% variance, {end-start} dims")

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
            "pc_start": cond["pc_start"],
            "pc_end": cond["pc_end"],
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

    print(f"\n  === Comparison with Mistral Phase 83 ===")
    print(f"  Mistral reference:  baseline 6.7%, random 46.7%, pca_top4 3.3%, pca_bottom 43.3%")
    print(f"  Qwen results:")
    for cond in all_results["conditions"]:
        print(f"    {cond['condition']:20s}: {cond['solve_rate']*100:.1f}%")

    # Save
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)

    results_path = os.path.join(RESULTS_DIR, "phase84_log.json")
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
    results, elapsed = run_phase84(model, tok, device)

    print(f"\n{'='*80}")
    print(f"  Phase 84 COMPLETE: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 84 complete.")
