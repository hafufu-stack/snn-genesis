"""
Phase 86: Causal Verification — Deterministic Offset vs Stochastic Noise
=========================================================================

Phase 83 showed: PC 257+ noise is beneficial (43.3%) on Mistral-7B.
But this is correlational evidence. Is it the DIRECTION that matters,
or the RANDOMNESS (exploration diversity)?

This experiment tests deterministic offsets vs stochastic noise:
  1. baseline:              No noise
  2. random_noise:          Standard random noise (σ=0.15, Flash 10) — control
  3. deterministic_bottom:  Fixed offset in PC 257+ direction (not random!)
  4. deterministic_mid65:   Fixed offset in PC 65-256 direction

For deterministic conditions:
  - Compute a FIXED unit vector in the target PCA subspace
  - Add α * fixed_vector to hidden states (α matches σ=0.15 energy)
  - Same Flash Annealing schedule (first-10 linear decay)
  - The vector is the SAME for every token (no randomness)

If deterministic works → the DIRECTION is causally important
If deterministic fails → RANDOMNESS (exploration diversity) is causally important

N=30 per condition, Mistral-7B, Layer 18
Uses Phase 83 PCA vectors if available, else collects fresh.
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
#  CAUSAL OFFSET HOOK
# ===================================================

class CausalOffsetHook:
    """Inject either random noise or a FIXED deterministic offset.

    For deterministic mode:
      - A fixed unit vector is computed in the target PCA subspace
      - The SAME vector (scaled by alpha) is added every time
      - No randomness involved → tests causal role of direction
    """
    def __init__(self):
        self.active = False
        self.alpha = BASE_SIGMA  # scale factor (matched to σ)
        self.mode = "none"
        self.fixed_offset = None  # (d,) fixed vector for deterministic mode
        self.band_basis = None    # for stochastic band noise comparison
        self.handle = None

    def setup_none(self):
        self.mode = "none"

    def setup_random(self):
        self.mode = "random"

    def setup_deterministic(self, Vt_band, device='cuda'):
        """Create a fixed deterministic offset vector in the given PCA band.

        The offset is the mean of the band's PC directions, normalized,
        then scaled to have the same L2 norm as typical noise (σ * √d).
        """
        self.mode = "deterministic"
        V = Vt_band  # (band_k, d)
        # Fixed direction: mean of all band PC directions
        fixed_dir = V.mean(dim=0)  # (d,)
        fixed_dir = fixed_dir / fixed_dir.norm()  # unit vector
        self.fixed_offset = fixed_dir.to(device)

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.alpha <= 0:
                return args
            hs = args[0]

            if hook_obj.mode == "random":
                # Standard stochastic noise (control)
                noise = torch.randn_like(hs) * hook_obj.alpha

            elif hook_obj.mode == "deterministic":
                # Fixed offset — same vector every time (no randomness)
                offset = hook_obj.fixed_offset  # (d,)
                # Scale to match typical noise energy: σ * √d
                scaled = offset * hook_obj.alpha * math.sqrt(hs.shape[-1])
                noise = scaled.unsqueeze(0).unsqueeze(0).expand_as(hs)

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
                hook.alpha = BASE_SIGMA * (1.0 - legal_move_count / 10.0)
            else:
                hook.active = False; hook.alpha = 0.0
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
#  MAIN
# ===================================================

CONDITIONS = [
    {"name": "baseline",              "mode": "none"},
    {"name": "random_noise",          "mode": "random"},
    {"name": "deterministic_bottom",  "mode": "deterministic",  "pc_start": 256, "pc_end": 4096},
    {"name": "deterministic_mid65",   "mode": "deterministic",  "pc_start": 64,  "pc_end": 256},
]


def run_phase86(model, tok, device):
    print(f"\n{'='*80}")
    print(f"  Phase 86: Causal Verification — Deterministic Offset")
    print(f"  {len(CONDITIONS)} conditions x N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load or compute PCA
    # Try Phase 83's saved data first (Mistral)
    pca_path_83 = os.path.join(RESULTS_DIR, "phase83_mistral_pca.npz")
    # Also look in results folder for an existing Mistral PCA
    found_pca = False

    # Check if we saved Mistral PCA somewhere
    for candidate in [pca_path_83,
                      os.path.join(RESULTS_DIR, "phase86_mistral_pca.npz")]:
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
        pca_save = os.path.join(RESULTS_DIR, "phase86_mistral_pca.npz")
        np.savez_compressed(pca_save, Vt=Vt, S=S, mean=mean)
        print(f"  PCA saved: {pca_save}")

    Vt_torch = torch.tensor(Vt, dtype=torch.float16, device=device)
    print(f"  PCA loaded: Vt shape {Vt_torch.shape}")

    hook = CausalOffsetHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 86: Causal Verification — Deterministic Offset",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "n_per_condition": N_PER_CONDITION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    for cond_idx, cond in enumerate(CONDITIONS):
        cond_name = cond["name"]
        print(f"\n  [{cond_idx+1}/{len(CONDITIONS)}] Condition: {cond_name}")

        if cond["mode"] == "none":
            use_flash = False
            hook.setup_none()
        elif cond["mode"] == "random":
            use_flash = True
            hook.setup_random()
        elif cond["mode"] == "deterministic":
            use_flash = True
            start, end = cond["pc_start"], cond["pc_end"]
            end = min(end, Vt_torch.shape[0])
            band_Vt = Vt_torch[start:end]
            hook.setup_deterministic(band_Vt, device)
            print(f"    Deterministic offset in PC {start+1}-{end} ({end-start} dims)")

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
            summary["pc_start"] = cond["pc_start"]
            summary["pc_end"] = cond["pc_end"]
        all_results["conditions"].append(summary)
        print(f"    Solve rate: {solved/len(games)*100:.1f}% ({solved}/{len(games)})")

        # Intermediate save
        results_path = os.path.join(RESULTS_DIR, "phase86_log.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

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
        print(f"    {cond['condition']:25s}: {cond['solve_rate']*100:5.1f}% "
              f"(delta={delta*100:+5.1f}pp, p={p:.4f}) {sig}")

    # Causal interpretation
    print(f"\n  === Causal Interpretation ===")
    det_bottom = next(c for c in all_results["conditions"] if c["condition"] == "deterministic_bottom")
    print(f"    Random noise:           {rand['solve_rate']*100:.1f}%")
    print(f"    Deterministic bottom:   {det_bottom['solve_rate']*100:.1f}%")
    if det_bottom["solve_rate"] > bl["solve_rate"] + 0.10:
        print(f"    VERDICT: DIRECTION is causally important (deterministic offset works)")
    elif det_bottom["solve_rate"] < bl["solve_rate"] + 0.05:
        print(f"    VERDICT: RANDOMNESS is causally important (deterministic offset fails)")
    else:
        print(f"    VERDICT: AMBIGUOUS (need larger N)")

    # Save final
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)

    results_path = os.path.join(RESULTS_DIR, "phase86_log.json")
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
    results, elapsed = run_phase86(model, tok, device)

    print(f"\n{'='*80}")
    print(f"  Phase 86 COMPLETE: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 86 complete.")
