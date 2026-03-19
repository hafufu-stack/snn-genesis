"""
Phase 82: Cross-Architecture PCA Comparison
=============================================

Phase 71 (Mistral-7B) showed that Top-4 PCs encode the reasoning axis.
Is this universal across LLM architectures?

This experiment runs the SAME PCA analysis on:
  1. Qwen2.5-7B-Instruct   (Alibaba, different architecture)
  2. Phi-3-mini-4k-instruct (Microsoft, 3.8B params, 32 layers)

For each model:
  Step 1: Collect hidden states from baseline Hanoi games (50 games)
  Step 2: PCA decomposition — measure variance in top-k PCs
  Step 3: Test: baseline, random, pca_top4, pca_bottom
  N=30 per condition

If Top-4 PC concentrations are similar → "Universal Reasoning Manifold" claim.
If different → architecture-specific representation.
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
SEED = 2026
MAX_STEPS = 50
BASE_SIGMA = 0.15
N_PER_CONDITION = 30
N_COLLECTION_GAMES = 50

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)




# Models to test — each with architecture-specific layer index
# Layer index chosen as ~56% depth (similar to L18/32 for Mistral)
MODELS = [
    {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "short": "Qwen2.5-7B",
        "layer_idx": 16,   # 28 layers total, ~57%
        "hidden_dim": 3584,
    },
    {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "short": "Phi-3-mini",
        "layer_idx": 18,   # 32 layers total, ~56%
        "hidden_dim": 3072,
    },
]


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
#  MODEL + GENERATION (generic)
# ===================================================

def load_model(model_name):
    print(f"\n Loading {model_name}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    # Fix rope_scaling compatibility for Phi-3 and similar models
    # When using trust_remote_code=True, the model's own code handles RoPE.
    # HF's built-in rope init may crash on non-standard rope_scaling configs,
    # so we disable it and let the custom code handle it.
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
        rs = config.rope_scaling
        if 'type' not in rs:
            # Custom models (Phi-3 etc) handle RoPE themselves
            # Setting to None prevents HF's _init_rope from crashing
            config.rope_scaling = None
            print(f"  Note: Disabled rope_scaling for custom model (was: {rs})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    n_layers = len(model.model.layers)
    print(f"  Done: {n_layers} layers")
    return model, tok, n_layers

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
#  HIDDEN STATE COLLECTION (architecture-generic)
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


def collect_hidden_states(model, tok, layer_idx, n_games=N_COLLECTION_GAMES):
    print(f"\n  Collecting hidden states from {n_games} baseline games (Layer {layer_idx})...")
    collector = CollectorHook()
    collector.register(model, layer_idx)

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
    baseline_rate = solved_count / n_games
    states = np.array(collector.collected)
    print(f"  Collected {states.shape[0]} vectors of dim {states.shape[1]}")
    print(f"  Baseline solve rate: {baseline_rate*100:.1f}%")
    return states, baseline_rate


# ===================================================
#  PCA NOISE HOOK (generic)
# ===================================================

class PCANoiseHook:
    """Test noise aligned with or avoiding specific PCA bands."""
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.mode = "random"
        self.pca_basis = None  # (k, d) for pca_top mode
        self.handle = None

    def setup_random(self):
        self.mode = "random"
        self.pca_basis = None

    def setup_pca_top(self, Vt_top, device='cuda'):
        self.mode = "pca_top"
        self.pca_basis = Vt_top.to(device)

    def setup_pca_bottom(self, Vt_bottom, hidden_dim, device='cuda'):
        self.mode = "pca_bottom"
        self.pca_basis = Vt_bottom.to(device)
        self.hidden_dim = hidden_dim

    def register(self, model, layer_idx):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            hs = args[0]

            if hook_obj.mode == "random":
                noise = torch.randn_like(hs) * hook_obj.sigma

            elif hook_obj.mode == "pca_top":
                V = hook_obj.pca_basis
                k = V.shape[0]
                b, s, d = hs.shape
                coeffs = torch.randn(b, s, k, dtype=hs.dtype, device=hs.device) * hook_obj.sigma
                noise = coeffs @ V
                noise = noise * math.sqrt(d / k)

            elif hook_obj.mode == "pca_bottom":
                V = hook_obj.pca_basis
                k = V.shape[0]
                b, s, d = hs.shape
                coeffs = torch.randn(b, s, k, dtype=hs.dtype, device=hs.device) * hook_obj.sigma
                noise = coeffs @ V
                noise = noise * math.sqrt(d / k)

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
#  RUN ONE MODEL
# ===================================================

def run_model_experiment(model_config):
    model_name = model_config["name"]
    model_short = model_config["short"]
    layer_idx = model_config["layer_idx"]
    hidden_dim = model_config["hidden_dim"]

    print(f"\n{'='*80}")
    print(f"  Phase 82: {model_short} (Layer {layer_idx})")
    print(f"{'='*80}")

    t0 = time.time()

    model, tok, n_layers = load_model(model_name)
    device = next(model.parameters()).device

    # Step 1: Collect hidden states
    states, baseline_rate = collect_hidden_states(model, tok, layer_idx)

    # Step 2: PCA
    print(f"\n  Computing PCA on {states.shape[0]} x {states.shape[1]} matrix...")
    mean = states.mean(axis=0)
    centered = states - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = S**2 / (S**2).sum()
    cumvar = np.cumsum(explained_var)

    print(f"  Variance explained (Top-k PCs):")
    for k in [4, 16, 64, 256]:
        if k <= len(cumvar):
            print(f"    Top-{k}: {cumvar[k-1]*100:.1f}%")

    Vt_torch = torch.tensor(Vt, dtype=torch.float16, device=device)

    pca_info = {
        "n_samples": states.shape[0],
        "n_layers": n_layers,
        "hidden_dim": states.shape[1],
        "layer_idx": layer_idx,
        "explained_variance_top4": round(cumvar[3] * 100, 2),
        "explained_variance_top16": round(cumvar[15] * 100, 2) if len(cumvar) > 15 else None,
        "explained_variance_top64": round(cumvar[63] * 100, 2) if len(cumvar) > 63 else None,
    }

    # Step 3: Test conditions
    conditions = [
        {"name": "baseline",    "mode": "baseline"},
        {"name": "random",      "mode": "random"},
        {"name": "pca_top4",    "mode": "pca_top",    "k": 4},
        {"name": "pca_bottom",  "mode": "pca_bottom", "k": 4},
    ]

    hook = PCANoiseHook()
    hook.register(model, layer_idx)

    model_results = {
        "model": model_short,
        "model_full": model_name,
        "n_layers": n_layers,
        "hidden_dim": states.shape[1],
        "layer_idx": layer_idx,
        "pca_info": pca_info,
        "baseline_collection_rate": round(baseline_rate, 3),
        "conditions": []
    }

    for cond_idx, cond in enumerate(conditions):
        cond_name = cond["name"]
        print(f"\n  [{cond_idx+1}/{len(conditions)}] Condition: {cond_name}")

        if cond["mode"] == "baseline":
            use_flash = False
            hook.setup_random()
        elif cond["mode"] == "random":
            use_flash = True
            hook.setup_random()
        elif cond["mode"] == "pca_top":
            use_flash = True
            k = cond["k"]
            hook.setup_pca_top(Vt_torch[:k], device)
        elif cond["mode"] == "pca_bottom":
            use_flash = True
            k = cond["k"]
            hook.setup_pca_bottom(Vt_torch[k:], states.shape[1], device)

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
        model_results["conditions"].append(summary)
        print(f"    Solve rate: {solved/len(games)*100:.1f}% ({solved}/{len(games)})")

    hook.remove()

    # Cleanup
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    model_results["elapsed_seconds"] = round(elapsed, 1)

    return model_results


# ===================================================
#  MAIN EXPERIMENT
# ===================================================

def run_phase82():
    print(f"\n{'='*80}")
    print(f"  Phase 82: Cross-Architecture PCA Comparison")
    print(f"  {len(MODELS)} models × 4 conditions × N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    all_results = {
        "experiment": "Phase 82: Cross-Architecture PCA Comparison",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_per_condition": N_PER_CONDITION,
        "sigma": BASE_SIGMA,
        "models": []
    }

    for model_config in MODELS:
        try:
            model_results = run_model_experiment(model_config)
            all_results["models"].append(model_results)
            # Save intermediate results after each model
            interim_path = os.path.join(RESULTS_DIR, "phase82_log.json")
            with open(interim_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Intermediate results saved: {interim_path}")
        except Exception as e:
            print(f"  ERROR with {model_config['short']}: {e}")
            import traceback; traceback.print_exc()
            all_results["models"].append({"model": model_config['short'], "error": str(e)})
            continue

    # Cross-model comparison
    print(f"\n{'='*80}")
    print(f"  === Cross-Architecture Comparison ===")
    print(f"{'='*80}")

    # Add Mistral reference data from Phase 71
    print(f"\n  Reference: Mistral-7B (Phase 71)")
    print(f"    Top-4 PC variance: 21.5% (from Phase 71)")
    print(f"    PCA-top4: 6.7%, Random: 46.7%")

    for mr in all_results["models"]:
        print(f"\n  {mr['model']}:")
        print(f"    Layers: {mr['n_layers']}, Hidden dim: {mr['hidden_dim']}")
        print(f"    Top-4 PC variance: {mr['pca_info']['explained_variance_top4']}%")

        bl = next(c for c in mr["conditions"] if c["condition"] == "baseline")
        rand = next(c for c in mr["conditions"] if c["condition"] == "random")
        top4 = next(c for c in mr["conditions"] if c["condition"] == "pca_top4")
        bottom = next(c for c in mr["conditions"] if c["condition"] == "pca_bottom")

        print(f"    Baseline: {bl['solve_rate']*100:.1f}%")
        print(f"    Random:   {rand['solve_rate']*100:.1f}%")
        print(f"    PCA-top4: {top4['solve_rate']*100:.1f}%")
        print(f"    PCA-bottom: {bottom['solve_rate']*100:.1f}%")

        # Fisher test: random vs baseline
        table = [[rand["n_solved"], rand["n_total"]-rand["n_solved"]],
                 [bl["n_solved"], bl["n_total"]-bl["n_solved"]]]
        _, p = fisher_exact(table)
        print(f"    Random vs Baseline: p={p:.4f}")

        # Key test: pca_top4 vs random
        table = [[top4["n_solved"], top4["n_total"]-top4["n_solved"]],
                 [rand["n_solved"], rand["n_total"]-rand["n_solved"]]]
        _, p = fisher_exact(table)
        gap = (rand["solve_rate"] - top4["solve_rate"]) * 100
        if gap > 10:
            verdict = "✅ CONFIRMED: Top-4 PCs encode reasoning (same as Mistral)"
        elif gap > 5:
            verdict = "⚠️ PARTIAL: Top-4 PCs may encode reasoning"
        else:
            verdict = "❌ NOT CONFIRMED: Top-4 PCs don't seem special here"
        print(f"    Random - PCA-top4 gap: {gap:.1f}pp (p={p:.4f}) → {verdict}")

    # Universality summary
    print(f"\n  === Universality Assessment ===")
    variances = [mr['pca_info']['explained_variance_top4'] for mr in all_results["models"]]
    print(f"    Top-4 PC variances: {', '.join(f'{v}%' for v in variances)} (Mistral: 21.5%)")
    if all(v > 10 for v in variances):
        print(f"    → UNIVERSAL: All models concentrate >10% variance in top-4 PCs")
    else:
        print(f"    → ARCHITECTURE-SPECIFIC: Variance concentration differs")

    # Save
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)

    results_path = os.path.join(RESULTS_DIR, "phase82_log.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  Total elapsed: {elapsed_total/60:.1f} min ({elapsed_total/3600:.1f} hours)")

    return all_results


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    results = run_phase82()

    print(f"\n{'='*80}")
    print(f"  Phase 82 COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 82 complete.")
