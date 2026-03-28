"""
Phase 108: Layer-Specific Prior Override
========================================

Gemini's proposal: Which layer is the "prior memory readout" layer?
Does concentrating noise injection at a specific layer maximize
the Prior Override effect?

Conditions (N=30 each, Modified Hanoi + Aha!+Noise):
  1. baseline:  No noise
  2. L8:        Early layer
  3. L12:       Early-mid
  4. L16:       Mid
  5. L18:       Default (v19)
  6. L22:       Late-mid
  7. L26:       Late-early
  8. L30:       Late

Mistral-7B-Instruct-v0.3 (32 layers), σ=0.15
Total: 8 x 30 = 240 games
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

LAYER_TARGETS = [8, 12, 16, 18, 22, 26, 30]

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
    n_layers = len(model.model.layers)
    print(f"  Done: {n_layers} layers, hidden_dim={model.config.hidden_size}")
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
#  AHA + NOISE HOOK (layer-switchable)
# ===================================================

class LayerAhaHook:
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.fixed_offset = None
        self.handle = None

    def setup(self, diff_unit_vec, device='cuda'):
        du = torch.tensor(diff_unit_vec, dtype=torch.float16, device=device)
        self.fixed_offset = du

    def register(self, model, layer_idx):
        """Register hook on specific layer. Removes previous hook first."""
        self.remove()
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            hs = args[0]
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
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, hook, use_noise=True):
    env = HanoiEnv(n_disks=3, modified=True)
    error = None; consec_fail = 0; legal_move_count = 0
    for step in range(MAX_STEPS):
        if use_noise:
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

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Phase 108: Layer-Specific Prior Override\n"
                 "Which layer is the 'prior memory readout' layer?",
                 fontsize=12, fontweight="bold")

    conds = all_results["conditions"]
    names = [c["condition"] for c in conds]
    rates = [c["solve_rate"] * 100 for c in conds]
    colors = ["#9E9E9E"] + ["#2196F3" if c["layer"] != 18 else "#9C27B0" for c in conds[1:]]
    bars = ax.bar(range(len(conds)), rates, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Solve Rate (%)", fontsize=11)
    ax.set_xlabel("Injection Layer", fontsize=11)
    ax.axhline(y=rates[0], color='gray', linestyle='--', alpha=0.5, label="Baseline")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase108_layer_specific_override.png")
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
    print(f"  Phase 108: Layer-Specific Prior Override")
    print(f"  {1 + len(LAYER_TARGETS)} conditions x N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load Diff-PCA
    diff_pca_path = os.path.join(RESULTS_DIR, "phase91_diff_pca.npz")
    if not os.path.exists(diff_pca_path):
        raise FileNotFoundError(f"Phase 91 Diff-PCA not found: {diff_pca_path}")
    data = np.load(diff_pca_path)
    diff_unit = data["diff_unit"]

    model, tok, n_layers = load_model()
    device = next(model.parameters()).device

    hook = LayerAhaHook()
    hook.setup(diff_unit, device)

    all_results = {
        "experiment": "Phase 108: Layer-Specific Prior Override",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "n_layers": n_layers,
        "sigma": BASE_SIGMA,
        "n_per_condition": N_PER_CONDITION,
        "layer_targets": LAYER_TARGETS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": [],
    }

    results_path = os.path.join(RESULTS_DIR, "phase108_log.json")

    # Baseline (no noise — register on L18 but keep inactive)
    print(f"\n  === Baseline (no noise) ===")
    hook.register(model, 18)
    games = []
    for trial in range(N_PER_CONDITION):
        stats = play_game(model, tok, hook, use_noise=False)
        games.append(stats)
    solved = sum(1 for g in games if g["solved"])
    all_results["conditions"].append({
        "condition": "baseline",
        "layer": None,
        "solve_rate": round(solved / len(games), 4),
        "n_solved": solved,
        "n_total": len(games),
        "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
        "avg_illegal_moves": round(np.mean([g["illegal_moves"] for g in games]), 2),
        "games": games,
    })
    print(f"    Result: {solved}/{len(games)} = {solved/len(games)*100:.1f}%")
    hook.remove()

    # Layer sweep
    for layer_idx in LAYER_TARGETS:
        print(f"\n  === Layer {layer_idx} ===")
        hook.register(model, layer_idx)

        games = []
        for trial in range(N_PER_CONDITION):
            stats = play_game(model, tok, hook, use_noise=True)
            games.append(stats)
            if (trial + 1) % 15 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                elapsed = time.time() - t0
                print(f"    [{trial+1}/{N_PER_CONDITION}] {sr:.1f}% | {elapsed/60:.1f}min")

        solved = sum(1 for g in games if g["solved"])
        all_results["conditions"].append({
            "condition": f"L{layer_idx}",
            "layer": layer_idx,
            "solve_rate": round(solved / len(games), 4),
            "n_solved": solved,
            "n_total": len(games),
            "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
            "avg_illegal_moves": round(np.mean([g["illegal_moves"] for g in games]), 2),
            "games": games,
        })
        print(f"    Result: {solved}/{len(games)} = {solved/len(games)*100:.1f}%")
        hook.remove()

        # Intermediate save
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    bl_rate = all_results["conditions"][0]["solve_rate"]
    best_layer = None
    best_rate = 0
    for c in all_results["conditions"]:
        delta = c["solve_rate"] - bl_rate
        print(f"    {c['condition']:12s}: {c['solve_rate']*100:5.1f}% (Δ={delta*100:+.1f}pp)")
        if c["layer"] is not None and c["solve_rate"] > best_rate:
            best_rate = c["solve_rate"]
            best_layer = c["layer"]

    print(f"\n  Best layer: L{best_layer} ({best_rate*100:.1f}%)")

    # Fisher exact: best layer vs baseline
    best_cond = next(c for c in all_results["conditions"] if c["layer"] == best_layer)
    bl_cond = all_results["conditions"][0]
    table = [[best_cond["n_solved"], best_cond["n_total"]-best_cond["n_solved"]],
             [bl_cond["n_solved"], bl_cond["n_total"]-bl_cond["n_solved"]]]
    _, p_val = fisher_exact(table)
    print(f"  Fisher exact (L{best_layer} vs baseline): p={p_val:.4f}")

    # Layer profile pattern
    layer_rates = [(c["layer"], c["solve_rate"]) for c in all_results["conditions"] if c["layer"] is not None]
    peak_layer = max(layer_rates, key=lambda x: x[1])[0]

    if peak_layer <= 14:
        verdict = "EARLY_LAYER_OPTIMAL"
        print(f"\n  VERDICT: {verdict} — Prior readout happens in early-mid layers")
    elif peak_layer <= 20:
        verdict = "MID_LAYER_OPTIMAL"
        print(f"\n  VERDICT: {verdict} — Prior readout happens in middle layers (L{peak_layer})")
    else:
        verdict = "LATE_LAYER_OPTIMAL"
        print(f"\n  VERDICT: {verdict} — Prior readout happens in late layers (L{peak_layer})")

    all_results["verdict"] = verdict
    all_results["best_layer"] = best_layer
    all_results["fisher_best_vs_baseline"] = round(p_val, 6)

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
    print(f"\n Phase 108 complete.")
