"""
Phase 102: Standard vs Modified Hanoi — Neural Prior Override Control
=====================================================================

All v12-v18 experiments used MODIFIED rules (larger-on-smaller).
This control experiment tests: does Flash Annealing / Aha! help when
the LLM does NOT need to override its prior knowledge?

Hypothesis: Noise injection is specifically beneficial when the task
requires overriding trained priors (Modified Hanoi). For Standard Hanoi
(where prior knowledge is CORRECT), noise should be neutral or harmful.

Conditions (N=30 each, Mistral-7B):
  Standard Hanoi:
    1. standard_baseline:     No noise, standard rules
    2. standard_flash:        Flash Annealing σ=0.15, standard rules
    3. standard_aha+noise:    Aha!+noise, standard rules

  Modified Hanoi (V18 replication):
    4. modified_baseline:     No noise, modified rules
    5. modified_flash:        Flash Annealing σ=0.15, modified rules
    6. modified_aha+noise:    Aha!+noise, modified rules

Mistral-7B-Instruct-v0.3, Layer 18, σ=0.15

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
    if env.modified:
        rules = "MODIFIED RULES: You can ONLY place a LARGER disk onto a SMALLER disk. The opposite of standard."
    else:
        rules = "STANDARD RULES: You can ONLY place a SMALLER disk onto a LARGER disk."
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
#  AHA + NOISE HOOK
# ===================================================

class AhaNoiseHook:
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.mode = "baseline"
        self.fixed_offset = None
        self.handle = None

    def setup_baseline(self):
        self.mode = "baseline"

    def setup_flash(self):
        self.mode = "flash"

    def setup_aha_noise(self, diff_unit_vec, device='cuda'):
        self.mode = "aha_noise"
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
            d = hs.shape[-1]

            if hook_obj.mode == "flash":
                noise = torch.randn_like(hs) * hook_obj.sigma
                return (hs + noise,) + args[1:]

            elif hook_obj.mode == "aha_noise":
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

def play_game(model, tok, hook, modified=True, use_flash=True):
    env = HanoiEnv(n_disks=3, modified=modified)
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 102: Standard vs Modified Hanoi — Neural Prior Override\n"
                 "Does noise help when prior knowledge is already correct?",
                 fontsize=12, fontweight="bold")

    for idx, rule_type in enumerate(["standard", "modified"]):
        ax = axes[idx]
        conds = [c for c in all_results["conditions"] if c["rule_type"] == rule_type]
        names = [c["condition"].replace(f"{rule_type}_", "") for c in conds]
        rates = [c["solve_rate"] * 100 for c in conds]
        colors = ["#9E9E9E", "#2196F3", "#9C27B0"]
        bars = ax.bar(range(len(conds)), rates, color=colors[:len(conds)], alpha=0.85,
                      edgecolor="white", linewidth=2)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel("Solve Rate (%)", fontsize=11)
        title = "Standard Hanoi\n(prior knowledge correct)" if rule_type == "standard" \
                else "Modified Hanoi\n(prior knowledge must be overridden)"
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylim(0, max(max(rates) + 15, 25))
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase102_standard_vs_modified.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

CONDITIONS = [
    # Standard Hanoi (prior knowledge is correct)
    {"name": "standard_baseline",    "modified": False, "mode": "baseline",  "flash": False},
    {"name": "standard_flash",       "modified": False, "mode": "flash",     "flash": True},
    {"name": "standard_aha+noise",   "modified": False, "mode": "aha_noise", "flash": True},
    # Modified Hanoi (prior knowledge must be overridden)
    {"name": "modified_baseline",    "modified": True,  "mode": "baseline",  "flash": False},
    {"name": "modified_flash",       "modified": True,  "mode": "flash",     "flash": True},
    {"name": "modified_aha+noise",   "modified": True,  "mode": "aha_noise", "flash": True},
]


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*80}")
    print(f"  Phase 102: Standard vs Modified Hanoi — Neural Prior Override")
    print(f"  {len(CONDITIONS)} conditions x N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load Diff-PCA from Phase 91 (Mistral)
    diff_pca_path = os.path.join(RESULTS_DIR, "phase91_diff_pca.npz")
    if not os.path.exists(diff_pca_path):
        raise FileNotFoundError(f"Phase 91 Diff-PCA not found: {diff_pca_path}")
    data = np.load(diff_pca_path)
    diff_unit = data["diff_unit"]
    print(f"  Loaded diff_unit: norm={np.linalg.norm(diff_unit):.4f}")

    model, tok = load_model()
    device = next(model.parameters()).device
    hook = AhaNoiseHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 102: Standard vs Modified Hanoi",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "layer": LAYER_IDX,
        "hidden_dim": HIDDEN_DIM,
        "sigma": BASE_SIGMA,
        "n_per_condition": N_PER_CONDITION,
        "hypothesis": "Noise injection helps specifically when prior override is needed",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    for cond_idx, cond in enumerate(CONDITIONS):
        cond_name = cond["name"]
        rule_type = "modified" if cond["modified"] else "standard"
        print(f"\n  [{cond_idx+1}/{len(CONDITIONS)}] {cond_name} ({rule_type} rules)")

        if cond["mode"] == "baseline":
            hook.setup_baseline()
        elif cond["mode"] == "flash":
            hook.setup_flash()
        elif cond["mode"] == "aha_noise":
            hook.setup_aha_noise(diff_unit, device)

        games = []
        for trial in range(N_PER_CONDITION):
            stats = play_game(model, tok, hook,
                            modified=cond["modified"],
                            use_flash=cond["flash"])
            games.append(stats)
            if (trial + 1) % 10 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                elapsed = time.time() - t0
                print(f"    [{trial+1}/{N_PER_CONDITION}] Solve rate: {sr:.1f}% | {elapsed/60:.1f}min")

        solved = sum(1 for g in games if g["solved"])
        summary = {
            "condition": cond_name,
            "rule_type": rule_type,
            "mode": cond["mode"],
            "solve_rate": round(solved / len(games), 4),
            "n_solved": solved,
            "n_total": len(games),
            "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
            "avg_illegal_moves": round(np.mean([g["illegal_moves"] for g in games]), 2),
            "avg_self_corrections": round(np.mean([g["self_corrections"] for g in games]), 2),
            "games": games
        }
        all_results["conditions"].append(summary)
        print(f"    Result: {solved}/{len(games)} = {solved/len(games)*100:.1f}%")

        # Intermediate save
        results_path = os.path.join(RESULTS_DIR, "phase102_log.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    hook.remove()

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    std_conds = [c for c in all_results["conditions"] if c["rule_type"] == "standard"]
    mod_conds = [c for c in all_results["conditions"] if c["rule_type"] == "modified"]

    print(f"\n  Standard Hanoi (prior correct):")
    std_bl = next(c for c in std_conds if "baseline" in c["condition"])
    for c in std_conds:
        delta = c["solve_rate"] - std_bl["solve_rate"]
        print(f"    {c['condition']:25s}: {c['solve_rate']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    print(f"\n  Modified Hanoi (prior override needed):")
    mod_bl = next(c for c in mod_conds if "baseline" in c["condition"])
    for c in mod_conds:
        delta = c["solve_rate"] - mod_bl["solve_rate"]
        print(f"    {c['condition']:25s}: {c['solve_rate']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    # Interaction: does noise help more in modified than standard?
    std_best = max(std_conds, key=lambda c: c["solve_rate"])
    mod_best = max(mod_conds, key=lambda c: c["solve_rate"])
    std_gain = std_best["solve_rate"] - std_bl["solve_rate"]
    mod_gain = mod_best["solve_rate"] - mod_bl["solve_rate"]

    print(f"\n  === Prior Override Interaction ===")
    print(f"  Standard: best gain = {std_gain*100:+.1f}pp ({std_best['condition']})")
    print(f"  Modified: best gain = {mod_gain*100:+.1f}pp ({mod_best['condition']})")
    print(f"  Differential: {(mod_gain - std_gain)*100:+.1f}pp")

    if mod_gain > std_gain + 0.05:
        verdict = "PRIOR_OVERRIDE_CONFIRMED"
        print(f"\n  VERDICT: {verdict}")
        print(f"  → Noise helps MORE when prior knowledge must be overridden")
    elif abs(mod_gain - std_gain) <= 0.05:
        verdict = "GENERAL_BENEFIT"
        print(f"\n  VERDICT: {verdict}")
        print(f"  → Noise helps equally regardless of prior alignment")
    else:
        verdict = "UNEXPECTED"
        print(f"\n  VERDICT: {verdict}")
        print(f"  → Noise helps MORE with standard rules (unexpected!)")

    all_results["verdict"] = verdict
    all_results["interaction"] = {
        "standard_gain": round(std_gain, 4),
        "modified_gain": round(mod_gain, 4),
        "differential": round(mod_gain - std_gain, 4),
    }

    # Fisher exact for key comparison
    mod_aha = next(c for c in mod_conds if "aha" in c["condition"])
    std_aha = next(c for c in std_conds if "aha" in c["condition"])
    table = [[mod_aha["n_solved"], mod_aha["n_total"]-mod_aha["n_solved"]],
             [std_aha["n_solved"], std_aha["n_total"]-std_aha["n_solved"]]]
    _, p = fisher_exact(table)
    print(f"\n  Fisher exact (mod_aha vs std_aha): p={p:.4f}")
    all_results["fisher_mod_vs_std_aha"] = round(p, 6)

    fig_path = visualize(all_results)
    all_results["figure"] = fig_path

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed, 1)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  Total elapsed: {elapsed/60:.1f} min")

    return all_results, elapsed


if __name__ == "__main__":
    main()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n Phase 102 complete.")
