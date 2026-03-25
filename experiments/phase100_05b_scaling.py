"""
Phase 100: 0.5B Model Scaling — Does Noise Help Smaller Models?
================================================================

Phases 86-95 used 7B models (Qwen-7B, Mistral-7B). This experiment
tests whether Flash Annealing and Aha! steering work on a SMALLER
model (Qwen2.5-0.5B), testing the scale-dependence of SNN-LLM.

Key question: Does the 0.5B model have enough representational
capacity for noise injection to help, or does it just degrade?

Conditions (N=30 each):
  1. baseline:       No noise
  2. flash_sigma015: Flash Annealing σ=0.15 (7B optimal)
  3. flash_sigma005: Flash Annealing σ=0.05 (scaled for 0.5B)
  4. flash_sigma030: Flash Annealing σ=0.30 (stress test)

Qwen2.5-0.5B-Instruct, Layer 12 (middle layer), σ varied

Total: 4 x 30 = 120 games
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
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_SHORT = "Qwen2.5-0.5B"
SEED = 2026
MAX_STEPS = 50
N_PER_CONDITION = 30

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
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True,
                                         trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
        trust_remote_code=True, local_files_only=True)
    model.eval()
    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"  Done: {n_layers} layers, hidden_dim={hidden_dim}")
    return model, tok, n_layers, hidden_dim

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
#  FLASH ANNEALING HOOK
# ===================================================

class FlashAnnealingHook:
    def __init__(self):
        self.active = False
        self.sigma = 0.15
        self.handle = None

    def register(self, model, layer_idx):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            hs = args[0]
            noise = torch.randn_like(hs) * hook_obj.sigma
            return (hs + noise,) + args[1:]
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, hook, base_sigma, use_flash=True):
    env = HanoiEnv(n_disks=3, modified=True)
    error = None; consec_fail = 0; legal_move_count = 0
    for step in range(MAX_STEPS):
        if use_flash:
            if legal_move_count < 10:
                hook.active = True
                hook.sigma = base_sigma * (1.0 - legal_move_count / 10.0)
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
    fig.suptitle("Phase 100: 0.5B Model Scaling — Does Flash Annealing Help Small Models?\n"
                 f"Qwen2.5-0.5B, N={N_PER_CONDITION} per condition",
                 fontsize=12, fontweight="bold")

    conds = all_results["conditions"]
    names = [c["condition"] for c in conds]
    rates = [c["solve_rate"] * 100 for c in conds]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(range(len(conds)), rates, color=colors[:len(conds)], alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_ylim(0, max(max(rates) + 15, 25))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase100_05b_scaling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

CONDITIONS = [
    {"name": "baseline",        "sigma": 0.0,  "use_flash": False},
    {"name": "flash_sigma015",  "sigma": 0.15, "use_flash": True},
    {"name": "flash_sigma005",  "sigma": 0.05, "use_flash": True},
    {"name": "flash_sigma030",  "sigma": 0.30, "use_flash": True},
]


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*80}")
    print(f"  Phase 100: 0.5B Model Scaling Test")
    print(f"  {len(CONDITIONS)} conditions x N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    model, tok, n_layers, hidden_dim = load_model()
    device = next(model.parameters()).device

    # Use middle layer
    layer_idx = n_layers // 2
    print(f"  Using layer {layer_idx} (middle of {n_layers})")

    hook = FlashAnnealingHook()
    hook.register(model, layer_idx)

    all_results = {
        "experiment": "Phase 100: 0.5B Model Scaling",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "n_layers": n_layers,
        "layer": layer_idx,
        "hidden_dim": hidden_dim,
        "n_per_condition": N_PER_CONDITION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    for cond in CONDITIONS:
        cond_name = cond["name"]
        print(f"\n  === Condition: {cond_name} (σ={cond['sigma']}) ===")

        games = []
        for trial in range(N_PER_CONDITION):
            stats = play_game(model, tok, hook, base_sigma=cond["sigma"],
                            use_flash=cond["use_flash"])
            games.append(stats)
            if (trial + 1) % 10 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                elapsed = time.time() - t0
                print(f"    [{trial+1}/{N_PER_CONDITION}] Solve rate: {sr:.1f}% | {elapsed/60:.1f}min")

        solved = sum(1 for g in games if g["solved"])
        summary = {
            "condition": cond_name,
            "sigma": cond["sigma"],
            "solve_rate": round(solved / len(games), 4),
            "n_solved": solved,
            "n_total": len(games),
            "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
            "avg_illegal_moves": round(np.mean([g["illegal_moves"] for g in games]), 2),
            "games": games
        }
        all_results["conditions"].append(summary)
        print(f"    Result: {solved}/{len(games)} = {solved/len(games)*100:.1f}%")

        # Intermediate save
        results_path = os.path.join(RESULTS_DIR, "phase100_log.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    hook.remove()

    # Analysis
    bl = all_results["conditions"][0]
    print(f"\n  === Results Summary ===")
    for cond in all_results["conditions"]:
        print(f"    {cond['condition']:20s}: {cond['solve_rate']*100:5.1f}% (σ={cond['sigma']})")

    best_noise = max(all_results["conditions"][1:], key=lambda c: c["solve_rate"])
    print(f"\n  Best noise σ: {best_noise['sigma']} → {best_noise['solve_rate']*100:.1f}%")

    # Compare with 7B baseline (~27% from Phase 86)
    print(f"\n  === Scale Comparison ===")
    print(f"  0.5B baseline: {bl['solve_rate']*100:.1f}%")
    print(f"  0.5B best noise: {best_noise['solve_rate']*100:.1f}%")
    print(f"  (7B Qwen baseline was ~27%, 7B+flash was ~43%)")

    if best_noise["solve_rate"] > bl["solve_rate"] + 0.05:
        verdict = "NOISE_HELPS_SMALL"
        print(f"\n  VERDICT: {verdict} — Flash Annealing benefits 0.5B model too!")
    elif best_noise["solve_rate"] > bl["solve_rate"]:
        verdict = "MARGINAL_BENEFIT"
        print(f"\n  VERDICT: {verdict} — Small improvement, scaling effect exists")
    else:
        verdict = "SCALE_DEPENDENT"
        print(f"\n  VERDICT: {verdict} — Flash Annealing requires sufficient model capacity")

    all_results["verdict"] = verdict

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
    print(f"\n Phase 100 complete.")
