"""
Phase 101: Aha! Synthesis on 0.5B — Cross-Scale Reasoning Vector Transfer
==========================================================================

Phase 100 establishes whether 0.5B benefits from noise.
This experiment tests a radical idea: can we TRANSFER the 7B Aha! vector
to the 0.5B model?

Since 0.5B has different hidden_dim than 7B, direct transfer is impossible.
Instead, we:
  1. Collect labeled hidden states from 0.5B on the same task
  2. Compute 0.5B's own Diff-PCA
  3. Compare 0.5B's discriminant direction with 7B's (via PCA coefficient overlap)
  4. Test Aha! steering on 0.5B with its native discriminant direction

Conditions (N=30 each):
  1. baseline:       No noise
  2. flash_optimal:  Best σ from Phase 100
  3. aha_0.5b:       Aha! steering using 0.5B's own Diff-PCA
  4. aha_flash:      Combined Aha! + Flash with 0.5B's Diff-PCA

Qwen2.5-0.5B-Instruct, middle layer

Total: 100 collection + 4 x 30 = 220 forward passes
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
N_COLLECTION = 100
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
#  LABELED HIDDEN STATE COLLECTION
# ===================================================

class LabeledCollectorHook:
    def __init__(self):
        self.current_states = []
        self.all_items = []
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
                hook_obj.current_states.append(last_hs)
        self.handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    def start_game(self):
        self.current_states = []
        self.recording = True

    def end_game(self, success):
        self.recording = False
        if self.current_states:
            states = np.array(self.current_states)
            self.all_items.append((states, success))
        self.current_states = []

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  AHA! + FLASH HOOK
# ===================================================

class AhaFlashHook:
    def __init__(self):
        self.active = False
        self.sigma = 0.15
        self.mode = "baseline"
        self.diff_unit = None
        self.handle = None

    def setup_baseline(self):
        self.mode = "baseline"

    def setup_flash(self, sigma):
        self.mode = "flash"
        self.sigma = sigma

    def setup_aha(self, diff_unit_vec, sigma, device='cuda'):
        self.mode = "aha"
        du = torch.tensor(diff_unit_vec, dtype=torch.float16, device=device)
        du_norm = du.norm()
        if du_norm > 1e-8:
            du = du / du_norm
        self.diff_unit = du
        self.sigma = sigma

    def setup_aha_flash(self, diff_unit_vec, sigma, device='cuda'):
        self.mode = "aha_flash"
        du = torch.tensor(diff_unit_vec, dtype=torch.float16, device=device)
        du_norm = du.norm()
        if du_norm > 1e-8:
            du = du / du_norm
        self.diff_unit = du
        self.sigma = sigma

    def register(self, model, layer_idx):
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

            elif hook_obj.mode == "aha":
                offset = hook_obj.diff_unit
                scaled = offset * hook_obj.sigma * math.sqrt(d)
                if hs.dim() == 3:
                    scaled = scaled.unsqueeze(0).unsqueeze(0).expand_as(hs)
                else:
                    scaled = scaled.unsqueeze(0).expand_as(hs)
                return (hs + scaled,) + args[1:]

            elif hook_obj.mode == "aha_flash":
                # Aha! half + Flash half
                offset = hook_obj.diff_unit
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
#  GAME FUNCTIONS
# ===================================================

def play_game_collect(model, tok, collector):
    """Play game, collecting hidden states."""
    env = HanoiEnv(n_disks=3, modified=True)
    error = None; consec_fail = 0
    collector.start_game()
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
    collector.end_game(env.is_solved())
    return env.is_solved()


def play_game_hook(model, tok, hook, base_sigma, use_flash=True):
    """Play game with noise hook."""
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
            hook.active = True
            hook.sigma = base_sigma
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle("Phase 101: Aha! Synthesis on 0.5B — Cross-Scale Reasoning Transfer",
                 fontsize=13, fontweight="bold")

    # Panel 1: Steering results
    ax = axes[0]
    conds = all_results["steering_results"]
    names = [c["condition"] for c in conds]
    rates = [c["solve_rate"] * 100 for c in conds]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#9C27B0"]
    bars = ax.bar(range(len(conds)), rates, color=colors[:len(conds)], alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title(f"Aha! Steering on 0.5B (N={N_PER_CONDITION})", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2: Scale comparison
    ax = axes[1]
    scale_data = all_results.get("scale_comparison", {})
    if scale_data:
        models = list(scale_data.keys())
        bl_rates = [scale_data[m].get("baseline", 0) * 100 for m in models]
        best_rates = [scale_data[m].get("best_noise", 0) * 100 for m in models]
        x = range(len(models))
        w = 0.35
        ax.bar([i-w/2 for i in x], bl_rates, w, label="Baseline", color="#9E9E9E", alpha=0.85)
        ax.bar([i+w/2 for i in x], best_rates, w, label="Best Noise", color="#4CAF50", alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, fontsize=10)
        for i, (bl, best) in enumerate(zip(bl_rates, best_rates)):
            ax.text(i-w/2, bl+1, f"{bl:.0f}%", ha="center", fontsize=9)
            ax.text(i+w/2, best+1, f"{best:.0f}%", ha="center", fontsize=9)
        ax.legend(fontsize=9)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("Scale Comparison: 0.5B vs 7B", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase101_aha_synthesis_05b.png")
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
    print(f"  Phase 101: Aha! Synthesis on 0.5B")
    print(f"  {N_COLLECTION} collection + 4 conditions x N={N_PER_CONDITION} steering")
    print(f"{'='*80}")

    t0 = time.time()

    model, tok, n_layers, hidden_dim = load_model()
    device = next(model.parameters()).device
    layer_idx = n_layers // 2
    print(f"  Using layer {layer_idx}, hidden_dim={hidden_dim}")

    # === Step 1: Collect labeled hidden states ===
    print(f"\n  === Step 1: Collecting {N_COLLECTION} games ===")
    collector = LabeledCollectorHook()
    collector.register(model, layer_idx)

    for gi in range(N_COLLECTION):
        solved = play_game_collect(model, tok, collector)
        if (gi + 1) % 20 == 0:
            ns = sum(1 for _, s in collector.all_items if s)
            print(f"    [{gi+1}/{N_COLLECTION}] Solved: {ns}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    collector.remove()
    n_solved_coll = sum(1 for _, s in collector.all_items if s)
    n_total_coll = len(collector.all_items)
    print(f"  Collection: {n_solved_coll}/{n_total_coll} solved "
          f"({n_solved_coll/n_total_coll*100:.1f}%)")

    # === Step 2: Compute Differential PCA ===
    print(f"\n  === Step 2: Computing Differential PCA ===")
    solved_means = []
    failed_means = []
    for states, success in collector.all_items:
        game_mean = states.mean(axis=0)
        if success:
            solved_means.append(game_mean)
        else:
            failed_means.append(game_mean)

    if len(solved_means) == 0 or len(failed_means) == 0:
        print(f"  WARNING: Need both solved and failed games for Diff-PCA!")
        print(f"  Solved: {len(solved_means)}, Failed: {len(failed_means)}")
        # Use random direction as fallback
        diff_unit = np.random.randn(hidden_dim).astype(np.float32)
        diff_unit /= np.linalg.norm(diff_unit)
        diff_norm = 0.0
    else:
        solved_mean = np.mean(solved_means, axis=0)
        failed_mean = np.mean(failed_means, axis=0)
        diff = solved_mean - failed_mean
        diff_norm = float(np.linalg.norm(diff))
        diff_unit = diff / (diff_norm + 1e-8)
        print(f"  Diff vector norm: {diff_norm:.4f}")
        print(f"  Solved mean norm: {np.linalg.norm(solved_mean):.4f}")
        print(f"  Failed mean norm: {np.linalg.norm(failed_mean):.4f}")

    # Save 0.5B diff vector
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "phase101_05b_diff_pca.npz"),
        diff_unit=diff_unit,
    )

    del collector
    gc.collect()

    # === Step 3: Determine optimal σ from Phase 100 ===
    p100_path = os.path.join(RESULTS_DIR, "phase100_log.json")
    optimal_sigma = 0.05  # Default for small model
    if os.path.exists(p100_path):
        with open(p100_path) as f:
            p100 = json.load(f)
        best_cond = max(p100.get("conditions", [{}])[1:] or [{"sigma": 0.05}],
                       key=lambda c: c.get("solve_rate", 0))
        optimal_sigma = best_cond.get("sigma", 0.05)
        print(f"  Phase 100 best σ: {optimal_sigma}")
    else:
        print(f"  Phase 100 not found, using default σ={optimal_sigma}")

    # === Step 4: Steering test ===
    print(f"\n  === Step 3: Steering Test ({N_PER_CONDITION} games per condition) ===")

    hook = AhaFlashHook()
    hook.register(model, layer_idx)

    conditions = [
        {"name": "baseline",      "setup": "baseline"},
        {"name": "flash_optimal", "setup": "flash"},
        {"name": "aha_0.5b",      "setup": "aha"},
        {"name": "aha_flash",     "setup": "aha_flash"},
    ]

    steering_results = []
    for cond in conditions:
        print(f"\n  Condition: {cond['name']}")
        if cond["setup"] == "baseline":
            hook.setup_baseline()
            use_flash = False
        elif cond["setup"] == "flash":
            hook.setup_flash(optimal_sigma)
            use_flash = True
        elif cond["setup"] == "aha":
            hook.setup_aha(diff_unit, optimal_sigma, device)
            use_flash = True
        elif cond["setup"] == "aha_flash":
            hook.setup_aha_flash(diff_unit, optimal_sigma, device)
            use_flash = True

        games = []
        for trial in range(N_PER_CONDITION):
            stats = play_game_hook(model, tok, hook, base_sigma=optimal_sigma,
                                  use_flash=use_flash)
            games.append(stats)
            if (trial + 1) % 10 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                print(f"    [{trial+1}/{N_PER_CONDITION}] Solve rate: {sr:.1f}%")

        solved = sum(1 for g in games if g["solved"])
        steering_results.append({
            "condition": cond["name"],
            "solve_rate": round(solved / len(games), 4),
            "n_solved": solved,
            "n_total": len(games),
            "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
            "avg_illegal_moves": round(np.mean([g["illegal_moves"] for g in games]), 2),
            "games": games
        })
        print(f"    Result: {solved}/{len(games)} = {solved/len(games)*100:.1f}%")

        # Intermediate save
        results_path = os.path.join(RESULTS_DIR, "phase101_log.json")
        inter_results = {
            "experiment": "Phase 101: Aha! Synthesis on 0.5B",
            "model": MODEL_SHORT,
            "steering_results": steering_results,
        }
        with open(results_path, 'w') as f:
            json.dump(inter_results, f, indent=2, default=str)

    hook.remove()

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    bl_rate = steering_results[0]["solve_rate"]
    for sr in steering_results:
        delta = sr["solve_rate"] - bl_rate
        print(f"    {sr['condition']:20s}: {sr['solve_rate']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    # Scale comparison
    scale_comparison = {
        "0.5B": {
            "baseline": bl_rate,
            "best_noise": max(s["solve_rate"] for s in steering_results),
        },
    }
    # Load 7B data if available
    for p_path, model_name in [
        ("phase95_log.json", "7B-Mistral"),
        ("phase92_log.json", "7B-Qwen"),
    ]:
        full_path = os.path.join(RESULTS_DIR, p_path)
        if os.path.exists(full_path):
            with open(full_path) as f:
                p_data = json.load(f)
            bl_7b = next((c["solve_rate"] for c in p_data.get("conditions", [])
                         if c.get("condition") == "baseline"), 0)
            best_7b = max((c["solve_rate"] for c in p_data.get("conditions", [])), default=0)
            scale_comparison[model_name] = {"baseline": bl_7b, "best_noise": best_7b}

    # Compile final results
    all_results = {
        "experiment": "Phase 101: Aha! Synthesis on 0.5B",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "n_layers": n_layers,
        "layer": layer_idx,
        "hidden_dim": hidden_dim,
        "optimal_sigma": optimal_sigma,
        "diff_vector_norm": round(diff_norm, 4),
        "collection_stats": {
            "n_total": n_total_coll,
            "n_solved": n_solved_coll,
            "solve_rate": round(n_solved_coll / n_total_coll, 4) if n_total_coll > 0 else 0,
        },
        "steering_results": steering_results,
        "scale_comparison": scale_comparison,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Verdict
    best_steer = max(steering_results, key=lambda s: s["solve_rate"])
    if best_steer["solve_rate"] > bl_rate + 0.1:
        verdict = "AHA_SCALES_DOWN"
        print(f"\n  VERDICT: {verdict} — Aha! steering works on 0.5B too!")
    elif best_steer["solve_rate"] > bl_rate + 0.03:
        verdict = "MARGINAL_EFFECT"
        print(f"\n  VERDICT: {verdict} — Small Aha! effect on 0.5B")
    else:
        verdict = "CAPACITY_LIMITED"
        print(f"\n  VERDICT: {verdict} — 0.5B lacks capacity for Aha! steering")

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
    print(f"\n Phase 101 complete.")
