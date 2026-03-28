"""
Phase 109: Trajectory Distillation — Extract "Rule-Override" Trajectories
=========================================================================

Opus Proposal: Record the sequence of hidden-state vectors at each
reasoning step of SUCCESSFUL modified-Hanoi games under Aha!+noise.
Then distill this into a "prior override trajectory template."

Protocol:
  1. Run 100 Modified Hanoi games with Aha!+noise
  2. Record hidden states at layer 18 for each step
  3. From solved games, extract the trajectory pattern
  4. Compute a "trajectory template" via averaging + PCA
  5. Test if replaying this template (without Aha!) improves performance

Mistral-7B-Instruct-v0.3, Layer 18, σ=0.15
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
N_COLLECTION = 100  # Games for trajectory collection
N_TEST = 50         # Games for template testing
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

def generate_with_hidden(model, tok, prompt, temperature=0.5, max_tokens=80):
    """Generate response and capture last hidden state at target layer."""
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    captured = {}
    def capture_hook(module, args):
        hs = args[0]
        # Capture mean of last token's hidden state
        if hs.dim() == 3:
            captured["hidden"] = hs[0, -1, :].detach().cpu().float().numpy()
        else:
            captured["hidden"] = hs[-1, :].detach().cpu().float().numpy()
        return args

    handle = model.model.layers[LAYER_IDX].register_forward_pre_hook(capture_hook)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperature, top_p=0.9,
            repetition_penalty=1.2, pad_token_id=tok.pad_token_id)
    handle.remove()

    full = tok.decode(out[0], skip_special_tokens=True)
    inp = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
    resp = full[len(inp):].strip()
    return resp, captured.get("hidden")

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

    def setup_aha_noise(self, diff_unit_vec, device='cuda'):
        self.mode = "aha_noise"
        du = torch.tensor(diff_unit_vec, dtype=torch.float16, device=device)
        self.fixed_offset = du

    def setup_trajectory(self, trajectory_vec, device='cuda'):
        """Use a distilled trajectory template as the steering vector."""
        self.mode = "trajectory"
        tv = torch.tensor(trajectory_vec, dtype=torch.float16, device=device)
        self.fixed_offset = tv

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            if hook_obj.mode == "baseline":
                return args
            hs = args[0]
            d = hs.shape[-1]

            if hook_obj.mode in ("aha_noise", "trajectory"):
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
#  PHASE 1: COLLECT TRAJECTORIES
# ===================================================

def collect_trajectories(model, tok, hook, n_games):
    """Run games with Aha!+noise and record hidden states from solved games."""
    trajectories = []  # List of (trajectory_vectors, stats)
    all_stats = []

    for game_idx in range(n_games):
        env = HanoiEnv(n_disks=3, modified=True)
        error = None; consec_fail = 0; legal_move_count = 0
        game_hiddens = []

        for step in range(MAX_STEPS):
            if legal_move_count < 10:
                hook.active = True
                hook.sigma = BASE_SIGMA * (1.0 - legal_move_count / 10.0)
            else:
                hook.active = False; hook.sigma = 0.0

            prompt = build_chat_prompt(tok, env, error)
            resp, hidden = generate_with_hidden(model, tok, prompt)
            if hidden is not None:
                game_hiddens.append(hidden)

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
        all_stats.append(stats)

        if stats["solved"] and len(game_hiddens) >= 3:
            trajectories.append(np.array(game_hiddens))

        if (game_idx + 1) % 20 == 0:
            sr = sum(1 for s in all_stats if s["solved"]) / len(all_stats) * 100
            print(f"    [{game_idx+1}/{n_games}] Solved: {sr:.1f}%, Trajectories: {len(trajectories)}")

    return trajectories, all_stats


# ===================================================
#  PHASE 2: DISTILL TEMPLATE
# ===================================================

def distill_trajectory_template(trajectories):
    """Compute a trajectory template from solved game trajectories."""
    # Compute per-step mean direction
    max_len = max(len(t) for t in trajectories)
    step_means = []

    for step_i in range(min(max_len, 15)):  # First 15 steps max
        step_vecs = []
        for traj in trajectories:
            if step_i < len(traj):
                step_vecs.append(traj[step_i])
        if step_vecs:
            mean_vec = np.mean(step_vecs, axis=0)
            step_means.append(mean_vec)

    if len(step_means) < 2:
        print("  WARNING: Too few trajectory steps for distillation")
        return None, {}

    # Stack and do PCA
    step_matrix = np.array(step_means)
    # Compute trajectory direction: last - first (direction of "prior override")
    trajectory_direction = step_matrix[-1] - step_matrix[0]
    traj_norm = np.linalg.norm(trajectory_direction)
    if traj_norm > 1e-8:
        trajectory_unit = trajectory_direction / traj_norm
    else:
        trajectory_unit = trajectory_direction

    # Also compute PCA on the trajectory
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(5, len(step_means)))
    pca.fit(step_matrix)

    info = {
        "n_trajectories": len(trajectories),
        "max_steps": max_len,
        "n_steps_used": len(step_means),
        "trajectory_norm": float(traj_norm),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    }

    return trajectory_unit, info


# ===================================================
#  PHASE 3: TEST TEMPLATE
# ===================================================

def play_game_simple(model, tok, hook, use_noise=True):
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Phase 109: Trajectory Distillation — Extracting Prior Override Patterns",
                 fontsize=12, fontweight="bold")

    # Panel 1: Collection phase solve rate
    ax = axes[0]
    sr = all_results["collection"]["solve_rate"] * 100
    n_traj = all_results["collection"]["n_trajectories"]
    ax.bar(["Aha!+Noise\n(Collection)"], [sr], color="#9C27B0", alpha=0.85)
    ax.text(0, sr + 2, f"{sr:.1f}%\n({n_traj} trajs)", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("Phase 1: Trajectory Collection", fontweight="bold")
    ax.set_ylim(0, max(sr + 15, 50))
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Template testing comparison
    ax = axes[1]
    test = all_results["test_results"]
    names = [t["condition"] for t in test]
    rates = [t["solve_rate"] * 100 for t in test]
    colors = ["#9E9E9E", "#9C27B0", "#4CAF50"]
    bars = ax.bar(range(len(test)), rates, color=colors[:len(test)], alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(test)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("Phase 3: Template Testing", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Trajectory PCA variance
    ax = axes[2]
    if "distillation" in all_results and "pca_explained_variance" in all_results["distillation"]:
        var = all_results["distillation"]["pca_explained_variance"]
        cum_var = np.cumsum(var) * 100
        ax.bar(range(len(var)), [v * 100 for v in var], color="#2196F3", alpha=0.85,
               label="Individual")
        ax.plot(range(len(var)), cum_var, 'ro-', label="Cumulative")
        ax.set_xlabel("PC Component")
        ax.set_ylabel("Explained Variance (%)")
        ax.legend(fontsize=9)
    ax.set_title("Phase 2: Trajectory PCA", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for a in axes:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase109_trajectory_distillation.png")
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
    print(f"  Phase 109: Trajectory Distillation")
    print(f"  Collection: {N_COLLECTION} games, Testing: {N_TEST} games x 3 conditions")
    print(f"{'='*80}")

    t0 = time.time()

    # Load Diff-PCA
    diff_pca_path = os.path.join(RESULTS_DIR, "phase91_diff_pca.npz")
    if not os.path.exists(diff_pca_path):
        raise FileNotFoundError(f"Phase 91 Diff-PCA not found: {diff_pca_path}")
    data = np.load(diff_pca_path)
    diff_unit = data["diff_unit"]

    model, tok = load_model()
    device = next(model.parameters()).device
    hook = AhaNoiseHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 109: Trajectory Distillation",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.join(RESULTS_DIR, "phase109_log.json")

    # === Phase 1: Collect trajectories ===
    print(f"\n  === Phase 1: Collecting Trajectories ({N_COLLECTION} games) ===")
    hook.setup_aha_noise(diff_unit, device)
    trajectories, collection_stats = collect_trajectories(model, tok, hook, N_COLLECTION)

    solved = sum(1 for s in collection_stats if s["solved"])
    all_results["collection"] = {
        "n_games": N_COLLECTION,
        "n_solved": solved,
        "solve_rate": round(solved / N_COLLECTION, 4),
        "n_trajectories": len(trajectories),
    }
    print(f"  Collection: {solved}/{N_COLLECTION} solved, {len(trajectories)} trajectories captured")

    # === Phase 2: Distill template ===
    print(f"\n  === Phase 2: Distilling Trajectory Template ===")
    trajectory_unit, distill_info = distill_trajectory_template(trajectories)
    all_results["distillation"] = distill_info
    print(f"  Distilled from {distill_info.get('n_trajectories', 0)} trajectories")

    # Save trajectory template
    if trajectory_unit is not None:
        traj_path = os.path.join(RESULTS_DIR, "phase109_trajectory_template.npz")
        np.savez(traj_path, trajectory_unit=trajectory_unit)
        print(f"  Template saved: {traj_path}")

    # === Phase 3: Test template ===
    print(f"\n  === Phase 3: Testing Template ({N_TEST} games x 3 conditions) ===")

    test_conditions = [
        {"name": "baseline",            "setup": "baseline",    "use_noise": False},
        {"name": "aha_noise",            "setup": "aha_noise",   "use_noise": True},
        {"name": "trajectory_template",  "setup": "trajectory",  "use_noise": True},
    ]

    test_results = []
    for cfg in test_conditions:
        print(f"\n    Condition: {cfg['name']}")
        if cfg["setup"] == "baseline":
            hook.setup_baseline()
        elif cfg["setup"] == "aha_noise":
            hook.setup_aha_noise(diff_unit, device)
        elif cfg["setup"] == "trajectory" and trajectory_unit is not None:
            hook.setup_trajectory(trajectory_unit, device)
        elif cfg["setup"] == "trajectory":
            hook.setup_baseline()  # Fallback if template failed
            print("    WARNING: No trajectory template, using baseline")

        games = []
        for trial in range(N_TEST):
            stats = play_game_simple(model, tok, hook, use_noise=cfg["use_noise"])
            games.append(stats)
            if (trial + 1) % 25 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                print(f"      [{trial+1}/{N_TEST}] {sr:.1f}%")

        solved = sum(1 for g in games if g["solved"])
        test_results.append({
            "condition": cfg["name"],
            "solve_rate": round(solved / len(games), 4),
            "n_solved": solved,
            "n_total": len(games),
            "games": games,
        })
        print(f"    Result: {solved}/{len(games)} = {solved/len(games)*100:.1f}%")

    all_results["test_results"] = test_results

    hook.remove()

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    bl = test_results[0]
    for tr in test_results:
        delta = tr["solve_rate"] - bl["solve_rate"]
        print(f"    {tr['condition']:25s}: {tr['solve_rate']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    # Fisher exact: trajectory vs baseline
    traj_r = test_results[2]
    table = [[traj_r["n_solved"], traj_r["n_total"]-traj_r["n_solved"]],
             [bl["n_solved"], bl["n_total"]-bl["n_solved"]]]
    _, p_val = fisher_exact(table)
    print(f"\n  Fisher exact (trajectory vs baseline): p={p_val:.4f}")
    all_results["fisher_traj_vs_baseline"] = round(p_val, 6)

    # Verdict
    traj_rate = traj_r["solve_rate"]
    aha_rate = test_results[1]["solve_rate"]
    bl_rate = bl["solve_rate"]

    if traj_rate > bl_rate + 0.1 and traj_rate >= aha_rate * 0.8:
        verdict = "DISTILLATION_SUCCESS"
        print(f"\n  VERDICT: {verdict} — Trajectory template captures prior-override pattern!")
    elif traj_rate > bl_rate + 0.05:
        verdict = "PARTIAL_DISTILLATION"
        print(f"\n  VERDICT: {verdict} — Template captures some of the Aha! effect")
    else:
        verdict = "DISTILLATION_FAILED"
        print(f"\n  VERDICT: {verdict} — Template does not replicate Aha! effect")

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
    print(f"\n Phase 109 complete.")
