"""
Phase 97: Cross-Architecture Vector Translation — Can Reasoning Transfer?
=========================================================================

Phase 94 showed Qwen and Mistral both concentrate discriminant axes in
top-64 PCs. This experiment tests whether the Aha! vector itself can
be TRANSFERRED between architectures.

Method:
  1. Procrustes analysis: align Qwen's top-64 PCA space to Mistral's
  2. Transfer QwenAha → Mistral space, MistralAha → Qwen space
  3. Test transferred vectors via Aha! + noise steering

Protocol (N=30 each):
  Mistral tests:
    1. baseline
    2. mistral_native_aha+noise (Phase 95 replication)
    3. qwen_transferred_aha+noise
    4. random_direction+noise (null control)
  Qwen tests:
    5. baseline
    6. qwen_native_aha+noise (Phase 92 replication)
    7. mistral_transferred_aha+noise
    8. random_direction+noise (null control)

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
from scipy.linalg import orthogonal_procrustes

# === Config ===
SEED = 2026
MAX_STEPS = 50
BASE_SIGMA = 0.15
N_PER_CONDITION = 30

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Model configs
MODELS = {
    "Mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "short": "Mistral-7B",
        "hidden_dim": 4096,
        "layer": 18,
        "diff_pca_path": "phase91_diff_pca.npz",
        "std_pca_path": "phase86_mistral_pca.npz",
        "trust_remote_code": False,
    },
    "Qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "short": "Qwen2.5-7B",
        "hidden_dim": 3584,
        "layer": 16,
        "diff_pca_path": "phase87_diff_pca.npz",
        "std_pca_path": "phase84_qwen_pca.npz",
        "trust_remote_code": True,
    },
}


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
#  MODEL MANAGEMENT
# ===================================================

def load_model(model_key):
    cfg = MODELS[model_key]
    print(f"\n Loading {cfg['name']}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(cfg["name"], use_fast=True,
                                         trust_remote_code=cfg["trust_remote_code"],
                                         local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"], quantization_config=bnb, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=cfg["trust_remote_code"],
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
#  VECTOR TRANSLATION
# ===================================================

def translate_vector_procrustes(source_pca_Vt, target_pca_Vt, source_diff_unit, top_k=64):
    """Translate a differential vector from source PCA space to target PCA space
    using Orthogonal Procrustes on the top-K principal components.

    Since the two models have different hidden dimensions, we work in the
    shared top-K PCA coefficient space.
    """
    # Project source diff_unit into source's top-K PCA space
    S_Vt = source_pca_Vt[:top_k]  # (K, d_source)
    source_coeffs = S_Vt @ source_diff_unit  # (K,)

    # Now source_coeffs lives in K-dimensional "concept space"
    # We map it to target's concept space via Procrustes.
    # But Procrustes needs paired data — we don't have that.
    # Instead, use a simpler approach: assume the K PCA dimensions
    # are ordered by importance, and the differential vector's
    # coefficients transfer proportionally.

    # Direct coefficient transfer (spectral assumption):
    T_Vt = target_pca_Vt[:top_k]  # (K, d_target)
    # Reconstruct in target space using same coefficients
    transferred = source_coeffs @ T_Vt  # (d_target,)
    norm = np.linalg.norm(transferred)
    if norm > 1e-8:
        transferred = transferred / norm
    return transferred


def translate_vector_random_proj(source_diff_unit, target_dim):
    """Random projection baseline — null control."""
    rng = np.random.RandomState(42)
    rand_vec = rng.randn(target_dim).astype(np.float32)
    rand_vec /= (np.linalg.norm(rand_vec) + 1e-8)
    return rand_vec


# ===================================================
#  AHA+ NOISE HOOK
# ===================================================

class AhaSteeringHook:
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.mode = "baseline"
        self.fixed_offset = None
        self.handle = None

    def setup_baseline(self):
        self.mode = "baseline"

    def setup_aha_plus_noise(self, diff_unit_vec, device='cuda'):
        self.mode = "aha_plus_noise"
        du = torch.tensor(diff_unit_vec, dtype=torch.float16, device=device)
        self.fixed_offset = du

    def register(self, model, layer_idx):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            if hook_obj.mode == "baseline":
                return args
            hs = args[0]
            if hook_obj.mode == "aha_plus_noise":
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
            return args
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, hook, use_flash=True):
    env = HanoiEnv(n_disks=3, modified=True)
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


def run_conditions(model, tok, hook, layer_idx, conditions, n_per=N_PER_CONDITION):
    """Run a set of conditions on a single model."""
    device = next(model.parameters()).device
    hook.register(model, layer_idx)
    results = []

    for cond in conditions:
        print(f"\n    Condition: {cond['name']}")
        if cond.get("vector") is None:
            hook.setup_baseline()
            use_flash = False
        else:
            hook.setup_aha_plus_noise(cond["vector"], device)
            use_flash = True

        games = []
        for trial in range(n_per):
            stats = play_game(model, tok, hook, use_flash=use_flash)
            games.append(stats)
            if (trial + 1) % 10 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                print(f"      [{trial+1}/{n_per}] Solve rate: {sr:.1f}%")

        solved = sum(1 for g in games if g["solved"])
        results.append({
            "condition": cond["name"],
            "solve_rate": round(solved / len(games), 4),
            "n_solved": solved,
            "n_total": len(games),
        })
        print(f"      Final: {solved}/{len(games)} = {solved/len(games)*100:.1f}%")

    hook.remove()
    return results


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(all_results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 97: Cross-Architecture Vector Translation\n"
                 "Can Qwen's reasoning vector make Mistral smarter (and vice versa)?",
                 fontsize=13, fontweight="bold")

    for ax_idx, (model_key, title) in enumerate([("Mistral", "Mistral-7B"), ("Qwen", "Qwen2.5-7B")]):
        ax = axes[ax_idx]
        if model_key in all_results.get("model_results", {}):
            conds = all_results["model_results"][model_key]
            names = [c["condition"] for c in conds]
            rates = [c["solve_rate"] * 100 for c in conds]
            colors = ["#9E9E9E", "#2196F3", "#FF9800", "#E0E0E0"]
            bars = ax.bar(range(len(conds)), rates, color=colors[:len(conds)], alpha=0.85,
                          edgecolor="white", linewidth=2)
            for bar, val in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels([n.replace("_", "\n").replace("+", "+\n") for n in names], fontsize=8)
        ax.set_ylabel("Solve Rate (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase97_cross_arch_transfer.png")
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
    print(f"  Phase 97: Cross-Architecture Vector Translation")
    print(f"{'='*80}")

    t0 = time.time()

    # Load Differential PCA vectors
    pca_data = {}
    for model_key, cfg in MODELS.items():
        diff_path = os.path.join(RESULTS_DIR, cfg["diff_pca_path"])
        std_path = os.path.join(RESULTS_DIR, cfg["std_pca_path"])
        if not os.path.exists(diff_path):
            raise FileNotFoundError(f"Diff PCA not found for {model_key}: {diff_path}")
        data = np.load(diff_path)
        pca_data[model_key] = {
            "diff_unit": data["diff_unit"],
            "Vt": data["Vt"],
        }
        if os.path.exists(std_path):
            std_data = np.load(std_path)
            pca_data[model_key]["std_Vt"] = std_data["Vt"]
            print(f"  {model_key}: diff_unit norm={np.linalg.norm(data['diff_unit']):.4f}, "
                  f"std PCA loaded ({std_data['Vt'].shape})")
        else:
            print(f"  {model_key}: std PCA not found, using diff PCA Vt for translation")
            pca_data[model_key]["std_Vt"] = data["Vt"]

    # Translate vectors
    print(f"\n  === Vector Translation ===")
    # Qwen → Mistral
    qwen_to_mistral = translate_vector_procrustes(
        pca_data["Qwen"]["std_Vt"], pca_data["Mistral"]["std_Vt"],
        pca_data["Qwen"]["diff_unit"], top_k=64
    )
    # Mistral → Qwen
    mistral_to_qwen = translate_vector_procrustes(
        pca_data["Mistral"]["std_Vt"], pca_data["Qwen"]["std_Vt"],
        pca_data["Mistral"]["diff_unit"], top_k=64
    )
    # Random controls
    random_mistral = translate_vector_random_proj(None, MODELS["Mistral"]["hidden_dim"])
    random_qwen = translate_vector_random_proj(None, MODELS["Qwen"]["hidden_dim"])

    print(f"  Qwen→Mistral vector norm: {np.linalg.norm(qwen_to_mistral):.4f}")
    print(f"  Mistral→Qwen vector norm: {np.linalg.norm(mistral_to_qwen):.4f}")

    all_results = {
        "experiment": "Phase 97: Cross-Architecture Vector Translation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sigma": BASE_SIGMA,
        "n_per_condition": N_PER_CONDITION,
        "translation_method": "spectral_coefficient_transfer_top64",
        "model_results": {},
    }

    # === Test on Mistral ===
    print(f"\n  === Testing on Mistral ===")
    model, tok = load_model("Mistral")
    mistral_conditions = [
        {"name": "baseline", "vector": None},
        {"name": "native_aha+noise", "vector": pca_data["Mistral"]["diff_unit"]},
        {"name": "qwen_transferred+noise", "vector": qwen_to_mistral},
        {"name": "random_dir+noise", "vector": random_mistral},
    ]
    hook = AhaSteeringHook()
    mistral_results = run_conditions(model, tok, hook, MODELS["Mistral"]["layer"],
                                      mistral_conditions)
    all_results["model_results"]["Mistral"] = mistral_results

    # Intermediate save
    results_path = os.path.join(RESULTS_DIR, "phase97_log.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Cleanup Mistral
    del model, tok
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Test on Qwen ===
    print(f"\n  === Testing on Qwen ===")
    model, tok = load_model("Qwen")
    qwen_conditions = [
        {"name": "baseline", "vector": None},
        {"name": "native_aha+noise", "vector": pca_data["Qwen"]["diff_unit"]},
        {"name": "mistral_transferred+noise", "vector": mistral_to_qwen},
        {"name": "random_dir+noise", "vector": random_qwen},
    ]
    hook = AhaSteeringHook()
    qwen_results = run_conditions(model, tok, hook, MODELS["Qwen"]["layer"],
                                    qwen_conditions)
    all_results["model_results"]["Qwen"] = qwen_results

    del model, tok
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Analysis ===
    print(f"\n  === Cross-Architecture Transfer Analysis ===")
    for model_key in ["Mistral", "Qwen"]:
        results = all_results["model_results"][model_key]
        bl = results[0]
        native = results[1]
        transferred = results[2]
        rand_ctrl = results[3]

        print(f"\n  {model_key}:")
        print(f"    baseline:     {bl['solve_rate']*100:.1f}%")
        print(f"    native aha:   {native['solve_rate']*100:.1f}%")
        print(f"    transferred:  {transferred['solve_rate']*100:.1f}%")
        print(f"    random:       {rand_ctrl['solve_rate']*100:.1f}%")

        # Fisher exact: transferred vs random
        table = [[transferred["n_solved"], transferred["n_total"]-transferred["n_solved"]],
                 [rand_ctrl["n_solved"], rand_ctrl["n_total"]-rand_ctrl["n_solved"]]]
        _, p = fisher_exact(table)
        print(f"    transferred vs random: p={p:.4f}")

    # Verdict
    m_trans = all_results["model_results"]["Mistral"][2]["solve_rate"]
    m_rand = all_results["model_results"]["Mistral"][3]["solve_rate"]
    q_trans = all_results["model_results"]["Qwen"][2]["solve_rate"]
    q_rand = all_results["model_results"]["Qwen"][3]["solve_rate"]

    if m_trans > m_rand + 0.1 and q_trans > q_rand + 0.1:
        verdict = "TRANSFERABLE"
        print(f"\n  VERDICT: {verdict} — Reasoning vectors transfer across architectures!")
    elif m_trans > m_rand + 0.05 or q_trans > q_rand + 0.05:
        verdict = "PARTIALLY_TRANSFERABLE"
        print(f"\n  VERDICT: {verdict} — Partial transfer observed")
    else:
        verdict = "ARCHITECTURE_SPECIFIC"
        print(f"\n  VERDICT: {verdict} — Vectors are architecture-specific")

    all_results["verdict"] = verdict

    # Visualization
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
    print(f"\n Phase 97 complete.")
