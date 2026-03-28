"""
Phase 111: Self-Evolving SNN — Adaptive σ Control
==================================================

Capstone experiment: The SNN detector from Phase 110 controls
not just WHETHER to inject noise, but also HOW MUCH (adaptive σ).

Architecture:
  - SNN conflict detector outputs conflict probability
  - σ = σ_base * conflict_probability (proportional control)
  - Additionally: SNN learns online from each game's outcome

Conditions (N=50 each, Modified Hanoi):
  1. baseline:             No noise
  2. fixed_aha:            Fixed σ=0.15
  3. adaptive_binary:      SNN binary (inject/no-inject from Phase 110)
  4. adaptive_proportional: SNN proportional σ control
  5. adaptive_online:       SNN with online learning (updates after each game)

Mistral-7B-Instruct-v0.3, Layer 18
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
LAYER_IDX = 18
PCA_DIM = 64
WINDOW_SIZE = 5
N_TEST = 50

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================
#  LIF SNN  (reused from Phase 110)
# ===================================================

class LIFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, tau=0.9, threshold=1.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.tau = tau
        self.threshold = threshold

    def forward(self, x, membrane=None):
        if membrane is None:
            membrane = torch.zeros(x.size(0), self.fc.out_features, device=x.device)
        current = self.fc(x)
        membrane = self.tau * membrane + current
        spikes = (membrane >= self.threshold).float()
        membrane = membrane * (1 - spikes)
        return spikes, membrane


class SNNConflictDetector(nn.Module):
    def __init__(self, input_dim=PCA_DIM * WINDOW_SIZE, hidden_dim=32, output_dim=2):
        super().__init__()
        self.lif1 = LIFLayer(input_dim, hidden_dim, tau=0.9)
        self.lif2 = LIFLayer(hidden_dim, output_dim, tau=0.85)
        self.readout = nn.Linear(output_dim, 2)

    def forward(self, x):
        s1, _ = self.lif1(x)
        s2, _ = self.lif2(s1)
        out = self.readout(s2)
        return out

    def get_conflict_prob(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            return probs[:, 1].item()


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
                return False, f"Illegal"
            if not self.modified and disk >= top:
                self.illegal_count += 1; self._prev_illegal = True
                return False, f"Illegal"
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
        return {"solved": self.is_solved(), "legal_moves": len(self.moves),
                "illegal_moves": self.illegal_count, "self_corrections": self.self_corrections}


# ===================================================
#  PROMPT & PARSER
# ===================================================

def build_chat_prompt(tokenizer, env, error=None):
    rules = "MODIFIED RULES: You can ONLY place a LARGER disk onto a SMALLER disk. The opposite of standard."
    system = (
        f"You are solving Tower of Hanoi with {env.n_disks} disks. "
        f"{rules} "
        f"Goal: move ALL disks from A to C. "
        f"Respond with EXACTLY one move in format: Move: X->Y (e.g. Move: A->C). "
        f"You may add a brief Think: line before it."
    )
    msg = f"State: {env.state_str()}\n"
    legal = env.legal_moves()
    msg += f"Legal moves: {', '.join(legal)}\n"
    if env.moves:
        recent = env.moves[-3:]
        msg += f"Your last moves: {'; '.join(recent)}\n"
    if error:
        msg += f"ERROR: {error}. Pick from legal moves above.\n"
    msg += "Your move:"
    messages = [{"role": "user", "content": system + "\n\n" + msg}]
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
    print(f"  Done: {len(model.model.layers)} layers")
    return model, tok

def gen(model, tok, prompt, temperature=0.5, max_tokens=80):
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
#  ADAPTIVE σ HOOK
# ===================================================

class AdaptiveSigmaHook:
    def __init__(self):
        self.mode = "off"  # off, fixed, binary, proportional
        self.sigma = BASE_SIGMA
        self.diff_offset = None
        self.handle = None
        self.captured_states = []
        self.window_buffer = []
        self.pca_proj = None      # (PCA_DIM, HIDDEN_DIM)
        self.snn = None
        self.device = 'cuda'
        self.inject_log = []      # Per-step injection log

    def setup_off(self):
        self.mode = "off"

    def setup_fixed(self, diff_unit, device):
        self.mode = "fixed"
        self.diff_offset = torch.tensor(diff_unit, dtype=torch.float16, device=device)
        self.device = device

    def setup_binary(self, diff_unit, pca_proj, snn, device):
        self.mode = "binary"
        self.diff_offset = torch.tensor(diff_unit, dtype=torch.float16, device=device)
        self.pca_proj = pca_proj
        self.snn = snn
        self.device = device

    def setup_proportional(self, diff_unit, pca_proj, snn, device):
        self.mode = "proportional"
        self.diff_offset = torch.tensor(diff_unit, dtype=torch.float16, device=device)
        self.pca_proj = pca_proj
        self.snn = snn
        self.device = device

    def reset_game(self):
        self.captured_states = []
        self.window_buffer = []
        self.inject_log = []

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            hs = args[0]

            # Always capture for adaptive modes
            if hook_obj.mode in ("binary", "proportional"):
                if hs.dim() == 3:
                    state = hs[0, -1, :].detach().cpu().float().numpy()
                else:
                    state = hs[-1, :].detach().cpu().float().numpy()
                hook_obj.captured_states.append(state)
                if len(hook_obj.captured_states) >= 2:
                    delta = hook_obj.captured_states[-1] - hook_obj.captured_states[-2]
                    hook_obj.window_buffer.append(delta)
                    if len(hook_obj.window_buffer) > WINDOW_SIZE:
                        hook_obj.window_buffer.pop(0)

            if hook_obj.mode == "off":
                return args

            effective_sigma = hook_obj.sigma

            if hook_obj.mode == "fixed":
                pass  # use sigma as-is

            elif hook_obj.mode == "binary":
                if len(hook_obj.window_buffer) >= WINDOW_SIZE:
                    conflict_p = hook_obj._get_conflict_prob()
                    if conflict_p <= 0.5:
                        hook_obj.inject_log.append(0)
                        return args
                    hook_obj.inject_log.append(1)
                else:
                    hook_obj.inject_log.append(1)

            elif hook_obj.mode == "proportional":
                if len(hook_obj.window_buffer) >= WINDOW_SIZE:
                    conflict_p = hook_obj._get_conflict_prob()
                    effective_sigma = hook_obj.sigma * conflict_p
                    hook_obj.inject_log.append(conflict_p)
                else:
                    effective_sigma = hook_obj.sigma
                    hook_obj.inject_log.append(1.0)

            if effective_sigma <= 0:
                return args

            d = hs.shape[-1]
            offset = hook_obj.diff_offset
            det_scale = effective_sigma * math.sqrt(d) * 0.5
            det_noise = offset * det_scale
            if hs.dim() == 3:
                det_noise = det_noise.unsqueeze(0).unsqueeze(0).expand_as(hs)
            else:
                det_noise = det_noise.unsqueeze(0).expand_as(hs)
            stoch_noise = torch.randn_like(hs) * (effective_sigma * 0.5)
            return (hs + det_noise + stoch_noise,) + args[1:]

        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def _get_conflict_prob(self):
        window = np.array(self.window_buffer[-WINDOW_SIZE:])
        projected = window @ self.pca_proj.T
        flat = projected.flatten()
        x = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.snn.get_conflict_prob(x)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, hook, use_annealing=True):
    env = HanoiEnv(n_disks=3, modified=True)
    hook.reset_game()
    error = None; consec_fail = 0; legal_move_count = 0
    for step in range(MAX_STEPS):
        if use_annealing:
            if legal_move_count < 10:
                hook.sigma = BASE_SIGMA * (1.0 - legal_move_count / 10.0)
            else:
                hook.sigma = 0.0
        else:
            hook.sigma = BASE_SIGMA
        prompt = build_chat_prompt(tok, env, error)
        resp = gen(model, tok, prompt)
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
    stats = env.stats()
    stats["steps_taken"] = step + 1
    stats["inject_log"] = list(hook.inject_log)
    return stats


# ===================================================
#  ONLINE LEARNING
# ===================================================

def online_update(snn, optimizer, game_windows, solved, device):
    """Update SNN weights based on game outcome."""
    if not game_windows:
        return
    # If game was solved → current labels were "correct enough"
    # If game failed → inject_needed labels should have been different
    # Simple heuristic: reinforce current predictions if solved, penalize if not
    X = torch.tensor(np.array(game_windows), dtype=torch.float32).to(device)
    if solved:
        # Reinforce: treat current predictions as correct
        with torch.no_grad():
            labels = snn(X).argmax(dim=1)
    else:
        # Penalize: flip predictions (should have injected more)
        with torch.no_grad():
            preds = snn(X).argmax(dim=1)
            labels = 1 - preds  # Flip

    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    logits = snn(X)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(all_results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Phase 111: Self-Evolving SNN — Adaptive σ Control\n"
                 "SNN controls noise intensity based on detected conflict level",
                 fontsize=12, fontweight="bold")

    # Panel 1: Solve rates
    ax = axes[0]
    test = all_results.get("test_results", [])
    names = [t["condition"] for t in test]
    rates = [t["solve_rate"] * 100 for t in test]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax.bar(range(len(test)), rates, color=colors[:len(test)], alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(test)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("5-Way Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Online learning curve (if available)
    ax = axes[1]
    online = all_results.get("online_learning", {})
    if online and "cumulative_solve_rate" in online:
        cum_rate = online["cumulative_solve_rate"]
        ax.plot(range(1, len(cum_rate)+1), [r*100 for r in cum_rate],
                color="#9C27B0", linewidth=2, label="Online SNN")
        bl_rate = all_results.get("test_results", [{}])[0].get("solve_rate", 0) * 100
        ax.axhline(y=bl_rate, color='gray', linestyle='--', alpha=0.5, label="Baseline")
        ax.legend(fontsize=9)
    ax.set_xlabel("Game Number")
    ax.set_ylabel("Cumulative Solve Rate (%)")
    ax.set_title("Online Learning Progress", fontweight="bold")
    ax.grid(alpha=0.3)

    for a in axes:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase111_self_evolving_snn.png")
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
    print(f"  Phase 111: Self-Evolving SNN (Adaptive σ Control)")
    print(f"  5 conditions x N={N_TEST}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load Diff-PCA
    diff_pca_path = os.path.join(RESULTS_DIR, "phase91_diff_pca.npz")
    if not os.path.exists(diff_pca_path):
        raise FileNotFoundError(f"Phase 91 Diff-PCA not found: {diff_pca_path}")
    data = np.load(diff_pca_path)
    diff_unit = data["diff_unit"]

    # Load SNN from Phase 110
    snn_path = os.path.join(RESULTS_DIR, "phase110_snn_detector.pt")
    pca_path = os.path.join(RESULTS_DIR, "phase110_pca_projection.npz")

    model, tok = load_model()
    device = next(model.parameters()).device

    # Initialize SNN (either from Phase 110 or fresh)
    snn = SNNConflictDetector(input_dim=PCA_DIM * WINDOW_SIZE).to(device)
    pca_proj = None

    if os.path.exists(snn_path) and os.path.exists(pca_path):
        snn.load_state_dict(torch.load(snn_path, map_location=device))
        pca_data = np.load(pca_path)
        pca_proj = pca_data["pca_components"]
        print(f"  Loaded SNN from Phase 110")
    else:
        print(f"  WARNING: Phase 110 artifacts not found, training fresh SNN")
        # Create random PCA projection as fallback
        pca_proj = np.random.randn(PCA_DIM, HIDDEN_DIM).astype(np.float32)
        pca_proj /= np.linalg.norm(pca_proj, axis=1, keepdims=True)

    snn.eval()

    hook = AdaptiveSigmaHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 111: Self-Evolving SNN",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "n_test": N_TEST,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.join(RESULTS_DIR, "phase111_log.json")

    # === Test 5 conditions ===
    test_configs = [
        {"name": "baseline",              "setup": "off"},
        {"name": "fixed_aha",             "setup": "fixed"},
        {"name": "adaptive_binary",       "setup": "binary"},
        {"name": "adaptive_proportional", "setup": "proportional"},
    ]

    test_results = []
    for cfg in test_configs:
        print(f"\n  === Condition: {cfg['name']} ===")
        if cfg["setup"] == "off":
            hook.setup_off()
        elif cfg["setup"] == "fixed":
            hook.setup_fixed(diff_unit, device)
        elif cfg["setup"] == "binary":
            hook.setup_binary(diff_unit, pca_proj, snn, device)
        elif cfg["setup"] == "proportional":
            hook.setup_proportional(diff_unit, pca_proj, snn, device)

        games = []
        for trial in range(N_TEST):
            stats = play_game(model, tok, hook, use_annealing=True)
            games.append(stats)
            if (trial + 1) % 25 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                print(f"    [{trial+1}/{N_TEST}] {sr:.1f}%")

        solved = sum(1 for g in games if g["solved"])
        # Compute injection stats
        all_logs = [g.get("inject_log", []) for g in games]
        avg_inject_rate = 0.0
        if all_logs and any(all_logs):
            flat_logs = [v for log in all_logs for v in log if isinstance(v, (int, float))]
            avg_inject_rate = np.mean(flat_logs) if flat_logs else 0.0

        test_results.append({
            "condition": cfg["name"],
            "solve_rate": round(solved / len(games), 4),
            "n_solved": solved,
            "n_total": len(games),
            "avg_inject_rate": round(float(avg_inject_rate), 4),
            "games": games,
        })
        print(f"    Result: {solved}/{len(games)} = {solved/len(games)*100:.1f}%")

    # === Online learning condition ===
    print(f"\n  === Condition: adaptive_online (with online learning) ===")
    snn_online = SNNConflictDetector(input_dim=PCA_DIM * WINDOW_SIZE).to(device)
    if os.path.exists(snn_path):
        snn_online.load_state_dict(torch.load(snn_path, map_location=device))
    snn_online.train()
    online_optimizer = torch.optim.Adam(snn_online.parameters(), lr=0.0005)

    hook.setup_proportional(diff_unit, pca_proj, snn_online, device)

    online_games = []
    cumulative_solve_rate = []
    for trial in range(N_TEST):
        snn_online.eval()
        stats = play_game(model, tok, hook, use_annealing=True)
        online_games.append(stats)

        # Online update
        snn_online.train()
        if hook.window_buffer:
            windows = []
            for i in range(len(hook.window_buffer) - WINDOW_SIZE + 1):
                w = np.array(hook.window_buffer[i:i+WINDOW_SIZE]) @ pca_proj.T
                windows.append(w.flatten())
            if windows:
                online_update(snn_online, online_optimizer, windows, stats["solved"], device)

        # Track cumulative solve rate
        solved_so_far = sum(1 for g in online_games if g["solved"])
        cumulative_solve_rate.append(solved_so_far / len(online_games))

        if (trial + 1) % 10 == 0:
            sr = sum(1 for g in online_games if g["solved"]) / len(online_games) * 100
            print(f"    [{trial+1}/{N_TEST}] Cumulative: {sr:.1f}%")

    solved = sum(1 for g in online_games if g["solved"])
    test_results.append({
        "condition": "adaptive_online",
        "solve_rate": round(solved / len(online_games), 4),
        "n_solved": solved,
        "n_total": len(online_games),
        "avg_inject_rate": 0.0,  # Varies per game
        "games": online_games,
    })
    print(f"    Final: {solved}/{len(online_games)} = {solved/len(online_games)*100:.1f}%")

    all_results["test_results"] = test_results
    all_results["online_learning"] = {
        "cumulative_solve_rate": [round(r, 4) for r in cumulative_solve_rate],
        "improvement": round(cumulative_solve_rate[-1] - cumulative_solve_rate[0], 4) if cumulative_solve_rate else 0,
    }

    hook.remove()

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    bl = test_results[0]
    for tr in test_results:
        delta = tr["solve_rate"] - bl["solve_rate"]
        print(f"    {tr['condition']:25s}: {tr['solve_rate']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    # Best adaptive method
    adaptive_results = [t for t in test_results if t["condition"] not in ("baseline", "fixed_aha")]
    best_adaptive = max(adaptive_results, key=lambda t: t["solve_rate"])
    fixed_rate = test_results[1]["solve_rate"]

    if best_adaptive["solve_rate"] >= fixed_rate:
        verdict = "ADAPTIVE_SUPERIOR"
        print(f"\n  VERDICT: {verdict} — Adaptive SNN matches or surpasses fixed injection!")
    elif best_adaptive["solve_rate"] > bl["solve_rate"] + 0.05:
        verdict = "ADAPTIVE_EFFECTIVE"
        print(f"\n  VERDICT: {verdict} — Adaptive SNN helps but doesn't beat fixed injection")
    else:
        verdict = "NEEDS_MORE_DEVELOPMENT"
        print(f"\n  VERDICT: {verdict} — Self-evolving SNN needs more refinement")

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
    print(f"\n Phase 111 complete.")
