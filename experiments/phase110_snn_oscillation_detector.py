"""
Phase 110: SNN Oscillation Detector — Spike-Based Prior Conflict Detection
==========================================================================

Opus Proposal: Use a lightweight SNN (Spiking Neural Network) to detect
oscillation patterns in the hidden-state trajectory that signal "prior conflict."
When detected, automatically trigger Aha!+noise injection.

Architecture:
  - Input: rolling window of 5 hidden-state Δ vectors (∈ R^64 via PCA projection)
  - SNN: 2-layer LIF network (64→32→2) trained to classify "conflicted" vs "aligned"
  - Training: supervised from Phase 109 trajectory data + Phase 91/102 labels
  - Evaluation: adaptive injection vs fixed injection on Modified Hanoi (N=50)

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
PCA_DIM = 64       # Project to 64-dim for SNN input
WINDOW_SIZE = 5     # Rolling window of Δ vectors
N_COLLECT = 80      # Collection games (40 standard + 40 modified)
N_TEST = 50         # Test games per condition

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================
#  LIF SNN DETECTOR
# ===================================================

class LIFLayer(nn.Module):
    """Leaky Integrate-and-Fire neuron layer."""
    def __init__(self, in_dim, out_dim, tau=0.9, threshold=1.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.tau = tau
        self.threshold = threshold

    def forward(self, x, membrane=None):
        """x: (batch, in_dim), returns spikes and new membrane."""
        if membrane is None:
            membrane = torch.zeros(x.size(0), self.fc.out_features, device=x.device)
        current = self.fc(x)
        membrane = self.tau * membrane + current
        spikes = (membrane >= self.threshold).float()
        membrane = membrane * (1 - spikes)  # Reset on spike
        return spikes, membrane


class SNNOscillationDetector(nn.Module):
    """2-layer LIF network for detecting prior-conflict oscillations."""
    def __init__(self, input_dim=PCA_DIM * WINDOW_SIZE, hidden_dim=32, output_dim=2):
        super().__init__()
        self.lif1 = LIFLayer(input_dim, hidden_dim, tau=0.9)
        self.lif2 = LIFLayer(hidden_dim, output_dim, tau=0.85)
        self.readout = nn.Linear(output_dim, 2)  # conflict vs aligned

    def forward(self, x):
        """x: (batch, window_size * pca_dim)"""
        s1, _ = self.lif1(x)
        s2, _ = self.lif2(s1)
        out = self.readout(s2)
        return out

    def predict_conflict(self, x):
        """Returns probability of conflict state."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            return probs[:, 1]  # probability of "conflict"


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

def build_chat_prompt(tokenizer, env, error=None):
    if env.modified:
        rules = "MODIFIED RULES: You can ONLY place a LARGER disk onto a SMALLER disk. The opposite of standard."
    else:
        rules = "STANDARD RULES: You can ONLY place a SMALLER disk onto a LARGER disk."
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
    print(f"  Done: {len(model.model.layers)} layers, hidden_dim={model.config.hidden_size}")
    return model, tok


# ===================================================
#  HIDDEN STATE CAPTURE + ADAPTIVE INJECTION
# ===================================================

class CaptureAndInjectHook:
    """Combined hook: captures hidden states AND optionally injects noise."""
    def __init__(self):
        self.capture_mode = False
        self.inject_mode = False
        self.adaptive_mode = False
        self.sigma = BASE_SIGMA
        self.diff_offset = None
        self.handle = None
        self.captured_states = []
        self.pca_projection = None  # (HIDDEN_DIM, PCA_DIM) projection matrix
        self.snn_detector = None
        self.window_buffer = []  # Rolling Δ buffer

    def setup_capture_only(self):
        self.capture_mode = True
        self.inject_mode = False
        self.adaptive_mode = False

    def setup_fixed_inject(self, diff_unit, device='cuda'):
        self.capture_mode = False
        self.inject_mode = True
        self.adaptive_mode = False
        self.diff_offset = torch.tensor(diff_unit, dtype=torch.float16, device=device)

    def setup_adaptive(self, diff_unit, pca_proj, snn_model, device='cuda'):
        self.capture_mode = True
        self.inject_mode = True
        self.adaptive_mode = True
        self.diff_offset = torch.tensor(diff_unit, dtype=torch.float16, device=device)
        self.pca_projection = torch.tensor(pca_proj, dtype=torch.float32, device=device)
        self.snn_detector = snn_model.to(device)
        self.snn_detector.eval()
        self.window_buffer = []

    def setup_off(self):
        self.capture_mode = False
        self.inject_mode = False
        self.adaptive_mode = False

    def reset_buffer(self):
        self.window_buffer = []
        self.captured_states = []

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            hs = args[0]
            # Capture
            if hook_obj.capture_mode:
                if hs.dim() == 3:
                    state = hs[0, -1, :].detach().cpu().float().numpy()
                else:
                    state = hs[-1, :].detach().cpu().float().numpy()
                hook_obj.captured_states.append(state)

                # Update Δ buffer for adaptive mode
                if len(hook_obj.captured_states) >= 2:
                    delta = hook_obj.captured_states[-1] - hook_obj.captured_states[-2]
                    hook_obj.window_buffer.append(delta)
                    if len(hook_obj.window_buffer) > WINDOW_SIZE:
                        hook_obj.window_buffer.pop(0)

            # Inject (fixed or adaptive)
            if not hook_obj.inject_mode or hook_obj.sigma <= 0:
                return args

            should_inject = True
            if hook_obj.adaptive_mode and hook_obj.snn_detector is not None:
                if len(hook_obj.window_buffer) >= WINDOW_SIZE:
                    # Project deltas to PCA space
                    window = np.array(hook_obj.window_buffer[-WINDOW_SIZE:])
                    projected = window @ hook_obj.pca_projection.cpu().numpy().T
                    flat = projected.flatten()
                    x = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(hs.device)
                    conflict_prob = hook_obj.snn_detector.predict_conflict(x).item()
                    should_inject = conflict_prob > 0.5
                else:
                    should_inject = True  # Default inject at start

            if not should_inject:
                return args

            d = hs.shape[-1]
            offset = hook_obj.diff_offset
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

def generate_response(model, tok, prompt, temperature=0.5, max_tokens=80):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperature, top_p=0.9,
            repetition_penalty=1.2, pad_token_id=tok.pad_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    inp = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(inp):].strip()


def play_game(model, tok, hook, modified=True, use_annealing=True):
    env = HanoiEnv(n_disks=3, modified=modified)
    hook.reset_buffer()
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
        resp = generate_response(model, tok, prompt)
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
    return stats


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(all_results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Phase 110: SNN Oscillation Detector — Adaptive Noise Injection",
                 fontsize=12, fontweight="bold")

    # Panel 1: SNN training accuracy
    ax = axes[0]
    if "snn_training" in all_results:
        losses = all_results["snn_training"].get("loss_history", [])
        if losses:
            ax.plot(losses, color="#2196F3")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
    ax.set_title("SNN Detector Training", fontweight="bold")
    ax.grid(alpha=0.3)

    # Panel 2: Test results comparison
    ax = axes[1]
    test = all_results.get("test_results", [])
    if test:
        names = [t["condition"] for t in test]
        rates = [t["solve_rate"] * 100 for t in test]
        colors = ["#9E9E9E", "#2196F3", "#4CAF50"]
        bars = ax.bar(range(len(test)), rates, color=colors[:len(test)], alpha=0.85,
                      edgecolor="white", linewidth=2)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(test)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("Adaptive vs Fixed Injection", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Conflict detection rate
    ax = axes[2]
    if "adaptive_stats" in all_results:
        stats = all_results["adaptive_stats"]
        ax.bar(["Inject Rate"], [stats.get("avg_inject_rate", 0) * 100],
               color="#FF9800", alpha=0.85)
        ax.text(0, stats.get("avg_inject_rate", 0) * 100 + 2,
                f"{stats.get('avg_inject_rate', 0)*100:.1f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Injection Rate (%)")
    ax.set_title("Adaptive Injection Frequency", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    for a in axes:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase110_snn_oscillation.png")
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
    print(f"  Phase 110: SNN Oscillation Detector")
    print(f"  Collection: {N_COLLECT} games, SNN training, Testing: {N_TEST} x 3")
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
    hook = CaptureAndInjectHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 110: SNN Oscillation Detector",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.join(RESULTS_DIR, "phase110_log.json")

    # === Phase 1: Collect labeled trajectories ===
    print(f"\n  === Phase 1: Collecting labeled trajectories ===")
    all_deltas_std = []  # Label 0 = aligned
    all_deltas_mod = []  # Label 1 = conflict

    # Standard Hanoi (aligned priors)
    print(f"  Standard Hanoi ({N_COLLECT//2} games)...")
    hook.setup_capture_only()
    for gi in range(N_COLLECT // 2):
        hook.reset_buffer()
        stats = play_game(model, tok, hook, modified=False, use_annealing=False)
        if len(hook.window_buffer) >= WINDOW_SIZE:
            all_deltas_std.extend(hook.window_buffer)

    # Modified Hanoi (conflicting priors)
    print(f"  Modified Hanoi ({N_COLLECT//2} games)...")
    for gi in range(N_COLLECT // 2):
        hook.reset_buffer()
        stats = play_game(model, tok, hook, modified=True, use_annealing=False)
        if len(hook.window_buffer) >= WINDOW_SIZE:
            all_deltas_mod.extend(hook.window_buffer)

    print(f"  Collected: {len(all_deltas_std)} aligned, {len(all_deltas_mod)} conflict Δ-vectors")

    # === Phase 2: PCA projection + SNN training ===
    print(f"\n  === Phase 2: PCA + SNN Training ===")
    from sklearn.decomposition import PCA

    # Combine and fit PCA
    all_deltas = np.array(all_deltas_std + all_deltas_mod)
    pca = PCA(n_components=PCA_DIM)
    pca.fit(all_deltas)
    pca_proj = pca.components_  # (PCA_DIM, HIDDEN_DIM)
    print(f"  PCA: {pca.explained_variance_ratio_[:5].sum()*100:.1f}% variance in first 5 PCs")

    # Create windowed training data
    def make_windows(deltas, label):
        projected = deltas @ pca_proj.T  # (N, PCA_DIM)
        X, y = [], []
        for i in range(len(projected) - WINDOW_SIZE + 1):
            window = projected[i:i+WINDOW_SIZE].flatten()
            X.append(window)
            y.append(label)
        return X, y

    X_std, y_std = make_windows(np.array(all_deltas_std), 0)
    X_mod, y_mod = make_windows(np.array(all_deltas_mod), 1)
    X_train = torch.tensor(np.array(X_std + X_mod), dtype=torch.float32).to(device)
    y_train = torch.tensor(np.array(y_std + y_mod), dtype=torch.long).to(device)

    # Shuffle
    perm = torch.randperm(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    print(f"  Training data: {len(X_train)} windows ({len(X_std)} aligned + {len(X_mod)} conflict)")

    # Train SNN
    snn = SNNOscillationDetector(input_dim=PCA_DIM * WINDOW_SIZE).to(device)
    optimizer = torch.optim.Adam(snn.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    batch_size = min(64, len(X_train))
    for epoch in range(50):
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            optimizer.zero_grad()
            logits = snn(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / (len(X_train) // batch_size + 1)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                preds = snn(X_train).argmax(dim=1)
                acc = (preds == y_train).float().mean().item()
            print(f"    Epoch {epoch+1}/50: loss={avg_loss:.4f}, acc={acc*100:.1f}%")

    # Final accuracy
    with torch.no_grad():
        preds = snn(X_train).argmax(dim=1)
        train_acc = (preds == y_train).float().mean().item()
    print(f"  Final train accuracy: {train_acc*100:.1f}%")

    all_results["snn_training"] = {
        "n_windows": len(X_train),
        "final_accuracy": round(train_acc, 4),
        "loss_history": [round(l, 4) for l in loss_history],
        "pca_variance_top5": round(pca.explained_variance_ratio_[:5].sum(), 4),
    }

    # Save SNN model
    snn_path = os.path.join(RESULTS_DIR, "phase110_snn_detector.pt")
    torch.save(snn.state_dict(), snn_path)
    pca_path = os.path.join(RESULTS_DIR, "phase110_pca_projection.npz")
    np.savez(pca_path, pca_components=pca_proj)

    # === Phase 3: Test adaptive vs fixed injection ===
    print(f"\n  === Phase 3: Testing ({N_TEST} games x 3 conditions) ===")

    test_conditions = [
        {"name": "baseline",       "mode": "off"},
        {"name": "fixed_inject",   "mode": "fixed"},
        {"name": "adaptive_snn",   "mode": "adaptive"},
    ]

    test_results = []
    for cfg in test_conditions:
        print(f"\n    Condition: {cfg['name']}")
        if cfg["mode"] == "off":
            hook.setup_off()
        elif cfg["mode"] == "fixed":
            hook.setup_fixed_inject(diff_unit, device)
        elif cfg["mode"] == "adaptive":
            hook.setup_adaptive(diff_unit, pca_proj, snn, device)

        games = []
        for trial in range(N_TEST):
            stats = play_game(model, tok, hook, modified=True, use_annealing=True)
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
        print(f"    {tr['condition']:20s}: {tr['solve_rate']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    # Verdict
    adaptive = test_results[2]["solve_rate"]
    fixed = test_results[1]["solve_rate"]
    baseline = test_results[0]["solve_rate"]

    if adaptive >= fixed - 0.02 and adaptive > baseline + 0.05:
        verdict = "SNN_ADAPTIVE_WORKS"
        print(f"\n  VERDICT: {verdict} — SNN adaptively matches or beats fixed injection!")
    elif adaptive > baseline + 0.03:
        verdict = "SNN_PARTIAL_SUCCESS"
        print(f"\n  VERDICT: {verdict} — SNN helps but doesn't fully match fixed injection")
    else:
        verdict = "SNN_NEEDS_MORE_DATA"
        print(f"\n  VERDICT: {verdict} — SNN detector needs more training data or refinement")

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
    print(f"\n Phase 110 complete.")
