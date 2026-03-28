"""
Phase 70c: Noise Source Type Sweep (k=4 fixed)
================================================

Phase 76 answered "WHEN to inject noise" → Flash Annealing (first 10 steps)
Phase 70c answers "WHAT kind of noise to inject" → Structured vs random

All conditions use k=4 rank subspace (Phase 70b champion) with σ=0.15,
Layer 18, Flash Annealing schedule (first 10 steps only).

Noise sources:
  1. Gaussian (baseline)     — torch.randn, i.i.d., maximum entropy
  2. Quasi-periodic           — sin(t) + sin(√2·t), never repeats, deterministic
  3. Logistic map (chaos)     — x_{n+1} = 4x_n(1-x_n), low-dim chaos, fractal structure
  4. 1/f (pink) noise         — power-spectral density ∝ 1/f, neural-realistic
  5. Uniform                  — torch.rand, flat distribution (control)
  6. Gaussian (no Flash)      — always-on Gaussian (Phase 70b replication, control)

Conditions (6, N=30 each + baseline N=30):
  Total: 210 games, ~3.5 hours

Key question: "Given optimal k=4 and Flash schedule, does the STRUCTURE
of noise matter, or is any perturbation equally effective?"

Usage:
    python experiments/phase70c_noise_source.py
"""

import torch
import torch.nn as nn
import os, json, gc, time, random, re, math, csv
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
RANK_K = 4  # Phase 70b champion

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CSV_PATH = r"C:\tmp\experiment_control.csv"

def should_hibernate(phase_num):
    if not os.path.exists(CSV_PATH):
        return True
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['phase']) == phase_num:
                return int(row['hibernate']) == 1
    return True


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
#  NOISE SOURCE GENERATORS
# ===================================================

def generate_basis(k, d=HIDDEN_DIM, device='cuda'):
    """Generate k-dimensional orthogonal basis via QR."""
    torch.manual_seed(42)
    A = torch.randn(d, k, dtype=torch.float32, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q.to(torch.float16)


class NoiseSourceBase:
    """Base class for noise source generators."""
    def __init__(self, name):
        self.name = name
        self.call_count = 0

    def generate(self, shape, dtype, device):
        """Generate noise coefficients of shape (batch, seq, k)."""
        raise NotImplementedError

    def reset(self):
        self.call_count = 0


class GaussianSource(NoiseSourceBase):
    """Standard i.i.d. Gaussian — maximum entropy, memoryless."""
    def __init__(self):
        super().__init__("gaussian")

    def generate(self, shape, dtype, device):
        self.call_count += 1
        return torch.randn(shape, dtype=dtype, device=device)


class QuasiPeriodicSource(NoiseSourceBase):
    """Quasi-periodic: sin(t) + sin(√2·t) + sin(π·t) + sin(e·t).
    Never repeats, deterministic, dense trajectory on torus."""
    def __init__(self):
        super().__init__("quasi_periodic")
        # Irrational frequency ratios
        self.freqs = [1.0, math.sqrt(2), math.pi, math.e]

    def generate(self, shape, dtype, device):
        self.call_count += 1
        b, s, k = shape
        # Use call count as time parameter
        t = self.call_count * 0.1
        values = []
        for i in range(k):
            # Each dimension uses a different irrational frequency combo
            val = 0
            for j, freq in enumerate(self.freqs):
                val += math.sin(freq * (t + i * 7.3 + j * 3.1))
            values.append(val / len(self.freqs))  # Normalize to [-1, 1]

        # Create tensor: same value across batch and seq (deterministic)
        coeffs = torch.tensor(values, dtype=dtype, device=device)
        # Add slight per-token variation using golden ratio
        token_offsets = torch.arange(s, dtype=dtype, device=device) * 0.618033988749895
        # Shape: (1, s, k) → broadcast to (b, s, k)
        noise = coeffs.unsqueeze(0).unsqueeze(0).expand(b, s, k).clone()
        for dim in range(k):
            phase_shift = torch.sin(token_offsets * self.freqs[dim % len(self.freqs)] + dim)
            noise[:, :, dim] += phase_shift.unsqueeze(0) * 0.3
        return noise


class LogisticMapSource(NoiseSourceBase):
    """Logistic map: x_{n+1} = 4·x_n·(1-x_n).
    Chaotic, deterministic, sensitive to initial conditions.
    Has fractal structure in its trajectories."""
    def __init__(self):
        super().__init__("logistic_chaos")
        # Initialize k independent chaotic trajectories
        self.states = None

    def _init_states(self, k, device, dtype):
        # Use different initial conditions for each dimension
        # Avoid exact 0, 0.25, 0.5, 0.75, 1.0 (fixed points)
        x0 = torch.tensor([0.1 + 0.15 * i for i in range(k)], dtype=torch.float64)
        # Warm up: iterate 100 times to reach chaotic regime
        for _ in range(100):
            x0 = 4.0 * x0 * (1.0 - x0)
        self.states = x0.to(dtype).to(device)

    def generate(self, shape, dtype, device):
        self.call_count += 1
        b, s, k = shape
        if self.states is None or len(self.states) != k:
            self._init_states(k, device, dtype)

        # Generate s steps of chaos for each of k dimensions
        noise = torch.zeros(s, k, dtype=torch.float64, device=device)
        states = self.states.to(torch.float64)
        for t in range(s):
            states = 4.0 * states * (1.0 - states)
            # Map from [0,1] to [-1,1] (zero mean)
            noise[t] = states * 2.0 - 1.0

        self.states = states.to(dtype)
        # Expand to batch: (1, s, k) → (b, s, k)
        return noise.unsqueeze(0).expand(b, s, k).to(dtype)


class PinkNoiseSource(NoiseSourceBase):
    """1/f (pink) noise via spectral filtering.
    Neural-realistic: brain noise follows 1/f power spectrum."""
    def __init__(self):
        super().__init__("pink_1f")

    def generate(self, shape, dtype, device):
        self.call_count += 1
        b, s, k = shape
        # Generate in float32 for FFT, then convert
        noise_np = np.zeros((s, k), dtype=np.float32)
        for dim in range(k):
            # White noise in frequency domain
            white = np.random.randn(s) + 1j * np.random.randn(s)
            # 1/f filter (avoid division by zero at DC)
            freqs = np.fft.fftfreq(s)
            freqs[0] = 1.0  # Avoid div by zero
            filt = 1.0 / np.sqrt(np.abs(freqs))
            filt[0] = 0.0  # Zero DC component
            # Apply filter and inverse FFT
            pink = np.real(np.fft.ifft(white * filt))
            # Normalize to unit variance
            if pink.std() > 1e-8:
                pink = pink / pink.std()
            noise_np[:, dim] = pink

        noise = torch.tensor(noise_np, dtype=dtype, device=device)
        return noise.unsqueeze(0).expand(b, s, k).clone()


class UniformSource(NoiseSourceBase):
    """Uniform distribution on [-1, 1]. Control for distribution shape."""
    def __init__(self):
        super().__init__("uniform")

    def generate(self, shape, dtype, device):
        self.call_count += 1
        # Uniform [-1, 1], scaled to have same std as Gaussian
        # std of Uniform[-1,1] = 1/√3 ≈ 0.577, so multiply by √3 to match Gaussian std=1
        return (torch.rand(shape, dtype=dtype, device=device) * 2 - 1) * math.sqrt(3)


# ===================================================
#  RANK-K NOISE HOOK WITH PLUGGABLE SOURCE
# ===================================================

class SourcedRankKHook:
    """Rank-k noise hook with pluggable noise source."""
    def __init__(self):
        self.active = False
        self.sigma = BASE_SIGMA
        self.rank_k = RANK_K
        self.basis = None
        self.source = None
        self.handle = None

    def setup(self, basis, source):
        self.basis = basis
        self.source = source

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.source is None:
                return args
            hs = args[0]
            b, s, d = hs.shape
            # Get structured noise coefficients from source
            z = hook_obj.source.generate((b, s, hook_obj.rank_k), hs.dtype, hs.device)
            z = z * hook_obj.sigma
            # Project into full space via basis
            noise = z @ hook_obj.basis.T
            return (hs + noise,) + args[1:]
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


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
#  GAME FUNCTION (with Flash Annealing)
# ===================================================

def play_game(model, tok, env, hook, flash_steps=10):
    """Play a Hanoi game with Flash Annealing noise schedule."""
    env.reset()
    error = None
    consec_fail = 0
    legal_move_count = 0

    for step in range(MAX_STEPS):
        # Flash Annealing: noise ON for first N steps only
        hook.active = (step < flash_steps)

        prompt = build_chat_prompt(tok, env, error)
        resp = generate(model, tok, prompt)
        move = parse_move(resp)

        if move is None:
            env.illegal_count += 1
            env.total_attempts += 1
            env._prev_illegal = True
            error = "Parse fail. Use Move: X->Y"
            consec_fail += 1
            if consec_fail >= 10:
                break
            continue

        ok, msg = env.try_move(move[0], move[1])
        if ok:
            legal_move_count += 1
            error = None
            consec_fail = 0
            if env.is_solved():
                break
        else:
            error = msg
            consec_fail += 1
            if consec_fail >= 10:
                break

    stats = env.stats()
    stats["steps_taken"] = step + 1
    return stats


# ===================================================
#  EXPERIMENT CONDITIONS
# ===================================================

CONDITIONS = [
    {"name": "baseline_no_noise",  "source": None,                  "flash": 0,    "desc": "No noise (control)"},
    {"name": "gaussian_flash",     "source": GaussianSource,        "flash": 10,   "desc": "Gaussian + Flash10 (Phase 70b champion)"},
    {"name": "quasiperiodic_flash","source": QuasiPeriodicSource,   "flash": 10,   "desc": "Quasi-periodic sin(√2·t) + Flash10"},
    {"name": "chaos_flash",        "source": LogisticMapSource,     "flash": 10,   "desc": "Logistic map chaos + Flash10"},
    {"name": "pink_1f_flash",      "source": PinkNoiseSource,       "flash": 10,   "desc": "1/f pink noise + Flash10"},
    {"name": "uniform_flash",      "source": UniformSource,         "flash": 10,   "desc": "Uniform [-1,1] + Flash10 (distribution control)"},
    {"name": "gaussian_always",    "source": GaussianSource,        "flash": 999,  "desc": "Gaussian always-on (Phase 70b replication)"},
]


# ===================================================
#  MAIN EXPERIMENT
# ===================================================

def run_phase70c(model, tok, device):
    print(f"\n{'='*80}")
    print(f"  Phase 70c: Noise Source Type Sweep")
    print(f"  k={RANK_K} fixed, σ={BASE_SIGMA}, Layer {LAYER_IDX}")
    print(f"  {len(CONDITIONS)} conditions × N={N_PER_CONDITION}")
    print(f"{'='*80}")

    # Generate shared orthogonal basis
    basis = generate_basis(RANK_K, HIDDEN_DIM, device)
    print(f"  Basis: {basis.shape} (k={RANK_K})")

    hook = SourcedRankKHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 70c: Noise Source Type Sweep",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "rank_k": RANK_K,
        "n_per_condition": N_PER_CONDITION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    t0 = time.time()

    for cond_idx, cond in enumerate(CONDITIONS):
        cond_name = cond["name"]
        flash_steps = cond["flash"]
        source_cls = cond["source"]

        print(f"\n  [{cond_idx+1}/{len(CONDITIONS)}] {cond_name}: {cond['desc']}")

        # Setup noise source
        if source_cls is not None:
            source = source_cls()
            hook.setup(basis, source)
        else:
            hook.setup(basis, None)

        games = []
        for trial in range(N_PER_CONDITION):
            # Reset source state for each game (quasi-periodic and chaos have state)
            if source_cls is not None:
                source = source_cls()
                hook.source = source

            env = HanoiEnv(n_disks=3, modified=True)
            stats = play_game(model, tok, env, hook, flash_steps=flash_steps)
            games.append(stats)

            if (trial + 1) % 10 == 0:
                solved_so_far = sum(1 for g in games if g["solved"])
                rate = solved_so_far / len(games) * 100
                print(f"    [{trial+1}/{N_PER_CONDITION}] Solve rate: {rate:.1f}%")

        # Condition summary
        solved = sum(1 for g in games if g["solved"])
        solve_rate = solved / len(games)
        avg_moves = np.mean([g["legal_moves"] for g in games])
        avg_illegal = np.mean([g["illegal_moves"] for g in games])
        avg_corrections = np.mean([g["self_corrections"] for g in games])

        summary = {
            "condition": cond_name,
            "description": cond["desc"],
            "flash_steps": flash_steps,
            "noise_source": source_cls.__name__ if source_cls else "none",
            "solve_rate": solve_rate,
            "n_solved": solved,
            "n_total": len(games),
            "avg_legal_moves": round(avg_moves, 2),
            "avg_illegal_moves": round(avg_illegal, 2),
            "avg_self_corrections": round(avg_corrections, 2),
            "games": games
        }

        all_results["conditions"].append(summary)

        elapsed = time.time() - t0
        print(f"    Solve rate: {solve_rate*100:.1f}% ({solved}/{len(games)})")
        print(f"    Avg moves: {avg_moves:.1f} legal, {avg_illegal:.1f} illegal, {avg_corrections:.1f} corrections")
        print(f"    Elapsed: {elapsed/60:.1f} min")

    hook.remove()

    # Fisher exact: each condition vs baseline
    baseline_solved = all_results["conditions"][0]["n_solved"]
    baseline_total = all_results["conditions"][0]["n_total"]
    print(f"\n  === Statistical Comparisons vs Baseline ===")
    for cond_result in all_results["conditions"][1:]:
        table = [
            [cond_result["n_solved"], cond_result["n_total"] - cond_result["n_solved"]],
            [baseline_solved, baseline_total - baseline_solved]
        ]
        _, p = fisher_exact(table)
        delta = cond_result["solve_rate"] - all_results["conditions"][0]["solve_rate"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {cond_result['condition']:25s}: {cond_result['solve_rate']*100:5.1f}% (d={delta*100:+5.1f}pp, p={p:.4f}) {sig}")

    # Cross-comparison: gaussian_flash vs all other flash sources
    gauss_flash = all_results["conditions"][1]  # gaussian_flash
    print(f"\n  === vs Gaussian Flash (champion) ===")
    for cond_result in all_results["conditions"][2:]:
        if cond_result["flash_steps"] == 10:  # Only compare flash conditions
            table = [
                [cond_result["n_solved"], cond_result["n_total"] - cond_result["n_solved"]],
                [gauss_flash["n_solved"], gauss_flash["n_total"] - gauss_flash["n_solved"]]
            ]
            _, p = fisher_exact(table)
            delta = cond_result["solve_rate"] - gauss_flash["solve_rate"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {cond_result['condition']:25s}: {cond_result['solve_rate']*100:5.1f}% (d={delta*100:+5.1f}pp vs gauss, p={p:.4f}) {sig}")

    # Save results
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)

    results_path = os.path.join(RESULTS_DIR, "phase70c_log.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  Total elapsed: {elapsed_total/60:.1f} min ({elapsed_total/3600:.1f} hours)")

    return all_results, elapsed_total


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model()

    results, elapsed = run_phase70c(model, tok, device)

    print(f"\n{'='*80}")
    print(f"  Phase 70c COMPLETE")
    print(f"  Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 70c complete.")

    print("  Phase 70c done.")
