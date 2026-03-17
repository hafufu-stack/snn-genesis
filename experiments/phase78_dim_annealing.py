"""
Phase 78: Dimensional Flash Annealing — Time x Space Fusion
=============================================================

Phase 63 showed Flash Annealing (σ decay over time) is optimal.
Phase 70b showed low-rank noise (k control) is efficient.
Phase 78 fuses BOTH: anneal the DIMENSIONALITY (k) over time.

Idea: Start with full-rank shotgun exploration, then narrow to
      laser-precise low-rank perturbation as reasoning converges.

Schedules (all with σ=0.15 per-dim, first-10 linear σ decay):
  1. baseline:      No noise
  2. fixed_full:    k=4096 throughout (Phase 70b replication)
  3. fixed_k16:     k=16 throughout
  4. fixed_k4:      k=4 throughout
  5. dim_anneal_A:  k=4096→64→4   (3-stage, exponential shrink)
  6. dim_anneal_B:  k=4096→4      (linear k shrink over 10 steps)
  7. dim_anneal_C:  k=4→4096      (REVERSE: narrow→wide)
  8. dual_anneal:   k=4096→4 AND σ=0.15→0 simultaneously

Total: 8 conditions × N=30 = 240 games, ~4 hours
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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


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
#  DYNAMIC RANK NOISE HOOK
# ===================================================

class DynRankHook:
    """Noise hook with dynamically changing rank during a game."""
    def __init__(self, device='cuda'):
        self.active = False
        self.sigma = BASE_SIGMA
        self.current_k = HIDDEN_DIM
        self.device = device
        self.handle = None
        # Pre-generate bases for various ranks
        self._bases = {}

    def precompute_bases(self, ranks):
        """Pre-compute orthogonal bases for all needed ranks."""
        for k in ranks:
            if k < HIDDEN_DIM:
                torch.manual_seed(42)
                A = torch.randn(HIDDEN_DIM, k, dtype=torch.float32, device=self.device)
                Q, _ = torch.linalg.qr(A)
                self._bases[k] = Q.to(torch.float16)  # (d, k)
            else:
                self._bases[k] = None  # full rank

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            hs = args[0]
            k = hook_obj.current_k
            basis = hook_obj._bases.get(k)
            if basis is None:
                # Full rank
                noise = torch.randn_like(hs) * hook_obj.sigma
            else:
                b, s, d = hs.shape
                z = torch.randn(b, s, k, dtype=hs.dtype, device=hs.device) * hook_obj.sigma
                noise = z @ basis.T  # (b,s,k) @ (k,d)
            return (hs + noise,) + args[1:]
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  SCHEDULE DEFINITIONS
# ===================================================

def get_schedule(name):
    """Returns a function: f(legal_move_count) -> (active, sigma, rank_k)"""

    if name == "baseline":
        def fn(lm):
            return False, 0.0, HIDDEN_DIM
        return fn

    elif name == "fixed_full":
        def fn(lm):
            if lm < 10:
                sigma = BASE_SIGMA * (1.0 - lm / 10.0)
                return True, sigma, 4096
            return False, 0.0, 4096
        return fn

    elif name == "fixed_k16":
        def fn(lm):
            if lm < 10:
                sigma = BASE_SIGMA * (1.0 - lm / 10.0)
                return True, sigma, 16
            return False, 0.0, 16
        return fn

    elif name == "fixed_k4":
        def fn(lm):
            if lm < 10:
                sigma = BASE_SIGMA * (1.0 - lm / 10.0)
                return True, sigma, 4
            return False, 0.0, 4
        return fn

    elif name == "dim_anneal_A":
        # 3-stage: k=4096 (moves 0-3), k=64 (moves 4-6), k=4 (moves 7-9)
        def fn(lm):
            if lm >= 10:
                return False, 0.0, 4
            sigma = BASE_SIGMA * (1.0 - lm / 10.0)
            if lm < 4:
                return True, sigma, 4096
            elif lm < 7:
                return True, sigma, 64
            else:
                return True, sigma, 4
        return fn

    elif name == "dim_anneal_B":
        # Linear k shrink: k = 4096 * (1 - lm/10), clamped to {4,16,64,256,1024,4096}
        rank_ladder = [4, 16, 64, 256, 1024, 4096]
        def fn(lm):
            if lm >= 10:
                return False, 0.0, 4
            sigma = BASE_SIGMA * (1.0 - lm / 10.0)
            frac = 1.0 - lm / 10.0  # 1.0 -> 0.0
            target_k = max(4, int(4096 * frac))
            # Snap to nearest rank in ladder
            k = min(rank_ladder, key=lambda x: abs(x - target_k))
            return True, sigma, k
        return fn

    elif name == "dim_anneal_C":
        # REVERSE: k=4 (moves 0-3), k=64 (moves 4-6), k=4096 (moves 7-9)
        def fn(lm):
            if lm >= 10:
                return False, 0.0, 4096
            sigma = BASE_SIGMA * (1.0 - lm / 10.0)
            if lm < 4:
                return True, sigma, 4
            elif lm < 7:
                return True, sigma, 64
            else:
                return True, sigma, 4096
        return fn

    elif name == "dual_anneal":
        # Both σ AND k decay: σ=0.15→0 AND k=4096→4 simultaneously
        rank_ladder = [4, 16, 64, 256, 1024, 4096]
        def fn(lm):
            if lm >= 10:
                return False, 0.0, 4
            sigma = BASE_SIGMA * (1.0 - lm / 10.0)
            frac = 1.0 - lm / 10.0
            target_k = max(4, int(4096 * frac))
            k = min(rank_ladder, key=lambda x: abs(x - target_k))
            return True, sigma, k
        return fn

    else:
        raise ValueError(f"Unknown schedule: {name}")


# ===================================================
#  GAME FUNCTION
# ===================================================

def play_game_dynamic(model, tok, env, hook, schedule_fn):
    env.reset()
    error = None
    consec_fail = 0
    legal_move_count = 0
    k_log = []

    for step in range(MAX_STEPS):
        active, sigma, k = schedule_fn(legal_move_count)
        hook.active = active
        hook.sigma = sigma
        hook.current_k = k
        k_log.append({"step": step, "lm": legal_move_count, "k": k, "sigma": round(sigma, 4), "active": active})

        prompt = build_chat_prompt(tok, env, error)
        resp = generate(model, tok, prompt)
        move = parse_move(resp)

        if move is None:
            env.illegal_count += 1; env.total_attempts += 1; env._prev_illegal = True
            error = "Parse fail. Use Move: X->Y"
            consec_fail += 1
            if consec_fail >= 10: break
            continue

        ok, msg = env.try_move(move[0], move[1])
        if ok:
            legal_move_count += 1
            error = None; consec_fail = 0
            if env.is_solved(): break
        else:
            error = msg
            consec_fail += 1
            if consec_fail >= 10: break

    hook.active = False
    stats = env.stats()
    stats["steps_taken"] = step + 1
    return stats


# ===================================================
#  MAIN EXPERIMENT
# ===================================================

SCHEDULES = ["baseline", "fixed_full", "fixed_k16", "fixed_k4",
             "dim_anneal_A", "dim_anneal_B", "dim_anneal_C", "dual_anneal"]

def run_phase78(model, tok, device):
    print(f"\n{'='*80}")
    print(f"  Phase 78: Dimensional Flash Annealing")
    print(f"  {len(SCHEDULES)} schedules x N={N_PER_CONDITION}")
    print(f"{'='*80}")

    t0 = time.time()

    # Pre-compute all bases
    needed_ranks = [4, 16, 64, 256, 1024, 4096]
    hook = DynRankHook(device=device)
    hook.precompute_bases(needed_ranks)
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 78: Dimensional Flash Annealing",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "n_per_condition": N_PER_CONDITION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    for sched_idx, sched_name in enumerate(SCHEDULES):
        print(f"\n  [{sched_idx+1}/{len(SCHEDULES)}] Schedule: {sched_name}")
        schedule_fn = get_schedule(sched_name)
        games = []

        for trial in range(N_PER_CONDITION):
            env = HanoiEnv(n_disks=3, modified=True)
            stats = play_game_dynamic(model, tok, env, hook, schedule_fn)
            games.append(stats)

            if (trial + 1) % 10 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                elapsed = time.time() - t0
                print(f"    [{trial+1}/{N_PER_CONDITION}] Solve rate: {sr:.1f}% | {elapsed/60:.1f}min")

        solved = sum(1 for g in games if g["solved"])
        summary = {
            "schedule": sched_name,
            "solve_rate": solved / len(games),
            "n_solved": solved,
            "n_total": len(games),
            "avg_legal_moves": round(np.mean([g["legal_moves"] for g in games]), 2),
            "avg_illegal_moves": round(np.mean([g["illegal_moves"] for g in games]), 2),
            "games": games
        }
        all_results["conditions"].append(summary)
        print(f"    Solve rate: {solved/len(games)*100:.1f}% ({solved}/{len(games)})")

    hook.remove()

    # Statistical analysis
    bl = all_results["conditions"][0]
    print(f"\n  === Fisher Exact Tests vs Baseline ===")
    for cond in all_results["conditions"][1:]:
        table = [[cond["n_solved"], cond["n_total"]-cond["n_solved"]],
                 [bl["n_solved"], bl["n_total"]-bl["n_solved"]]]
        _, p = fisher_exact(table)
        delta = cond["solve_rate"] - bl["solve_rate"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    {cond['schedule']:20s}: {cond['solve_rate']*100:5.1f}% "
              f"(delta={delta*100:+5.1f}pp, p={p:.4f}) {sig}")

    # Key comparison: dim_anneal vs fixed
    print(f"\n  === Dimensional Annealing vs Fixed Rank ===")
    fixed_full = next(c for c in all_results["conditions"] if c["schedule"] == "fixed_full")
    for an in ["dim_anneal_A", "dim_anneal_B", "dim_anneal_C", "dual_anneal"]:
        cond = next(c for c in all_results["conditions"] if c["schedule"] == an)
        table = [[cond["n_solved"], cond["n_total"]-cond["n_solved"]],
                 [fixed_full["n_solved"], fixed_full["n_total"]-fixed_full["n_solved"]]]
        _, p = fisher_exact(table)
        delta = cond["solve_rate"] - fixed_full["solve_rate"]
        print(f"    {an:20s}: {cond['solve_rate']*100:.1f}% vs fixed_full {fixed_full['solve_rate']*100:.1f}% "
              f"(delta={delta*100:+.1f}pp, p={p:.4f})")

    # Save
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)

    results_path = os.path.join(RESULTS_DIR, "phase78_log.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  Total elapsed: {elapsed_total/60:.1f} min ({elapsed_total/3600:.1f} hours)")

    return all_results, elapsed_total


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    model, tok = load_model()
    device = next(model.parameters()).device
    results, elapsed = run_phase78(model, tok, device)

    print(f"\n{'='*80}")
    print(f"  Phase 78 COMPLETE: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 78 complete.")

    if should_hibernate(78):
        print("  Hibernating...")
    else:
        import winsound, subprocess
        for _ in range(3):
            time.sleep(0.2)
        print("  Phase 78 done. Chaining to Phase 79...")
        phase79_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase79_orthogonal_noise.py")
        if os.path.exists(phase79_script):
            subprocess.Popen(["python", phase79_script],
                           cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            print(f"  Started: {phase79_script}")
        else:
            print(f"  WARNING: Phase 79 script not found: {phase79_script}")

