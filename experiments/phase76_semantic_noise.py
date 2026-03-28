"""
Phase 76: Semantic Phase Noise Injection
=========================================

Instead of injecting noise at fixed steps (Flash Annealing) or fixed layers,
this experiment controls noise based on the SEMANTIC PHASE of reasoning:

  Phase A: PLANNING  — first 2 legal moves (the LLM is forming its strategy)
  Phase B: EXECUTING — moves 3+ (the LLM is carrying out its plan)
  Phase C: RECOVERY  — immediately after an illegal move (error correction)

For each phase, noise is either ON (σ=0.15, Layer 18) or OFF (σ=0).
This gives 2³ = 8 conditions (+ 1 baseline with noise always off,
+ 1 control with noise always on = 10 conditions total).

Key question: "Does noise help more during planning, execution, or recovery?"

If noise helps most during PLANNING, it suggests stochastic resonance aids
initial strategy formation. If during RECOVERY, it helps escape local minima.

Conditions (10, N=30 each):
  0: baseline       (always OFF)
  1: plan_only      (A=ON,  B=OFF, C=OFF)
  2: exec_only      (A=OFF, B=ON,  C=OFF)
  3: recover_only   (A=OFF, B=OFF, C=ON)
  4: plan+exec      (A=ON,  B=ON,  C=OFF)
  5: plan+recover   (A=ON,  B=OFF, C=ON)
  6: exec+recover   (A=OFF, B=ON,  C=ON)
  7: all_on         (A=ON,  B=ON,  C=ON)
  8: flash_10       (first 10 steps ON, then OFF — Phase 60 replication)
  9: anti_flash     (first 10 steps OFF, then ON — control)

Total: 300 games, ~5 hours

Usage:
    python experiments/phase76_semantic_noise.py
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
BASE_SIGMA = 0.15  # Phase 63 champion
N_PER_CONDITION = 30
LAYER_IDX = 18

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
#  NOISE HOOK (simple Layer 18 σ toggle)
# ===================================================

class NoiseHook:
    """Simple noise hook with adjustable sigma."""
    def __init__(self):
        self.active = False
        self.sigma = 0.0
        self.handle = None

    def register(self, model, layer_idx=LAYER_IDX):
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
#  SEMANTIC PHASE DEFINITIONS
# ===================================================

# Each condition defines noise ON/OFF for three semantic phases:
#   plan (first 2 legal moves), exec (move 3+), recover (after illegal)
SEMANTIC_CONDITIONS = [
    {"name": "baseline",       "plan": False, "exec": False, "recover": False},
    {"name": "plan_only",      "plan": True,  "exec": False, "recover": False},
    {"name": "exec_only",      "plan": False, "exec": True,  "recover": False},
    {"name": "recover_only",   "plan": False, "exec": False, "recover": True},
    {"name": "plan+exec",      "plan": True,  "exec": True,  "recover": False},
    {"name": "plan+recover",   "plan": True,  "exec": False, "recover": True},
    {"name": "exec+recover",   "plan": False, "exec": True,  "recover": True},
    {"name": "all_on",         "plan": True,  "exec": True,  "recover": True},
    {"name": "flash_10",       "plan": None,  "exec": None,  "recover": None},  # step-based
    {"name": "anti_flash",     "plan": None,  "exec": None,  "recover": None},  # step-based
]


def make_semantic_noise_fn(hook, condition):
    """Create a noise control function based on semantic phase."""
    plan_on  = condition.get("plan", False)
    exec_on  = condition.get("exec", False)
    recov_on = condition.get("recover", False)
    name     = condition["name"]

    def noise_fn(step, legal_move_count, is_recovery=False):
        if name == "flash_10":
            # Flash Annealing replication (first 10 steps)
            hook.active = (step < 10)
            hook.sigma = BASE_SIGMA if hook.active else 0.0
            return
        elif name == "anti_flash":
            # Anti-flash: noise AFTER step 10
            hook.active = (step >= 10)
            hook.sigma = BASE_SIGMA if hook.active else 0.0
            return

        # Semantic phase detection
        if is_recovery:
            hook.active = recov_on
        elif legal_move_count < 2:
            # Planning phase (first 2 legal moves)
            hook.active = plan_on
        else:
            # Execution phase (move 3+)
            hook.active = exec_on

        hook.sigma = BASE_SIGMA if hook.active else 0.0

    return noise_fn


# ===================================================
#  GAME FUNCTION (with semantic phase tracking)
# ===================================================

def play_game_semantic(model, tok, env, hook, noise_fn):
    """Play a Hanoi game with semantic phase noise control."""
    env.reset()
    error = None
    consec_fail = 0
    legal_move_count = 0
    phase_log = []  # Track which phases had noise

    for step in range(MAX_STEPS):
        is_recovery = (error is not None)
        noise_fn(step, legal_move_count, is_recovery)
        phase_log.append({
            "step": step,
            "legal_moves_so_far": legal_move_count,
            "is_recovery": is_recovery,
            "noise_active": hook.active,
            "sigma": hook.sigma
        })

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

    # Count noise-on steps per phase
    plan_noise_steps = sum(1 for p in phase_log if p["legal_moves_so_far"] < 2 and not p["is_recovery"] and p["noise_active"])
    exec_noise_steps = sum(1 for p in phase_log if p["legal_moves_so_far"] >= 2 and not p["is_recovery"] and p["noise_active"])
    recov_noise_steps = sum(1 for p in phase_log if p["is_recovery"] and p["noise_active"])
    total_noise_steps = sum(1 for p in phase_log if p["noise_active"])

    stats["plan_noise_steps"] = plan_noise_steps
    stats["exec_noise_steps"] = exec_noise_steps
    stats["recovery_noise_steps"] = recov_noise_steps
    stats["total_noise_steps"] = total_noise_steps

    return stats


# ===================================================
#  MAIN EXPERIMENT
# ===================================================

def run_phase76(model, tok, device):
    print(f"\n{'='*80}")
    print(f"  Phase 76: Semantic Phase Noise Injection")
    print(f"  {len(SEMANTIC_CONDITIONS)} conditions × N={N_PER_CONDITION}")
    print(f"{'='*80}")

    hook = NoiseHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 76: Semantic Phase Noise Injection",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "n_per_condition": N_PER_CONDITION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": []
    }

    t0 = time.time()

    for cond_idx, cond in enumerate(SEMANTIC_CONDITIONS):
        cond_name = cond["name"]
        print(f"\n  [{cond_idx+1}/{len(SEMANTIC_CONDITIONS)}] Condition: {cond_name}")
        if cond_name not in ("flash_10", "anti_flash"):
            print(f"    Plan={'ON' if cond['plan'] else 'OFF'}, "
                  f"Exec={'ON' if cond['exec'] else 'OFF'}, "
                  f"Recover={'ON' if cond['recover'] else 'OFF'}")

        noise_fn = make_semantic_noise_fn(hook, cond)
        games = []

        for trial in range(N_PER_CONDITION):
            env = HanoiEnv(n_disks=3, modified=True)
            stats = play_game_semantic(model, tok, env, hook, noise_fn)
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
        avg_noise_steps = np.mean([g["total_noise_steps"] for g in games])
        avg_plan_noise = np.mean([g["plan_noise_steps"] for g in games])
        avg_exec_noise = np.mean([g["exec_noise_steps"] for g in games])
        avg_recov_noise = np.mean([g["recovery_noise_steps"] for g in games])

        summary = {
            "condition": cond_name,
            "config": {k: v for k, v in cond.items() if k != "name"},
            "solve_rate": solve_rate,
            "n_solved": solved,
            "n_total": len(games),
            "avg_legal_moves": round(avg_moves, 2),
            "avg_illegal_moves": round(avg_illegal, 2),
            "avg_total_noise_steps": round(avg_noise_steps, 1),
            "avg_plan_noise_steps": round(avg_plan_noise, 1),
            "avg_exec_noise_steps": round(avg_exec_noise, 1),
            "avg_recovery_noise_steps": round(avg_recov_noise, 1),
            "games": games
        }

        all_results["conditions"].append(summary)

        elapsed = time.time() - t0
        print(f"    Solve rate: {solve_rate*100:.1f}% ({solved}/{len(games)})")
        print(f"    Avg moves: {avg_moves:.1f} legal, {avg_illegal:.1f} illegal")
        print(f"    Avg noise steps: {avg_noise_steps:.1f} (plan={avg_plan_noise:.1f}, exec={avg_exec_noise:.1f}, recover={avg_recov_noise:.1f})")
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
        print(f"    {cond_result['condition']:20s}: {cond_result['solve_rate']*100:5.1f}% (Δ={delta*100:+5.1f}pp, p={p:.4f}) {sig}")

    # Save results
    elapsed_total = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed_total, 1)

    results_path = os.path.join(RESULTS_DIR, "phase76_log.json")
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

    results, elapsed = run_phase76(model, tok, device)

    print(f"\n{'='*80}")
    print(f"  Phase 76 COMPLETE")
    print(f"  Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n Phase 76 complete.")

    print("  Phase 76 done.")
