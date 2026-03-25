"""
Phase 96: Cross-Task Meta-Aha! Vector — Task Generalization of Reasoning Direction
===================================================================================

Phase 87/91 discovered discriminant axes for specific tasks (Tower of Hanoi).
This experiment tests whether the Aha! vector is task-universal or task-specific
by extracting differential vectors from MULTIPLE reasoning tasks and measuring
their cosine similarity.

Tasks:
  1. Tower of Hanoi (Modified — existing)
  2. Arithmetic Chain (multi-step calculation)
  3. Logical Deduction (variable ordering)

Protocol:
  1. Play 100 games per task on Mistral, collect labeled hidden states
  2. Compute Differential PCA → Aha! vector per task
  3. Measure cosine similarity and Top-64 PC overlap between task vectors
  4. Cross-task steering test: Hanoi vector → Arithmetic, Arithmetic vector → Hanoi

Mistral-7B-Instruct-v0.3, Layer 18, N=30 per behavioral condition

Total: 300 collection + 4x30=120 steering test = 420 forward passes
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
N_COLLECTION_GAMES = 100
LAYER_IDX = 18

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================
#  TASK 1: TOWER OF HANOI (Modified)
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
#  TASK 2: ARITHMETIC CHAIN
# ===================================================

class ArithmeticTask:
    """Multi-step arithmetic chain: (a op b) op c = ?"""
    def __init__(self):
        self.reset()

    def reset(self):
        ops = ['+', '-', '*']
        self.a = random.randint(2, 15)
        self.b = random.randint(2, 10)
        self.c = random.randint(2, 8)
        self.op1 = random.choice(ops)
        self.op2 = random.choice(ops)
        intermediate = eval(f"{self.a} {self.op1} {self.b}")
        self.answer = eval(f"{intermediate} {self.op2} {self.c}")
        self.expression = f"({self.a} {self.op1} {self.b}) {self.op2} {self.c}"

    def check(self, response):
        # Extract number from response
        numbers = re.findall(r'-?\d+', response)
        if numbers:
            return int(numbers[-1]) == self.answer
        return False


# ===================================================
#  TASK 3: LOGICAL DEDUCTION
# ===================================================

class LogicalDeductionTask:
    """3-variable ordering: Given A>B, B>C, who is smallest?"""
    def __init__(self):
        self.reset()

    def reset(self):
        names = random.sample(["Alice", "Bob", "Carol", "Dave", "Eve"], 3)
        self.order = names[:]  # names[0] > names[1] > names[2]
        self.clues = []
        # Generate 2 comparison clues
        self.clues.append(f"{names[0]} is taller than {names[1]}")
        self.clues.append(f"{names[1]} is taller than {names[2]}")
        # Randomly ask about different positions
        q_type = random.choice(["shortest", "tallest", "middle"])
        if q_type == "shortest":
            self.question = "Who is the shortest?"
            self.answer = names[2]
        elif q_type == "tallest":
            self.question = "Who is the tallest?"
            self.answer = names[0]
        else:
            self.question = "Who is in the middle?"
            self.answer = names[1]

    def check(self, response):
        return self.answer.lower() in response.lower()


# ===================================================
#  PROMPT BUILDERS
# ===================================================

def build_hanoi_system_msg(env):
    rules = "MODIFIED RULES: You can ONLY place a LARGER disk onto a SMALLER disk. The opposite of standard."
    return (
        f"You are solving Tower of Hanoi with {env.n_disks} disks. "
        f"{rules} "
        f"Goal: move ALL disks from A to C. "
        f"Respond with EXACTLY one move in format: Move: X->Y (e.g. Move: A->C). "
        f"You may add a brief Think: line before it."
    )

def build_hanoi_user_msg(env, error=None):
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

def build_arithmetic_prompt(task):
    return (
        f"Solve this step by step: {task.expression}\n"
        f"Show your work, then give the final answer as a single number.\n"
        f"Answer:"
    )

def build_logic_prompt(task):
    clues_str = ". ".join(task.clues) + "."
    return (
        f"Given the following facts: {clues_str}\n"
        f"{task.question}\n"
        f"Think step by step, then give the name.\n"
        f"Answer:"
    )

def build_chat_prompt(tokenizer, content):
    messages = [{"role": "user", "content": content}]
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

def generate(model, tok, prompt, temperature=0.5, max_tokens=120):
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
        self.all_items = []  # [(states_array, success_bool)]
        self.recording = False
        self.handle = None

    def register(self, model, layer_idx=LAYER_IDX):
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

    def start_item(self):
        self.current_states = []
        self.recording = True

    def end_item(self, success):
        self.recording = False
        if self.current_states:
            states = np.array(self.current_states)
            self.all_items.append((states, success))
        self.current_states = []

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


def compute_diff_vector(labeled_items):
    """Compute discriminant direction from labeled hidden states."""
    solved_means = []
    failed_means = []
    for states, success in labeled_items:
        game_mean = states.mean(axis=0)
        if success:
            solved_means.append(game_mean)
        else:
            failed_means.append(game_mean)

    n_s = len(solved_means)
    n_f = len(failed_means)
    print(f"    Success: {n_s}, Failed: {n_f}")

    if n_s == 0 or n_f == 0:
        print(f"    WARNING: Single class only, using random direction")
        return np.random.randn(HIDDEN_DIM).astype(np.float32), 0.0

    solved_mean = np.mean(solved_means, axis=0)
    failed_mean = np.mean(failed_means, axis=0)
    diff = solved_mean - failed_mean
    norm = np.linalg.norm(diff)
    diff_unit = diff / (norm + 1e-8)
    return diff_unit, norm


# ===================================================
#  TASK-SPECIFIC GAME RUNNERS (for collection)
# ===================================================

def collect_hanoi_states(model, tok, collector, n_games):
    print(f"\n  Collecting Hanoi hidden states ({n_games} games)...")
    for gi in range(n_games):
        env = HanoiEnv(n_disks=3, modified=True)
        error = None; consec_fail = 0
        collector.start_item()
        for step in range(MAX_STEPS):
            content = build_hanoi_system_msg(env) + "\n\n" + build_hanoi_user_msg(env, error)
            prompt = build_chat_prompt(tok, content)
            resp = generate(model, tok, prompt, max_tokens=80)
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
        collector.end_item(env.is_solved())
        if (gi + 1) % 20 == 0:
            ns = sum(1 for _, s in collector.all_items if s)
            print(f"    [{gi+1}/{n_games}] Solved so far: {ns}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def collect_arithmetic_states(model, tok, collector, n_items):
    print(f"\n  Collecting Arithmetic hidden states ({n_items} problems)...")
    for gi in range(n_items):
        task = ArithmeticTask()
        collector.start_item()
        prompt = build_chat_prompt(tok, build_arithmetic_prompt(task))
        resp = generate(model, tok, prompt, max_tokens=120)
        success = task.check(resp)
        collector.end_item(success)
        if (gi + 1) % 20 == 0:
            ns = sum(1 for _, s in collector.all_items if s)
            print(f"    [{gi+1}/{n_items}] Correct so far: {ns}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def collect_logic_states(model, tok, collector, n_items):
    print(f"\n  Collecting Logic hidden states ({n_items} problems)...")
    for gi in range(n_items):
        task = LogicalDeductionTask()
        collector.start_item()
        prompt = build_chat_prompt(tok, build_logic_prompt(task))
        resp = generate(model, tok, prompt, max_tokens=120)
        success = task.check(resp)
        collector.end_item(success)
        if (gi + 1) % 20 == 0:
            ns = sum(1 for _, s in collector.all_items if s)
            print(f"    [{gi+1}/{n_items}] Correct so far: {ns}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()


# ===================================================
#  AHA! STEERING HOOK (for cross-task test)
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
        """Combined: deterministic Aha! offset + full-space stochastic noise."""
        self.mode = "aha_plus_noise"
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

            if hook_obj.mode == "aha_plus_noise":
                d = hs.shape[-1]
                # Deterministic offset
                offset = hook_obj.fixed_offset
                det_scale = hook_obj.sigma * math.sqrt(d) * 0.5
                det_noise = offset * det_scale
                if hs.dim() == 3:
                    det_noise = det_noise.unsqueeze(0).unsqueeze(0).expand_as(hs)
                else:
                    det_noise = det_noise.unsqueeze(0).expand_as(hs)
                # Stochastic noise
                stoch_noise = torch.randn_like(hs) * (hook_obj.sigma * 0.5)
                return (hs + det_noise + stoch_noise,) + args[1:]

            return args
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  HANOI GAME WITH HOOK (for steering test)
# ===================================================

def play_hanoi_with_hook(model, tok, hook):
    env = HanoiEnv(n_disks=3, modified=True)
    error = None; consec_fail = 0; legal_move_count = 0
    for step in range(MAX_STEPS):
        # Flash annealing
        if legal_move_count < 10:
            hook.active = True
            hook.sigma = BASE_SIGMA * (1.0 - legal_move_count / 10.0)
        else:
            hook.active = False; hook.sigma = 0.0

        content = build_hanoi_system_msg(env) + "\n\n" + build_hanoi_user_msg(env, error)
        prompt = build_chat_prompt(tok, content)
        resp = generate(model, tok, prompt, max_tokens=80)
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
    fig.suptitle("Phase 96: Cross-Task Meta-Aha! Vector — Is Reasoning Direction Universal?",
                 fontsize=13, fontweight="bold")

    # Panel 1: Task baselines
    ax = axes[0]
    task_names = [t["task"] for t in all_results["task_baselines"]]
    solve_rates = [t["success_rate"] * 100 for t in all_results["task_baselines"]]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(range(len(task_names)), solve_rates, color=colors, alpha=0.85)
    for bar, val in zip(bars, solve_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels(task_names, fontsize=9)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Task Baselines (N=100)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2: Cosine similarity heatmap
    ax = axes[1]
    cos_matrix = np.array(all_results["cosine_similarity_matrix"])
    im = ax.imshow(cos_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    for i in range(cos_matrix.shape[0]):
        for j in range(cos_matrix.shape[1]):
            ax.text(j, i, f"{cos_matrix[i,j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(task_names)))
    ax.set_yticks(range(len(task_names)))
    ax.set_xticklabels(task_names, fontsize=9)
    ax.set_yticklabels(task_names, fontsize=9)
    ax.set_title("Aha! Vector Cosine Similarity", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 3: Cross-task steering results
    ax = axes[2]
    if "steering_test" in all_results:
        steer = all_results["steering_test"]
        names_s = [c["condition"] for c in steer]
        rates_s = [c["solve_rate"] * 100 for c in steer]
        colors_s = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF9800"]
        bars = ax.bar(range(len(steer)), rates_s, color=colors_s[:len(steer)], alpha=0.85)
        for bar, val in zip(bars, rates_s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(steer)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names_s], fontsize=8)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("Cross-Task Steering (Hanoi)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase96_cross_task_meta_aha.png")
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

    model, tok = load_model()
    device = next(model.parameters()).device

    print(f"\n{'='*80}")
    print(f"  Phase 96: Cross-Task Meta-Aha! Vector")
    print(f"  3 tasks x {N_COLLECTION_GAMES} games → Differential PCA → cosine similarity")
    print(f"{'='*80}")

    t0 = time.time()
    task_names = ["Hanoi", "Arithmetic", "Logic"]
    diff_units = {}
    task_baselines = []

    # === Step 1: Collect labeled hidden states for each task ===
    for task_name, collect_fn in [
        ("Hanoi", lambda col: collect_hanoi_states(model, tok, col, N_COLLECTION_GAMES)),
        ("Arithmetic", lambda col: collect_arithmetic_states(model, tok, col, N_COLLECTION_GAMES)),
        ("Logic", lambda col: collect_logic_states(model, tok, col, N_COLLECTION_GAMES)),
    ]:
        print(f"\n  === Task: {task_name} ===")
        collector = LabeledCollectorHook()
        collector.register(model, LAYER_IDX)
        collect_fn(collector)
        collector.remove()

        n_success = sum(1 for _, s in collector.all_items if s)
        n_total = len(collector.all_items)
        success_rate = n_success / n_total if n_total > 0 else 0
        print(f"  {task_name}: {n_success}/{n_total} = {success_rate*100:.1f}%")

        task_baselines.append({
            "task": task_name,
            "n_success": n_success,
            "n_total": n_total,
            "success_rate": round(success_rate, 4),
        })

        # Compute differential vector
        diff_unit, diff_norm = compute_diff_vector(collector.all_items)
        diff_units[task_name] = diff_unit
        print(f"  Diff vector norm: {diff_norm:.4f}")
        del collector
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Step 2: Cosine similarity matrix ===
    print(f"\n  === Cosine Similarity Matrix ===")
    cos_matrix = np.zeros((3, 3))
    for i, t1 in enumerate(task_names):
        for j, t2 in enumerate(task_names):
            cos_sim = float(np.dot(diff_units[t1], diff_units[t2]) /
                          (np.linalg.norm(diff_units[t1]) * np.linalg.norm(diff_units[t2]) + 1e-8))
            cos_matrix[i, j] = cos_sim
    print(f"  Cosine similarity matrix:")
    for i, t1 in enumerate(task_names):
        row = [f"{cos_matrix[i,j]:.4f}" for j in range(3)]
        print(f"    {t1:12s}: {', '.join(row)}")

    # Top-64 PC overlap analysis
    pca_path = os.path.join(RESULTS_DIR, "phase86_mistral_pca.npz")
    top64_overlaps = {}
    if os.path.exists(pca_path):
        pca_data = np.load(pca_path)
        Vt_std = pca_data["Vt"]
        print(f"\n  === Top-64 PC Overlap per Task ===")
        for task_name in task_names:
            du = diff_units[task_name]
            cos_sims = np.abs(Vt_std @ du)
            t64_overlap = float(np.sum(cos_sims[:64]**2))
            mid_overlap = float(np.sum(cos_sims[64:256]**2))
            bot_overlap = float(np.sum(cos_sims[256:]**2))
            top64_overlaps[task_name] = {
                "top64": round(t64_overlap, 4),
                "mid_band": round(mid_overlap, 4),
                "pc257_plus": round(bot_overlap, 4),
            }
            print(f"    {task_name}: top-64={t64_overlap*100:.1f}%, "
                  f"mid={mid_overlap*100:.1f}%, pc257+={bot_overlap*100:.1f}%")

    # === Step 3: Cross-task steering test on Hanoi ===
    print(f"\n  === Cross-Task Steering Test (on Hanoi) ===")
    steering_conditions = [
        {"name": "baseline", "vector": None},
        {"name": "hanoi_aha+noise", "vector": diff_units["Hanoi"]},
        {"name": "arith_aha+noise", "vector": diff_units["Arithmetic"]},
        {"name": "logic_aha+noise", "vector": diff_units["Logic"]},
    ]

    hook = AhaSteeringHook()
    hook.register(model, LAYER_IDX)
    steering_results = []

    for cond in steering_conditions:
        print(f"\n  Condition: {cond['name']}")
        if cond["vector"] is None:
            hook.setup_baseline()
        else:
            hook.setup_aha_plus_noise(cond["vector"], device)

        games = []
        for trial in range(N_PER_CONDITION):
            stats = play_hanoi_with_hook(model, tok, hook)
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
        })
        print(f"    Final: {solved}/{len(games)} = {solved/len(games)*100:.1f}%")

    hook.remove()

    # === Compile Results ===
    elapsed = time.time() - t0
    all_results = {
        "experiment": "Phase 96: Cross-Task Meta-Aha! Vector",
        "model": MODEL_SHORT,
        "model_full": MODEL_NAME,
        "layer": LAYER_IDX,
        "hidden_dim": HIDDEN_DIM,
        "sigma": BASE_SIGMA,
        "n_collection_games": N_COLLECTION_GAMES,
        "n_per_steering": N_PER_CONDITION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_baselines": task_baselines,
        "cosine_similarity_matrix": cos_matrix.tolist(),
        "cosine_similarity_labels": task_names,
        "top64_overlaps": top64_overlaps,
        "steering_test": steering_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    # Interpretation
    hanoi_arith = abs(cos_matrix[0, 1])
    hanoi_logic = abs(cos_matrix[0, 2])
    arith_logic = abs(cos_matrix[1, 2])
    avg_cross = (hanoi_arith + hanoi_logic + arith_logic) / 3

    print(f"\n  === Interpretation ===")
    if avg_cross > 0.5:
        verdict = "UNIVERSAL_META_REASONING"
        print(f"  VERDICT: {verdict} — avg cross-task cosine = {avg_cross:.3f} > 0.5")
        print(f"  A universal meta-reasoning axis exists!")
    elif avg_cross > 0.2:
        verdict = "PARTIALLY_SHARED"
        print(f"  VERDICT: {verdict} — avg cross-task cosine = {avg_cross:.3f}")
        print(f"  Partial overlap: reasoning shares some directions across tasks")
    else:
        verdict = "TASK_SPECIFIC"
        print(f"  VERDICT: {verdict} — avg cross-task cosine = {avg_cross:.3f} < 0.2")
        print(f"  Reasoning directions are task-specific")

    all_results["verdict"] = verdict
    all_results["avg_cross_task_cosine"] = round(avg_cross, 4)

    # Visualization
    fig_path = visualize(all_results)
    all_results["figure"] = fig_path

    # Save
    results_path = os.path.join(RESULTS_DIR, "phase96_log.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  Total elapsed: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")

    # Save diff vectors for reuse
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "phase96_diff_vectors.npz"),
        hanoi=diff_units["Hanoi"],
        arithmetic=diff_units["Arithmetic"],
        logic=diff_units["Logic"],
    )

    return all_results, elapsed


if __name__ == "__main__":
    main()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n Phase 96 complete.")
