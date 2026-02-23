"""
Phase 50: The Illusion Breaker — Tower of Hanoi
================================================

Apple "The Illusion of Thinking" (2025) argues that LLMs
  (1) Fail when rules are modified (mimicry collapse)
  (2) Cannot self-correct errors
  (3) Give up (shorten output) on hard problems

This experiment tests whether SNN perturbation at L18
can break the mimicry pattern and enable genuine reasoning
on standard and rule-modified Tower of Hanoi puzzles.

Conditions:
  1. Baseline: No SNN injection
  2. L18-Echo: Continuous σ=0.02 at L18 (Physical Prompting)
  3. Conflict-Triggered: σ=0.05 spike at L18 only when illegal move detected

Tasks:
  - Standard Hanoi (3 disks, 4 disks)
  - Modified Hanoi (3 disks, 4 disks) — inverted rule: can only move larger onto smaller

Usage:
    python experiments/phase50_illusion_breaker.py
"""

import torch
import torch.nn as nn
import os, json, gc, time, random, re, copy
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ═══ Config ═══
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
SIGMA_NORMAL = 0.05
SIGMA_ECHO = 0.02
SIGMA_SPIKE = 0.05
TARGET_LAYER = 18
ECHO_TOKENS = 2
MAX_NEW_TOKENS = 200
SEED = 2026
N_TRIALS = 10

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
#  TOWER OF HANOI ENVIRONMENT
# ═══════════════════════════════════════════════════

class HanoiEnvironment:
    """Tower of Hanoi game engine with standard and modified rules."""

    def __init__(self, n_disks=3, modified_rules=False):
        self.n_disks = n_disks
        self.modified_rules = modified_rules  # If True: can only place LARGER on SMALLER
        self.pegs = {"A": list(range(n_disks, 0, -1)), "B": [], "C": []}
        self.move_history = []
        self.illegal_moves = 0
        self.total_attempts = 0
        self.self_corrections = 0  # Correct move immediately after illegal

    def reset(self):
        self.pegs = {"A": list(range(self.n_disks, 0, -1)), "B": [], "C": []}
        self.move_history = []
        self.illegal_moves = 0
        self.total_attempts = 0
        self.self_corrections = 0

    def is_solved(self):
        return len(self.pegs["C"]) == self.n_disks

    def is_legal_move(self, from_peg, to_peg):
        if from_peg not in self.pegs or to_peg not in self.pegs:
            return False
        if not self.pegs[from_peg]:
            return False  # No disk on source peg
        disk = self.pegs[from_peg][-1]
        if not self.pegs[to_peg]:
            return True  # Empty target is always legal

        top_target = self.pegs[to_peg][-1]
        if self.modified_rules:
            # INVERTED: can only place LARGER disk onto SMALLER
            return disk > top_target
        else:
            # STANDARD: can only place SMALLER disk onto LARGER
            return disk < top_target

    def make_move(self, from_peg, to_peg):
        """Attempt a move. Returns (success, error_msg)."""
        self.total_attempts += 1
        from_peg = from_peg.upper().strip()
        to_peg = to_peg.upper().strip()

        if from_peg not in self.pegs or to_peg not in self.pegs:
            self.illegal_moves += 1
            return False, f"Invalid peg name: {from_peg} or {to_peg}"
        if from_peg == to_peg:
            self.illegal_moves += 1
            return False, f"Cannot move to same peg"
        if not self.pegs[from_peg]:
            self.illegal_moves += 1
            return False, f"Peg {from_peg} is empty"
        if not self.is_legal_move(from_peg, to_peg):
            self.illegal_moves += 1
            if self.modified_rules:
                return False, f"ILLEGAL: In modified rules, you can only place a LARGER disk onto a SMALLER one"
            else:
                return False, f"ILLEGAL: Cannot place larger disk on smaller"

        # Check if this is a self-correction (legal move right after illegal)
        if self.illegal_moves > 0 and len(self.move_history) > 0:
            last = self.move_history[-1]
            if last.get("illegal", False):
                self.self_corrections += 1

        disk = self.pegs[from_peg].pop()
        self.pegs[to_peg].append(disk)
        self.move_history.append({
            "from": from_peg, "to": to_peg, "disk": disk,
            "illegal": False, "step": len(self.move_history) + 1
        })
        return True, f"Moved disk {disk} from {from_peg} to {to_peg}"

    def record_illegal(self, from_peg, to_peg, reason):
        self.move_history.append({
            "from": from_peg, "to": to_peg, "disk": None,
            "illegal": True, "reason": reason,
            "step": len(self.move_history) + 1
        })

    def render_ascii(self):
        """Render the current state as a text representation for LLM."""
        lines = []
        max_h = self.n_disks + 1
        for peg_name in ["A", "B", "C"]:
            disks = self.pegs[peg_name]
            disk_str = str(disks) if disks else "[]"
            lines.append(f"  Peg {peg_name}: {disk_str}  (top → right)")
        return "\n".join(lines)

    def optimal_moves(self):
        """Calculate optimal number of moves."""
        if self.modified_rules:
            # Modified rules (inverted): more complex, approximately 3^n - 1
            return 3**self.n_disks - 1
        else:
            # Standard: 2^n - 1
            return 2**self.n_disks - 1

    def get_state_summary(self):
        return {
            "solved": self.is_solved(),
            "total_attempts": self.total_attempts,
            "legal_moves": self.total_attempts - self.illegal_moves,
            "illegal_moves": self.illegal_moves,
            "self_corrections": self.self_corrections,
            "optimal": self.optimal_moves(),
            "move_count": len([m for m in self.move_history if not m.get("illegal")]),
        }


# ═══════════════════════════════════════════════════
#  LLM PLAYER
# ═══════════════════════════════════════════════════

def build_hanoi_prompt(env, error_feedback=None):
    """Build the prompt for the LLM to play Tower of Hanoi."""

    if env.modified_rules:
        rule_desc = (
            "MODIFIED RULES (IMPORTANT — read carefully!):\n"
            "- You can ONLY place a LARGER disk onto a SMALLER disk.\n"
            "- This is the OPPOSITE of the standard Tower of Hanoi rules.\n"
            "- You CANNOT place a smaller disk on top of a larger disk.\n"
            "- Goal: Move ALL disks from peg A to peg C."
        )
    else:
        rule_desc = (
            "STANDARD RULES:\n"
            "- You can ONLY place a SMALLER disk onto a LARGER disk.\n"
            "- You cannot place a larger disk on top of a smaller disk.\n"
            "- Goal: Move ALL disks from peg A to peg C."
        )

    prompt = f"""You are solving a Tower of Hanoi puzzle with {env.n_disks} disks.
Disks are numbered 1 (smallest) to {env.n_disks} (largest).
There are 3 pegs: A, B, C.

{rule_desc}

Current state:
{env.render_ascii()}
"""

    # Add move history (last 5 moves)
    legal_moves = [m for m in env.move_history if not m.get("illegal")]
    if legal_moves:
        recent = legal_moves[-5:]
        prompt += "\nRecent moves:\n"
        for m in recent:
            prompt += f"  Step {m['step']}: Moved disk {m['disk']} from {m['from']} to {m['to']}\n"

    if error_feedback:
        prompt += f"\n⚠️ ERROR: {error_feedback}\nPlease choose a LEGAL move this time.\n"

    prompt += (
        "\nWhat is your next move? Respond with EXACTLY this format:\n"
        "Move disk [N] from [X] to [Y]\n"
        "where N is the disk number, X and Y are peg letters (A, B, or C).\n"
        "Give ONLY the move, nothing else."
    )
    return prompt


def parse_move(response):
    """Parse LLM response to extract move. Returns (from_peg, to_peg) or None."""
    response = response.strip().upper()

    # Pattern: "Move disk N from X to Y"
    m = re.search(r'MOVE\s+DISK\s+\d+\s+FROM\s+([ABC])\s+TO\s+([ABC])', response)
    if m:
        return m.group(1), m.group(2)

    # Fallback: "from A to C" or "A to C" or "A -> C"
    m = re.search(r'FROM\s+([ABC])\s+TO\s+([ABC])', response)
    if m:
        return m.group(1), m.group(2)

    m = re.search(r'([ABC])\s*(?:->|→|TO)\s*([ABC])', response)
    if m:
        return m.group(1), m.group(2)

    return None


# ═══════════════════════════════════════════════════
#  SNN HOOKS
# ═══════════════════════════════════════════════════

class L18EchoHook:
    """Continuous low-dose noise at L18 (Physical Prompting)."""
    def __init__(self, sigma=0.02):
        self.sigma = sigma
        self.active = True

    def __call__(self, module, args):
        if not self.active:
            return args
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype) * self.sigma
        return (hs + noise,) + args[1:]


class ConflictTriggeredHook:
    """Spike injection at L18 — only triggered when illegal move detected."""
    def __init__(self, sigma_spike=0.05):
        self.sigma_spike = sigma_spike
        self.triggered = False
        self.spike_tokens_remaining = 0
        self.spike_tokens_max = 3  # Spike lasts 3 tokens

    def __call__(self, module, args):
        if self.triggered and self.spike_tokens_remaining > 0:
            hs = args[0]
            noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype) * self.sigma_spike
            self.spike_tokens_remaining -= 1
            if self.spike_tokens_remaining <= 0:
                self.triggered = False
            return (hs + noise,) + args[1:]
        return args

    def fire_spike(self):
        """Trigger a spike (called when illegal move detected)."""
        self.triggered = True
        self.spike_tokens_remaining = self.spike_tokens_max

    def reset(self):
        self.triggered = False
        self.spike_tokens_remaining = 0


# ═══════════════════════════════════════════════════
#  MODEL LOADING & GENERATION
# ═══════════════════════════════════════════════════

def load_model():
    print(f"\n📦 Loading {MODEL_NAME}...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
    n_layers = len(model.model.layers)
    print(f"  ✅ Loaded: {n_layers} layers")
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Generate text from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.7, top_p=0.9, top_k=50,
            repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(input_text):].strip()


def compute_response_entropy(model, tokenizer, prompt):
    """Compute average per-token entropy for a prompt (measure of model certainty)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    # Compute entropy for last 20 tokens (most relevant to decision)
    n_tokens = min(20, logits.shape[0])
    entropies = []
    for i in range(logits.shape[0] - n_tokens, logits.shape[0]):
        probs = torch.softmax(logits[i].float(), dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        entropies.append(ent)
    return np.mean(entropies)


# ═══════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════

def play_hanoi(model, tokenizer, env, condition, hook=None, handle=None, max_steps=30):
    """
    Have the LLM play one game of Tower of Hanoi.
    Returns game stats dict.
    """
    env.reset()
    error_feedback = None
    consecutive_illegal = 0
    entropy_trajectory = []

    for step in range(max_steps):
        prompt = build_hanoi_prompt(env, error_feedback)

        # Measure entropy before generating move
        ent = compute_response_entropy(model, tokenizer, prompt)
        entropy_trajectory.append(ent)

        # Generate move
        response = generate_text(model, tokenizer, prompt, max_new_tokens=60)

        move = parse_move(response)
        if move is None:
            # Unparseable response — count as illegal
            env.illegal_moves += 1
            env.total_attempts += 1
            env.record_illegal("?", "?", f"Unparseable: {response[:80]}")
            error_feedback = f"Could not understand your move. You said: '{response[:60]}'. Please use format: Move disk N from X to Y"
            consecutive_illegal += 1

            # Conflict-triggered spike
            if condition == "conflict_triggered" and hook:
                hook.fire_spike()

            if consecutive_illegal >= 5:
                break  # Give up if 5 consecutive failures
            continue

        from_peg, to_peg = move
        success, msg = env.make_move(from_peg, to_peg)

        if success:
            error_feedback = None
            consecutive_illegal = 0
            if env.is_solved():
                break
        else:
            env.record_illegal(from_peg, to_peg, msg)
            error_feedback = msg
            consecutive_illegal += 1

            # Conflict-triggered spike on illegal move
            if condition == "conflict_triggered" and hook:
                hook.fire_spike()

            if consecutive_illegal >= 5:
                break

    state = env.get_state_summary()
    state["entropy_trajectory"] = entropy_trajectory
    state["condition"] = condition
    state["n_disks"] = env.n_disks
    state["modified_rules"] = env.modified_rules
    return state


def run_experiment(model, tokenizer, n_disks, modified_rules, condition, n_trials=N_TRIALS):
    """Run n_trials games for one condition."""
    task_name = f"{'Modified' if modified_rules else 'Standard'}-{n_disks}"
    print(f"\n  🎮 {task_name} | Condition: {condition} | Trials: {n_trials}")

    layers = model.model.layers
    hook = None
    handle = None

    if condition == "l18_echo":
        hook = L18EchoHook(sigma=SIGMA_ECHO)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)
    elif condition == "conflict_triggered":
        hook = ConflictTriggeredHook(sigma_spike=SIGMA_SPIKE)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    trial_results = []
    for trial in range(n_trials):
        env = HanoiEnvironment(n_disks=n_disks, modified_rules=modified_rules)
        result = play_hanoi(model, tokenizer, env, condition, hook, handle)
        trial_results.append(result)

        status = "✅" if result["solved"] else "❌"
        moves = result["move_count"]
        illegal = result["illegal_moves"]
        print(f"    Trial {trial+1:2d}: {status} moves={moves} illegal={illegal}")

        # Reset hook state between trials
        if hook and hasattr(hook, 'reset'):
            hook.reset()

    if handle:
        handle.remove()

    # Aggregate stats
    solve_rate = np.mean([r["solved"] for r in trial_results]) * 100
    avg_illegal = np.mean([r["illegal_moves"] for r in trial_results])
    avg_moves = np.mean([r["move_count"] for r in trial_results])
    avg_self_corrections = np.mean([r["self_corrections"] for r in trial_results])
    completed = np.mean([1 if r["move_count"] > 0 else 0 for r in trial_results]) * 100

    # Average entropy
    all_ent = [np.mean(r["entropy_trajectory"]) for r in trial_results if r["entropy_trajectory"]]
    avg_entropy = np.mean(all_ent) if all_ent else 0

    summary = {
        "task": task_name,
        "condition": condition,
        "n_trials": n_trials,
        "solve_rate": round(solve_rate, 1),
        "avg_moves": round(avg_moves, 1),
        "avg_illegal_moves": round(avg_illegal, 1),
        "avg_self_corrections": round(avg_self_corrections, 1),
        "completion_rate": round(completed, 1),
        "avg_entropy": round(avg_entropy, 2),
        "optimal_moves": trial_results[0]["optimal"] if trial_results else 0,
    }

    print(f"    📊 Solve={solve_rate:.0f}% Illegal={avg_illegal:.1f} "
          f"SelfCorr={avg_self_corrections:.1f} Entropy={avg_entropy:.2f}")

    return summary, trial_results


# ═══════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════

def visualize(all_results):
    """Create comprehensive visualization of results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Phase 50: The Illusion Breaker — Tower of Hanoi\n"
                 "SNN perturbation vs Apple's 'Illusion of Thinking'",
                 fontsize=15, fontweight="bold", y=0.98)

    conditions = ["baseline", "l18_echo", "conflict_triggered"]
    cond_labels = ["Baseline\n(No SNN)", "L18 Echo\n(σ=0.02)", "Conflict\nTriggered"]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]
    tasks = ["Standard-3", "Standard-4", "Modified-3", "Modified-4"]

    # Organize data: all_results is list of summaries
    data = {}
    for r in all_results:
        key = (r["task"], r["condition"])
        data[key] = r

    # ── Panel 1: Solve Rate by task ──
    ax = axes[0, 0]
    x = np.arange(len(tasks))
    width = 0.25
    for i, (cond, label, color) in enumerate(zip(conditions, cond_labels, colors)):
        vals = [data.get((t, cond), {}).get("solve_rate", 0) for t in tasks]
        bars = ax.bar(x + i * width, vals, width, label=label.replace('\n', ' '),
                      color=color, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{v:.0f}%", ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("① Solve Rate (Apple criticism: mimicry collapse)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel 2: Illegal Moves ──
    ax = axes[0, 1]
    for i, (cond, label, color) in enumerate(zip(conditions, cond_labels, colors)):
        vals = [data.get((t, cond), {}).get("avg_illegal_moves", 0) for t in tasks]
        ax.bar(x + i * width, vals, width, label=label.replace('\n', ' '),
               color=color, alpha=0.85, edgecolor='white')
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("Avg Illegal Moves")
    ax.set_title("② Illegal Moves (rule comprehension)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel 3: Self-Corrections ──
    ax = axes[0, 2]
    for i, (cond, label, color) in enumerate(zip(conditions, cond_labels, colors)):
        vals = [data.get((t, cond), {}).get("avg_self_corrections", 0) for t in tasks]
        ax.bar(x + i * width, vals, width, label=label.replace('\n', ' '),
               color=color, alpha=0.85, edgecolor='white')
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("Avg Self-Corrections")
    ax.set_title("③ Self-Correction (Apple: 'cannot self-correct')")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel 4: Move Efficiency ──
    ax = axes[1, 0]
    for i, (cond, label, color) in enumerate(zip(conditions, cond_labels, colors)):
        vals = [data.get((t, cond), {}).get("avg_moves", 0) for t in tasks]
        ax.bar(x + i * width, vals, width, label=label.replace('\n', ' '),
               color=color, alpha=0.85, edgecolor='white')
    # Add optimal line
    optimals = [data.get((t, "baseline"), {}).get("optimal_moves", 0) for t in tasks]
    for j, opt in enumerate(optimals):
        ax.axhline(y=opt, xmin=(j/len(tasks))+0.02, xmax=((j+1)/len(tasks))-0.02,
                    color='gold', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("Avg Legal Moves")
    ax.set_title("④ Move Count (gold line = optimal)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel 5: Average Entropy ──
    ax = axes[1, 1]
    for i, (cond, label, color) in enumerate(zip(conditions, cond_labels, colors)):
        vals = [data.get((t, cond), {}).get("avg_entropy", 0) for t in tasks]
        ax.bar(x + i * width, vals, width, label=label.replace('\n', ' '),
               color=color, alpha=0.85, edgecolor='white')
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("Avg Entropy")
    ax.set_title("⑤ Entropy (low = mimicry, high = reasoning)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel 6: Summary table ──
    ax = axes[1, 2]
    ax.axis('off')
    table_data = []
    headers = ["Task", "Condition", "Solve%", "Illegal", "Self-Corr"]
    for task in tasks:
        for cond, label in zip(conditions, ["Base", "Echo", "Spike"]):
            r = data.get((task, cond), {})
            table_data.append([
                task, label,
                f"{r.get('solve_rate', 0):.0f}%",
                f"{r.get('avg_illegal_moves', 0):.1f}",
                f"{r.get('avg_self_corrections', 0):.1f}",
            ])
    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    # Color code rows
    for i in range(len(table_data)):
        row_idx = i + 1  # +1 for header
        cond_idx = i % 3
        for j in range(len(headers)):
            cell = table[row_idx, j]
            cell.set_facecolor(colors[cond_idx] + "20")  # Light version

    ax.set_title("⑥ Summary", fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(FIGURES_DIR, "phase50_illusion_breaker.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model, tokenizer = load_model()

    all_summaries = []
    all_details = {}

    # Define task configurations
    task_configs = [
        {"n_disks": 3, "modified": False, "name": "Standard-3"},
        {"n_disks": 4, "modified": False, "name": "Standard-4"},
        {"n_disks": 3, "modified": True,  "name": "Modified-3"},
        {"n_disks": 4, "modified": True,  "name": "Modified-4"},
    ]

    conditions = ["baseline", "l18_echo", "conflict_triggered"]

    for task_cfg in task_configs:
        print(f"\n{'═'*60}")
        print(f"  🏰 TASK: {task_cfg['name']} (disks={task_cfg['n_disks']}, "
              f"modified={task_cfg['modified']})")
        print(f"{'═'*60}")

        for condition in conditions:
            summary, details = run_experiment(
                model, tokenizer,
                n_disks=task_cfg["n_disks"],
                modified_rules=task_cfg["modified"],
                condition=condition,
                n_trials=N_TRIALS,
            )
            all_summaries.append(summary)
            all_details[f"{task_cfg['name']}_{condition}"] = details

    elapsed = time.time() - t_start
    fig_path = visualize(all_summaries)

    # Save results
    output = {
        "experiment": "Phase 50: The Illusion Breaker",
        "model": MODEL_SHORT,
        "reference": "Apple 'The Illusion of Thinking' (2025)",
        "elapsed_minutes": round(elapsed / 60, 1),
        "summaries": all_summaries,
        "figure_path": fig_path,
    }

    log_path = os.path.join(RESULTS_DIR, "phase50_illusion_breaker_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print final summary
    print(f"\n{'═'*70}")
    print(f"  ⚡ PHASE 50: THE ILLUSION BREAKER — FINAL VERDICT")
    print(f"{'═'*70}")
    print(f"  Apple's claim: 'LLMs cannot reason, only mimic'")
    print(f"  Counterclaim: 'SNN perturbation breaks mimicry → enables reasoning'")
    print(f"{'─'*70}")
    print(f"  {'Task':<15} {'Condition':<20} {'Solve%':>8} {'Illegal':>8} {'SelfCorr':>8}")
    print(f"{'─'*70}")
    for s in all_summaries:
        print(f"  {s['task']:<15} {s['condition']:<20} "
              f"{s['solve_rate']:>7.0f}% {s['avg_illegal_moves']:>8.1f} "
              f"{s['avg_self_corrections']:>8.1f}")
    print(f"{'─'*70}")

    # Key comparisons
    for task in ["Standard-3", "Standard-4", "Modified-3", "Modified-4"]:
        base = next((s for s in all_summaries if s["task"] == task and s["condition"] == "baseline"), None)
        best = max(
            [s for s in all_summaries if s["task"] == task],
            key=lambda s: s["solve_rate"]
        )
        if base and best["condition"] != "baseline":
            delta = best["solve_rate"] - base["solve_rate"]
            print(f"  {task}: Best={best['condition']} (Δsolve={delta:+.0f}%)")

    print(f"\n  ⏱ Total time: {elapsed/60:.1f} minutes")
    print(f"  💾 Results: {log_path}")
    print(f"  📊 Figure: {fig_path}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
