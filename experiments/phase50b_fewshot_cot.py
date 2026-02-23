"""
Phase 50b: The Illusion Breaker v2 — Few-Shot + CoT
====================================================

Phase 50 found: Mistral-7B (4-bit) scores 0% on all Hanoi conditions.
Root cause: LLM fails to output moves in the required format.

Phase 50b fixes:
  1. Few-shot: Show 2-3 example moves to stabilize output format
  2. CoT: Ask LLM to think step-by-step before choosing a move
  3. Simplified output: "A->C" format instead of "Move disk N from A to C"
  4. Focus: Standard-3 only (the easiest case)

Conditions:
  1. Baseline (no SNN) + Few-shot + CoT
  2. L18-Echo (σ=0.02) + Few-shot + CoT
  3. Conflict-Triggered (σ=0.05 on illegal) + Few-shot + CoT

Usage:
    python experiments/phase50b_fewshot_cot.py
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
SIGMA_ECHO = 0.02
SIGMA_SPIKE = 0.05
TARGET_LAYER = 18
MAX_NEW_TOKENS = 150
SEED = 2026
N_TRIALS = 10
MAX_STEPS = 40   # More generous (optimal for 3-disk = 7 moves)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
#  TOWER OF HANOI ENVIRONMENT
# ═══════════════════════════════════════════════════

class HanoiEnvironment:
    """Tower of Hanoi game engine."""

    def __init__(self, n_disks=3, modified_rules=False):
        self.n_disks = n_disks
        self.modified_rules = modified_rules
        self.reset()

    def reset(self):
        self.pegs = {"A": list(range(self.n_disks, 0, -1)), "B": [], "C": []}
        self.move_history = []
        self.illegal_moves = 0
        self.total_attempts = 0
        self.self_corrections = 0
        self._last_was_illegal = False

    def is_solved(self):
        return len(self.pegs["C"]) == self.n_disks

    def is_legal_move(self, from_peg, to_peg):
        if from_peg not in self.pegs or to_peg not in self.pegs:
            return False
        if not self.pegs[from_peg]:
            return False
        disk = self.pegs[from_peg][-1]
        if not self.pegs[to_peg]:
            return True
        top_target = self.pegs[to_peg][-1]
        if self.modified_rules:
            return disk > top_target
        else:
            return disk < top_target

    def get_legal_moves(self):
        """Return list of all currently legal moves as (from, to) tuples."""
        moves = []
        for f in ["A", "B", "C"]:
            for t in ["A", "B", "C"]:
                if f != t and self.is_legal_move(f, t):
                    moves.append((f, t))
        return moves

    def make_move(self, from_peg, to_peg):
        self.total_attempts += 1
        from_peg = from_peg.upper().strip()
        to_peg = to_peg.upper().strip()

        if from_peg not in self.pegs or to_peg not in self.pegs:
            self.illegal_moves += 1
            self._last_was_illegal = True
            return False, f"Invalid peg: {from_peg} or {to_peg}"
        if from_peg == to_peg:
            self.illegal_moves += 1
            self._last_was_illegal = True
            return False, "Cannot move to same peg"
        if not self.pegs[from_peg]:
            self.illegal_moves += 1
            self._last_was_illegal = True
            return False, f"Peg {from_peg} is empty"
        if not self.is_legal_move(from_peg, to_peg):
            self.illegal_moves += 1
            self._last_was_illegal = True
            if self.modified_rules:
                return False, f"ILLEGAL: Modified rules require placing LARGER on SMALLER"
            else:
                return False, f"ILLEGAL: Cannot place larger disk on smaller"

        # Check self-correction
        if self._last_was_illegal:
            self.self_corrections += 1
        self._last_was_illegal = False

        disk = self.pegs[from_peg].pop()
        self.pegs[to_peg].append(disk)
        self.move_history.append({"from": from_peg, "to": to_peg, "disk": disk})
        return True, f"OK: disk {disk} from {from_peg} to {to_peg}"

    def render_state(self):
        """Compact state representation for LLM."""
        lines = []
        for p in ["A", "B", "C"]:
            disks = self.pegs[p]
            lines.append(f"  {p}: {disks if disks else '[]'}")
        return "\n".join(lines)

    def optimal_moves(self):
        if self.modified_rules:
            return 3**self.n_disks - 1
        else:
            return 2**self.n_disks - 1

    def get_stats(self):
        return {
            "solved": self.is_solved(),
            "legal_moves": len(self.move_history),
            "illegal_moves": self.illegal_moves,
            "self_corrections": self.self_corrections,
            "total_attempts": self.total_attempts,
            "optimal": self.optimal_moves(),
        }


# ═══════════════════════════════════════════════════
#  FEW-SHOT + COT PROMPT BUILDER
# ═══════════════════════════════════════════════════

FEW_SHOT_STANDARD_3 = """Here is an example of solving a 3-disk Tower of Hanoi puzzle:

State: A: [3, 2, 1]  B: []  C: []
Think: I need to move disk 1 out of the way first.
Move: A->C

State: A: [3, 2]  B: []  C: [1]
Think: Now I can move disk 2 to B.
Move: A->B

State: A: [3]  B: [2]  C: [1]
Think: I need to put disk 1 on top of disk 2.
Move: C->B

State: A: [3]  B: [2, 1]  C: []
Think: Now I can move disk 3 to the goal peg C.
Move: A->C

(The game continues until all disks are on C.)
"""

FEW_SHOT_MODIFIED_3 = """Here is an example of the modified rules where you MUST place a LARGER disk onto a SMALLER one:

State: A: [3, 2, 1]  B: []  C: []
Think: In modified rules, I need to move the smallest disk first since larger disks go ON TOP of smaller ones.
Move: A->C

State: A: [3, 2]  B: []  C: [1]
Think: Now I move disk 2. In modified rules, disk 2 can go on peg with disk 1 (larger onto smaller) or empty B.
Move: A->C

State: A: [3]  B: []  C: [1, 2]
Think: Disk 2 is now on top of disk 1 (larger on smaller = legal in modified rules). Move disk 1 somewhere.
Move: A->B

(The game continues following modified rules: LARGER onto SMALLER only.)
"""


def build_prompt(env, error_feedback=None, move_number=0):
    """Build Few-shot + CoT prompt."""

    if env.modified_rules:
        rule_text = (
            "MODIFIED RULES: You can ONLY place a LARGER disk onto a SMALLER disk.\n"
            "This is the OPPOSITE of standard rules!"
        )
        few_shot = FEW_SHOT_MODIFIED_3
    else:
        rule_text = (
            "STANDARD RULES: You can ONLY place a SMALLER disk onto a LARGER disk."
        )
        few_shot = FEW_SHOT_STANDARD_3

    prompt = f"""Tower of Hanoi — {env.n_disks} disks
{rule_text}
Goal: Move ALL disks from A to C.

{few_shot}
---
Now it's YOUR turn. Current state:
{env.render_state()}
"""

    # Show recent move history
    if env.move_history:
        recent = env.move_history[-3:]
        prompt += "\nYour previous moves:\n"
        for m in recent:
            prompt += f"  {m['from']}->{m['to']} (disk {m['disk']})\n"

    # Show available legal moves
    legal = env.get_legal_moves()
    if legal:
        prompt += f"\nLegal moves: {', '.join(f'{f}->{t}' for f, t in legal)}\n"

    if error_feedback:
        prompt += f"\n⚠️ {error_feedback}\nPick a LEGAL move from the list above.\n"

    prompt += (
        "\nThink step-by-step, then give your move.\n"
        "Format your response EXACTLY as:\n"
        "Think: [your reasoning]\n"
        "Move: [X]->[Y]\n"
        "where X and Y are peg letters (A, B, or C)."
    )
    return prompt


def parse_move_v2(response):
    """Parse LLM response — more robust than v1."""
    response_upper = response.strip().upper()

    # Primary: "Move: A->C"
    m = re.search(r'MOVE:\s*([ABC])\s*->\s*([ABC])', response_upper)
    if m:
        return m.group(1), m.group(2)

    # Fallback: "A->C" anywhere
    m = re.search(r'([ABC])\s*->\s*([ABC])', response_upper)
    if m:
        return m.group(1), m.group(2)

    # Fallback: "Move disk N from X to Y"
    m = re.search(r'MOVE\s+DISK\s+\d+\s+FROM\s+([ABC])\s+TO\s+([ABC])', response_upper)
    if m:
        return m.group(1), m.group(2)

    # Fallback: "from A to C"
    m = re.search(r'FROM\s+([ABC])\s+TO\s+([ABC])', response_upper)
    if m:
        return m.group(1), m.group(2)

    return None


# ═══════════════════════════════════════════════════
#  SNN HOOKS (same as Phase 50)
# ═══════════════════════════════════════════════════

class L18EchoHook:
    def __init__(self, sigma=0.02):
        self.sigma = sigma

    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn(hs.shape, device=hs.device, dtype=hs.dtype) * self.sigma
        return (hs + noise,) + args[1:]


class ConflictTriggeredHook:
    def __init__(self, sigma_spike=0.05):
        self.sigma_spike = sigma_spike
        self.triggered = False
        self.spike_tokens_remaining = 0
        self.spike_tokens_max = 5  # Longer spike (was 3 in Phase 50)

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
        self.triggered = True
        self.spike_tokens_remaining = self.spike_tokens_max

    def reset(self):
        self.triggered = False
        self.spike_tokens_remaining = 0


# ═══════════════════════════════════════════════════
#  MODEL + GENERATION
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
    print(f"  ✅ Loaded: {len(model.model.layers)} layers")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.6, top_p=0.85, top_k=40,
            repetition_penalty=1.3, pad_token_id=tokenizer.pad_token_id
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(input_text):].strip()


def compute_response_entropy(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    n = min(20, logits.shape[0])
    ents = []
    for i in range(logits.shape[0] - n, logits.shape[0]):
        probs = torch.softmax(logits[i].float(), dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        ents.append(ent)
    return float(np.mean(ents))


# ═══════════════════════════════════════════════════
#  GAME LOOP
# ═══════════════════════════════════════════════════

def play_one_game(model, tokenizer, env, condition, hook=None):
    """Play one game of Hanoi with Few-shot + CoT."""
    env.reset()
    error_feedback = None
    consecutive_fail = 0
    entropies = []
    cot_thoughts = []

    for step in range(MAX_STEPS):
        prompt = build_prompt(env, error_feedback, step)

        # Measure entropy
        ent = compute_response_entropy(model, tokenizer, prompt)
        entropies.append(ent)

        # Generate
        response = generate_response(model, tokenizer, prompt, max_new_tokens=100)

        # Extract CoT thought
        think_match = re.search(r'Think:\s*(.+?)(?:\n|Move:)', response, re.DOTALL | re.IGNORECASE)
        if think_match:
            cot_thoughts.append(think_match.group(1).strip()[:100])

        # Parse move
        move = parse_move_v2(response)

        if move is None:
            env.illegal_moves += 1
            env.total_attempts += 1
            env._last_was_illegal = True
            error_feedback = f"Could not parse your move. You said: '{response[:60]}'. Use format: Move: X->Y"
            consecutive_fail += 1
            if condition == "conflict_triggered" and hook:
                hook.fire_spike()
            if consecutive_fail >= 8:
                break
            continue

        from_peg, to_peg = move
        success, msg = env.make_move(from_peg, to_peg)

        if success:
            error_feedback = None
            consecutive_fail = 0
            if env.is_solved():
                break
        else:
            error_feedback = msg
            consecutive_fail += 1
            if condition == "conflict_triggered" and hook:
                hook.fire_spike()
            if consecutive_fail >= 8:
                break

    stats = env.get_stats()
    stats["entropies"] = entropies
    stats["avg_entropy"] = float(np.mean(entropies)) if entropies else 0.0
    stats["cot_samples"] = cot_thoughts[:5]
    stats["condition"] = condition
    return stats


def run_condition(model, tokenizer, n_disks, modified, condition, n_trials=N_TRIALS):
    """Run all trials for one condition."""
    task_label = f"{'Modified' if modified else 'Standard'}-{n_disks}"
    print(f"\n  🎮 {task_label} | {condition} | {n_trials} trials")

    layers = model.model.layers
    hook = None
    handle = None

    if condition == "l18_echo":
        hook = L18EchoHook(sigma=SIGMA_ECHO)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)
    elif condition == "conflict_triggered":
        hook = ConflictTriggeredHook(sigma_spike=SIGMA_SPIKE)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    results = []
    for trial in range(n_trials):
        env = HanoiEnvironment(n_disks=n_disks, modified_rules=modified)
        r = play_one_game(model, tokenizer, env, condition, hook)
        results.append(r)

        icon = "✅" if r["solved"] else "❌"
        print(f"    Trial {trial+1:2d}: {icon} legal={r['legal_moves']} "
              f"illegal={r['illegal_moves']} self_corr={r['self_corrections']}")

        if hook and hasattr(hook, 'reset'):
            hook.reset()

    if handle:
        handle.remove()

    # Aggregate
    solve_rate = np.mean([r["solved"] for r in results]) * 100
    avg_legal = np.mean([r["legal_moves"] for r in results])
    avg_illegal = np.mean([r["illegal_moves"] for r in results])
    avg_self_corr = np.mean([r["self_corrections"] for r in results])
    avg_ent = np.mean([r["avg_entropy"] for r in results])

    summary = {
        "task": task_label, "condition": condition, "n_trials": n_trials,
        "solve_rate": round(solve_rate, 1),
        "avg_legal_moves": round(avg_legal, 1),
        "avg_illegal_moves": round(avg_illegal, 1),
        "avg_self_corrections": round(avg_self_corr, 1),
        "avg_entropy": round(avg_ent, 2),
        "optimal": results[0]["optimal"],
        "cot_samples": results[0].get("cot_samples", []),
    }

    print(f"    📊 Solve={solve_rate:.0f}% Legal={avg_legal:.1f} "
          f"Illegal={avg_illegal:.1f} SelfCorr={avg_self_corr:.1f}")

    return summary, results


# ═══════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════

def visualize(summaries, detail_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 50b: The Illusion Breaker v2\n"
                 "Few-Shot + Chain-of-Thought + SNN Perturbation",
                 fontsize=14, fontweight="bold", y=0.98)

    conditions = ["baseline", "l18_echo", "conflict_triggered"]
    labels = ["Baseline\n(No SNN)", "L18 Echo\n(σ=0.02)", "Conflict\nTriggered"]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]

    # Group by task
    tasks = list(dict.fromkeys(s["task"] for s in summaries))
    x = np.arange(len(tasks))
    width = 0.25

    # Panel 1: Solve Rate
    ax = axes[0, 0]
    for i, (cond, label, color) in enumerate(zip(conditions, labels, colors)):
        vals = [next((s["solve_rate"] for s in summaries
                      if s["task"] == t and s["condition"] == cond), 0) for t in tasks]
        bars = ax.bar(x + i * width, vals, width, label=label.replace('\n', ' '),
                      color=color, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v:.0f}%", ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("① Solve Rate (with Few-shot + CoT)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Legal vs Illegal Moves
    ax = axes[0, 1]
    for i, (cond, label, color) in enumerate(zip(conditions, labels, colors)):
        legal = [next((s["avg_legal_moves"] for s in summaries
                       if s["task"] == t and s["condition"] == cond), 0) for t in tasks]
        illegal = [next((s["avg_illegal_moves"] for s in summaries
                         if s["task"] == t and s["condition"] == cond), 0) for t in tasks]
        ax.bar(x + i * width, legal, width, label=f"{label.split(chr(10))[0]} legal",
               color=color, alpha=0.85)
        ax.bar(x + i * width, illegal, width, bottom=legal,
               label=f"{label.split(chr(10))[0]} illegal",
               color=color, alpha=0.35, hatch='//')
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylabel("Avg Moves")
    ax.set_title("② Legal (solid) + Illegal (hatched) Moves")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 3: Self-Corrections
    ax = axes[1, 0]
    for i, (cond, label, color) in enumerate(zip(conditions, labels, colors)):
        vals = [next((s["avg_self_corrections"] for s in summaries
                      if s["task"] == t and s["condition"] == cond), 0) for t in tasks]
        bars = ax.bar(x + i * width, vals, width, label=label.replace('\n', ' '),
                      color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{v:.1f}", ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylabel("Avg Self-Corrections")
    ax.set_title("③ Self-Correction (Apple: 'cannot self-correct')")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 4: Entropy comparison
    ax = axes[1, 1]
    for i, (cond, label, color) in enumerate(zip(conditions, labels, colors)):
        vals = [next((s["avg_entropy"] for s in summaries
                      if s["task"] == t and s["condition"] == cond), 0) for t in tasks]
        bars = ax.bar(x + i * width, vals, width, label=label.replace('\n', ' '),
                      color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylabel("Avg Entropy")
    ax.set_title("④ Entropy (CoT reasoning indicator)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(FIGURES_DIR, "phase50b_fewshot_cot.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")
    return fig_path


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

def main():
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tokenizer = load_model()

    all_summaries = []
    all_details = {}

    # Focus on winnable configurations
    configs = [
        {"n_disks": 3, "modified": False, "name": "Standard-3"},
        {"n_disks": 3, "modified": True,  "name": "Modified-3"},
        {"n_disks": 4, "modified": False, "name": "Standard-4"},
    ]

    conditions = ["baseline", "l18_echo", "conflict_triggered"]

    for cfg in configs:
        print(f"\n{'═'*60}")
        print(f"  🏰 TASK: {cfg['name']} (modified={cfg['modified']})")
        print(f"{'═'*60}")

        for cond in conditions:
            summary, details = run_condition(
                model, tokenizer,
                n_disks=cfg["n_disks"],
                modified=cfg["modified"],
                condition=cond,
            )
            all_summaries.append(summary)
            all_details[f"{cfg['name']}_{cond}"] = details

    elapsed = time.time() - t_start
    fig_path = visualize(all_summaries, all_details)

    # Save results
    output = {
        "experiment": "Phase 50b: The Illusion Breaker v2 (Few-Shot + CoT)",
        "model": MODEL_SHORT,
        "improvements_over_50": [
            "Few-shot examples for output format",
            "Chain-of-Thought reasoning",
            "Simplified move format (A->C)",
            "Legal moves shown in prompt",
            "Longer spike duration (5 tokens)",
        ],
        "elapsed_minutes": round(elapsed / 60, 1),
        "summaries": all_summaries,
        "figure_path": fig_path,
    }
    log_path = os.path.join(RESULTS_DIR, "phase50b_fewshot_cot_log.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print verdict
    print(f"\n{'═'*70}")
    print(f"  ⚡ PHASE 50b: THE ILLUSION BREAKER v2 — FINAL VERDICT")
    print(f"{'═'*70}")
    print(f"  Improvements: Few-shot + CoT + simplified format + legal move hints")
    print(f"{'─'*70}")
    print(f"  {'Task':<15} {'Condition':<20} {'Solve%':>8} {'Legal':>8} "
          f"{'Illegal':>8} {'SelfCorr':>8} {'Entropy':>8}")
    print(f"{'─'*70}")
    for s in all_summaries:
        print(f"  {s['task']:<15} {s['condition']:<20} {s['solve_rate']:>7.0f}% "
              f"{s['avg_legal_moves']:>8.1f} {s['avg_illegal_moves']:>8.1f} "
              f"{s['avg_self_corrections']:>8.1f} {s['avg_entropy']:>8.2f}")
    print(f"{'─'*70}")

    # Show CoT samples
    print(f"\n  🧠 CoT Reasoning Samples:")
    for s in all_summaries:
        if s.get("cot_samples"):
            print(f"    [{s['task']} / {s['condition']}]")
            for thought in s["cot_samples"][:2]:
                print(f"      → {thought}")

    # Key comparisons
    print(f"\n  📊 Key Comparisons vs Phase 50 (no Few-shot/CoT):")
    for task in dict.fromkeys(s["task"] for s in all_summaries):
        base = next((s for s in all_summaries
                     if s["task"] == task and s["condition"] == "baseline"), None)
        if base:
            print(f"    {task} baseline: solve={base['solve_rate']}% "
                  f"(Phase 50: 0%)")

    print(f"\n  ⏱ Total: {elapsed/60:.1f} min | 💾 {log_path}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
