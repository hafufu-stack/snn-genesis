"""
Phase 50c: The Illusion Breaker v3 — Chat Template + CoT
=========================================================

Root cause of Phase 50/50b failure:
  Mistral-Instruct-v0.3 needs chat template (apply_chat_template).
  Without it, the model runs in text-completion mode and ignores instructions.

Fix: Use proper chat template for all prompts.

Debug evidence:
  raw_prompt → "(X is the tower and Y is where you want to move it)"  ← BAD
  chat_template → "Step 1: Move disk 1 from A to C (A=[3,2], B=[], C=[1])" ← GOOD

Conditions:
  1. Baseline (no SNN) + Chat template
  2. L18-Echo (σ=0.02) + Chat template
  3. Conflict-Triggered (σ=0.05 spike on illegal) + Chat template

Usage:
    python experiments/phase50c_chat_template.py
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
SEED = 2026
N_TRIALS = 10
MAX_STEPS = 50  # Generous limit

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
#  HANOI ENVIRONMENT
# ═══════════════════════════════════════════════════

class HanoiEnv:
    def __init__(self, n_disks=3, modified=False):
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
                return False, f"Modified: disk {disk} must be LARGER than {top}"
            if not self.modified and disk >= top:
                self.illegal_count += 1; self._prev_illegal = True
                return False, f"Standard: disk {disk} must be SMALLER than {top}"

        if self._prev_illegal:
            self.self_corrections += 1
        self._prev_illegal = False

        self.pegs[from_p].pop()
        self.pegs[to_p].append(disk)
        self.moves.append(f"{from_p}->{to_p} (disk {disk})")
        return True, f"Moved disk {disk}: {from_p}->{to_p}"

    def state_str(self):
        return f"A:{self.pegs['A']} B:{self.pegs['B']} C:{self.pegs['C']}"

    def optimal(self):
        return (3**self.n_disks - 1) if self.modified else (2**self.n_disks - 1)

    def stats(self):
        return {
            "solved": self.is_solved(),
            "legal_moves": len(self.moves),
            "illegal_moves": self.illegal_count,
            "self_corrections": self.self_corrections,
            "optimal": self.optimal(),
        }


# ═══════════════════════════════════════════════════
#  PROMPT BUILDER (Chat format)
# ═══════════════════════════════════════════════════

def build_system_msg(env):
    if env.modified:
        rules = "MODIFIED RULES: You can ONLY place a LARGER disk onto a SMALLER disk. The opposite of standard."
    else:
        rules = "STANDARD RULES: You can ONLY place a SMALLER disk onto a LARGER disk."
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
        msg += f"⚠️ ERROR: {error}. Pick from legal moves above.\n"

    msg += "Your move:"
    return msg


def build_chat_prompt(tokenizer, env, error=None):
    messages = [
        {"role": "user", "content": build_system_msg(env) + "\n\n" + build_user_msg(env, error)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ═══════════════════════════════════════════════════
#  MOVE PARSER
# ═══════════════════════════════════════════════════

def parse_move(response):
    resp = response.strip().upper()

    # "Move: A->C" or "Move:A->C"
    m = re.search(r'MOVE:\s*([ABC])\s*->\s*([ABC])', resp)
    if m: return m.group(1), m.group(2)

    # "A->C" standalone
    m = re.search(r'([ABC])\s*->\s*([ABC])', resp)
    if m: return m.group(1), m.group(2)

    # "Move disk N from X to Y"
    m = re.search(r'MOVE\s+DISK\s+\d+\s+FROM\s+([ABC])\s+TO\s+([ABC])', resp)
    if m: return m.group(1), m.group(2)

    # "from A to C"
    m = re.search(r'FROM\s+([ABC])\s+TO\s+([ABC])', resp)
    if m: return m.group(1), m.group(2)

    # "Step N: Move disk N from A to C"
    m = re.search(r'STEP\s+\d+:\s*MOVE\s+DISK\s+\d+\s+FROM\s+([ABC])\s+TO\s+([ABC])', resp)
    if m: return m.group(1), m.group(2)

    # Last resort: any two different A/B/C letters
    letters = re.findall(r'[ABC]', resp)
    if len(letters) >= 2 and letters[0] != letters[1]:
        return letters[0], letters[1]

    return None


# ═══════════════════════════════════════════════════
#  SNN HOOKS
# ═══════════════════════════════════════════════════

class EchoHook:
    def __init__(self, sigma=0.02):
        self.sigma = sigma
    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn_like(hs) * self.sigma
        return (hs + noise,) + args[1:]

class SpikeHook:
    def __init__(self, sigma=0.05):
        self.sigma = sigma
        self.active = False
        self.remaining = 0
    def __call__(self, module, args):
        if self.active and self.remaining > 0:
            hs = args[0]
            noise = torch.randn_like(hs) * self.sigma
            self.remaining -= 1
            if self.remaining <= 0: self.active = False
            return (hs + noise,) + args[1:]
        return args
    def fire(self):
        self.active = True; self.remaining = 5
    def reset(self):
        self.active = False; self.remaining = 0


# ═══════════════════════════════════════════════════
#  MODEL + GENERATION
# ═══════════════════════════════════════════════════

def load_model():
    print(f"\n📦 Loading {MODEL_NAME}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    print(f"  ✅ {len(model.model.layers)} layers")
    return model, tok


def generate(model, tok, prompt, max_tokens=100):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=0.5, top_p=0.9, top_k=40,
            repetition_penalty=1.2, pad_token_id=tok.pad_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    inp = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(inp):].strip()


def measure_entropy(model, tok, prompt):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    n = min(15, logits.shape[0])
    ents = []
    for i in range(logits.shape[0] - n, logits.shape[0]):
        p = torch.softmax(logits[i].float(), dim=-1)
        ents.append(-torch.sum(p * torch.log(p + 1e-10)).item())
    return float(np.mean(ents))


# ═══════════════════════════════════════════════════
#  GAME LOOP
# ═══════════════════════════════════════════════════

def play_game(model, tok, env, condition, hook=None):
    env.reset()
    error = None
    consec_fail = 0
    entropies = []
    thoughts = []
    raw_responses = []

    for step in range(MAX_STEPS):
        prompt = build_chat_prompt(tok, env, error)

        ent = measure_entropy(model, tok, prompt)
        entropies.append(ent)

        resp = generate(model, tok, prompt, max_tokens=80)
        raw_responses.append(resp[:120])

        # Extract think
        th = re.search(r'Think:\s*(.+?)(?:\n|Move:|$)', resp, re.IGNORECASE)
        if th: thoughts.append(th.group(1).strip()[:80])

        move = parse_move(resp)
        if move is None:
            env.illegal_count += 1
            env.total_attempts += 1
            env._prev_illegal = True
            error = f"Couldn't parse: '{resp[:50]}'. Use format Move: X->Y"
            consec_fail += 1
            if condition == "conflict" and hook: hook.fire()
            if consec_fail >= 10: break
            continue

        ok, msg = env.try_move(move[0], move[1])
        if ok:
            error = None; consec_fail = 0
            if env.is_solved(): break
        else:
            error = msg; consec_fail += 1
            if condition == "conflict" and hook: hook.fire()
            if consec_fail >= 10: break

    st = env.stats()
    st["avg_entropy"] = float(np.mean(entropies)) if entropies else 0
    st["thoughts"] = thoughts[:3]
    st["raw_samples"] = raw_responses[:3]
    return st


def run_condition(model, tok, n_disks, modified, cond_name, n_trials=N_TRIALS):
    task = f"{'Mod' if modified else 'Std'}-{n_disks}"
    print(f"\n  🎮 {task} | {cond_name} | {n_trials} trials")

    layers = model.model.layers
    hook, handle = None, None
    if cond_name == "echo":
        hook = EchoHook(SIGMA_ECHO)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)
    elif cond_name == "conflict":
        hook = SpikeHook(SIGMA_SPIKE)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    results = []
    for t in range(n_trials):
        env = HanoiEnv(n_disks, modified)
        r = play_game(model, tok, env, cond_name, hook)
        results.append(r)
        icon = "✅" if r["solved"] else "❌"
        print(f"    {t+1:2d}: {icon} legal={r['legal_moves']} illegal={r['illegal_moves']} "
              f"self_corr={r['self_corrections']}")
        if hook and hasattr(hook, 'reset'): hook.reset()

    if handle: handle.remove()

    s = lambda key: round(np.mean([r[key] for r in results]), 1)
    sr = round(np.mean([r["solved"] for r in results]) * 100, 1)
    summary = {
        "task": task, "condition": cond_name, "n_trials": n_trials,
        "solve_rate": sr,
        "avg_legal": s("legal_moves"), "avg_illegal": s("illegal_moves"),
        "avg_self_corr": s("self_corrections"), "avg_entropy": s("avg_entropy"),
        "optimal": results[0]["optimal"],
        "thoughts": results[0].get("thoughts", []),
        "raw_samples": results[0].get("raw_samples", []),
    }
    print(f"    📊 Solve={sr:.0f}% Legal={summary['avg_legal']:.1f} "
          f"Illegal={summary['avg_illegal']:.1f}")
    return summary, results


# ═══════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════

def visualize(summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 50c: The Illusion Breaker v3\n"
                 "Chat Template + CoT → SNN Perturbation",
                 fontsize=14, fontweight="bold", y=0.98)

    conds = ["baseline", "echo", "conflict"]
    labels = ["Baseline", "L18 Echo\n(σ=0.02)", "Conflict\nTriggered"]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]

    tasks = list(dict.fromkeys(s["task"] for s in summaries))
    x = np.arange(len(tasks))
    w = 0.25

    def plot_bars(ax, metric, title, ylabel, fmt="{:.0f}%"):
        for i, (c, l, col) in enumerate(zip(conds, labels, colors)):
            vals = [next((s[metric] for s in summaries
                         if s["task"]==t and s["condition"]==c), 0) for t in tasks]
            bars = ax.bar(x + i*w, vals, w, label=l.replace('\n',' '),
                         color=col, alpha=0.85, edgecolor='white')
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                       fmt.format(v), ha='center', fontsize=9, fontweight='bold')
        ax.set_xticks(x + w)
        ax.set_xticklabels(tasks, fontsize=11)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plot_bars(axes[0,0], "solve_rate",
             "① Solve Rate", "Solve Rate (%)")
    plot_bars(axes[0,1], "avg_illegal",
             "② Illegal Moves", "Avg Illegal", fmt="{:.1f}")
    plot_bars(axes[1,0], "avg_self_corr",
             "③ Self-Corrections (Apple: 'cannot self-correct')",
             "Avg Self-Corrections", fmt="{:.1f}")
    plot_bars(axes[1,1], "avg_entropy",
             "④ Entropy (reasoning intensity)",
             "Avg Entropy", fmt="{:.2f}")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(FIGURES_DIR, "phase50c_chat_template.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure: {path}")
    return path


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tok = load_model()

    configs = [
        (3, False, "Std-3"),
        (3, True,  "Mod-3"),
        (4, False, "Std-4"),
    ]
    conds = ["baseline", "echo", "conflict"]

    all_sum = []
    all_det = {}

    for n, mod, name in configs:
        print(f"\n{'═'*60}")
        print(f"  🏰 {name} ({'modified' if mod else 'standard'} rules)")
        print(f"{'═'*60}")
        for c in conds:
            s, d = run_condition(model, tok, n, mod, c)
            all_sum.append(s)
            all_det[f"{name}_{c}"] = d

    elapsed = time.time() - t0
    fig = visualize(all_sum)

    out = {
        "experiment": "Phase 50c: Chat Template + CoT",
        "model": MODEL_SHORT,
        "key_fix": "Using tokenizer.apply_chat_template() instead of raw prompt",
        "elapsed_min": round(elapsed/60, 1),
        "summaries": all_sum,
        "figure": fig,
    }
    log = os.path.join(RESULTS_DIR, "phase50c_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Verdict
    print(f"\n{'═'*70}")
    print(f"  ⚡ PHASE 50c: CHAT TEMPLATE FIX — VERDICT")
    print(f"{'═'*70}")
    print(f"  Key fix: apply_chat_template() for Mistral-Instruct")
    print(f"{'─'*70}")
    print(f"  {'Task':<8} {'Cond':<12} {'Solve%':>8} {'Legal':>8} "
          f"{'Illegal':>8} {'SelfCorr':>8}")
    print(f"{'─'*70}")
    for s in all_sum:
        print(f"  {s['task']:<8} {s['condition']:<12} {s['solve_rate']:>7.0f}% "
              f"{s['avg_legal']:>8.1f} {s['avg_illegal']:>8.1f} "
              f"{s['avg_self_corr']:>8.1f}")
    print(f"{'─'*70}")

    # Show raw output samples
    print(f"\n  🧠 Raw LLM Output Samples:")
    for s in all_sum[:3]:
        if s.get("raw_samples"):
            print(f"    [{s['task']}/{s['condition']}]")
            for r in s["raw_samples"][:2]:
                print(f"      → {r}")

    # Compare with Phase 50
    print(f"\n  📊 Phase 50 → 50c Improvement:")
    for t in tasks_order(all_sum):
        base50c = next((s for s in all_sum if s["task"]==t and s["condition"]=="baseline"), None)
        if base50c:
            print(f"    {t}: {base50c['solve_rate']:.0f}% (was 0% in Phase 50)")

    print(f"\n  ⏱ {elapsed/60:.1f} min | 💾 {log}")
    print(f"{'═'*70}")


def tasks_order(sums):
    return list(dict.fromkeys(s["task"] for s in sums))


if __name__ == "__main__":
    main()
