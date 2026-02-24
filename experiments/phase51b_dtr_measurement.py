"""
Phase 51b: Deep-Thinking Ratio (DTR) Measurement on Tower of Hanoi
===================================================================

Inspired by "Think Deep, Not Just Long" (arXiv:2602.13517)

Key idea:
  - For each generated token, measure HOW DEEP the model "thinks"
  - Shallow thinking: prediction converges early (low layers) → routine/intuitive
  - Deep thinking: prediction keeps changing until final layers → effortful reasoning
  - DTR = fraction of tokens that are "deep-thinking"

Hypothesis:
  Trials that SOLVE Modified-3 Hanoi will have higher DTR (more deep thinking)
  than trials that FAIL.

Method:
  1. Hook all 32 layers of Mistral-7B
  2. For each generated token, project hidden states at each layer through lm_head
  3. Compute JSD between consecutive layer distributions
  4. "Convergence depth" = last layer where JSD > threshold
  5. Token is "deep-thinking" if convergence depth > N_LAYERS * 0.7

Usage:
    python experiments/phase51b_dtr_measurement.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, json, gc, time, random, re, copy
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ═══ Config ═══
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
SEED = 2026
N_TRIALS = 20  # Fewer trials since DTR measurement is compute-heavy
MAX_STEPS = 50
MAX_GEN_TOKENS = 80
DTR_THRESHOLD_RATIO = 0.7  # Token is "deep" if convergence > 70% of layers
JSD_THRESHOLD = 0.01  # JSD below this = "converged"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
#  HANOI ENVIRONMENT
# ═══════════════════════════════════════════════════

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
        }


# ═══════════════════════════════════════════════════
#  PROMPT BUILDER
# ═══════════════════════════════════════════════════

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
    m = re.search(r'MOVE:\s*([ABC])\s*->\s*([ABC])', resp)
    if m: return m.group(1), m.group(2)
    m = re.search(r'([ABC])\s*->\s*([ABC])', resp)
    if m: return m.group(1), m.group(2)
    m = re.search(r'MOVE\s+DISK\s+\d+\s+FROM\s+([ABC])\s+TO\s+([ABC])', resp)
    if m: return m.group(1), m.group(2)
    m = re.search(r'FROM\s+([ABC])\s+TO\s+([ABC])', resp)
    if m: return m.group(1), m.group(2)
    letters = re.findall(r'[ABC]', resp)
    if len(letters) >= 2 and letters[0] != letters[1]:
        return letters[0], letters[1]
    return None


# ═══════════════════════════════════════════════════
#  DTR ENGINE — Layer-wise JSD Measurement
# ═══════════════════════════════════════════════════

class DTREngine:
    """
    Measures Deep-Thinking Ratio by computing JSD between consecutive layers.
    
    For each generated token:
    1. Capture hidden states at all 32 layers
    2. Project through lm_head to get logit distributions
    3. Compute JSD between consecutive layer distributions
    4. Find "convergence depth" — last layer where JSD > threshold
    5. Token is "deep-thinking" if conv_depth > 70% of total layers
    """
    
    def __init__(self, model, n_layers=32, jsd_threshold=JSD_THRESHOLD, 
                 depth_ratio=DTR_THRESHOLD_RATIO):
        self.model = model
        self.n_layers = n_layers
        self.jsd_threshold = jsd_threshold
        self.depth_ratio = depth_ratio
        self.deep_threshold = int(n_layers * depth_ratio)  # e.g., 22 for 32 layers
        
        # Storage for hidden states (captured by hooks)
        self.layer_hidden_states = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks on all transformer layers."""
        self.hooks = []
        for i, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self.hooks.append(hook)
        
    def _make_hook(self, layer_idx):
        def hook_fn(module, inputs, outputs):
            # outputs can be a tuple (hidden_states, ...) or just the tensor
            if isinstance(outputs, tuple):
                hs = outputs[0]
            else:
                hs = outputs
            # hs shape: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            if hs.dim() == 3:
                hs = hs[:, -1, :].detach()  # Take last token
            elif hs.dim() == 2:
                hs = hs.detach()  # Already (batch, hidden_dim)
            else:
                hs = hs.reshape(1, -1).detach()
            self.layer_hidden_states[layer_idx] = hs
        return hook_fn
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def compute_token_depth(self):
        """
        Compute convergence depth for the most recently generated token.
        Returns: (convergence_depth, layer_jsds, is_deep)
        """
        if len(self.layer_hidden_states) < 2:
            return 0, [], False
        
        # Project each layer's hidden state through lm_head
        # Use only top-k logits to save memory
        TOP_K = 100
        lm_head = self.model.lm_head
        
        layer_jsds = []
        last_significant_layer = 0
        
        prev_probs = None
        for layer_idx in range(self.n_layers):
            if layer_idx not in self.layer_hidden_states:
                layer_jsds.append(0.0)
                continue
                
            hs = self.layer_hidden_states[layer_idx]
            
            # Apply RMSNorm before lm_head (Mistral uses RMSNorm)
            if hasattr(self.model.model, 'norm'):
                hs = self.model.model.norm(hs.to(self.model.model.norm.weight.dtype))
            
            # Cast to lm_head weight dtype for compatibility with quantized models
            lm_dtype = lm_head.weight.dtype
            with torch.no_grad():
                logits = lm_head(hs.to(lm_dtype))  # (1, vocab_size)
                # Top-k for efficiency
                topk_logits, topk_indices = torch.topk(logits[0], TOP_K)
                probs = F.softmax(topk_logits.float(), dim=-1)
            
            if prev_probs is not None and prev_probs.shape == probs.shape:
                # Jensen-Shannon Divergence
                m = 0.5 * (prev_probs + probs)
                kl1 = F.kl_div(m.log(), prev_probs, reduction='sum', log_target=False)
                kl2 = F.kl_div(m.log(), probs, reduction='sum', log_target=False)
                jsd = (0.5 * (kl1 + kl2)).item()
                jsd = max(0, jsd)  # Numerical stability
                layer_jsds.append(jsd)
                
                if jsd > self.jsd_threshold:
                    last_significant_layer = layer_idx
            else:
                layer_jsds.append(0.0)
            
            prev_probs = probs
        
        # Clear stored states
        self.layer_hidden_states.clear()
        
        # Convergence depth = last layer with significant JSD
        conv_depth = last_significant_layer
        is_deep = conv_depth >= self.deep_threshold
        
        return conv_depth, layer_jsds, is_deep
    
    def clear(self):
        self.layer_hidden_states.clear()


# ═══════════════════════════════════════════════════
#  MODEL LOADING
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


# ═══════════════════════════════════════════════════
#  GENERATION WITH DTR MEASUREMENT
# ═══════════════════════════════════════════════════

def generate_with_dtr(model, tok, prompt, dtr_engine, temperature=0.5, max_tokens=80):
    """
    Manual autoregressive generation with per-token DTR measurement.
    """
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    input_ids = inputs["input_ids"]
    
    token_depths = []
    token_jsds_all = []
    generated_ids = []
    
    dtr_engine.register_hooks()
    
    try:
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            
            # Measure DTR for this token position
            conv_depth, layer_jsds, is_deep = dtr_engine.compute_token_depth()
            token_depths.append(conv_depth)
            token_jsds_all.append(layer_jsds)
            
            # Sample next token
            logits = logits / temperature
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            # Repetition penalty
            for prev_id in generated_ids[-20:]:
                logits[0, prev_id] /= 1.2
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop at EOS
            if next_token.item() == tok.eos_token_id:
                break
    finally:
        dtr_engine.remove_hooks()
    
    response = tok.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Compute DTR metrics
    n_tokens = len(token_depths)
    n_deep = sum(1 for d in token_depths if d >= dtr_engine.deep_threshold)
    dtr = n_deep / n_tokens if n_tokens > 0 else 0
    avg_depth = np.mean(token_depths) if token_depths else 0
    max_depth = max(token_depths) if token_depths else 0
    
    dtr_metrics = {
        "n_tokens": n_tokens,
        "n_deep_tokens": n_deep,
        "dtr": round(dtr, 4),
        "avg_convergence_depth": round(float(avg_depth), 2),
        "max_convergence_depth": int(max_depth),
        "depth_distribution": token_depths,
    }
    
    return response, dtr_metrics


# ═══════════════════════════════════════════════════
#  GAME LOOP WITH DTR
# ═══════════════════════════════════════════════════

def play_game_with_dtr(model, tok, env, dtr_engine):
    env.reset()
    error = None
    consec_fail = 0
    step_dtr_metrics = []

    for step in range(MAX_STEPS):
        prompt = build_chat_prompt(tok, env, error)
        resp, dtr_m = generate_with_dtr(model, tok, prompt, dtr_engine)
        
        # Track whether this step's move was legal
        move = parse_move(resp)
        move_legal = False
        
        if move is None:
            env.illegal_count += 1
            env.total_attempts += 1
            env._prev_illegal = True
            error = f"Couldn't parse: '{resp[:50]}'. Use format Move: X->Y"
            consec_fail += 1
            if consec_fail >= 10: break
        else:
            ok, msg = env.try_move(move[0], move[1])
            if ok:
                error = None; consec_fail = 0; move_legal = True
                if env.is_solved(): 
                    dtr_m["move_legal"] = True
                    dtr_m["step"] = step
                    step_dtr_metrics.append(dtr_m)
                    break
            else:
                error = msg; consec_fail += 1
                if consec_fail >= 10: break
        
        dtr_m["move_legal"] = move_legal
        dtr_m["step"] = step
        step_dtr_metrics.append(dtr_m)

    stats = env.stats()
    
    # Aggregate DTR across all steps
    all_dtrs = [m["dtr"] for m in step_dtr_metrics]
    legal_dtrs = [m["dtr"] for m in step_dtr_metrics if m["move_legal"]]
    illegal_dtrs = [m["dtr"] for m in step_dtr_metrics if not m["move_legal"]]
    all_depths = [m["avg_convergence_depth"] for m in step_dtr_metrics]
    
    stats["dtr_overall"] = round(float(np.mean(all_dtrs)), 4) if all_dtrs else 0
    stats["dtr_legal_moves"] = round(float(np.mean(legal_dtrs)), 4) if legal_dtrs else 0
    stats["dtr_illegal_moves"] = round(float(np.mean(illegal_dtrs)), 4) if illegal_dtrs else 0
    stats["avg_depth_overall"] = round(float(np.mean(all_depths)), 2) if all_depths else 0
    stats["n_steps"] = len(step_dtr_metrics)
    stats["step_details"] = step_dtr_metrics
    
    return stats


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model, tok = load_model()
    n_layers = len(model.model.layers)
    print(f"  🧠 DTR Engine: {n_layers} layers, deep threshold = layer {int(n_layers * DTR_THRESHOLD_RATIO)}")
    
    dtr_engine = DTREngine(model, n_layers=n_layers)
    
    print(f"\n{'═'*60}")
    print(f"  🏰 Modified-3 Hanoi | DTR Measurement | {N_TRIALS} trials")
    print(f"{'═'*60}")
    
    results = []
    solved_count = 0
    
    for t in range(N_TRIALS):
        env = HanoiEnv(3, modified=True)
        r = play_game_with_dtr(model, tok, env, dtr_engine)
        results.append(r)
        if r["solved"]: solved_count += 1
        
        icon = "✅" if r["solved"] else "❌"
        print(f"  {t+1:2d}/{N_TRIALS}: {icon} legal={r['legal_moves']:2d} "
              f"illegal={r['illegal_moves']:2d} self_corr={r['self_corrections']:2d} | "
              f"DTR={r['dtr_overall']:.3f} AvgDepth={r['avg_depth_overall']:.1f} "
              f"[{solved_count}/{t+1}={solved_count/(t+1)*100:.0f}%]")
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    
    # ═══ ANALYSIS: Solved vs Unsolved ═══
    solved_trials = [r for r in results if r["solved"]]
    failed_trials = [r for r in results if not r["solved"]]
    
    analysis = {
        "solved": {
            "n": len(solved_trials),
            "avg_dtr": round(float(np.mean([r["dtr_overall"] for r in solved_trials])), 4) if solved_trials else 0,
            "avg_depth": round(float(np.mean([r["avg_depth_overall"] for r in solved_trials])), 2) if solved_trials else 0,
            "avg_dtr_legal": round(float(np.mean([r["dtr_legal_moves"] for r in solved_trials])), 4) if solved_trials else 0,
            "avg_dtr_illegal": round(float(np.mean([r["dtr_illegal_moves"] for r in solved_trials])), 4) if solved_trials else 0,
        },
        "failed": {
            "n": len(failed_trials),
            "avg_dtr": round(float(np.mean([r["dtr_overall"] for r in failed_trials])), 4) if failed_trials else 0,
            "avg_depth": round(float(np.mean([r["avg_depth_overall"] for r in failed_trials])), 2) if failed_trials else 0,
            "avg_dtr_legal": round(float(np.mean([r["dtr_legal_moves"] for r in failed_trials])), 4) if failed_trials else 0,
            "avg_dtr_illegal": round(float(np.mean([r["dtr_illegal_moves"] for r in failed_trials])), 4) if failed_trials else 0,
        },
    }
    
    # Statistical test: Solved DTR vs Failed DTR
    if solved_trials and failed_trials:
        solved_dtrs = [r["dtr_overall"] for r in solved_trials]
        failed_dtrs = [r["dtr_overall"] for r in failed_trials]
        try:
            from scipy.stats import mannwhitneyu
            u_stat, p_val = mannwhitneyu(solved_dtrs, failed_dtrs, alternative='two-sided')
            analysis["mann_whitney"] = {
                "u_statistic": round(float(u_stat), 2),
                "p_value": round(float(p_val), 6),
                "significant": p_val < 0.05,
            }
        except ImportError:
            # Manual comparison
            analysis["mann_whitney"] = {"note": "scipy not available"}
        
        # Also test: legal move DTR vs illegal move DTR (across all trials)
        all_legal_dtrs = [r["dtr_legal_moves"] for r in results if r["dtr_legal_moves"] > 0]
        all_illegal_dtrs = [r["dtr_illegal_moves"] for r in results if r["dtr_illegal_moves"] > 0]
        try:
            from scipy.stats import mannwhitneyu
            if all_legal_dtrs and all_illegal_dtrs:
                u2, p2 = mannwhitneyu(all_legal_dtrs, all_illegal_dtrs, alternative='two-sided')
                analysis["legal_vs_illegal_dtr"] = {
                    "avg_legal_dtr": round(float(np.mean(all_legal_dtrs)), 4),
                    "avg_illegal_dtr": round(float(np.mean(all_illegal_dtrs)), 4),
                    "u_statistic": round(float(u2), 2),
                    "p_value": round(float(p2), 6),
                    "significant": p2 < 0.05,
                }
        except ImportError:
            pass

    # ═══ VISUALIZATION ═══
    fig_path = visualize(results, analysis)
    
    # ═══ SAVE ═══
    # Strip depth_distribution from step_details to save space
    save_results = []
    for r in results:
        r_copy = dict(r)
        if "step_details" in r_copy:
            for sd in r_copy["step_details"]:
                sd.pop("depth_distribution", None)
        save_results.append(r_copy)
    
    out = {
        "experiment": "Phase 51b: DTR Measurement",
        "model": MODEL_SHORT,
        "task": "Modified-3 Hanoi",
        "n_trials": N_TRIALS,
        "n_layers": n_layers,
        "jsd_threshold": JSD_THRESHOLD,
        "deep_threshold_layer": int(n_layers * DTR_THRESHOLD_RATIO),
        "elapsed_min": round(elapsed / 60, 1),
        "analysis": analysis,
        "figure": fig_path,
        "trial_summaries": [{
            "trial": i+1,
            "solved": r["solved"],
            "legal_moves": r["legal_moves"],
            "illegal_moves": r["illegal_moves"],
            "self_corrections": r["self_corrections"],
            "dtr_overall": r["dtr_overall"],
            "dtr_legal": r["dtr_legal_moves"],
            "dtr_illegal": r["dtr_illegal_moves"],
            "avg_depth": r["avg_depth_overall"],
        } for i, r in enumerate(results)],
    }
    log = os.path.join(RESULTS_DIR, "phase51b_dtr_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    
    # ═══ VERDICT ═══
    print(f"\n{'═'*70}")
    print(f"  🧠 PHASE 51b: DEEP-THINKING RATIO — VERDICT")
    print(f"{'═'*70}")
    print(f"  Task: Modified-3 Hanoi | N={N_TRIALS}")
    print(f"  Deep threshold: layer {int(n_layers * DTR_THRESHOLD_RATIO)} / {n_layers}")
    print(f"{'─'*70}")
    print(f"  {'Group':<10} {'N':>4} {'DTR':>8} {'AvgDepth':>10} {'DTR(legal)':>12} {'DTR(illegal)':>14}")
    print(f"{'─'*70}")
    print(f"  {'SOLVED':<10} {analysis['solved']['n']:>4} "
          f"{analysis['solved']['avg_dtr']:>8.4f} "
          f"{analysis['solved']['avg_depth']:>10.2f} "
          f"{analysis['solved']['avg_dtr_legal']:>12.4f} "
          f"{analysis['solved']['avg_dtr_illegal']:>14.4f}")
    print(f"  {'FAILED':<10} {analysis['failed']['n']:>4} "
          f"{analysis['failed']['avg_dtr']:>8.4f} "
          f"{analysis['failed']['avg_depth']:>10.2f} "
          f"{analysis['failed']['avg_dtr_legal']:>12.4f} "
          f"{analysis['failed']['avg_dtr_illegal']:>14.4f}")
    print(f"{'─'*70}")
    
    if "mann_whitney" in analysis and "p_value" in analysis["mann_whitney"]:
        p = analysis["mann_whitney"]["p_value"]
        sig = "★★★" if p < 0.001 else ("★★" if p < 0.01 else ("★" if p < 0.05 else "n.s."))
        print(f"  Mann-Whitney U test (Solved vs Failed DTR): p={p:.6f} {sig}")
    
    if "legal_vs_illegal_dtr" in analysis:
        la = analysis["legal_vs_illegal_dtr"]
        p = la["p_value"]
        sig = "★★★" if p < 0.001 else ("★★" if p < 0.01 else ("★" if p < 0.05 else "n.s."))
        print(f"  Legal vs Illegal DTR: {la['avg_legal_dtr']:.4f} vs {la['avg_illegal_dtr']:.4f} p={p:.6f} {sig}")
    
    print(f"\n  ⏱ {elapsed/60:.1f} min | 💾 {log}")
    print(f"{'═'*70}")


def visualize(results, analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 51b: Deep-Thinking Ratio on Modified-3 Hanoi\n"
                 "Do solved trials show deeper thinking?",
                 fontsize=14, fontweight="bold", y=0.98)
    
    solved = [r for r in results if r["solved"]]
    failed = [r for r in results if not r["solved"]]
    
    # Panel 1: DTR comparison (Solved vs Failed)
    ax = axes[0, 0]
    groups = ["Solved", "Failed"]
    dtrs = [
        analysis["solved"]["avg_dtr"],
        analysis["failed"]["avg_dtr"],
    ]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(groups, dtrs, color=colors, alpha=0.85, edgecolor="white", linewidth=2)
    for b, v in zip(bars, dtrs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
               f"{v:.4f}", ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel("Deep-Thinking Ratio", fontsize=12)
    ax.set_title("① DTR: Solved vs Failed", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 2: Average convergence depth
    ax = axes[0, 1]
    depths = [
        analysis["solved"]["avg_depth"],
        analysis["failed"]["avg_depth"],
    ]
    bars = ax.bar(groups, depths, color=colors, alpha=0.85, edgecolor="white", linewidth=2)
    for b, v in zip(bars, depths):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
               f"{v:.1f}", ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel("Avg Convergence Depth (layer)", fontsize=12)
    ax.set_title("② Convergence Depth: Solved vs Failed", fontsize=13, fontweight="bold")
    ax.axhline(y=int(32 * DTR_THRESHOLD_RATIO), color='red', linestyle='--', alpha=0.5, 
               label=f'Deep threshold (L{int(32*DTR_THRESHOLD_RATIO)})')
    ax.legend(); ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 3: DTR per trial (scatter)
    ax = axes[1, 0]
    for i, r in enumerate(results):
        color = "#2ecc71" if r["solved"] else "#e74c3c"
        marker = "★" if r["solved"] else "●"
        ax.scatter(i+1, r["dtr_overall"], c=color, s=80, alpha=0.7, edgecolors='white')
    ax.set_xlabel("Trial #", fontsize=12)
    ax.set_ylabel("DTR", fontsize=12)
    ax.set_title("③ DTR per Trial (green=solved, red=failed)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 4: Legal vs Illegal move DTR
    ax = axes[1, 1]
    legal_dtrs = [r["dtr_legal_moves"] for r in results if r["dtr_legal_moves"] > 0]
    illegal_dtrs = [r["dtr_illegal_moves"] for r in results if r["dtr_illegal_moves"] > 0]
    data = [legal_dtrs, illegal_dtrs]
    bp = ax.boxplot(data, labels=["Legal Moves", "Illegal Moves"],
                    patch_artist=True, medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_ylabel("DTR", fontsize=12)
    ax.set_title("④ DTR: Legal vs Illegal Moves", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(FIGURES_DIR, "phase51b_dtr_measurement.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure: {path}")
    return path


if __name__ == "__main__":
    main()
