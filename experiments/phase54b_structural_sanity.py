"""
Phase 54b: The Structural Sanity Check
========================================

Hypothesis:
  High-Temperature (1.2) breaks structured output (JSON formatting)
  while SNN Echo preserves structural integrity.

Design:
  - 3 conditions: Baseline (temp=0.5), High-Temp (temp=1.2), SNN Echo (σ=0.02, temp=0.5)
  - 20 structured extraction tasks per condition
  - Each task: extract info from text → output strict JSON
  - Evaluation: json.loads() success/failure (objective, zero ambiguity)

Expected: temp=1.2 → JSON parse errors (unclosed brackets, bad syntax)
"""

import torch
import os, json, gc, time, random, re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ═══ Config ═══
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
SEED = 2026
TARGET_LAYER = 18
SIGMA_ECHO = 0.02

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
#  STRUCTURED EXTRACTION TASKS (20 tasks)
# ═══════════════════════════════════════════════════

TASKS = [
    {
        "text": "John Smith is a 34-year-old software engineer from San Francisco. He works at Google.",
        "instruction": "Extract: name, age, job, city, company",
        "required_keys": ["name", "age", "job", "city", "company"],
    },
    {
        "text": "The iPhone 15 Pro costs $999, has 256GB storage, and comes in Titanium Blue.",
        "instruction": "Extract: product, price, storage, color",
        "required_keys": ["product", "price", "storage", "color"],
    },
    {
        "text": "Tokyo Tower was built in 1958, stands 333 meters tall, and is located in Minato, Tokyo.",
        "instruction": "Extract: name, year_built, height_meters, location",
        "required_keys": ["name", "year_built", "height_meters", "location"],
    },
    {
        "text": "Dr. Maria Garcia published 'Neural Networks in Medicine' in Nature, 2024. It has 45 citations.",
        "instruction": "Extract: author, title, journal, year, citations",
        "required_keys": ["author", "title", "journal", "year", "citations"],
    },
    {
        "text": "Flight AA123 departs from JFK at 14:30 and arrives at LAX at 17:45. Economy class costs $350.",
        "instruction": "Extract: flight_number, departure_airport, departure_time, arrival_airport, arrival_time, price",
        "required_keys": ["flight_number", "departure_airport", "arrival_airport", "price"],
    },
    {
        "text": "The recipe requires 2 cups flour, 1 cup sugar, 3 eggs, and 200ml milk. Bake at 180C for 25 minutes.",
        "instruction": "Extract: ingredients (as list), temperature_celsius, baking_time_minutes",
        "required_keys": ["ingredients", "temperature_celsius"],
    },
    {
        "text": "Tesla Model 3 Long Range: 0-60 mph in 4.2 seconds, 358 mile range, starting at $42,990.",
        "instruction": "Extract: car_model, acceleration_0_60, range_miles, price",
        "required_keys": ["car_model", "range_miles", "price"],
    },
    {
        "text": "Mount Everest, at 8,849 meters, is on the border of Nepal and Tibet. First summited in 1953 by Edmund Hillary.",
        "instruction": "Extract: mountain, height_meters, countries (list), first_summit_year, first_climber",
        "required_keys": ["mountain", "height_meters", "first_summit_year"],
    },
    {
        "text": "Python 3.12 was released on October 2, 2023. Key features include improved error messages and a new type system.",
        "instruction": "Extract: language, version, release_date, features (list)",
        "required_keys": ["language", "version", "release_date"],
    },
    {
        "text": "The Mona Lisa by Leonardo da Vinci, painted circa 1503-1519, measures 77cm x 53cm. Displayed at the Louvre, Paris.",
        "instruction": "Extract: title, artist, year_range, dimensions, museum, city",
        "required_keys": ["title", "artist", "museum"],
    },
    {
        "text": "Amazon stock (AMZN) closed at $178.25 on Feb 20, 2026. Market cap: $1.87 trillion. P/E ratio: 62.3.",
        "instruction": "Extract: company, ticker, price, date, market_cap, pe_ratio",
        "required_keys": ["ticker", "price", "market_cap"],
    },
    {
        "text": "Patient ID: P-4521. Blood pressure: 130/85. Heart rate: 72 bpm. Temperature: 37.2C. Diagnosis: hypertension stage 1.",
        "instruction": "Extract: patient_id, blood_pressure, heart_rate_bpm, temperature_c, diagnosis",
        "required_keys": ["patient_id", "diagnosis"],
    },
    {
        "text": "The concert is on March 15, 2026 at Madison Square Garden. Doors open at 7 PM. Tickets: $85-$250. Artist: Adele.",
        "instruction": "Extract: artist, date, venue, doors_open, ticket_price_range",
        "required_keys": ["artist", "date", "venue"],
    },
    {
        "text": "Server: us-east-1. CPU usage: 78%. Memory: 12.4GB/16GB. Disk: 450GB/1TB. Status: warning. Uptime: 45 days.",
        "instruction": "Extract: server, cpu_percent, memory_used_gb, memory_total_gb, disk_used_gb, status, uptime_days",
        "required_keys": ["server", "cpu_percent", "status"],
    },
    {
        "text": "The Great Wall of China stretches 21,196 km. Construction began in 7th century BC. UNESCO World Heritage Site since 1987.",
        "instruction": "Extract: name, length_km, construction_start, unesco_year",
        "required_keys": ["name", "length_km", "unesco_year"],
    },
    {
        "text": "Order #ORD-2026-789: 3x Widget A ($12.99 each), 1x Widget B ($24.50). Subtotal: $63.47. Tax: $5.08. Total: $68.55.",
        "instruction": "Extract: order_id, items (list with name, quantity, unit_price), subtotal, tax, total",
        "required_keys": ["order_id", "subtotal", "total"],
    },
    {
        "text": "Hurricane Katrina (Category 5) made landfall on August 29, 2005. Wind speed: 280 km/h. Damage: $125 billion.",
        "instruction": "Extract: name, category, date, wind_speed_kmh, damage_billion_usd",
        "required_keys": ["name", "category", "wind_speed_kmh"],
    },
    {
        "text": "WiFi network 'Office-5G' uses WPA3 encryption. Channel: 36. Frequency: 5GHz. Connected devices: 23. Signal: -42 dBm.",
        "instruction": "Extract: network_name, encryption, channel, frequency, connected_devices, signal_dbm",
        "required_keys": ["network_name", "encryption", "connected_devices"],
    },
    {
        "text": "Professor Tanaka teaches CS101 (Intro to Programming) on Mondays and Wednesdays, 10:00-11:30, Room 302, Building A.",
        "instruction": "Extract: professor, course_code, course_name, days (list), time, room, building",
        "required_keys": ["professor", "course_code", "course_name"],
    },
    {
        "text": "SpaceX Falcon 9 launched Starlink batch 7-10 on Feb 18, 2026 from Cape Canaveral. 23 satellites deployed. Booster B1081 landed successfully.",
        "instruction": "Extract: rocket, mission, launch_date, launch_site, satellites_deployed, booster_id, landing_success",
        "required_keys": ["rocket", "mission", "launch_date"],
    },
]


# ═══════════════════════════════════════════════════
#  SNN ECHO HOOK
# ═══════════════════════════════════════════════════

class EchoHook:
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, module, args):
        hs = args[0]
        noise = torch.randn_like(hs) * self.sigma
        return (hs + noise,) + args[1:]


# ═══════════════════════════════════════════════════
#  MODEL & GENERATION
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
    print(f"  ✅ Loaded ({len(model.model.layers)} layers)")
    return model, tok


def generate(model, tok, prompt, temperature=0.5, max_tokens=200):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperature, top_p=0.9, top_k=40,
            repetition_penalty=1.2, pad_token_id=tok.pad_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    inp = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(inp):].strip()


def build_json_prompt(tokenizer, task):
    messages = [
        {"role": "user", "content": (
            f"Extract the requested information from the text below and output ONLY valid JSON. "
            f"No explanation, no markdown, no extra text. Just the JSON object.\n\n"
            f"Text: \"{task['text']}\"\n"
            f"Extract: {task['instruction']}\n\n"
            f"Output ONLY the JSON:"
        )}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate_json(response, required_keys):
    """Evaluate JSON validity and key presence. Returns (valid_json, has_keys, parsed)."""
    # Try to extract JSON from response
    resp = response.strip()
    
    # Try to find JSON block
    # Sometimes model wraps in ```json ... ```
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', resp, re.DOTALL)
    if m:
        resp = m.group(1)
    else:
        # Find first { to last }
        start = resp.find('{')
        end = resp.rfind('}')
        if start != -1 and end != -1 and end > start:
            resp = resp[start:end+1]
    
    # Try parsing
    try:
        parsed = json.loads(resp)
        valid_json = True
    except (json.JSONDecodeError, ValueError):
        return False, False, None
    
    # Check required keys (case-insensitive)
    parsed_keys_lower = {k.lower() for k in parsed.keys()}
    has_all_keys = all(k.lower() in parsed_keys_lower for k in required_keys)
    
    return valid_json, has_all_keys, parsed


# ═══════════════════════════════════════════════════
#  RUN CONDITION
# ═══════════════════════════════════════════════════

def run_condition(model, tok, cond_name, temperature, tasks):
    print(f"\n  📝 {cond_name} | temp={temperature} | {len(tasks)} tasks")

    layers = model.model.layers
    hook, handle = None, None
    if cond_name == "snn_echo":
        hook = EchoHook(SIGMA_ECHO)
        handle = layers[TARGET_LAYER].register_forward_pre_hook(hook)

    results = []
    n_valid = 0
    n_keys = 0
    
    for i, task in enumerate(tasks):
        prompt = build_json_prompt(tok, task)
        response = generate(model, tok, prompt, temperature=temperature)

        valid_json, has_keys, parsed = evaluate_json(response, task["required_keys"])
        if valid_json: n_valid += 1
        if has_keys: n_keys += 1
        
        icon_json = "✅" if valid_json else "❌"
        icon_keys = "🔑" if has_keys else "⚠️"

        results.append({
            "task_idx": i,
            "instruction": task["instruction"][:50],
            "response": response[:150],
            "valid_json": valid_json,
            "has_required_keys": has_keys,
            "parsed": parsed,
        })

        print(f"    {i+1:2d}/{len(tasks)}: JSON:{icon_json} Keys:{icon_keys} "
              f"[{n_valid}/{i+1}={n_valid/(i+1)*100:.0f}%] "
              f"R: {response[:60].replace(chr(10), ' ')}")

    if handle: handle.remove()

    json_rate = n_valid / len(tasks) * 100
    key_rate = n_keys / len(tasks) * 100

    summary = {
        "condition": cond_name,
        "temperature": temperature,
        "n_tasks": len(tasks),
        "n_valid_json": n_valid,
        "json_validity_rate": round(json_rate, 1),
        "n_has_keys": n_keys,
        "key_completeness_rate": round(key_rate, 1),
    }

    print(f"    📊 JSON Valid: {json_rate:.0f}% ({n_valid}/{len(tasks)}) | "
          f"Keys Complete: {key_rate:.0f}% ({n_keys}/{len(tasks)})")
    return summary, results


# ═══════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════

def visualize(summaries, phase51_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Phase 54b: The Structural Sanity Check\n"
                 "Can High-Temperature maintain structured output?",
                 fontsize=14, fontweight="bold", y=1.02)

    conds = ["baseline", "high_temp", "snn_echo"]
    labels = ["Baseline\n(temp=0.5)", "High-Temp\n(temp=1.2)", "SNN Echo\n(σ=0.02)"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    # Panel 1: Hanoi Solve Rate (from Phase 51)
    ax = axes[0]
    hanoi_rates = [phase51_data[c]["solve_rate"] for c in conds]
    bars = ax.bar(range(3), hanoi_rates, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    for b, sr in zip(bars, hanoi_rates):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
               f"{sr:.0f}%", ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("① Hanoi Solve Rate\n(Phase 51, N=50)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 40); ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: JSON Validity Rate
    ax = axes[1]
    json_rates = [next(s["json_validity_rate"] for s in summaries if s["condition"]==c) for c in conds]
    bars = ax.bar(range(3), json_rates, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    for b, sr in zip(bars, json_rates):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
               f"{sr:.0f}%", ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("JSON Valid (%)", fontsize=12)
    ax.set_title("② JSON Structural Integrity\n(20 extraction tasks)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110); ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 3: Trade-off scatter
    ax = axes[2]
    for i, (c, label) in enumerate(zip(conds, ["Baseline", "High-Temp", "SNN Echo"])):
        ax.scatter(hanoi_rates[i], json_rates[i], c=colors[i], s=300, zorder=5,
                  edgecolors='black', linewidth=1.5)
        offset_y = -8 if i == 1 else 5
        ax.annotate(label, (hanoi_rates[i], json_rates[i]),
                   textcoords="offset points", xytext=(0, offset_y+8),
                   fontsize=12, fontweight='bold', ha='center')

    ax.set_xlabel("Hanoi Solve Rate (%)", fontsize=12)
    ax.set_ylabel("JSON Validity Rate (%)", fontsize=12)
    ax.set_title("③ Creativity vs. Structure\n(Top-right = ideal)", fontsize=13, fontweight='bold')
    ax.set_xlim(5, 35); ax.set_ylim(40, 110)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    from matplotlib.patches import Rectangle
    ideal = Rectangle((15, 85), 20, 25, linewidth=2, edgecolor='green',
                      facecolor='green', alpha=0.1)
    ax.add_patch(ideal)
    ax.text(25, 105, "IDEAL\nZONE", ha='center', fontsize=10, color='green', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGURES_DIR, "phase54b_structural_sanity.png")
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

    print(f"\n{'═'*60}")
    print(f"  🧪 Phase 54b: The Structural Sanity Check")
    print(f"  Does High-Temperature break JSON output?")
    print(f"{'═'*60}")

    conditions = [
        ("baseline", 0.5),
        ("high_temp", 1.2),
        ("snn_echo", 0.5),
    ]

    all_summaries = []
    all_results = {}

    for cond_name, temp in conditions:
        summary, results = run_condition(model, tok, cond_name, temp, TASKS)
        all_summaries.append(summary)
        all_results[cond_name] = results
        gc.collect(); torch.cuda.empty_cache()

    elapsed = time.time() - t0

    phase51_data = {
        "baseline": {"solve_rate": 16.0},
        "high_temp": {"solve_rate": 28.0},
        "snn_echo": {"solve_rate": 22.0},
    }

    fig_path = visualize(all_summaries, phase51_data)

    out = {
        "experiment": "Phase 54b: The Structural Sanity Check",
        "model": MODEL_NAME,
        "purpose": "Test JSON structural integrity under different temperature/SNN conditions",
        "elapsed_min": round(elapsed / 60, 1),
        "phase51_hanoi": phase51_data,
        "structural_results": all_summaries,
        "detailed_results": {k: v for k, v in all_results.items()},
        "figure": fig_path,
    }
    log = os.path.join(RESULTS_DIR, "phase54b_structural_log.json")
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Verdict
    print(f"\n{'═'*70}")
    print(f"  🧪 PHASE 54b: STRUCTURAL SANITY CHECK — VERDICT")
    print(f"{'═'*70}")
    print(f"  {'Condition':<15} {'Hanoi%':>8} {'JSON Valid%':>12} {'Keys OK%':>10}")
    print(f"{'─'*70}")
    for s in all_summaries:
        hanoi = phase51_data[s["condition"]]["solve_rate"]
        print(f"  {s['condition']:<15} {hanoi:>7.0f}% {s['json_validity_rate']:>11.0f}% {s['key_completeness_rate']:>9.0f}%")
    print(f"{'─'*70}")

    bl = next(s for s in all_summaries if s["condition"] == "baseline")
    ht = next(s for s in all_summaries if s["condition"] == "high_temp")
    sn = next(s for s in all_summaries if s["condition"] == "snn_echo")

    print(f"\n  🏆 ANALYSIS:")
    if ht["json_validity_rate"] < bl["json_validity_rate"] - 5:
        diff = bl["json_validity_rate"] - ht["json_validity_rate"]
        print(f"  🔥 High-Temp LOSES {diff:.0f}% JSON validity!")
        print(f"     → Temperature=1.2 breaks structured output")
        if sn["json_validity_rate"] >= bl["json_validity_rate"] - 3:
            print(f"  ✅ SNN Echo PRESERVES structure ({sn['json_validity_rate']:.0f}%)")
            print(f"     → 'Sober creativity': reasoning WITHOUT losing structural integrity")
            print(f"\n  🎯 THIS IS THE v10 RESULT!")
    elif ht["json_validity_rate"] >= bl["json_validity_rate"] - 5:
        print(f"  ⚠️ High-Temp maintains JSON validity ({ht['json_validity_rate']:.0f}%)")
        print(f"     → Mistral-7B's JSON generation is temperature-robust")
        print(f"     → Need even harder structural tasks")

    print(f"\n  ⏱ {elapsed/60:.1f} min | 💾 {log}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
