"""
Debug: See what the LLM actually outputs for Hanoi prompts.
Saves output to file for clean reading.
"""
import torch, os, json, warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "debug_hanoi_raw_output.json")

print("Loading model...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
model.eval()
print("Model loaded")

results = []

# === TEST 1: Raw prompt (no chat template) ===
raw_prompt = """Tower of Hanoi — 3 disks
STANDARD RULES: You can ONLY place a SMALLER disk onto a LARGER disk.
Goal: Move ALL disks from A to C.

Current state:
  A: [3, 2, 1]
  B: []
  C: []

Legal moves: A->B, A->C

Think step-by-step, then give your move.
Format: Move: X->Y"""

for trial in range(2):
    inputs = tokenizer(raw_prompt, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                             temperature=0.6, top_p=0.85, repetition_penalty=1.3,
                             pad_token_id=tokenizer.pad_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    response = full[len(input_text):]
    results.append({"type": "raw_prompt", "trial": trial+1, "response": response})
    print(f"  raw trial {trial+1} done: {len(response)} chars")

# === TEST 2: Chat template ===
messages = [{"role": "user", "content": raw_prompt}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

for trial in range(2):
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                             temperature=0.6, top_p=0.85, repetition_penalty=1.3,
                             pad_token_id=tokenizer.pad_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    response = full[len(input_text):]
    results.append({"type": "chat_template", "trial": trial+1, "response": response})
    print(f"  chat trial {trial+1} done: {len(response)} chars")

# === TEST 3: Very simple prompt ===
simple = """Solve Tower of Hanoi with 3 disks.
Pegs: A (has disks 3,2,1), B (empty), C (empty).
Move smallest disk first. Give one move:
Move: """

for trial in range(2):
    inputs = tokenizer(simple, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=True,
                             temperature=0.6, top_p=0.85, repetition_penalty=1.3,
                             pad_token_id=tokenizer.pad_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    response = full[len(input_text):]
    results.append({"type": "simple_prompt", "trial": trial+1, "response": response})
    print(f"  simple trial {trial+1} done: {len(response)} chars")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Saved to {OUT_FILE}")
