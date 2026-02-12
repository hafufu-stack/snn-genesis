"""Quick diagnostic: check training format vs evaluation format."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase5_scaleup import (
    load_model, generate_text, evaluate_accuracy,
    CLEAN_QUESTIONS, build_test_set, build_nightmare_questions,
    heal_nightmare, classify_nightmare,
    LORA_R, LORA_ALPHA, MAX_LENGTH, BATCH_SIZE
)
import torch

print("=" * 60)
print("DIAGNOSTIC: Train-Test Format Alignment Check")
print("=" * 60)

model, tokenizer = load_model()
test_clean, test_nightmare = build_test_set()

# 1. Baseline: evaluate with base model (no LoRA)
print("\n--- Step 1: Baseline (no LoRA) ---")
for i, q in enumerate(test_clean[:3]):
    resp = generate_text(model, tokenizer, q["q"], max_new=50)
    print(f"  Q: {q['q']}")
    print(f"  Expected: {q['a']}")
    print(f"  Got: {resp[:150]}")
    answer_words = q["a"].lower().split()
    hit = any(w in resp.lower() for w in answer_words if len(w) > 2)
    print(f"  Match: {hit}")
    print()

# 2. Show the chat template format
print("\n--- Step 2: Training Data Format ---")
sample_q = CLEAN_QUESTIONS[0]
chat = [
    {"role": "user", "content": sample_q["q"]},
    {"role": "assistant", "content": sample_q["a"]},
]
template_text = tokenizer.apply_chat_template(chat, tokenize=False)
print(f"  apply_chat_template output:")
print(f"  {repr(template_text[:300])}")

# Also show what generate_text sends to the model
eval_chat = [{"role": "user", "content": sample_q["q"]}]
eval_text = tokenizer.apply_chat_template(eval_chat, tokenize=False)
print(f"\n  Eval prompt (what generate_text sends):")
print(f"  {repr(eval_text[:200])}")

# 3. Train a tiny LoRA (just 3 steps) and check if it breaks
print("\n--- Step 3: Mini LoRA Training (3 steps) ---")
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
peft_model = get_peft_model(model, lora_cfg)

# Build training data with chat template
train_clean = [q for q in CLEAN_QUESTIONS if q not in test_clean]
texts = []
for item in train_clean[:10]:  # only 10 samples
    chat = [
        {"role": "user", "content": item['q']},
        {"role": "assistant", "content": item['a']},
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    texts.append(text)

print(f"  Training samples: {len(texts)}")
print(f"  Sample 0: {repr(texts[0][:300])}")

ds = Dataset.from_dict({"text": texts})

output_dir = "results/debug_tmp"
os.makedirs(output_dir, exist_ok=True)

sft_kwargs = dict(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    max_steps=3,  # ONLY 3 STEPS!
)
try:
    training_cfg = SFTConfig(**sft_kwargs, max_seq_length=MAX_LENGTH)
except TypeError:
    sft_kwargs["dataset_kwargs"] = {"max_seq_length": MAX_LENGTH}
    training_cfg = SFTConfig(**sft_kwargs)

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=ds,
    args=training_cfg,
)
result = trainer.train()
print(f"  Training loss: {result.training_loss:.4f}")

# 4. Evaluate after mini training
print("\n--- Step 4: Post-LoRA Evaluation (3 steps only) ---")
for i, q in enumerate(test_clean[:3]):
    resp = generate_text(peft_model, tokenizer, q["q"], max_new=50)
    print(f"  Q: {q['q']}")
    print(f"  Expected: {q['a']}")
    print(f"  Got: {resp[:150]}")
    answer_words = q["a"].lower().split()
    hit = any(w in resp.lower() for w in answer_words if len(w) > 2)
    print(f"  Match: {hit}")
    print()

# 5. Also test with just plain text format
print("\n--- Step 5: Disable LoRA and re-evaluate ---")
peft_model.disable_adapter_layers()
for i, q in enumerate(test_clean[:3]):
    resp = generate_text(peft_model, tokenizer, q["q"], max_new=50)
    print(f"  Q: {q['q']}")
    print(f"  Expected: {q['a']}")
    print(f"  Got: {resp[:150]}")
    answer_words = q["a"].lower().split()
    hit = any(w in resp.lower() for w in answer_words if len(w) > 2)
    print(f"  Match: {hit}")
    print()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
