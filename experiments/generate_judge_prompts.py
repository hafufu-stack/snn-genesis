"""
generate_judge_prompts.py — Create paste-ready prompts for manual Gemini evaluation.

Reads phase12_outputs_intermediate.json and generates text files
that can be pasted into Gemini Web UI for judging.

One file per condition, each containing all outputs for that condition.
"""

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
INPUT_FILE = os.path.join(RESULTS_DIR, "phase12_outputs_intermediate.json")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "judge_prompts")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# For each condition, create one prompt file
for idx, (condition, outputs) in enumerate(data.items(), 1):
    # Build batch prompt
    prompt_text = f"""You are an expert creative writing evaluator. A language model was given creative prompts and produced the following responses. Judge EACH response on exactly TWO dimensions.

## Evaluation Criteria
1. **Coherence** (1-5): Is the text grammatically correct, logically structured, and meaningful?
   - 5: Perfectly natural, well-structured text
   - 4: Mostly natural with minor awkwardness
   - 3: Understandable but noticeably unnatural
   - 2: Partially comprehensible, broken in places
   - 1: Complete gibberish / word salad / nonsense

2. **Creativity** (1-5): Does it show original, surprising, or imaginative ideas?
   - 5: Highly original and unexpected
   - 4: Fresh perspective or unique elements
   - 3: Average creativity
   - 2: Clichéd or predictable
   - 1: No originality

## CRITICAL RULE
If Coherence is 1 or 2 (broken/gibberish text), Creativity MUST be 1 regardless of how "unique" the garbage text appears. Destruction is not creation.

## Condition: {condition}

Evaluate all {len(outputs)} responses below and return a JSON array.

"""
    for i, item in enumerate(outputs):
        prompt_text += f"""---
### Response {i+1}
**Prompt:** "{item['prompt']}"

**Output:**
{item['output'][:400]}

"""

    prompt_text += f"""---

## Output Format
Return ONLY a JSON array with {len(outputs)} entries (no markdown code blocks, just raw JSON):

[
  {{"id": 1, "coherence": <1-5>, "creativity": <1-5>, "reasoning": "brief explanation"}},
  {{"id": 2, "coherence": <1-5>, "creativity": <1-5>, "reasoning": "brief explanation"}},
  ...
]
"""

    # Save to file
    safe_name = condition.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "").replace("\u03c3", "s")
    filename = f"prompt_{idx:02d}_{safe_name}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    
    print(f"  [{idx:2d}/10] {filename} ({len(outputs)} outputs, {len(prompt_text)} chars)")

print(f"\n\u2705 All prompt files saved to: {OUTPUT_DIR}")
print(f"\nUsage:")
print(f"  1. Paste each file's content into Gemini Web UI")
print(f"  2. Save the returned JSON responses to results/judge_responses/")
print(f"  3. Run phase12 script to aggregate and visualize")
