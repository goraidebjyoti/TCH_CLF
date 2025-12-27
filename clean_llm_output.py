#clean_llm_output.py

import json
from tqdm import tqdm

# === Paths ===
input_jsonl = "data/2022/reasoning/WholeQ_RM3_RETRIEVAL_T2022_deepseek_raw.jsonl"
output_jsonl = "data/2022/reasoning/WholeQ_RM3_RETRIEVAL_T2022_deepseek_clean.jsonl"
error_jsonl  = "data/2022/reasoning/WholeQ_RM3_RETRIEVAL_T2022_deepseek_errors.jsonl"


def extract_final_json(raw_output: str):
    """
    Extract the final JSON object (reasoning + relevance) from raw model output.
    Ensures we skip earlier examples in the prompt.
    Returns dict or None if parsing fails.
    """
    try:
        # Find the last {...} block in the text
        start = raw_output.rfind("{")
        end = raw_output.rfind("}") + 1
        if start == -1 or end == -1 or end <= start:
            return None
        json_str = raw_output[start:end].strip()

        # Parse JSON safely
        parsed = json.loads(json_str)

        # Make sure it really has the required keys
        if "reasoning" in parsed and "relevance" in parsed:
            return parsed
        else:
            return None
    except Exception:
        return None


# First count lines for tqdm total
with open(input_jsonl, "r", encoding="utf-8") as infile:
    total_lines = sum(1 for _ in infile)

with open(input_jsonl, "r", encoding="utf-8") as infile, \
     open(output_jsonl, "w", encoding="utf-8") as cleanfile, \
     open(error_jsonl, "w", encoding="utf-8") as errorfile:

    for line_num, line in enumerate(tqdm(infile, total=total_lines, desc="Sanitizing"), start=1):
        record = json.loads(line.strip())

        topic_id = record.get("topic_id")
        trial_id = record.get("trial_id")
        raw_output = record.get("raw_output", "")

        parsed = extract_final_json(raw_output)

        if parsed:  # ✅ successfully parsed
            clean_record = {
                "topic_id": topic_id,
                "trial_id": trial_id,
                "reasoning": parsed.get("reasoning", "").strip(),
                "relevance": parsed.get("relevance", "").strip()
            }
            cleanfile.write(json.dumps(clean_record) + "\n")

        else:  # ❌ failed parse → log error
            error_record = {
                "topic_id": topic_id,
                "trial_id": trial_id,
                "raw_output": raw_output
            }
            errorfile.write(json.dumps(error_record) + "\n")

print(f"✅ Clean file saved: {output_jsonl}")
print(f"⚠️ Errors logged to: {error_jsonl}")
