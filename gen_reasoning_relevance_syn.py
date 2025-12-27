# gen_reasoning_relevance_syn.py

import os

# === GPU Configuration ===
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import torch
import json
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

# ===============================
# Reproducibility
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# === Paths Configuration ===
input_jsonl = "dataset/train_1196.jsonl"   # new input file
trec_collection_path = "data/clinicaltrials/corpus.jsonl"
topics_csv = "data/dataset/synthetic_gold_queries.tsv"

# output file (jsonl) placed in same dir as input_jsonl
output_jsonl_path = os.path.join(
    "data/dataset/reasoning",
    f"{os.path.splitext(os.path.basename(input_jsonl))[0]}_deepseek_raw.jsonl"
)

# === Global Configurations ===
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
TEMPERATURE = 0.3
MAX_NEW_TOKENS = 2048   # reasoning + relevance can be long
USE_BFLOAT16 = False
FLOAT_TYPE = torch.float16

# === Load Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=FLOAT_TYPE,
)

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=FLOAT_TYPE,
    device_map={"": 0}
)

print("Model loaded successfully.")

# === Load Corpus ===
def load_corpus():
    corpus_dict = {}
    with open(trec_collection_path, "r", encoding="utf-8") as f:
        for line in f:
            trial = json.loads(line.strip())
            corpus_dict[trial["id"]] = trial.get("contents")
    print(f"Loaded {len(corpus_dict)} documents.")
    return corpus_dict

corpus_cache = load_corpus()

# === Load Topics TSV ===
topic_df = pd.read_csv(topics_csv, sep="\t", header=None, names=["id", "text"])
topic_map = {str(row["id"]): row["text"] for _, row in topic_df.iterrows()}
print(f"Loaded {len(topic_map)} topics.")

# === Build Prompt ===
def build_prompt(query, trial_text):
    prompt = f"""
### Role: You are an expert in biomedical AI with access to clinical trial data and the ability to assess the relevance of a given patient case description to a specific clinical trial. Your task is to provide reasoning and then determine whether the trial is relevant to the patient case.

### Instruction:
- Given a patient description and a clinical trial, first provide reasoning as a single paragraph with **at least three sentences**.  
- The reasoning must specifically discuss eligibility or non-eligibility factors such as patient demographics (age, gender), medical condition, interventions, and inclusion/exclusion criteria where applicable.  
- Do not use vague statements such as "the trial seems relevant" or "all eligibility criteria are satisfied." The explanation must clearly reference concrete eligibility aspects that support or contradict the match.  
- After the reasoning, decide the trial's relevance:
  - Use **"Relevant"** if the trial is a suitable match for the patient case based on eligibility, condition, intervention, or other criteria.  
  - Use **"Non-Relevant"** if the trial does not fit the patient case due to mismatched condition, intervention, demographics, or exclusion criteria.  
- The reasoning must **not** use bullet points, numbered lists, or line breaks; it should be continuous prose in one paragraph.  
- Your output must be in **strict JSONL format** with two fields:  
  - "reasoning": one paragraph of explanatory reasoning with multiple sentences (minimum 3)  
  - "relevance": one of ["Relevant", "Non-Relevant"]  
- Do not include any text outside the JSONL object.  

### Patient Description: {query}

### Clinical Trial: {trial_text}

### Output Format Example:
{{"reasoning": "The patient is a 45-year-old with the same type of cancer described in the trial. The trial allows participants in this age group and does not list gender as a restriction. The intervention being studied is consistent with the patient's treatment needs, and there are no exclusion criteria that would prevent enrollment.", "relevance": "Relevant"}}

### Output:
"""
    return prompt.strip()

# === Generate Raw Output Only ===
def generate_raw_output(query, trial_text, topic_no, trial_id):
    if trial_text == "Text not found." or trial_text is None:
        return {
            "topic_id": topic_no,
            "trial_id": trial_id,
            "raw_output": "Trial text missing; eligibility cannot be assessed."
        }

    prompt = build_prompt(query, trial_text)
    input_ids = TOKENIZER(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = MODEL.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True
        )

    raw_output = TOKENIZER.decode(outputs[0], skip_special_tokens=True).strip()

    return {
        "topic_id": topic_no,
        "trial_id": trial_id,
        "raw_output": raw_output
    }

# === Process and Save JSONL ===
with open(input_jsonl, "r", encoding="utf-8") as infile, open(output_jsonl_path, "a", encoding="utf-8") as out_f:
    for line in tqdm(infile, desc="Processing pairs"):
        ex = json.loads(line.strip())
        topic_no, trial_id = str(ex["topic_id"]), ex["trial_id"]

        topic_query = topic_map.get(topic_no, "Topic text not found.")
        trial_text = corpus_cache.get(trial_id, "Text not found.")

        result = generate_raw_output(topic_query, trial_text, topic_no, trial_id)

        out_f.write(json.dumps(result) + "\n")
        out_f.flush()
        print(f"Saved Topic {topic_no} | Trial {trial_id}")

print(f"All results saved to {output_jsonl_path}")
