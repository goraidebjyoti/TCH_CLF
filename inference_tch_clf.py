# ==========================================================
# inference_tch_clf.py  (Fixed Sorting)
# ----------------------------------------------------------
# Uses trained Clinical-Longformer (TeacherReranker) to run
# inference on 2021 and 2022 reasoning datasets for
# WholeQ and WholeQ+RM3 retrieval types.
# Generates numerically sorted TREC-style ranked outputs.
# ==========================================================

import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from train_teacher_longformer import TeacherReranker

# ====================== CONFIG ===========================
MODEL_PATH = "models_new/Teacher_ClinicalLongformer_1196/alpha0.2/best_teacher_alpha0.2.pt"
MODEL_NAME = "yikuan8/Clinical-Longformer"
MAX_LENGTH = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input files
DATASETS = {
    "2021_wholeq": {
        "queries": "data/dataset/ct_2021_queries.tsv",
        "reasonings": "data/2021/reasoning/WholeQ_RETRIEVAL_T2021_deepseek_clean.jsonl",
    },
    "2021_wholeq_rm3": {
        "queries": "data/dataset/ct_2021_queries.tsv",
        "reasonings": "data/2021/reasoning/WholeQ_RM3_RETRIEVAL_T2021_deepseek_clean.jsonl",
    },
    "2022_wholeq": {
        "queries": "data/dataset/ct_2022_queries.tsv",
        "reasonings": "data/2022/reasoning/WholeQ_RETRIEVAL_T2022_deepseek_clean.jsonl",
    },
    "2022_wholeq_rm3": {
        "queries": "data/dataset/ct_2022_queries.tsv",
        "reasonings": "data/2022/reasoning/WholeQ_RM3_RETRIEVAL_T2022_deepseek_clean.jsonl",
    }
}

# Common trial corpus
TRIALS_JSONL = "data/clinicaltrials/parsed/concatenated_trials.jsonl"

# Output directory
OUTPUT_DIR = "output/predictions_teacher_reasoning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
#                 Utility Functions
# =========================================================
def load_queries(tsv_path):
    queries = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            topic_id, text = line.strip().split("\t", 1)
            queries[topic_id] = text
    return queries


def load_trials(jsonl_path):
    trials = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            trials[obj["id"]] = obj["concatenated_text"]
    return trials


# =========================================================
#                    Inference Function
# =========================================================
def run_inference(dataset_name, queries_file, reasonings_file, trials):
    print(f"\n🚀 Running inference for: {dataset_name}")
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_teacher_run.txt")

    # Load queries + tokenizer
    queries = load_queries(queries_file)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load reasoning data
    with open(reasonings_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Run model
    model = TeacherReranker().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    trec_entries = []
    with torch.no_grad():
        for obj in tqdm(data, desc=f"Inference {dataset_name}"):
            topic_id = obj["topic_id"]
            trial_id = obj["trial_id"]
            reasoning = obj.get("reasoning", "")
            trial_text = trials.get(trial_id, "")

            second_text = f"{trial_text} {tokenizer.sep_token} Reasoning: {reasoning}" if reasoning else trial_text

            enc = tokenizer(
                queries[topic_id],
                second_text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)

            logit = model(enc["input_ids"], enc["attention_mask"]).item()
            score = torch.sigmoid(torch.tensor(logit)).item()

            trec_entries.append((topic_id, trial_id, score))

    # ✅ Sort numerically by topic_id instead of lexicographically
    unique_topics = sorted(set(t[0] for t in trec_entries), key=lambda x: int(x))

    with open(output_path, "w", encoding="utf-8") as f:
        for topic_id in unique_topics:
            topic_entries = [(t[1], t[2]) for t in trec_entries if t[0] == topic_id]
            topic_entries.sort(key=lambda x: x[1], reverse=True)
            for rank, (trial_id, score) in enumerate(topic_entries, start=1):
                f.write(f"{topic_id} Q0 {trial_id} {rank} {score:.6f} TeacherLongformer\n")

    print(f"✅ Saved ranked output: {output_path}")


# =========================================================
#                     Main Routine
# =========================================================
if __name__ == "__main__":
    print("========== TEACHER INFERENCE (LONGFORMER) ==========")
    print(f"Model: {MODEL_PATH}\nDevice: {DEVICE}\n")

    # Load trials once globally
    trials = load_trials(TRIALS_JSONL)
    print(f"Loaded {len(trials)} trials from corpus.\n")

    # Run for all datasets
    for dataset_name, files in DATASETS.items():
        run_inference(dataset_name, files["queries"], files["reasonings"], trials)

    print("\n🎯 All inference runs complete!")
