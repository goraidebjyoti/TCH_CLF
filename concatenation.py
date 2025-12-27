import json
from tqdm import tqdm

# ================= CONFIG =================
INPUT_FILE = "data/clinicaltrials/parsed/extracted_trials.jsonl"  # Input file from previous extraction
OUTPUT_FILE = "data/clinicaltrials/parsed/concatenated_trials.jsonl"  # Output file with concatenated strings
LOG_FILE = "data/clinicaltrials/parsed/concatenation_log.txt"  # Log file

# =============== MAIN PROCESSING ==================

def concatenate_trial(trial_json):
    """
    Concatenate non-null fields from a trial JSON into a single string with field names and || separator, excluding interventions.
    Returns a dictionary with trial ID and concatenated string.
    """
    trial_id = trial_json.get("id", "")
    fields = []
    # List of fields to concatenate, in order, excluding interventions
    field_order = [
        ("study_title", trial_json.get("study_title")),
        ("brief_summary", trial_json.get("brief_summary")),
        ("conditions", trial_json.get("conditions")),
        ("gender", trial_json.get("gender")),
        ("min_age", trial_json.get("min_age")),
        ("max_age", trial_json.get("max_age")),
        ("eligibility", trial_json.get("eligibility", {}).get("criteria"))
    ]

    # Add non-null fields to the list with field names
    for field_name, value in field_order:
        if value is not None:
            # Convert numbers to strings for concatenation
            if field_name in ["min_age", "max_age"]:
                value = str(value)
            fields.append(f"{field_name}: {value}")

    # Concatenate with || separator
    concatenated = " || ".join(fields)

    return {
        "id": trial_id,
        "concatenated_text": concatenated
    }

def main():
    total_trials = 0
    empty_trials = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile, \
         open(LOG_FILE, "w", encoding="utf-8") as logf:

        for line in tqdm(infile, desc="Concatenating trials"):
            total_trials += 1
            trial_json = json.loads(line)
            result = concatenate_trial(trial_json)

            if not result["concatenated_text"]:
                empty_trials += 1
                logf.write(f"Trial {result['id']} has no valid fields to concatenate\n")
                continue

            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Write summary to log
        logf.write(f"Total trials processed: {total_trials}\n")
        logf.write(f"Trials with empty concatenated text: {empty_trials}\n")

if __name__ == "__main__":
    main()