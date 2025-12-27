import json
import re
from tqdm import tqdm

# ================= CONFIG =================
INPUT_FILE = "data/clinicaltrials/corpus.jsonl"             # input file with raw trials
OUTPUT_FILE = "data/clinicaltrials/parsed/extracted_trials.jsonl"  # structured extracted output
LOG_FILE = "data/clinicaltrials/parsed/extraction_log.txt"         # log for missing field stats


# =============== HELPERS ==================

def normalize_age(age_str):
    """
    Convert age strings into years as float.
    Handles Years, Months, Days. Returns float in years or None.
    """
    if not age_str or age_str.strip().lower() in ["n/a", "na", "not applicable", "no limit"]:
        return None

    age_str = age_str.strip()
    match = re.match(r"(\d+(?:\.\d+)?)\s*(year|yr|years|yrs|month|months|mo|day|days|d)", age_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()

        if "year" in unit:
            return value
        elif "month" in unit:
            return value / 12.0
        elif "day" in unit:
            return value / 365.25
        else:
            return None
    else:
        # If no unit, assume years
        num_match = re.match(r"(\d+(?:\.\d+)?)", age_str, re.IGNORECASE)
        if num_match:
            return float(num_match.group(1))
        return None


def extract_field(pattern, text):
    """
    Utility to extract a field from trial text using regex.
    Returns the first match string or None.
    """
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def normalize_gender(gender_str, eligibility_text):
    """
    Normalize gender based on extracted gender field or eligibility criteria.
    Defaults to 'male and female' if unspecified.
    """
    text_to_check = ""
    if gender_str:
        text_to_check = gender_str.lower()
    elif eligibility_text:
        text_to_check = eligibility_text.lower()

    if not text_to_check:
        return "male and female"

    # Check for 'all'
    if 'all' in text_to_check:
        return "male and female"

    # Male terms
    male_terms = ['male', 'man', 'boy', 'men', 'boys']
    # Female terms
    female_terms = ['female', 'woman', 'girl', 'women', 'girls']

    has_male = any(term in text_to_check for term in male_terms)
    has_female = any(term in text_to_check for term in female_terms)

    if has_male and has_female:
        return "male and female"
    elif has_male:
        return "male"
    elif has_female:
        return "female"
    else:
        return "male and female"


# =============== MAIN EXTRACTION ==================

def process_trial(trial_json, missing_counts):
    """
    Process a single trial JSON, extract structured fields (excluding interventions).
    Updates missing_counts dict when a field is missing.
    """
    trial_id = trial_json.get("id")
    contents = trial_json.get("contents", "")

    # --- Study Title ---
    study_title = extract_field(r"Study Title:\s*(.+?)(?=\n[A-Z][a-zA-Z ]+:|$)", contents)
    if not study_title:
        missing_counts["study_title"] += 1

    # --- Brief Summary ---
    brief_summary = extract_field(r"Brief Summary:\s*(.+?)(?=\n[A-Z][a-zA-Z ]+:|$)", contents)
    if not brief_summary:
        missing_counts["brief_summary"] += 1

    # --- Conditions (singular/plural) ---
    conditions = extract_field(r"Conditions?:\s*(.+?)(?=\n[A-Z][a-zA-Z ]+:|$)", contents)
    if not conditions:
        missing_counts["conditions"] += 1

    # --- Gender (multiple variants) ---
    gender_raw = extract_field(r"(?:Gender|Sex(?:es)? Eligible?):\s*(.+?)(?=\n[A-Z][a-zA-Z ]+:|$)", contents)
    if not gender_raw:
        missing_counts["gender"] += 1

    # --- Age Range ---
    min_age_raw = extract_field(r"Minimum Age:\s*(.+?)(?=\n[A-Z][a-zA-Z ]+:|$)", contents)
    max_age_raw = extract_field(r"Maximum Age:\s*(.+?)(?=\n[A-Z][a-zA-Z ]+:|$)", contents)

    # Fallback: combined field like "Ages Eligible for Study: 18 Years to 65 Years"
    if not min_age_raw and not max_age_raw:
        combined_age = extract_field(r"Ages Eligible.*?:\s*(.+?)(?=\n[A-Z][a-zA-Z ]+:|$)", contents)
        if combined_age:
            range_match = re.match(r"(\d+.*?)(?:\s+to\s+(\d+.*))?$", combined_age, re.IGNORECASE)
            if range_match:
                min_age_raw = range_match.group(1).strip() if range_match.group(1) else None
                max_age_raw = range_match.group(2).strip() if range_match.group(2) else None
            else:
                min_age_raw = combined_age.strip()

    if not min_age_raw:
        missing_counts["min_age"] += 1
    if not max_age_raw:
        missing_counts["max_age"] += 1

    min_age_norm = normalize_age(min_age_raw) if min_age_raw else None
    max_age_norm = normalize_age(max_age_raw) if max_age_raw else None

    # Set defaults for ages if None or unparsable
    if min_age_norm is None:
        min_age_norm = 0.0
    if max_age_norm is None:
        max_age_norm = 150.0

    # --- Eligibility Criteria ---
    eligibility_text = extract_field(
        r"Eligibility Criteria:\s*(.+?)(?=\n[A-Z][a-zA-Z ]+:|$)", contents
    )

    if not eligibility_text:
        missing_counts["eligibility"] += 1

    eligibility_struct = {"criteria": eligibility_text}

    # Normalize gender
    gender = normalize_gender(gender_raw, eligibility_text)

    return {
        "id": trial_id,
        "study_title": study_title,
        "brief_summary": brief_summary,
        "conditions": conditions,
        "gender": gender,
        "min_age": min_age_norm,
        "max_age": max_age_norm,
        "eligibility": eligibility_struct
    }


def main():
    # Counters for missing data
    missing_counts = {
        "study_title": 0,
        "brief_summary": 0,
        "conditions": 0,
        "gender": 0,
        "min_age": 0,
        "max_age": 0,
        "eligibility": 0,
    }
    total_trials = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in tqdm(infile, desc="Processing trials"):
            total_trials += 1
            trial_json = json.loads(line)
            extracted = process_trial(trial_json, missing_counts)
            outfile.write(json.dumps(extracted, ensure_ascii=False) + "\n")

    # --- Write log ---
    with open(LOG_FILE, "w", encoding="utf-8") as logf:
        logf.write(f"Total trials processed: {total_trials}\n")
        logf.write("Missing field counts:\n")
        for k, v in missing_counts.items():
            logf.write(f"  {k}: {v}\n")


if __name__ == "__main__":
    main()