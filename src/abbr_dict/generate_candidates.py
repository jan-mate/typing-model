import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
from src.abbr_dict.candidates import generate_all_candidates
from src.abbr_dict.config import RPT_KEY, data_path

VOCABULARY_PATH = "data/abbr_dict/vocabulary.json"   # regime-independent input
OUTPUT_PATH = data_path("candidates.json")

CANDIDATES_PER_ITEM = 8000
MIN_ABBR_LEN = 1
MAX_REPLACEMENTS = 2

ENABLE_WORDS = True
ENABLE_SUFFIXES = True
RANDOM_SEED = 42


def main():
    import random
    random.seed(RANDOM_SEED)

    with open(VOCABULARY_PATH) as f:
        vocabulary = json.load(f)

    candidates = generate_all_candidates(
        vocabulary,
        n_per_item=CANDIDATES_PER_ITEM,
        min_len=MIN_ABBR_LEN,
        max_replacements=MAX_REPLACEMENTS,
        enable_words=ENABLE_WORDS,
        enable_suffixes=ENABLE_SUFFIXES,
        rpt_key=RPT_KEY,
    )

    candidates["rpt_key"] = RPT_KEY

    total = sum(len(item["candidates"]) for item in candidates["items"])
    print(f"Total candidates: {total}")
    print(f"rpt_key: {RPT_KEY}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"Saved candidates to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()