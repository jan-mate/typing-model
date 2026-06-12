import os
import sys
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocess_data import (
    filter_participants, annotate_typos, filter_typos, 
    normalize_data, enrich_linguistic, enrich_layout, split_data, calculate_frequencies
)


CONFIG = {
    "suffix": "",
    "use_custom_data": False,
    "subset_participants": None,
    "subset_enrichment": None, 
    "participant_filter": {"layout": "qwerty", "wpm": 80, "country": "US", "fingers": '9-10'},
    "typo_filter": {"Substitution": False, "Insertion": False, "Deletion": False, "Transposition": False, "Proofreading": False},
    "paths": {
        "keystrokes_raw": "data/raw/files/",
        "freq_dir": "data/frequencies/",
        "freqs": {
            "unigrams": "data/frequencies/unigrams_zipf.json", 
            "bigrams": "data/frequencies/bigrams_zipf.json", 
            "words": "data/frequencies/words_zipf.json"
        },
        "layout": {
            "layout": "data/layouts/qwerty_us.json", 
            "map": "data/layouts/layout_map.json", 
            "shifts": "data/layouts/shifts_us.json", 
            "movement": "data/layouts/movement_features.json"
        }
    }
}


# CONFIG = {
#     "suffix": "_dvorak",
#     "use_custom_data": False,
#     "subset_participants": None,
#     "subset_enrichment": None, 
#     "participant_filter": {"layout": "dvorak", "wpm": 80, "country": "US", "fingers": '9-10'},
#     "typo_filter": {"Substitution": False, "Insertion": False, "Deletion": False, "Transposition": False, "Proofreading": False},
#     "paths": {
#         "keystrokes_raw": "data/raw/files/",
#         "freq_dir": "data/frequencies/",
#         "freqs": {
#             "unigrams": "data/frequencies/unigrams_zipf.json", 
#             "bigrams": "data/frequencies/bigrams_zipf.json", 
#             "words": "data/frequencies/words_zipf.json"
#         },
#         "layout": {
#             "layout": "data/layouts/dvorak.json", 
#             "map": "data/layouts/layout_map.json", 
#             "shifts": "data/layouts/shifts_us.json", 
#             "movement": "data/layouts/movement_features.json"
#         }
#     }
# }

def main():
    s = CONFIG.get("suffix", "")
    paths = CONFIG["paths"]
    
    print("Step 0: Calculating Frequencies...")
    calculate_frequencies.run(paths["freq_dir"])

    if CONFIG["use_custom_data"]:
        print("Step 1: Preparing Custom Participants...")
        raw_dir = paths.get("custom_keystrokes_raw", "data/")
        part_file = os.path.join("data/interim", f"custom_participants{s}.txt")
        os.makedirs(os.path.dirname(part_file), exist_ok=True)
        ids =[f.replace("_keystrokes.txt", "") for f in os.listdir(raw_dir) if f.endswith("_keystrokes.txt")]
        pd.DataFrame({"PARTICIPANT_ID": ids}).to_csv(part_file, sep="\t", index=False)
    else:
        print("Step 1: Filtering Participants...")
        raw_dir = paths["keystrokes_raw"]
        part_file = filter_participants.run(**CONFIG["participant_filter"], subset_n=CONFIG["subset_participants"])
    
    print("Step 2: Annotating Keystrokes...")
    annotated_path = f"data/interim/annotated_sequences{s}.parquet"
    annotate_typos.run(part_file, raw_dir, annotated_path)
    
    if not os.path.exists(annotated_path) or pd.read_parquet(annotated_path).empty:
        return

    print("Step 3: Filtering Typos...")
    filtered_path = f"data/interim/filtered_sequences{s}.parquet"
    filter_typos.run(annotated_path, filtered_path, CONFIG["typo_filter"])
    
    print("Step 4: Normalizing IKI...")
    normalized_path = f"data/interim/normalized_sequences{s}.parquet"
    normalize_data.run(filtered_path, normalized_path)
    
    print("Step 5a: Enriching Linguistic Features...")
    linguistic_path = f"data/interim/linguistic_features{s}.parquet"
    enrich_linguistic.run(normalized_path, linguistic_path, paths["freqs"], subset_n=CONFIG["subset_enrichment"])
    
    print("Step 5b: Enriching Layout Features...")
    enriched_path = f"data/enriched/enriched_data{s}.parquet"
    enrich_layout.run(linguistic_path, enriched_path, paths["layout"])
    
    print("Step 6: Splitting Data into Folds...")
    final_path = f"data/enriched/enriched_with_folds{s}.parquet"
    meta_path = f"data/enriched/fold_metadata{s}.json"
    split_data.run(enriched_path, final_path, meta_path)

if __name__ == "__main__":
    main()