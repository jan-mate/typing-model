import os
import re
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def load_freq_map(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_balanced_folds(freq_map, n_folds=10):
    items = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
    
    folds_items = {i: [] for i in range(n_folds)}
    fold_weights = [0.0] * n_folds
    item_to_fold = {}

    for item, weight in items:
        min_fold_idx = fold_weights.index(min(fold_weights))
        folds_items[min_fold_idx].append(item)
        fold_weights[min_fold_idx] += weight
        item_to_fold[item] = min_fold_idx
        
    return item_to_fold, fold_weights, folds_items

def get_fold(item, fold_map, n_folds=10):
    if pd.isna(item) or item == "":
        return 0
    if item in fold_map:
        return fold_map[item]
    
    # deterministic fallback without hashlib for unseen items
    return sum(ord(c) for c in str(item)) % n_folds

def extract_core_word(raw_str):
    return re.sub(r'[^a-zA-Z]', '', str(raw_str)).lower()

def run(input_path, output_path, meta_path):
    tqdm.pandas()
    
    base_dir = Path(input_path).parent.parent
    freq_dir = base_dir / "frequencies"
    
    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    
    print("Loading frequency maps...")
    word_freqs = load_freq_map(freq_dir / "words.json")
    bigram_freqs = load_freq_map(freq_dir / "bigrams.json")
    
    print("Generating balanced folds...")
    w_map, w_totals, w_folds = generate_balanced_folds(word_freqs, n_folds=10)
    b_map, b_totals, b_folds = generate_balanced_folds(bigram_freqs, n_folds=10)

    df['prev1'] = df.groupby('sequence_id')['key'].shift(1).fillna("[PAD]")
    df['temp_bigram'] = df['prev1'] + df['key']

    df['temp_word_start_mask'] = df['word_index'] == 0
    df['word_group_id'] = df.groupby('sequence_id')['temp_word_start_mask'].cumsum()
    
    print("Extracting and mapping words...")
    valid_mask = df['key'] != '[PAD]'
    word_groups = df[valid_mask].groupby(['sequence_id', 'word_group_id'])['key'].apply(lambda x: ''.join(x)).reset_index(name='raw_word')
    word_groups['core_word'] = word_groups['raw_word'].apply(extract_core_word)
    word_groups['fold_word'] = word_groups['core_word'].progress_apply(lambda x: get_fold(x, w_map)).astype('int32')
    
    df = df.merge(word_groups[['sequence_id', 'word_group_id', 'fold_word']], on=['sequence_id', 'word_group_id'], how='left')
    df['fold_word'] = df['fold_word'].fillna(-1).astype('int32')

    print("Mapping bigrams...")
    df['fold_bigram'] = df['temp_bigram'].progress_apply(lambda x: get_fold(x, b_map)).astype('int32')

    df.loc[~valid_mask, 'fold_word'] = -1
    df.loc[~valid_mask, 'fold_bigram'] = -1

    cols_to_drop =['prev1', 'temp_bigram', 'temp_word_start_mask', 'word_group_id']
    df = df.drop(columns=cols_to_drop)
    
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False)
    
    metadata = {
        "word": {"totals": w_totals, "folds": w_folds},
        "bigram": {"totals": b_totals, "folds": b_folds}
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    run(
        "data/enriched/enriched_data.parquet",
        "data/enriched/enriched_with_folds.parquet",
        "data/enriched/fold_metadata.json"
    )