import os
import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.enrichment.features import sequence_position, word_position, syllable_position

def load_freq_map(path):
    if not os.path.exists(path): return {}
    with open(path, 'r') as f: return json.load(f)

def get_word_frequencies(ids, word_map):
    freqs = np.full(len(ids), np.nan, dtype=np.float32)
    delimiters = {32, 46, 44, 33, 63}
    i = 0
    while i < len(ids):
        if ids[i] == 0 or ids[i] in delimiters:
            i += 1
            continue
        start = i
        while i < len(ids) and ids[i] != 0 and ids[i] not in delimiters:
            i += 1
        end = i
        word = "".join(chr(ids[j]).lower() for j in range(start, end))
        val = word_map.get(word, 0.0)
        freqs[start:end] = val
    return freqs

def run(input_path, output_path, freq_paths, subset_n=None):
    if not os.path.exists(input_path): return
    df = pd.read_parquet(input_path)
    if subset_n:
        unique_ids = df['SEQUENCE_ID'].unique()[:subset_n]
        df = df[df['SEQUENCE_ID'].isin(unique_ids)].copy()

    u_map = load_freq_map(freq_paths['unigrams'])
    b_map = load_freq_map(freq_paths['bigrams'])
    w_map = load_freq_map(freq_paths['words'])

    n_pads = 8
    target_cols =['iki', 'iki_z', 'iki_log_z']
    res = { 
        "participant_id": [], "sequence_id":[], "original_sequence_id": [], "key_id": [], 
        "sequence_pos":[], "sequence_length":[], "sequence_relative_pos": [],
        "word_frequency": [], "word_index":[], "word_length": [], "word_relative_pos": [], 
        "is_word_start":[], "is_word_end": [], "is_syllable_start":[], "is_syllable_end": [],
        "repetition": [], "skipgram_repetition":[] 
    }
    for t in target_cols: res[t] =[]

    groups = df.groupby('SEQUENCE_ID', sort=False)
    for _, group in tqdm(groups, desc="Enriching Linguistic"):
        ids = group['KEY_ID'].values
        seq_len = len(ids)
        p_id = str(group['PARTICIPANT_ID'].iloc[0])
        s_id = str(group['SEQUENCE_ID'].iloc[0])
        os_id = str(group['ORIGINAL_SEQUENCE_ID'].iloc[0])

        res["participant_id"].extend(["[PAD]"] * n_pads +[p_id] * seq_len)
        res["sequence_id"].extend(["[PAD]"] * n_pads + [s_id] * seq_len)
        res["original_sequence_id"].extend(["[PAD]"] * n_pads + [os_id] * seq_len)
        res["key_id"].append(np.concatenate([np.zeros(n_pads, dtype=np.uint8), ids]))

        seq_pos, seq_len_arr, seq_rel_pos = sequence_position.get_features(seq_len)
        w_idx, w_len, w_rel_pos, w_start, w_end = word_position.get_features(ids)
        s_start, s_end = syllable_position.get_features(ids)

        res["sequence_pos"].append(np.concatenate([np.full(n_pads, -1.0, dtype=np.float32), seq_pos]))
        res["sequence_length"].append(np.concatenate([np.full(n_pads, -1.0, dtype=np.float32), seq_len_arr]))
        res["sequence_relative_pos"].append(np.concatenate([np.full(n_pads, -1.0, dtype=np.float32), seq_rel_pos]))
        
        res["word_frequency"].append(np.concatenate([np.full(n_pads, np.nan, dtype=np.float32), get_word_frequencies(ids, w_map)]))
        
        res["word_index"].append(np.concatenate([np.full(n_pads, -1.0, dtype=np.float32), w_idx]))
        res["word_length"].append(np.concatenate([np.full(n_pads, -1.0, dtype=np.float32), w_len]))
        res["word_relative_pos"].append(np.concatenate([np.full(n_pads, -1.0, dtype=np.float32), w_rel_pos]))
        res["is_word_start"].append(np.concatenate([np.zeros(n_pads, dtype=np.float32), w_start]))
        res["is_word_end"].append(np.concatenate([np.zeros(n_pads, dtype=np.float32), w_end]))
        res["is_syllable_start"].append(np.concatenate([np.zeros(n_pads, dtype=np.float32), s_start]))
        res["is_syllable_end"].append(np.concatenate([np.zeros(n_pads, dtype=np.float32), s_end]))

        rep = np.full(seq_len, np.nan, dtype=np.float32)
        skip_rep = np.full(seq_len, np.nan, dtype=np.float32)
        for i in range(1, seq_len):
            if ids[i] == ids[i-1]: rep[i] = 1.0
            else: rep[i] = 0.0
            if i >= 2 and ids[i] == ids[i-2]: skip_rep[i] = 1.0
            else: skip_rep[i] = 0.0
        res["repetition"].append(np.concatenate([np.full(n_pads, np.nan, dtype=np.float32), rep]))
        res["skipgram_repetition"].append(np.concatenate([np.full(n_pads, np.nan, dtype=np.float32), skip_rep]))
        for t in target_cols:
            res[t].append(np.concatenate([np.full(n_pads, np.nan, dtype=np.float32), group[t].values.astype(np.float32)]))

    data = { "participant_id": res["participant_id"], "sequence_id": res["sequence_id"], "original_sequence_id": res["original_sequence_id"] }
    feature_keys =[
        "key_id", "sequence_pos", "sequence_length", "sequence_relative_pos", 
        "word_frequency", "word_index", "word_length", "word_relative_pos", 
        "is_word_start", "is_word_end", "is_syllable_start", "is_syllable_end", 
        "repetition", "skipgram_repetition"
    ] + target_cols

    for k in feature_keys:
        data[k] = np.concatenate(res[k])

    keys_str = [chr(i) if i != 0 else "[PAD]" for i in data["key_id"]]
    data["key"] = keys_str
    data["unigram_frequency"] = pd.Series(keys_str).map(u_map).fillna(np.nan).values
    p_keys = pd.Series(keys_str).shift(1).fillna("[PAD]")
    data["bigram_frequency"] = (p_keys + pd.Series(keys_str)).map(b_map).fillna(np.nan).values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(data).to_parquet(output_path, index=False)

def main():
    run(
        input_path="data/interim/normalized_sequences.parquet",
        output_path="data/interim/linguistic_features.parquet",
        freq_paths={
            "unigrams": "data/frequencies/unigrams_zipf.json",
            "bigrams": "data/frequencies/bigrams_zipf.json",
            "words": "data/frequencies/words_zipf.json"
        }
    )

if __name__ == "__main__":
    main()