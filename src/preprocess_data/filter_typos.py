import pandas as pd
import os
from tqdm import tqdm

def run(input_path, output_path, config):
    df = pd.read_parquet(input_path)
    processed_chunks = []
    for seq_id, group in tqdm(df.groupby("SEQUENCE_ID"), desc="Filtering Typos"):
        group = group.sort_values("TIME").reset_index(drop=True)
        sub_id, sub_start_time, keep_indices, new_times, new_sids = 0, None, [], [], []
        for i, row in enumerate(group.itertuples()):
            t_type = row.TYPO_TYPE
            keep_row = config.get(t_type, True) if t_type else True
            if not keep_row:
                sub_id += 1
                sub_start_time = None
                continue
            if i > 0:
                if group.iloc[i-1]["TYPO_TYPE"] == "Deletion" and t_type == "Deletion" and config.get("Deletion", True):
                    sub_id += 1
                    sub_start_time = None
            if sub_start_time is None: sub_start_time = row.TIME
            keep_indices.append(i)
            new_times.append(int(row.TIME - sub_start_time))
            new_sids.append(f"{seq_id}_{sub_id}")
        if keep_indices:
            sub_df = group.iloc[keep_indices].copy()
            sub_df["ORIGINAL_SEQUENCE_ID"], sub_df["SEQUENCE_ID"], sub_df["TIME"] = seq_id, new_sids, new_times
            processed_chunks.append(sub_df)
    if processed_chunks:
        final_df = pd.concat(processed_chunks, ignore_index=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_parquet(output_path)

if __name__ == "__main__":
    cfg = {"Substitution": False, "Insertion": False, "Deletion": False, "Transposition": False, "Proofreading": False}
    run("data/interim/annotated_sequences.parquet", "data/interim/filtered_sequences.parquet", cfg)