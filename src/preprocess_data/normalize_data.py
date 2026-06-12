import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def run(input_path, output_path, sd_threshold=4):
    df = pd.read_parquet(input_path)
    df = df.sort_values(['PARTICIPANT_ID', 'SEQUENCE_ID', 'TIME'])
    df['KEY_ID'] = df['KEY'].astype(str).apply(lambda x: ord(x[0]) if len(x) > 0 and ord(x[0]) < 128 else 0).astype(np.uint8)
    df['iki_temp'] = df.groupby('SEQUENCE_ID')['TIME'].transform(lambda x: x.diff())
    p_stats = df.groupby('PARTICIPANT_ID')['iki_temp'].agg(['mean', 'std']).reset_index()
    p_stats.columns = ['PARTICIPANT_ID', 'p_mean', 'p_std']
    df = df.merge(p_stats, on='PARTICIPANT_ID')
    df['is_outlier'] = (df['iki_temp'] > (df['p_mean'] + sd_threshold * df['p_std'])).astype(int)
    participants, processed_data = df.groupby('PARTICIPANT_ID'), []
    for pid, p_group in tqdm(participants, desc="Normalizing IKI"):
        p_group = p_group.copy()
        p_group['segment_id'] = p_group.groupby('SEQUENCE_ID')['is_outlier'].cumsum()
        p_group['SEQUENCE_ID'] = p_group['SEQUENCE_ID'] + "_" + p_group['segment_id'].astype(str)
        s_groups = p_group.groupby('SEQUENCE_ID')
        p_group['TIME'] = s_groups['TIME'].transform(lambda x: x - x.min())
        p_group['iki'] = s_groups['TIME'].transform(lambda x: x.diff())
        mask = (~np.isnan(p_group['iki'].values)) & (p_group['iki'].values > 0)
        v_iki = p_group['iki'].values[mask]
        if len(v_iki) > 1:
            m, s = np.mean(v_iki), np.std(v_iki)
            p_group['iki_z'] = np.nan
            if s > 0: p_group.loc[mask, 'iki_z'] = (v_iki - m) / s
            iki_log = np.log(v_iki)
            lm, ls = np.mean(iki_log), np.std(iki_log)
            p_group['iki_log_z'] = np.nan
            if ls > 0: p_group.loc[mask, 'iki_log_z'] = (iki_log - lm) / ls
            processed_data.append(p_group)
    if processed_data:
        final_df = pd.concat(processed_data)
        cols = ['PARTICIPANT_ID', 'SEQUENCE_ID', 'ORIGINAL_SEQUENCE_ID', 'KEY', 'KEY_ID', 'TIME', 'IS_TYPO', 'TYPO_TYPE', 'iki', 'iki_z', 'iki_log_z']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df[cols].to_parquet(output_path, index=False)

if __name__ == "__main__":
    run("data/interim/filtered_sequences.parquet", "data/interim/normalized_sequences.parquet")