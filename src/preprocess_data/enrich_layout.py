import os
import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.enrichment.engine import EnrichmentEngine

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run(input_path, output_path, layout_paths):
    if not os.path.exists(input_path): return
    df = pd.read_parquet(input_path)
    
    engine = EnrichmentEngine(movement_features_path=layout_paths.get('movement'))
    with open(layout_paths['layout'], 'r') as f: l_data = json.load(f)
    with open(layout_paths['map'], 'r') as f: l_map = json.load(f)
    with open(layout_paths['shifts'], 'r') as f: s_data = json.load(f)

    features =["finger", "finger_type", "hand", "coords", "shift"]
    tables, move_matrix, _, sf_matrix, sh_matrix, sft_matrix, ir_matrix, or_matrix, _, _, sf_skip_matrix, in_triroll_matrix, out_triroll_matrix, redir_matrix, drj_matrix, scissor_matrix = engine._build_lookups(l_data, l_map, s_data, features)

    encoded = df['key_id'].values
    mask = (encoded == 0)
    
    df['is_pad'] = mask.astype(np.float32)

    for feat, table in tqdm(tables.items(), desc="Enriching Layout Features"):
        data = table[encoded].astype(np.float32)
        if data.ndim == 1:
            data[mask] = -1.0 if feat in ["shift"] else 0.0
            df[feat] = data
        else:
            data[mask, :] = np.nan if feat in["coords", "reach"] else 0.0
            if feat == 'coords':
                df['x'] = data[:, 0]
                df['y'] = data[:, 1]
            elif feat == 'reach':
                df['reach_dist'] = data[:, 0]
                df['reach_sin'] = data[:, 1]
                df['reach_cos'] = data[:, 2]
            elif feat == 'finger':
                for i in range(10): df[f'finger_{i}'] = data[:, i]
                df['finger'] = np.where(mask, -1.0, np.argmax(data, axis=1)).astype(np.float32)
            elif feat == 'finger_type':
                for i in range(5): df[f'finger_type_{i}'] = data[:, i]
                df['finger_type'] = np.where(mask, -1.0, np.argmax(data, axis=1)).astype(np.float32)
            elif feat == 'hand':
                for i in range(3): df[f'hand_{i}'] = data[:, i]
                df['hand'] = np.where(mask, -1.0, np.argmax(data, axis=1)).astype(np.float32)

    p = pd.Series(encoded).shift(1).fillna(0).astype(int).values
    pp = pd.Series(encoded).shift(2).fillna(0).astype(int).values
    valid_bi = (p != 0) & (encoded != 0) & (p < 128) & (encoded < 128)
    valid_tri = valid_bi & (pp != 0) & (pp < 128)

    if sf_matrix is not None: df['same_finger'] = np.where(valid_bi, sf_matrix[p, encoded], np.nan)
    if sh_matrix is not None: df['same_hand'] = np.where(valid_bi, sh_matrix[p, encoded], np.nan)
    if ir_matrix is not None: df['in_roll'] = np.where(valid_bi, ir_matrix[p, encoded], np.nan)
    if or_matrix is not None: df['out_roll'] = np.where(valid_bi, or_matrix[p, encoded], np.nan)
    if drj_matrix is not None: df['double_row_jump'] = np.where(valid_bi, drj_matrix[p, encoded], np.nan)
    if scissor_matrix is not None: df['scissors'] = np.where(valid_bi, scissor_matrix[p, encoded], np.nan)
    if sft_matrix is not None: df['same_finger_trigram'] = np.where(valid_tri, sft_matrix[pp, p, encoded], np.nan)
    if sf_skip_matrix is not None: df['same_finger_skipgram'] = np.where(valid_tri, sf_skip_matrix[pp, p, encoded], np.nan)
    if in_triroll_matrix is not None: df['in_triroll'] = np.where(valid_tri, in_triroll_matrix[pp, p, encoded], np.nan)
    if out_triroll_matrix is not None: df['out_triroll'] = np.where(valid_tri, out_triroll_matrix[pp, p, encoded], np.nan)
    if redir_matrix is not None: df['redirects'] = np.where(valid_tri, redir_matrix[pp, p, encoded], np.nan)

    if move_matrix is not None:
        m_data = np.full((len(encoded), 3), np.nan, dtype=np.float32)
        m_data[valid_bi] = move_matrix[p[valid_bi], encoded[valid_bi]]
        df['move_dist'] = m_data[:, 0]
        df['move_sin'] = m_data[:, 1]
        df['move_cos'] = m_data[:, 2]

    final_cols =[
        'participant_id', 'sequence_id', 'original_sequence_id',
        'key', 'key_id', 'is_pad', 'sequence_pos', 'sequence_length', 'sequence_relative_pos',
        'word_index', 'word_length', 'word_relative_pos', 
        'is_word_start', 'is_word_end', 'is_syllable_start', 'is_syllable_end',
        'hand', 'shift', 'finger', 'finger_type',
        'unigram_frequency', 'bigram_frequency', 'word_frequency',
        'same_finger', 'same_hand', 'same_finger_trigram',
        'repetition', 'skipgram_repetition', 'same_finger_skipgram',
        'in_roll', 'out_roll', 'in_triroll', 'out_triroll',
        'redirects', 'double_row_jump', 'scissors',
        'reach_dist', 'reach_sin', 'reach_cos',
        'move_dist', 'move_sin', 'move_cos',
        'x', 'y', 'iki', 'iki_z', 'iki_log_z'
    ]
    
    for i in range(10): final_cols.append(f'finger_{i}')
    for i in range(5): final_cols.append(f'finger_type_{i}')
    for i in range(3): final_cols.append(f'hand_{i}')

    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[final_cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

def main():
    run(
        input_path="data/interim/linguistic_features.parquet",
        output_path="data/enriched/enriched_data.parquet",
        layout_paths={
            "layout": "data/layouts/qwerty_us.json",
            "map": "data/layouts/layout_map.json",
            "shifts": "data/layouts/shifts_us.json",
            "movement": "data/layouts/movement_features.json"
        }
    )

if __name__ == "__main__":
    main()