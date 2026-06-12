import pytest
import pandas as pd
import numpy as np
import os
from src.preprocess_data.normalize_data import run

def create_synthetic_data(pid, intervals, seq_id):
    times = [0] + list(np.cumsum(intervals))
    return pd.DataFrame({
        "PARTICIPANT_ID": [pid] * len(times),
        "SEQUENCE_ID": [seq_id] * len(times),
        "ORIGINAL_SEQUENCE_ID": [seq_id] * len(times),
        "KEY": ["a"] * len(times),
        "TIME": times,
        "IS_TYPO": [False] * len(times),
        "TYPO_TYPE": [None] * len(times)
    })

def test_normalize_synthetic_data_splitting_and_zscore(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"

    p1_intervals = [90.0, 110.0] * 50 + [10000.0] + [90.0, 110.0] * 50
    p2_intervals = [115.0, 125.0] * 100

    df = pd.concat([
        create_synthetic_data("P1", p1_intervals, "P1_S1"),
        create_synthetic_data("P2", p2_intervals, "P2_S1")
    ])
    df.to_parquet(input_path)

    run(str(input_path), str(output_path), sd_threshold=4)
    res = pd.read_parquet(output_path)

    seqs_p1 = res[res['PARTICIPANT_ID'] == 'P1']['SEQUENCE_ID'].unique()
    assert len(seqs_p1) >= 2 
    
    seqs_p2 = res[res['PARTICIPANT_ID'] == 'P2']['SEQUENCE_ID'].unique()
    assert len(seqs_p2) == 1

    for pid in ['P1', 'P2']:
        p_data = res[res['PARTICIPANT_ID'] == pid]
        valid_z = p_data['iki_z'].dropna()
        assert len(valid_z) > 10
        assert np.isclose(np.mean(valid_z), 0, atol=1e-3)
        assert np.isclose(np.std(valid_z), 1, atol=1e-3)

@pytest.mark.skipif(
    not os.path.exists("data/interim/filtered_sequences.parquet"), 
    reason="Real data file not found"
)
def test_normalize_real_data_subset(tmp_path):
    # Arrange
    input_path = "data/interim/filtered_sequences.parquet"
    subset_input = tmp_path / "subset_input.parquet"
    output_path = tmp_path / "normalized.parquet"
    
    df = pd.read_parquet(input_path)
    # only first 20 participants, to run faster
    pids_to_keep = df['PARTICIPANT_ID'].unique()[:20]
    df[df['PARTICIPANT_ID'].isin(pids_to_keep)].to_parquet(subset_input)

    run(str(subset_input), str(output_path), sd_threshold=4)
    res = pd.read_parquet(output_path)

    for pid, group in res.groupby('PARTICIPANT_ID'):
        valid_z = group['iki_z'].dropna()
        if len(valid_z) > 1:
            assert np.isclose(np.mean(valid_z), 0, atol=1e-3)
            assert np.isclose(np.std(valid_z), 1, atol=1e-3)