import pytest
import pandas as pd
import numpy as np
import os
import re
import json
from src.preprocess_data.split_data import run

def extract_core_word(raw_str):
    return re.sub(r'[^a-zA-Z]', '', str(raw_str)).lower()

def create_synthetic_data():
    return pd.DataFrame({
        "sequence_id": ["S1"] * 12 +["S2"] * 8,
        "key": list("hello world!") + list("test run"),
        "word_index":[0,0,0,0,0, 0, 1,1,1,1,1,1] +[0,0,0,0, 1,1,1,1],
    })

def test_split_synthetic_data(tmp_path):
    enriched_dir = tmp_path / "data" / "enriched"
    freq_dir = tmp_path / "data" / "frequencies"
    enriched_dir.mkdir(parents=True)
    freq_dir.mkdir(parents=True)
    
    with open(freq_dir / "words.json", "w") as f:
        json.dump({"hello": 100, "world": 50, "test": 25, "run": 10}, f)
    with open(freq_dir / "bigrams.json", "w") as f:
        json.dump({"he": 10, "el": 10, "ll": 10, "lo": 10, "wo": 5, "or": 5}, f)
        
    input_path = enriched_dir / "input.parquet"
    output_path = enriched_dir / "output.parquet"
    meta_path = enriched_dir / "meta.json"
    
    create_synthetic_data().to_parquet(input_path)
    run(str(input_path), str(output_path), str(meta_path))
    
    res = pd.read_parquet(output_path)
    assert 'fold_word' in res.columns
    assert 'fold_bigram' in res.columns
    assert res['fold_word'].min() >= -1 and res['fold_word'].max() <= 9

@pytest.fixture(scope="module")
def real_data():
    parquet_path = "data/enriched/enriched_with_folds.parquet"
    meta_path = "data/enriched/fold_metadata.json"
    
    if not os.path.exists(parquet_path) or not os.path.exists(meta_path):
        pytest.skip("Real enriched_with_folds.parquet or metadata not found. Run split_data.py first.")
        
    df = pd.read_parquet(parquet_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return df, meta

def test_metadata_balance(real_data):
    _, meta = real_data
    
    for fold_type in['word', 'bigram']:
        totals = np.array(meta[fold_type]['totals'])
        mean_val = totals.mean()
        max_dev = np.abs(totals - mean_val).max() / mean_val
        
        # folds shouldn't deviate much from the mean (bigrams may deviate more than words)
        assert max_dev < 0.10, f"{fold_type.upper()} folds are highly unbalanced! Max dev: {max_dev:.2%}"

def test_padding_and_bounds(real_data):
    df, _ = real_data
    
    # PAD rows must be exactly -1
    pad_mask = df['key'] == '[PAD]'
    assert (df.loc[pad_mask, 'fold_word'] == -1).all()
    assert (df.loc[pad_mask, 'fold_bigram'] == -1).all()

    # valid keys must be exactly 0-9
    valid_mask = ~pad_mask
    assert df.loc[valid_mask, 'fold_word'].between(0, 9).all()
    assert df.loc[valid_mask, 'fold_bigram'].between(0, 9).all()

def test_no_bigram_leakage(real_data):
    df, _ = real_data
    
    df['prev1'] = df.groupby('sequence_id')['key'].shift(1).fillna("[PAD]")
    df['temp_bigram'] = df['prev1'] + df['key']
    
    valid_bigrams = df[(df['key'] != '[PAD]') & (df['fold_bigram'] != -1)]
    bigram_nunique = valid_bigrams.groupby('temp_bigram')['fold_bigram'].nunique()
    
    leaked = bigram_nunique[bigram_nunique > 1]
    assert len(leaked) == 0, f"Leakage detected! {len(leaked)} bigrams mapped to multiple folds."

def test_no_word_leakage(real_data):
    df, _ = real_data
    
    df['temp_word_start_mask'] = df['word_index'] == 0
    df['word_group_id'] = df.groupby('sequence_id')['temp_word_start_mask'].cumsum()
    
    word_groups = df[df['key'] != '[PAD]'].groupby(['sequence_id', 'word_group_id'])['key'].apply(lambda x: ''.join(x)).reset_index(name='raw_word')
    word_groups['core_word'] = word_groups['raw_word'].apply(extract_core_word)
    
    word_groups = word_groups.merge(
        df[['sequence_id', 'word_group_id', 'fold_word']].drop_duplicates(), 
        on=['sequence_id', 'word_group_id']
    )
    
    word_nunique = word_groups.groupby('core_word')['fold_word'].nunique()
    leaked = word_nunique[word_nunique > 1]
    
    assert len(leaked) == 0, f"Leakage detected! {len(leaked)} core words mapped to multiple folds."

def test_punctuation_grouping(real_data):
    df, _ = real_data
    
    df['temp_word_start_mask'] = df['word_index'] == 0
    df['word_group_id'] = df.groupby('sequence_id')['temp_word_start_mask'].cumsum()
    
    word_groups = df[df['key'] != '[PAD]'].groupby(['sequence_id', 'word_group_id'])['key'].apply(lambda x: ''.join(x)).reset_index(name='raw_word')
    word_groups['core_word'] = word_groups['raw_word'].apply(extract_core_word)
    
    word_groups = word_groups.merge(
        df[['sequence_id', 'word_group_id', 'fold_word']].drop_duplicates(), 
        on=['sequence_id', 'word_group_id']
    )
    
    punc_mask = word_groups['raw_word'].str.contains(r'[^a-zA-Z]')
    punc_words = word_groups[punc_mask]
    
    # punctuated words should land in the same fold as their clean counterpart
    for _, row in punc_words.head(20).iterrows():
        expected_fold_series = word_groups[word_groups['core_word'] == row['core_word']]['fold_word']
        if not expected_fold_series.empty:
            assert row['fold_word'] == expected_fold_series.iloc[0], \
                f"Punctuation word '{row['raw_word']}' did not match core word fold '{row['core_word']}'"