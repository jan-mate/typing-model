import json
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocess_data.calculate_frequencies import run

def test_calculate_frequencies_synthetic(tmp_path):
    out_dir = tmp_path / "frequencies"
    corpus_path = tmp_path / "dummy_corpus.parquet"
    
    df = pd.DataFrame({"body": ["the that"] * 5})
    df.to_parquet(corpus_path)
    
    run(str(out_dir), str(corpus_path))
    
    with open(out_dir / "unigrams.json", "r") as f:
        unigrams = json.load(f)
    
    assert unigrams["t"] > unigrams["h"]
    assert unigrams["h"] > unigrams["e"]
    
    with open(out_dir / "bigrams.json", "r") as f:
        bigrams = json.load(f)
        
    assert "th" in bigrams
    assert "ha" in bigrams
    assert bigrams["th"] > bigrams["at"]

@pytest.mark.skipif(
    not os.path.exists("data/raw/combined_corpus_2M.parquet"),
    reason="Real corpus file not found"
)
def test_calculate_frequencies_real_data_relative_commonality(tmp_path):
    out_dir = tmp_path / "frequencies"
    corpus_path = "data/raw/combined_corpus_2M.parquet"
    
    run(str(out_dir), corpus_path)
    
    with open(out_dir / "unigrams.json", "r") as f:
        uni = json.load(f)
    assert uni.get("e", 0) > uni.get("z", 0)
    assert uni.get("t", 0) > uni.get("q", 0)
    assert uni.get(" ", 0) > uni.get("a", 0)

    with open(out_dir / "bigrams.json", "r") as f:
        bi = json.load(f)
    assert bi.get("th", 0) > bi.get("xd", 0)
    assert bi.get("he", 0) > bi.get("qx", 0)
    assert bi.get("in", 0) > bi.get("zj", 0)

    with open(out_dir / "trigrams.json", "r") as f:
        tri = json.load(f)
    assert tri.get("the", 0) > tri.get(" xd", 0)
    assert tri.get("ing", 0) > tri.get("qzz", 0)

    with open(out_dir / "words.json", "r") as f:
        w = json.load(f)
    assert w.get("the", 0) > w.get("keyboard", 0)
    assert w.get("it", 0) > w.get("optimization", 0)

@pytest.mark.skipif(
    not os.path.exists("data/raw/combined_corpus_2M.parquet"),
    reason="Real corpus file not found"
)
def test_calculate_frequencies_zipf_ranges(tmp_path):
    out_dir = tmp_path / "frequencies"
    corpus_path = "data/raw/combined_corpus_2M.parquet"
    
    run(str(out_dir), corpus_path)
    
    zipf_files = ["unigrams_zipf.json", "bigrams_zipf.json", "trigrams_zipf.json", "words_zipf.json"]
    
    for fname in zipf_files:
        with open(out_dir / fname, "r") as f:
            data = json.load(f)
        for val in data.values():
            assert 0 <= val <= 10