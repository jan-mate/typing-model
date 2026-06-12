import os
import sys
import pytest
import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import get_all_features, prepare_sequential_data, get_categorical_indices

import pyarrow.parquet as pq

@pytest.fixture
def dummy_data():
    path = "data/enriched/enriched_with_folds.parquet"
    if not os.path.exists(path):
        pytest.skip(f"Real data file not found at {path}, skipping tests.")
    
    pf = pq.ParquetFile(path)
    return next(pf.iter_batches(batch_size=5000)).to_pandas()

def test_get_all_features_toggles():
    features_ohe = get_all_features(categorical_features=False)
    assert "hand_0" in features_ohe
    assert "finger_0" in features_ohe
    assert "hand" not in features_ohe
    assert "is_pad" in features_ohe
    
    features_cat = get_all_features(categorical_features=True)
    assert "hand" in features_cat
    assert "finger" in features_cat
    assert "hand_0" not in features_cat

def test_get_categorical_indices():
    features = get_all_features(categorical_features=True)
    w_back, w_ahead = 2, 1
    
    indices = get_categorical_indices(features, w_back, w_ahead)
    assert isinstance(indices, list)
    
    assert len(indices) == 3 * (w_back + w_ahead + 1)
    
    for idx in indices:
        feat_name = features[idx % len(features)]
        assert feat_name in ["finger", "finger_type", "hand"]

def test_prepare_sequential_data_shapes(dummy_data):
    features = get_all_features(categorical_features=True)
    w_back, w_ahead = 2, 1
    
    X, y, f_b, f_w = prepare_sequential_data(dummy_data, "iki_z", w_back, w_ahead, features)
    
    expected_cols = len(features) * (w_back + w_ahead + 1)
    assert X.shape[1] == expected_cols
    
    assert len(y) == X.shape[0]
    assert len(f_b) == X.shape[0]
    assert len(f_w) == X.shape[0]
    
    assert not np.isnan(X).any(), "Found NaNs in processed X array!"