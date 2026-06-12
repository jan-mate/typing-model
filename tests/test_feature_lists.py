import pytest
from src.models.wrappers import (
    MLP_MAIN_FEATURES, MLP_DL_FEATURES, MLP_LINGUISTIC_FEATURES,
    LGBM_FEATURES, LINREG_FEATURES, INFERENCE_FEATURES,
)


def test_mlp_main_features_length():
    assert len(MLP_MAIN_FEATURES) == 35


def test_mlp_dl_features_length():
    assert len(MLP_DL_FEATURES) == 11


def test_mlp_linguistic_features_length():
    assert len(MLP_LINGUISTIC_FEATURES) == 7


def test_lgbm_features_length():
    assert len(LGBM_FEATURES) == 23


def test_linreg_features_length():
    assert len(LINREG_FEATURES) == 41


def test_known_members():
    assert "move_dist" in MLP_MAIN_FEATURES
    assert "finger" in LGBM_FEATURES
    assert "finger" not in MLP_MAIN_FEATURES  # MLP uses one-hot, not categorical
    assert "finger_0" in LINREG_FEATURES
    assert "bigram_frequency" in MLP_LINGUISTIC_FEATURES


def test_inference_features_is_union():
    for f in MLP_MAIN_FEATURES:
        assert f in INFERENCE_FEATURES
    for f in LGBM_FEATURES:
        assert f in INFERENCE_FEATURES


def test_no_duplicates():
    for lst, name in [
        (MLP_MAIN_FEATURES, "MLP_MAIN"),
        (MLP_DL_FEATURES, "MLP_DL"),
        (MLP_LINGUISTIC_FEATURES, "MLP_LINGUISTIC"),
        (LGBM_FEATURES, "LGBM"),
        (LINREG_FEATURES, "LINREG"),
    ]:
        assert len(lst) == len(set(lst)), f"{name}_FEATURES has duplicates"
