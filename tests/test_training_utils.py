import json
import random

import numpy as np
import pytest

from src.utils.training_utils import (
    SEED,
    load_params,
    mae_with_ci,
    seed_everything,
    seed_worker,
)




def test_load_params_missing_file():
    with pytest.raises(FileNotFoundError):
        load_params("/nonexistent/path/params.json")


def test_load_params_returns_dict(tmp_path, capsys):
    params = {"w_back": 3, "w_ahead": 1, "lr": 0.001}
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))

    result = load_params(str(path))
    assert result == params
    assert isinstance(result, dict)


def test_load_params_prints_contents(tmp_path, capsys):
    params = {"learning_rate": 0.01, "n_layers": 3}
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))

    load_params(str(path))
    captured = capsys.readouterr()
    assert "learning_rate" in captured.out
    assert "n_layers" in captured.out




def test_mae_with_ci_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mae, lo, hi = mae_with_ci(y, y)
    assert mae == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_mae_with_ci_known_errors():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, -1.0, 2.0, -2.0])
    mae, _, _ = mae_with_ci(y_true, y_pred)
    assert mae == pytest.approx(1.5)


def test_mae_with_ci_ordering():
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(500)
    y_pred = rng.standard_normal(500)
    mae, lo, hi = mae_with_ci(y_true, y_pred)
    assert lo <= mae <= hi


def test_mae_with_ci_finite():
    rng = np.random.default_rng(1)
    y_true = rng.standard_normal(100)
    y_pred = rng.standard_normal(100)
    mae, lo, hi = mae_with_ci(y_true, y_pred)
    for v in (mae, lo, hi):
        assert np.isfinite(v)


def test_mae_with_ci_width_shrinks_with_n():
    rng = np.random.default_rng(42)
    big = rng.standard_normal(10_000)
    small = rng.standard_normal(100)
    _, lo_big, hi_big = mae_with_ci(np.zeros_like(big), big)
    _, lo_small, hi_small = mae_with_ci(np.zeros_like(small), small)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_mae_with_ci_confidence_level():
    rng = np.random.default_rng(2)
    y_true = rng.standard_normal(1000)
    y_pred = rng.standard_normal(1000)
    _, lo_95, hi_95 = mae_with_ci(y_true, y_pred, confidence=0.95)
    _, lo_99, hi_99 = mae_with_ci(y_true, y_pred, confidence=0.99)
    assert (hi_99 - lo_99) > (hi_95 - lo_95)




def test_seed_everything_numpy_deterministic():
    seed_everything()
    a = np.random.rand(5)
    seed_everything()
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_seed_everything_random_deterministic():
    seed_everything()
    a = [random.random() for _ in range(5)]
    seed_everything()
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_torch_deterministic():
    torch = pytest.importorskip("torch")
    seed_everything(use_torch=True)
    a = torch.rand(5)
    seed_everything(use_torch=True)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_seed_everything_uses_default_seed_constant():
    # the module-level SEED is what gets applied
    seed_everything()
    a = np.random.rand(3)
    np.random.seed(SEED)
    b = np.random.rand(3)
    assert np.array_equal(a, b)


def test_seed_worker_callable():
    pytest.importorskip("torch")
    seed_worker(0)
    seed_worker(7)  # should not raise


def test_seed_worker_makes_workers_distinct():
    pytest.importorskip("torch")
    seed_worker(0)
    a = np.random.rand(5)
    seed_worker(7)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)
