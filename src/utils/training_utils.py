import json
import os

import numpy as np

SEED = 42


def load_params(json_path: str) -> dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Missing hyperparameter file: {json_path}\n"
            "Run the corresponding HPO script first to generate it."
        )
    with open(json_path) as f:
        params = json.load(f)
    print("\n" + "=" * 50)
    print("Loaded Hyperparameters")
    print("=" * 50)
    for k, v in params.items():
        print(f"{k}: {v}")
    return params


def mae_with_ci(y_true, y_pred, confidence: float = 0.95):
    # gaussian SE for the MAE CI
    from scipy.stats import norm

    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    mae = float(errors.mean())
    se = float(errors.std(ddof=1) / np.sqrt(len(errors)))
    z = float(norm.ppf((1 + confidence) / 2))
    return mae, mae - z * se, mae + z * se


def seed_everything(use_torch: bool = False) -> None:
    import random

    random.seed(SEED)
    np.random.seed(SEED)
    if use_torch:
        import torch

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    import random

    import torch

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
