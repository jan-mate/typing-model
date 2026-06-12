import sys
sys.path.append('/content/typing-model')

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score
import warnings

from src.config import ENRICHED_DATA_PATH
from src.utils.data_loader import prepare_sequential_data, get_train_val_masks, get_categorical_indices
from src.utils.training_utils import SEED, mae_with_ci, seed_everything

warnings.filterwarnings('ignore')

CFG = {
    "data_path": ENRICHED_DATA_PATH,
    "target": "iki_z",
    "w_back": 2,
    "w_ahead": 1,
    "fold_id": 0
}

ACTIVE_FEATURES =[
  "move_dist",
  "x", "y", "shift",
  "bigram_frequency", "word_frequency",
  "same_hand", "same_finger",
  "repetition", "same_finger_skipgram",
  "in_roll","out_roll",
  "double_row_jump",
  "sequence_pos",
  "word_length", "word_relative_pos",
  "finger"]

def main():
    seed_everything()
    df_raw = pd.read_parquet(CFG["data_path"])

    X, y, f_b, f_w = prepare_sequential_data(df_raw, CFG["target"], CFG["w_back"], CFG["w_ahead"], ACTIVE_FEATURES)
    del df_raw

    cat_indices = get_categorical_indices(ACTIVE_FEATURES, CFG["w_back"], CFG["w_ahead"])

    train_mask, val_mask = get_train_val_masks(f_b, f_w, CFG["fold_id"])

    print(f"Training LightGBM on {sum(train_mask)} samples...")
    model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=500, verbosity=-1, n_jobs=-1, device='gpu', seed=SEED)
    model.fit(X[train_mask], y[train_mask], categorical_feature=cat_indices)

    preds = model.predict(X[val_mask])
    mae, ci_low, ci_high = mae_with_ci(y[val_mask], preds)
    r2 = r2_score(y[val_mask], preds)

    print("\n=== LightGBM Subset Result ===")
    print(f"ACTIVE_FEATURES ({len(ACTIVE_FEATURES)}): {ACTIVE_FEATURES}")
    print(f"Val MAE: {mae:.4f}  [95% CI: {ci_low:.4f}, {ci_high:.4f}]")
    print(f"Val R2:  {r2:.4f}")

if __name__ == "__main__":
    main()