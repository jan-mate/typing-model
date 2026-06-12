import sys
sys.path.append('/content/typing-model')

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import warnings

from src.utils.data_loader import prepare_sequential_data, get_train_val_masks, get_categorical_indices, get_all_features
from src.utils.training_utils import SEED, seed_everything, mae_with_ci

warnings.filterwarnings('ignore')

DRIVE_BASE_PATH = "/content/drive/MyDrive/typing-model"

CFG = {
    "data_path": f"{DRIVE_BASE_PATH}/enriched_data/enriched_with_folds.parquet",
    "model_save_dir": f"{DRIVE_BASE_PATH}/models/lgbm",
    "target": "iki_z",
    "w_back": 2,
    "w_ahead": 1,
    "fold_id": 0,
    "add_noise": True,
    "use_manual_features": True 
}

MANUAL_FEATURES = [
    'move_dist', 'move_cos', 'move_sin', 'x', 'y', 'shift', 
    'bigram_frequency', 'word_frequency', 'same_hand', 'same_finger', 
    'repetition', 'skipgram_repetition', 'same_finger_skipgram', 
    'in_roll', 'out_roll', 'redirects', 'double_row_jump', 
    'sequence_pos', 'word_index', 'word_length', 'word_relative_pos', 
    'finger', 'hand'
]

if CFG["use_manual_features"]:
    ACTIVE_FEATURES = list(MANUAL_FEATURES)
    if CFG["add_noise"] and "random_noise" not in ACTIVE_FEATURES:
        ACTIVE_FEATURES.append("random_noise")
else:
    ACTIVE_FEATURES = get_all_features(add_noise=CFG["add_noise"], categorical_features=True)

def main():
    seed_everything()
    df_raw = pd.read_parquet(CFG["data_path"])
    
    X, y, f_b, f_w = prepare_sequential_data(df_raw, CFG["target"], CFG["w_back"], CFG["w_ahead"], ACTIVE_FEATURES, add_noise=CFG["add_noise"])
    del df_raw
    
    cat_indices = get_categorical_indices(ACTIVE_FEATURES, CFG["w_back"], CFG["w_ahead"])

    train_mask, val_mask = get_train_val_masks(f_b, f_w, CFG["fold_id"])
    
    model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=500, verbosity=-1, n_jobs=-1, device='gpu', seed=SEED)
    
    print(f"Training LightGBM on {sum(train_mask)} samples with {len(ACTIVE_FEATURES)} features...")
    model.fit(X[train_mask], y[train_mask], categorical_feature=cat_indices)
    
    preds = model.predict(X[val_mask])
    mae, ci_low, ci_high = mae_with_ci(y[val_mask], preds)
    print(f"\nWord Fold Validation MAE: {mae:.4f} [95% CI: {ci_low:.4f}, {ci_high:.4f}]")
    
    print("Calculating SHAP values...")
    X_val_subset = X[val_mask]
    sample_size = min(50000, X_val_subset.shape[0])
    np.random.seed(42)
    sample_indices = np.random.choice(X_val_subset.shape[0], sample_size, replace=False)
    X_sample = X_val_subset[sample_indices]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    num_features = len(ACTIVE_FEATURES)
    num_steps = CFG["w_back"] + CFG["w_ahead"] + 1
    
    feature_importances = {}
    for i, feat in enumerate(ACTIVE_FEATURES):
        cols =[i + (step * num_features) for step in range(num_steps)]
        feature_importances[feat] = np.sum(mean_abs_shap[cols])
        
    print("\n--- SHAP Feature Importances ---")
    for i, (feat, imp) in enumerate(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)):
        print(f"{i+1:2d}. {feat:<25}: {imp:.5f}")

if __name__ == "__main__":
    main()