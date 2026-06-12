import sys
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import json
import warnings
import torch
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import ENRICHED_DATA_PATH, DVORAK_DATA_PATH, STORAGE_ROOT, model_dir
from src.models.mlp_arch import DynamicMLP
from src.models.wrappers import (
    LGBM_FEATURES,
    LINREG_FEATURES,
    MLP_MAIN_FEATURES,
    MLP_DL_FEATURES,
)
from src.utils.data_loader import PAD_VALUES, build_clusters, cluster_sums_counts, bootstrap_cluster_mean

QWERTY_PATH = ENRICHED_DATA_PATH
DVORAK_PATH = DVORAK_DATA_PATH

def prepare_sequential_data(df_raw, target_col, w_back, w_ahead, active_features):
    df = df_raw.copy()
    for col in active_features:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].fillna(PAD_VALUES.get(col, -99.0)).astype(np.float32)
        
    f_vals = df[active_features].values
    t_vals = df[target_col].values.astype(np.float32)
    keys = df['key'].values
    
    valid = np.where((keys != '[PAD]') & (~np.isnan(t_vals)))[0]
    valid = valid[(valid >= w_back) & (valid < len(df) - w_ahead)]
    
    num_feat = len(active_features)
    X = np.zeros((len(valid), (w_back + w_ahead + 1) * num_feat), dtype=np.float32)
    for i, idx in enumerate(valid):
        X[i] = f_vals[idx - w_back : idx + w_ahead + 1].flatten()
        
    fb = df['fold_bigram'].values[valid] if 'fold_bigram' in df.columns else np.full(len(valid), -1)
    fw = df['fold_word'].values[valid] if 'fold_word' in df.columns else np.full(len(valid), -1)
    seq = df['original_sequence_id'].values[valid]

    return X, t_vals[valid], fb, fw, seq

def get_train_val_masks(fold_bigram, fold_word, val_fold):
    val_mask = (fold_bigram == val_fold) | (fold_word == val_fold)
    train_mask = ~val_mask & (fold_bigram != -1) & (fold_word != -1)
    return train_mask, val_mask

def compute_extended_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    if np.std(y_pred) == 0: return mae, bias, np.mean(y_pred), 0.0, 0.0
    return mae, bias, np.mean(y_pred), pearsonr(y_true, y_pred)[0], spearmanr(y_true, y_pred)[0]

def bootstrap_ensemble_bias_shift(fold_q_biases, fold_d_biases, fold_q_seqs, fold_d_seqs, n_iterations=2000):
    # sentence-level cluster bootstrap; precompute per-cluster sums/counts so each draw
    # is a vectorized lookup
    q_stats = [cluster_sums_counts(b, build_clusters(s)) for b, s in zip(fold_q_biases, fold_q_seqs)]
    d_stats = [cluster_sums_counts(b, build_clusters(s)) for b, s in zip(fold_d_biases, fold_d_seqs)]
    boot_shifts = []
    for _ in range(n_iterations):
        iter_shifts = []
        for (q_sums, q_counts), (d_sums, d_counts) in zip(q_stats, d_stats):
            q_mean = bootstrap_cluster_mean(q_sums, q_counts)
            d_mean = bootstrap_cluster_mean(d_sums, d_counts)
            iter_shifts.append(d_mean - q_mean)
        boot_shifts.append(np.mean(iter_shifts))
    boot_shifts = np.array(boot_shifts)
    return boot_shifts.mean(), np.percentile(boot_shifts, 2.5), np.percentile(boot_shifts, 97.5)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_q = pd.read_parquet(QWERTY_PATH)
    df_d = pd.read_parquet(DVORAK_PATH)
    
    if 'fold_bigram' not in df_d.columns: df_d['fold_bigram'] = -1
    if 'fold_word' not in df_d.columns: df_d['fold_word'] = -1
    
    results = []

    MODELS = [
        ("Linear Regression", LINREG_FEATURES,   model_dir("linreg"),    "final_linreg",    "pkl", model_dir("linreg", "") + "/best_linreg_params.json"),
        ("LightGBM",          LGBM_FEATURES,     model_dir("lgbm"),      "final_lgbm",      "txt", model_dir("lgbm", "") + "/best_optuna_lgbm_params.json"),
        ("MLP (Main)",        MLP_MAIN_FEATURES,  model_dir("mlp_main"),  "final_mlp_main",  "pth", model_dir("mlp_main", "") + "/best_optuna_mlp_params.json"),
        ("MLP (DL)",          MLP_DL_FEATURES,    model_dir("mlp_dl"),    "final_mlp_dl",    "pth", model_dir("mlp_dl", "") + "/best_dl_params.json"),
    ]

    for name, feats, ens_dir, prefix, ext, cfg_path in MODELS:
        if not os.path.exists(cfg_path):
            print(f"Skipping {name}: Config not found at {cfg_path}")
            continue

        p = json.load(open(cfg_path))
        w_back = p.get("w_back", 3)
        w_ahead = p.get("w_ahead", 1)

        X_q, y_q, fb_q, fw_q, seq_q = prepare_sequential_data(df_q, "iki_z", w_back, w_ahead, feats)
        X_d, y_d, _, _, seq_d = prepare_sequential_data(df_d, "iki_z", w_back, w_ahead, feats)

        fold_q_biases = []
        fold_d_biases = []
        fold_q_seqs = []
        fold_d_seqs = []
        fold_val_biases = []
        fold_d_point_biases = []

        for fold_id in tqdm(range(10), desc=f"Evaluating {name}"):
            path = os.path.join(ens_dir, f"{prefix}_fold_{fold_id}.{ext}")
            if not os.path.exists(path):
                print(f"\n[Warning] Model missing: {path}")
                continue

            _, val_mask = get_train_val_masks(fb_q, fw_q, fold_id)
            if sum(val_mask) == 0:
                print(f"\n[Warning] No validation data for fold {fold_id}")
                continue

            if ext == "txt":
                model = lgb.Booster(model_file=path)
                val_preds = model.predict(X_q[val_mask])
                d_preds = model.predict(X_d)
            elif ext == "pkl":
                model = joblib.load(path)
                scaler_path = path.replace("_fold", "_scaler_fold")
                if not os.path.exists(scaler_path):
                    print(f"\n[Warning] Scaler missing: {scaler_path}")
                    continue
                scaler = joblib.load(scaler_path)

                X_q_scaled = scaler.transform(X_q[val_mask])
                X_d_scaled = scaler.transform(X_d)

                val_preds = model.predict(X_q_scaled)
                d_preds = model.predict(X_d_scaled)
            else:
                scaler_path = path.replace(".pth", "").replace("_fold", "_scaler_fold") + ".pkl"
                if not os.path.exists(scaler_path):
                    print(f"\n[Warning] MLP Scaler missing: {scaler_path}")
                    continue
                scaler = joblib.load(scaler_path)
                model = DynamicMLP(X_q.shape[1], p['n_layers'], p['hidden_dim'], p['dropout'], p['activation']).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                with torch.no_grad():
                    val_preds = model(torch.tensor(scaler.transform(X_q[val_mask]), dtype=torch.float32, device=device)).cpu().numpy()
                    d_preds = model(torch.tensor(scaler.transform(X_d), dtype=torch.float32, device=device)).cpu().numpy()

            val_y = y_q[val_mask]
            fold_q_biases.append(val_preds - val_y)
            fold_d_biases.append(d_preds - y_d)
            fold_q_seqs.append(seq_q[val_mask])
            fold_d_seqs.append(seq_d)
            fold_val_biases.append(np.mean(val_preds - val_y))
            fold_d_point_biases.append(np.mean(d_preds - y_d))

        if fold_q_biases:
            shift, low, high = bootstrap_ensemble_bias_shift(fold_q_biases, fold_d_biases, fold_q_seqs, fold_d_seqs)
            results.append({
                "Model": name,
                "Val Bias": np.mean(fold_val_biases),
                "Dvorak Bias": np.mean(fold_d_point_biases),
                "Bias Shift (Mean)": np.mean(fold_d_point_biases) - np.mean(fold_val_biases),
                "Bias Shift (2.5%)": low,
                "Bias Shift (97.5%)": high
            })
            
    if results:
        print("\n" + "="*120)
        print("BOOTSTRAPPED BIAS SHIFT EVALUATION (2000 ITERATIONS, ENSEMBLE)")
        print("="*120)
        print(pd.DataFrame(results).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        print("=" * 120)

if __name__ == "__main__":
    main()