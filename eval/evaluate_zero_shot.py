import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import json
import warnings
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

from src.config import model_dir, STORAGE_ROOT, ENRICHED_DATA_PATH, DVORAK_DATA_PATH
from src.models.mlp_arch import DynamicMLP
from src.models.wrappers import (
    LGBM_FEATURES, LINREG_FEATURES,
    MLP_MAIN_FEATURES, MLP_DL_FEATURES, MLP_LINGUISTIC_FEATURES,
)
from src.utils.data_loader import (
    prepare_sequential_data, get_train_val_masks,
    build_clusters, cluster_resample_idx,
)

warnings.filterwarnings('ignore')

DVORAK_PATH = DVORAK_DATA_PATH

def compute_metrics(y_true, y_pred, seq_ids, n_boot=2000):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    me = np.mean(y_pred - y_true)
    # sentence-level cluster bootstrap for the MAE CI (resample whole sequences)
    abs_err = np.abs(y_pred - y_true)
    clusters = build_clusters(seq_ids)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = abs_err[cluster_resample_idx(clusters)].mean()
    return mae, r2, me, np.percentile(boot, 2.5), np.percentile(boot, 97.5)

def mean_of_models_ci(y_true, preds_list, seq_ids, n_boot=2000):
    # same cluster bootstrap as compute_metrics, but for the mean of submodels' MAEs
    abs_errs = [np.abs(p - y_true) for p in preds_list]
    clusters = build_clusters(seq_ids)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = cluster_resample_idx(clusters)
        boot[b] = np.mean([ae[idx].mean() for ae in abs_errs])
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading data...")
    df_q = pd.read_parquet(ENRICHED_DATA_PATH)
    df_d = pd.read_parquet(DVORAK_PATH)

    if 'fold_bigram' not in df_d.columns: df_d['fold_bigram'] = -1
    if 'fold_word' not in df_d.columns: df_d['fold_word'] = -1

    results = []

    # linear regression
    lr_params_path = os.path.join(model_dir("linreg", subdir=""), "best_linreg_params.json")
    if os.path.exists(lr_params_path):
        lr_params = json.load(open(lr_params_path))
        w_b, w_a = lr_params.get("w_back", 1), lr_params.get("w_ahead", 1)
        X_q, y_q, fb_q, fw_q = prepare_sequential_data(df_q, "iki_z", w_b, w_a, LINREG_FEATURES)
        X_d, y_d, _, _, seq_d = prepare_sequential_data(df_d, "iki_z", w_b, w_a, LINREG_FEATURES, return_seq_ids=True)

        base_mae_q = mean_absolute_error(y_q, np.zeros_like(y_q))
        base_r2_q = r2_score(y_q, np.zeros_like(y_q))
        base_mae_d, base_r2_d, base_me_d, base_lb_d, base_ub_d = compute_metrics(y_d, np.zeros_like(y_d), seq_d)
        results.append({
            "Model": "Baseline (Predict Mean)", "Val MAE": base_mae_q, "Val R2": base_r2_q,
            "Dvorak MAE": base_mae_d, "Dvorak R2": base_r2_d, "Dvorak ME": base_me_d,
            "Dvorak 95% CI Lower": base_lb_d, "Dvorak 95% CI Upper": base_ub_d,
        })

        ensemble_dir = model_dir("linreg")
        oof_sum, oof_cnt, preds_d, v_folds = np.zeros(len(y_q)), np.zeros(len(y_q)), np.zeros(len(y_d)), 0
        pf_mae_d, pf_me_d, preds_d_list = [], [], []
        for fold_id in tqdm(range(10), desc="Evaluating Linear Regression"):
            model_path = os.path.join(ensemble_dir, f"final_linreg_fold_{fold_id}.pkl")
            scaler_path = os.path.join(ensemble_dir, f"final_linreg_scaler_fold_{fold_id}.pkl")
            if not (os.path.exists(model_path) and os.path.exists(scaler_path)): continue
            m, scaler = joblib.load(model_path), joblib.load(scaler_path)
            _, val_mask = get_train_val_masks(fb_q, fw_q, fold_id)
            coef = torch.tensor(m.coef_, device=device, dtype=torch.float32)
            intercept = float(m.intercept_)
            oof_sum[val_mask] += (torch.tensor(scaler.transform(X_q[val_mask]), device=device, dtype=torch.float32) @ coef + intercept).cpu().numpy()
            oof_cnt[val_mask] += 1
            pred_d_fold = (torch.tensor(scaler.transform(X_d), device=device, dtype=torch.float32) @ coef + intercept).cpu().numpy()
            preds_d += pred_d_fold
            preds_d_list.append(pred_d_fold)
            pf_mae_d.append(mean_absolute_error(y_d, pred_d_fold))
            pf_me_d.append(float(np.mean(pred_d_fold - y_d)))
            v_folds += 1
        if v_folds > 0:
            val_mae = mean_absolute_error(y_q[oof_cnt>0], oof_sum[oof_cnt>0]/oof_cnt[oof_cnt>0])
            val_r2 = r2_score(y_q[oof_cnt>0], oof_sum[oof_cnt>0]/oof_cnt[oof_cnt>0])
            mae_d, r2_d, me_d, lb, ub = compute_metrics(y_d, preds_d/v_folds, seq_d)
            mlb, mub = mean_of_models_ci(y_d, preds_d_list, seq_d)
            results.append({"Model": "Linear Regression", "Val MAE": val_mae, "Val R2": val_r2,
                            "Dvorak MAE": mae_d, "Dvorak R2": r2_d, "Dvorak ME": me_d,
                            "Dvorak 95% CI Lower": lb, "Dvorak 95% CI Upper": ub,
                            "Dvorak MAE (mean models)": float(np.mean(pf_mae_d)),
                            "Dvorak MAE (model std)": float(np.std(pf_mae_d)),
                            "Dvorak MAE (mean) CI Lower": float(mlb),
                            "Dvorak MAE (mean) CI Upper": float(mub),
                            "Dvorak ME (mean models)": float(np.mean(pf_me_d))})

    # lightgbm
    lgb_params_path = os.path.join(model_dir("lgbm", subdir=""), "best_optuna_lgbm_params.json")
    if os.path.exists(lgb_params_path):
        lgb_params = json.load(open(lgb_params_path))
        w_b, w_a = lgb_params.get("w_back", 3), lgb_params.get("w_ahead", 1)
        X_q, y_q, fb_q, fw_q = prepare_sequential_data(df_q, "iki_z", w_b, w_a, LGBM_FEATURES)
        X_d, y_d, _, _, seq_d = prepare_sequential_data(df_d, "iki_z", w_b, w_a, LGBM_FEATURES, return_seq_ids=True)

        ensemble_dir = model_dir("lgbm", subdir="ensemble")
        oof_sum, oof_cnt, preds_d, v_folds = np.zeros(len(y_q)), np.zeros(len(y_q)), np.zeros(len(y_d)), 0
        pf_mae_d, pf_me_d, preds_d_list = [], [], []
        for fold_id in tqdm(range(10), desc="Evaluating LightGBM"):
            path = os.path.join(ensemble_dir, f"final_lgbm_fold_{fold_id}.txt")
            if not os.path.exists(path): continue
            m = lgb.Booster(model_file=path)
            _, val_mask = get_train_val_masks(fb_q, fw_q, fold_id)
            oof_sum[val_mask] += m.predict(X_q[val_mask])
            oof_cnt[val_mask] += 1
            pred_d_fold = m.predict(X_d)
            preds_d += pred_d_fold
            preds_d_list.append(pred_d_fold)
            pf_mae_d.append(mean_absolute_error(y_d, pred_d_fold))
            pf_me_d.append(float(np.mean(pred_d_fold - y_d)))
            v_folds += 1
        if v_folds > 0:
            val_mae = mean_absolute_error(y_q[oof_cnt>0], oof_sum[oof_cnt>0]/oof_cnt[oof_cnt>0])
            val_r2 = r2_score(y_q[oof_cnt>0], oof_sum[oof_cnt>0]/oof_cnt[oof_cnt>0])
            mae_d, r2_d, me_d, lb, ub = compute_metrics(y_d, preds_d/v_folds, seq_d)
            mlb, mub = mean_of_models_ci(y_d, preds_d_list, seq_d)
            results.append({"Model": "LightGBM", "Val MAE": val_mae, "Val R2": val_r2,
                            "Dvorak MAE": mae_d, "Dvorak R2": r2_d, "Dvorak ME": me_d,
                            "Dvorak 95% CI Lower": lb, "Dvorak 95% CI Upper": ub,
                            "Dvorak MAE (mean models)": float(np.mean(pf_mae_d)),
                            "Dvorak MAE (model std)": float(np.std(pf_mae_d)),
                            "Dvorak MAE (mean) CI Lower": float(mlb),
                            "Dvorak MAE (mean) CI Upper": float(mub),
                            "Dvorak ME (mean models)": float(np.mean(pf_me_d))})

    # mlp variants
    mlp_configs = [
        {"name": "MLP (Main)",  "model": "mlp_main",       "json": "best_optuna_mlp_params.json", "feats": MLP_MAIN_FEATURES,       "prefix": "final_mlp_main"},
        {"name": "MLP (DL)",    "model": "mlp_dl",         "json": "best_dl_params.json",         "feats": MLP_DL_FEATURES,         "prefix": "final_mlp_dl"},
        {"name": "MLP (Ling)",  "model": "mlp_linguistic", "json": "best_linguistic_params.json", "feats": MLP_LINGUISTIC_FEATURES, "prefix": "final_mlp_linguistic"},
    ]

    for cfg in mlp_configs:
        json_path = os.path.join(model_dir(cfg["model"], subdir=""), cfg["json"])
        if not os.path.exists(json_path): continue
        ensemble_dir = model_dir(cfg["model"])
        p = json.load(open(json_path))
        w_b, w_a = p.get("w_back", 3), p.get("w_ahead", 1)
        X_q, y_q, fb_q, fw_q = prepare_sequential_data(df_q, "iki_z", w_b, w_a, cfg["feats"])
        X_d, y_d, _, _, seq_d = prepare_sequential_data(df_d, "iki_z", w_b, w_a, cfg["feats"], return_seq_ids=True)

        oof_sum, oof_cnt, preds_d, v_folds = np.zeros(len(y_q)), np.zeros(len(y_q)), np.zeros(len(y_d)), 0
        pf_mae_d, pf_me_d, preds_d_list = [], [], []
        for fold_id in tqdm(range(10), desc=f"Evaluating {cfg['name']}"):
            model_path = os.path.join(ensemble_dir, f"{cfg['prefix']}_fold_{fold_id}.pth")
            scaler_path = os.path.join(ensemble_dir, f"{cfg['prefix']}_scaler_fold_{fold_id}.pkl")
            if not (os.path.exists(model_path) and os.path.exists(scaler_path)): continue
            scaler = joblib.load(scaler_path)
            m = DynamicMLP(X_q.shape[1], p['n_layers'], p['hidden_dim'], p['dropout'], p['activation']).to(device)
            m.load_state_dict(torch.load(model_path, map_location=device))
            m.eval()
            _, val_mask = get_train_val_masks(fb_q, fw_q, fold_id)
            with torch.no_grad():
                oof_sum[val_mask] += m(torch.tensor(scaler.transform(X_q[val_mask]), dtype=torch.float32, device=device)).cpu().numpy()
                oof_cnt[val_mask] += 1
                pred_d_fold = m(torch.tensor(scaler.transform(X_d), dtype=torch.float32, device=device)).cpu().numpy()
                preds_d += pred_d_fold
                preds_d_list.append(pred_d_fold)
                pf_mae_d.append(mean_absolute_error(y_d, pred_d_fold))
                pf_me_d.append(float(np.mean(pred_d_fold - y_d)))
                v_folds += 1
        if v_folds > 0:
            val_mae = mean_absolute_error(y_q[oof_cnt>0], oof_sum[oof_cnt>0]/oof_cnt[oof_cnt>0])
            val_r2 = r2_score(y_q[oof_cnt>0], oof_sum[oof_cnt>0]/oof_cnt[oof_cnt>0])
            mae_d, r2_d, me_d, lb, ub = compute_metrics(y_d, preds_d/v_folds, seq_d)
            mlb, mub = mean_of_models_ci(y_d, preds_d_list, seq_d)
            results.append({"Model": cfg['name'], "Val MAE": val_mae, "Val R2": val_r2,
                            "Dvorak MAE": mae_d, "Dvorak R2": r2_d, "Dvorak ME": me_d,
                            "Dvorak 95% CI Lower": lb, "Dvorak 95% CI Upper": ub,
                            "Dvorak MAE (mean models)": float(np.mean(pf_mae_d)),
                            "Dvorak MAE (model std)": float(np.std(pf_mae_d)),
                            "Dvorak MAE (mean) CI Lower": float(mlb),
                            "Dvorak MAE (mean) CI Upper": float(mub),
                            "Dvorak ME (mean models)": float(np.mean(pf_me_d))})

    print("\n" + "="*125)
    print("FINAL ZERO-SHOT EVALUATION METRICS")
    print("="*125)
    if results: print(pd.DataFrame(results).to_string(index=False))
    else: print("No models were found to evaluate.")
    print("="*125)

if __name__ == "__main__":
    main()