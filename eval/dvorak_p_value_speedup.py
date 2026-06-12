# bootstrapped p-value for the Dvorak vs QWERTY speedup, combining two uncertainty
# sources: the raw speed advantage (on the corpus) and the layout bias shift (on test data)

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.wrappers import LGBM as LGBMSpeedModel, LGBM_FEATURES
from src.config import CORPUS_PATH, ENRICHED_DATA_PATH, DVORAK_DATA_PATH, STORAGE_ROOT, model_dir
from src.enrichment.engine import EnrichmentEngine
from src.utils.corpus_eval import predict_corpus, sample_corpus
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

def main():
    print("=" * 60)
    print("Exact Bootstrapped P-Value for Dvorak Speedup (LightGBM)")
    print("=" * 60)

    print("\n1. Evaluating Corpus (Raw Advantage)...")
    sentences = sample_corpus(CORPUS_PATH, n_sentences=25_000)
    
    engine = EnrichmentEngine(
        unigrams_path=f"{project_root}/data/frequencies/unigrams_zipf.json",
        bigrams_path=f"{project_root}/data/frequencies/bigrams_zipf.json",
        words_path=f"{project_root}/data/frequencies/words_zipf.json",
        movement_features_path=f"{project_root}/data/layouts/movement_features.json",
    )
    
    model = LGBMSpeedModel(model_dir("lgbm"))
    
    qwerty_paths = {
        "layout_path":     f"{project_root}/data/layouts/qwerty_us.json",
        "layout_map_path": f"{project_root}/data/layouts/layout_map.json",
        "shifts_path":     f"{project_root}/data/layouts/shifts_us.json",
    }
    dvorak_paths = {
        "layout_path":     f"{project_root}/data/layouts/dvorak.json",
        "layout_map_path": f"{project_root}/data/layouts/layout_map.json",
        "shifts_path":     f"{project_root}/data/layouts/shifts_us.json",
    }
    
    mean_z_q, _, _ = predict_corpus(model, sentences, engine, qwerty_paths)
    mean_z_d, _, _ = predict_corpus(model, sentences, engine, dvorak_paths)
    raw_diffs = mean_z_q - mean_z_d
    print(f"-> Mean Raw Advantage (QWERTY - Dvorak IKI_z): {raw_diffs.mean():.6f}")

    print("\n2. Evaluating Test Sets (Bias Shift)...")
    df_q = pd.read_parquet(QWERTY_PATH)
    df_d = pd.read_parquet(DVORAK_PATH)
    if 'fold_bigram' not in df_d.columns: df_d['fold_bigram'] = -1
    if 'fold_word' not in df_d.columns: df_d['fold_word'] = -1

    cfg_path = os.path.join(model_dir("lgbm", ""), "best_optuna_lgbm_params.json")
    with open(cfg_path) as f: p = json.load(f)
    w_back, w_ahead = p.get("w_back", 3), p.get("w_ahead", 1)

    X_q, y_q, fb_q, fw_q, seq_q = prepare_sequential_data(df_q, "iki_z", w_back, w_ahead, LGBM_FEATURES)
    X_d, y_d, _, _, seq_d = prepare_sequential_data(df_d, "iki_z", w_back, w_ahead, LGBM_FEATURES)

    fold_q_biases = []
    fold_d_biases = []
    fold_q_seqs = []
    fold_d_seqs = []

    for fold_id in tqdm(range(10), desc="Evaluating Folds"):
        path = os.path.join(model_dir("lgbm"), f"final_lgbm_fold_{fold_id}.txt")
        if not os.path.exists(path): continue
        _, val_mask = get_train_val_masks(fb_q, fw_q, fold_id)
        if sum(val_mask) == 0: continue

        m = lgb.Booster(model_file=path)

        val_preds = m.predict(X_q[val_mask])
        val_y = y_q[val_mask]
        fold_q_biases.append(val_preds - val_y)
        fold_q_seqs.append(seq_q[val_mask])

        d_preds = m.predict(X_d)
        fold_d_biases.append(d_preds - y_d)
        fold_d_seqs.append(seq_d)

    # point estimate matches model_bias.py
    point_shifts = [np.mean(d) - np.mean(q) for d, q in zip(fold_d_biases, fold_q_biases)]
    mean_bias_shift = np.mean(point_shifts)
    print(f"-> Mean Bias Shift (Dvorak Bias - QWERTY Bias): {mean_bias_shift:.6f}")

    print("\n3. Bootstrapping Adjusted Advantage (2000 iterations)...")
    n_iterations = 2000
    boot_adj_adv = np.zeros(n_iterations)

    # precompute per-cluster sums/counts so each bootstrap draw is a vectorized lookup
    q_stats = [cluster_sums_counts(b, build_clusters(s)) for b, s in zip(fold_q_biases, fold_q_seqs)]
    d_stats = [cluster_sums_counts(b, build_clusters(s)) for b, s in zip(fold_d_biases, fold_d_seqs)]

    for i in tqdm(range(n_iterations)):
        idx_corp = np.random.randint(0, len(raw_diffs), len(raw_diffs))
        boot_raw = np.mean(raw_diffs[idx_corp])

        # bias shift: resample whole sentences
        boot_shifts = []
        for (q_sums, q_counts), (d_sums, d_counts) in zip(q_stats, d_stats):
            q_mean = bootstrap_cluster_mean(q_sums, q_counts)
            d_mean = bootstrap_cluster_mean(d_sums, d_counts)
            boot_shifts.append(d_mean - q_mean)
        boot_bias = np.mean(boot_shifts)

        boot_adj_adv[i] = boot_raw + boot_bias

    mean_adj = boot_adj_adv.mean()
    ci_lower = np.percentile(boot_adj_adv, 2.5)
    ci_upper = np.percentile(boot_adj_adv, 97.5)

    p_val_one_sided = np.mean(boot_adj_adv <= 0) if mean_adj > 0 else np.mean(boot_adj_adv >= 0)
    p_val = min(p_val_one_sided * 2, 1.0)

    # convert z to speedup %, using IKI std=50.4ms and mean=110.5ms
    pct_mean = (mean_adj * 50.4) / 110.5 * 100
    pct_lower = (ci_lower * 50.4) / 110.5 * 100
    pct_upper = (ci_upper * 50.4) / 110.5 * 100

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Adjusted Advantage (Z-Score): {mean_adj:.4f}")
    print(f"95% Confidence Interval:      [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"P-Value (Two-Sided):          {p_val:.4f}")
    print("-" * 60)
    print(f"Typing Speedup (%):           {pct_mean:.1f}%")
    print(f"95% CI (%):                   [{pct_lower:.1f}%, {pct_upper:.1f}%]")
    print("=" * 60)

if __name__ == "__main__":
    main()