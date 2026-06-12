import numpy as np
from tqdm import tqdm

PAD_VALUES = {
    "is_pad": 1.0,
    "move_dist": -1.0, "move_sin": -2.0, "move_cos": -2.0,
    "coords": -5.0, "x": -5.0, "y": -5.0, 
    "hand": -1.0, "shift": -1.0, "finger": -1.0, "finger_type": -1.0,
    "unigram_frequency": -1.0, "bigram_frequency": -1.0, "word_frequency": -1.0,
    "same_hand": -1.0, "same_finger": -1.0, "same_finger_trigram": -1.0,
    "repetition": -1.0, "skipgram_repetition": -1.0, "same_finger_skipgram": -1.0,
    "in_roll": -1.0, "out_roll": -1.0, "in_triroll": -1.0, "out_triroll": -1.0,
    "redirects": -1.0, "double_row_jump": -1.0, "scissors": -1.0,
    "sequence_pos": -1.0, "sequence_length": -1.0, "sequence_relative_pos": -1.0,
    "word_index": -1.0, "word_length": -1.0, "word_relative_pos": -1.0,
    "is_word_start": -1.0, "is_word_end": -1.0,
    "is_syllable_start": -1.0, "is_syllable_end": -1.0,
    "random_noise": -5.0
}

for i in range(10): PAD_VALUES[f'finger_{i}'] = -1.0
for i in range(5): PAD_VALUES[f'finger_type_{i}'] = -1.0
for i in range(3): PAD_VALUES[f'hand_{i}'] = -1.0

def get_all_features(add_noise=False, categorical_features=False):
    features =[
        "is_pad", "move_dist", "move_sin", "move_cos",
        "x", "y", "shift",
        "unigram_frequency", "bigram_frequency", "word_frequency",
        "same_hand", "same_finger", "same_finger_trigram",
        "repetition", "skipgram_repetition", "same_finger_skipgram",
        "in_roll", "out_roll", "in_triroll", "out_triroll",
        "redirects", "double_row_jump", "scissors",
        "sequence_pos", "sequence_length", "sequence_relative_pos", 
        "word_index", "word_length", "word_relative_pos",
        "is_word_start", "is_word_end", 
        "is_syllable_start", "is_syllable_end"
    ]
    
    if categorical_features:
        features.extend(["finger", "finger_type", "hand"])
    else:
        for i in range(10): features.append(f'finger_{i}')
        for i in range(5): features.append(f'finger_type_{i}')
        for i in range(3): features.append(f'hand_{i}')
        
    if add_noise: 
        features.append("random_noise")
    return features

def get_categorical_indices(features, w_back, w_ahead):
    cat_cols = {"finger", "finger_type", "hand"}
    num_features = len(features)
    num_steps = w_back + w_ahead + 1
    return [
        step * num_features + i 
        for step in range(num_steps) 
        for i, f in enumerate(features) if f in cat_cols
    ]

def prepare_sequential_data(df, target_col, w_back, w_ahead, features, add_noise=False, return_seq_ids=False):
    features = list(features)
    if add_noise and "random_noise" not in features:
        features.append("random_noise")

    has_noise = "random_noise" in features
    real_features = [f for f in features if f != "random_noise"]

    for col in real_features:
        if col in df.columns:
            df[col] = df[col].fillna(PAD_VALUES.get(col, -99.0)).astype(np.float32)
        else:
            df[col] = np.full(len(df), PAD_VALUES.get(col, -99.0), dtype=np.float32)

    f_vals = df[real_features].values.astype(np.float32)
    t_vals = df[target_col].values.astype(np.float32)
    keys = df['key'].values
    vb = df['fold_bigram'].values
    vw = df['fold_word'].values

    # the target key must be non-pad and have a target + folds; the window may include
    # pad rows, which is fine
    valid = np.where((keys != '[PAD]') & (~np.isnan(t_vals)) & (vb != -1) & (vw != -1))[0]
    # clamp to the array ends so the window doesn't index out of bounds
    valid = valid[(valid >= w_back) & (valid < len(df) - w_ahead)]

    num_steps = w_back + w_ahead + 1
    num_real = len(real_features)
    X_real = np.zeros((len(valid), num_steps * num_real), dtype=np.float32)
    for i, idx in enumerate(tqdm(valid, desc="Vectorizing Sequences")):
        X_real[i] = f_vals[idx - w_back : idx + w_ahead + 1].flatten()

    seq_ids = df['original_sequence_id'].values[valid] if return_seq_ids else None

    if not has_noise:
        if return_seq_ids:
            return X_real, t_vals[valid], vb[valid], vw[valid], seq_ids
        return X_real, t_vals[valid], vb[valid], vw[valid]

    # interleave fresh per-step noise into the real features, keeping the noise
    # independent across steps and samples
    num_total = len(features)
    noise_idx = features.index("random_noise")
    real_indices = [i for i, f in enumerate(features) if f != "random_noise"]

    X = np.zeros((len(valid), num_steps * num_total), dtype=np.float32)
    np.random.seed(42)

    for step in range(num_steps):
        X[:, step * num_total + noise_idx] = np.random.randn(len(valid)).astype(np.float32)
        for j, real_pos in enumerate(real_indices):
            X[:, step * num_total + real_pos] = X_real[:, step * num_real + j]

    if return_seq_ids:
        return X, t_vals[valid], vb[valid], vw[valid], seq_ids
    return X, t_vals[valid], vb[valid], vw[valid]

def get_train_val_masks(f_b, f_w, fold_id):
    val_mask = (f_w == fold_id)
    train_mask = (f_w != fold_id) & (f_b != fold_id)
    return train_mask, val_mask


# grouping rows by sentence and resampling whole sentences preserves
# within-sentence autocorrelation in the bootstrap
def build_clusters(seq_ids):
    seq_ids = np.asarray(seq_ids)
    order = np.argsort(seq_ids, kind="stable")
    s = seq_ids[order]
    cuts = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    return np.split(order, cuts[1:])


def cluster_resample_idx(clusters):
    pick = np.random.randint(0, len(clusters), len(clusters))
    return np.concatenate([clusters[i] for i in pick])


def cluster_sums_counts(values, clusters):
    sums = np.array([values[c].sum() for c in clusters])
    counts = np.array([len(c) for c in clusters])
    return sums, counts


def bootstrap_cluster_mean(sums, counts):
    pick = np.random.randint(0, len(sums), len(sums))
    return sums[pick].sum() / counts[pick].sum()
