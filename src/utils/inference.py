import numpy as np
import pandas as pd

PAD_MAP = {
    "move_dist": -1.0, "x": -5.0, "y": -5.0, "shift": -1.0,
    "bigram_frequency": -1.0, "word_frequency": -1.0, "same_hand": -1.0,
    "same_finger": -1.0, "repetition": -1.0, "same_finger_skipgram": -1.0,
    "skipgram_repetition": -1.0, "in_roll": -1.0, "out_roll": -1.0,
    "in_triroll": -1.0, "double_row_jump": -1.0, "sequence_pos": -1.0,
    "word_length": -1.0, "word_relative_pos": -1.0,
    "finger": -1.0,
    "finger_type_0": -1.0, "finger_type_1": -1.0, "finger_type_2": -1.0,
    "finger_type_3": -1.0, "finger_type_4": -1.0,
    "hand_0": -1.0, "hand_1": -1.0, "hand_2": -1.0,
}

def build_synthetic_df(texts, w_back=3, w_ahead=1):
    # one DataFrame for all texts. each text is bracketed with pad keys (key id 0) so
    # every real keystroke has a full window; returns the df plus (start_idx, length) per text
    all_keys = []
    offsets = []
    seq_ids = []

    for text_idx, text in enumerate(texts):
        pad_before = [0] * w_back
        keys = [ord(c) for c in text]
        pad_after = [0] * w_ahead

        start = len(all_keys) + w_back
        all_keys.extend(pad_before + keys + pad_after)
        offsets.append((start, len(text)))

        seq_ids.extend([f"S{text_idx}"] * (w_back + len(text) + w_ahead))

    n = len(all_keys)
    df = pd.DataFrame({
        "PARTICIPANT_ID": ["SYNTH"] * n,
        "SEQUENCE_ID": seq_ids,
        "ORIGINAL_SEQUENCE_ID": seq_ids,
        "KEY_ID": all_keys,
        "TIME": np.arange(n) * 100,
        "iki": [0.0] * n,
        "iki_z": [0.0] * n,
        "iki_log_z": [0.0] * n,
    })
    return df, offsets


def extract_features(enriched, active_features):
    df_feat = pd.DataFrame(index=range(len(enriched["sequence_id"])))
    for f in active_features:
        if f == "move_dist":
            df_feat[f] = enriched["movement"][:, 0]
        elif f == "x":
            df_feat[f] = enriched["x"] if "x" in enriched else enriched["coords"][:, 0]
        elif f == "y":
            df_feat[f] = enriched["y"] if "y" in enriched else enriched["coords"][:, 1]
        elif f == "finger":
            df_feat[f] = (
                np.argmax(enriched["finger"], axis=1)
                if enriched["finger"].ndim == 2
                else enriched["finger"]
            )
        elif f == "hand":
            df_feat[f] = (
                np.argmax(enriched["hand"], axis=1)
                if enriched["hand"].ndim == 2
                else enriched["hand"]
            )
        elif f.startswith("finger_type_"):
            i = int(f.split("_")[-1])
            df_feat[f] = enriched["finger_type"][:, i] if enriched["finger_type"].ndim == 2 else (enriched["finger_type"] == i).astype(np.float32)
        elif f.startswith("hand_"):
            i = int(f.split("_")[-1])
            df_feat[f] = enriched["hand"][:, i] if enriched["hand"].ndim == 2 else (enriched["hand"] == i).astype(np.float32)
        elif f in enriched:
            df_feat[f] = enriched[f]
        else:
            df_feat[f] = 0.0
            
        df_feat[f] = df_feat[f].fillna(PAD_MAP.get(f, -1.0))
        
    return df_feat.astype(np.float32)


def build_linguistic_frame(texts, engine, w_back=3, w_ahead=1):
    # layout-agnostic step of inference, built once and reused across layouts
    if not texts:
        return None, []
    df, offsets = build_synthetic_df(texts, w_back, w_ahead)
    base_dict = engine.enrich_linguistics(df, n_pads=0)
    return base_dict, offsets


def predict_from_base(base_dict, offsets, model, engine, layout_paths, skip_second=None):
    # layout-specific step: enrich for this layout, then predict
    if base_dict is None or not offsets:
        return []

    enriched, _ = engine.enrich_layout(base_dict, **layout_paths)

    df_feats = extract_features(enriched, list(model.features))
    return compute_costs(df_feats, offsets, model, skip_second)


def predict_speed(texts, model, engine, layout_paths, skip_second=None):
    # returns one (mean, std) per text, summing IKI z-scores over its keystrokes
    base_dict, offsets = build_linguistic_frame(texts, engine, model.w_back, model.w_ahead)
    return predict_from_base(base_dict, offsets, model, engine, layout_paths, skip_second)


def compute_costs(df_feats, offsets, model, skip_second=None):
    # skip_second: texts whose second keystroke is also left uncounted (free, like the first)
    if not offsets:
        return []

    feat_matrix = df_feats[list(model.features)].values.astype(np.float32)

    predict_indices = []
    text_mapping = []

    for text_idx, (start, length) in enumerate(offsets):
        first = 2 if (skip_second and text_idx in skip_second) else 1
        for i in range(first, length):
            predict_indices.append(start + i)
            text_mapping.append(text_idx)

    if not predict_indices:
        return [(0.0, 0.0)] * len(offsets)

    predict_indices = np.array(predict_indices)

    rel_indices = np.arange(-model.w_back, model.w_ahead + 1)
    absolute_indices = predict_indices[:, None] + rel_indices

    X_raw = feat_matrix[absolute_indices]
    X = X_raw.reshape(len(predict_indices), -1)

    means, stds = model.predict(X)

    text_mapping = np.array(text_mapping)
    results = []
    if len(text_mapping) > 0:
        sum_means = np.bincount(text_mapping, weights=means, minlength=len(offsets))
        sum_vars = np.bincount(text_mapping, weights=stds**2, minlength=len(offsets))
        sum_stds = np.sqrt(sum_vars)

        for m, s in zip(sum_means, sum_stds):
            results.append((float(m), float(s)))
    else:
        results = [(0.0, 0.0)] * len(offsets)

    return results

def enrich_texts(texts, engine, layout_paths, active_features, max_w_back=3, max_w_ahead=1):
    if not texts:
        return pd.DataFrame(columns=active_features), []
    df, offsets = build_synthetic_df(texts, max_w_back, max_w_ahead)
    enriched, _ = engine.enrich(df, **layout_paths, n_pads=0)
    df_feats = extract_features(enriched, active_features)
    return df_feats, offsets
