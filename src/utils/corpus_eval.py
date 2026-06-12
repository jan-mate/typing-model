import re
import numpy as np
import pandas as pd

from src.utils.inference import build_synthetic_df, extract_features, compute_costs


_SENTENCE_SPLIT = re.compile(r"[.\n!?]+")


def sample_corpus(
    corpus_path: str,
    n_sentences: int,
    seed: int = 42,
    min_len: int = 2,
    max_len: int = 70,
    text_column: str = "body",
) -> list[str]:
    df = pd.read_parquet(corpus_path)
    if text_column not in df.columns:
        text_column = "text" if "text" in df.columns else df.columns[0]

    sentences: list[str] = []
    for text in df[text_column].dropna():
        for s in _SENTENCE_SPLIT.split(text):
            s = s.strip()
            if min_len <= len(s) <= max_len:
                sentences.append(s)

    rng = np.random.RandomState(seed)
    rng.shuffle(sentences)
    return sentences[:n_sentences]


def predict_corpus(
    model,
    sentences: list[str],
    engine,
    layout_paths: dict,
    batch_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    # returns per-sentence (mean_z, std_z) and timings
    if not sentences:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), {"linguistic": 0.0, "layout": 0.0, "predict": 0.0}

    all_means, all_stds = [], []
    total_timings = {"linguistic": 0.0, "layout": 0.0, "predict": 0.0}
    
    batch, batch_kk = [], 0
    for sent in sentences:
        batch.append(sent)
        batch_kk += len(sent)
        if batch_kk >= batch_size:
            m, s, t = _predict_batch(model, batch, engine, layout_paths)
            all_means.append(m)
            all_stds.append(s)
            for k in total_timings: total_timings[k] += t[k]
            batch, batch_kk = [], 0
    if batch:
        m, s, t = _predict_batch(model, batch, engine, layout_paths)
        all_means.append(m)
        all_stds.append(s)
        for k in total_timings: total_timings[k] += t[k]

    return np.concatenate(all_means), np.concatenate(all_stds), total_timings


def _predict_batch(model, sentences, engine, layout_paths):
    import time
    from src.utils.inference import build_linguistic_frame, extract_features, compute_costs

    t0 = time.perf_counter()
    base_dict, offsets = build_linguistic_frame(sentences, engine, model.w_back, model.w_ahead)
    t1 = time.perf_counter()

    enriched, _ = engine.enrich_layout(base_dict, **layout_paths)
    df_feats = extract_features(enriched, list(model.features))
    t2 = time.perf_counter()

    pairs = compute_costs(df_feats, offsets, model)
    t3 = time.perf_counter()

    # per-sentence -> per-keystroke: mean z-score (÷n), and typical per-key ensemble std (RMS, ÷√n)
    means = np.zeros(len(offsets), dtype=np.float32)
    stds = np.zeros(len(offsets), dtype=np.float32)
    for i, ((sum_m, sum_s), (_, length)) in enumerate(zip(pairs, offsets)):
        n_pred = max(length - 1, 1)
        means[i] = sum_m / n_pred
        stds[i] = sum_s / np.sqrt(n_pred)
        
    timings = {"linguistic": t1 - t0, "layout": t2 - t1, "predict": t3 - t2}
    return means, stds, timings


def corpus_stats(
    z_scores: np.ndarray,
    ci: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    z = np.asarray(z_scores, dtype=np.float64)
    n = len(z)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_bootstrap, n))
    boot_means = z[idx].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(z.mean()),
        "std": float(z.std()),
        "ci_low": float(np.percentile(boot_means, 100 * alpha)),
        "ci_high": float(np.percentile(boot_means, 100 * (1 - alpha))),
        "n": n,
    }