import os
import pytest
import numpy as np
import pandas as pd
from src.enrichment.engine import EnrichmentEngine, PAD_VALUES

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

ENGINE = EnrichmentEngine(
    unigrams_path=os.path.join(PROJECT_ROOT, 'data/frequencies/unigrams.json'),
    bigrams_path=os.path.join(PROJECT_ROOT, 'data/frequencies/bigrams.json'),
    words_path=os.path.join(PROJECT_ROOT, 'data/frequencies/words_zipf.json'),
    movement_features_path=os.path.join(PROJECT_ROOT, 'data/layouts/movement_features.json')
)

LAYOUTS = {
    "qwerty": {
        "layout_path": os.path.join(PROJECT_ROOT, "data/layouts/qwerty_us.json"),
        "layout_map_path": os.path.join(PROJECT_ROOT, "data/layouts/layout_map.json"),
        "shifts_path": os.path.join(PROJECT_ROOT, "data/layouts/shifts_us.json")
    },
    "dvorak": {
        "layout_path": os.path.join(PROJECT_ROOT, "data/layouts/dvorak.json"),
        "layout_map_path": os.path.join(PROJECT_ROOT, "data/layouts/layout_map.json"),
        "shifts_path": os.path.join(PROJECT_ROOT, "data/layouts/shifts_us.json")
    }
}

_parquet_cache: dict = {}


def load_real_data(target_ids, parquet_path=None):
    if parquet_path is None:
        parquet_path = os.path.join(PROJECT_ROOT, 'data/interim/normalized_sequences.parquet')

    if not os.path.exists(parquet_path):
        return pd.DataFrame()

    if parquet_path not in _parquet_cache:
        _parquet_cache[parquet_path] = pd.read_parquet(parquet_path)
    df_full = _parquet_cache[parquet_path]
    mask = df_full['SEQUENCE_ID'].str.startswith(tuple(target_ids))
    df_test = df_full[mask].copy()
    
    if df_test.empty:
        return df_test

    df_test['ID_BASE'] = df_test['SEQUENCE_ID'].apply(lambda x: next(t for t in target_ids if x.startswith(t)))
    df_test['ID_BASE'] = pd.Categorical(df_test['ID_BASE'], categories=target_ids, ordered=True)
    return df_test.sort_values(['ID_BASE', 'SEQUENCE_ID', 'TIME'])


class FeatureTester:
    def __init__(self, feature_name, is_one_hot=False):
        self.feature_name = feature_name
        self.is_one_hot = is_one_hot

    def _parse_expected(self, expected):
        if isinstance(expected, str):
            return[np.nan if c == 'N' else (-1.0 if c == 'P' else float(c)) for c in expected]
        return expected

    def check(self, sequence, expected, layout="qwerty", n_pads=1):
        sequences = [sequence] if isinstance(sequence, str) else sequence
        expected = [expected] if isinstance(expected, str) or (isinstance(expected, list) and not isinstance(expected[0], (str, list))) else expected

        rows =[]
        for seq_idx, text in enumerate(sequences):
            for t, char in enumerate(text):
                rows.append({"PARTICIPANT_ID": "SYNTH", "SEQUENCE_ID": f"S_{seq_idx}", "ORIGINAL_SEQUENCE_ID": f"S_{seq_idx}", "KEY_ID": ord(char), "TIME": t*100, "iki": 0.0, "iki_z": 0.0, "iki_log_z": 0.0})
        df = pd.DataFrame(rows)

        enriched, keys = ENGINE.enrich(df, n_pads=n_pads, **LAYOUTS[layout])
        actual = enriched[self.feature_name]

        if self.is_one_hot:
            actual = np.where(actual[:, 0] == -1.0, -1, np.argmax(actual, axis=1))

        pad_val = -1 if self.is_one_hot else PAD_VALUES.get(self.feature_name, np.nan) # <-- CHANGED HERE
        exp_full, chars_full = [],[]
        
        for i, exp_seq in enumerate(expected):
            exp_full.extend([pad_val] * n_pads)
            chars_full.extend(["PAD"] * n_pads)
            exp_full.extend(self._parse_expected(exp_seq))
            chars_full.extend(list(sequences[i]))

        self._assert_match(np.array(actual, dtype=np.float32), np.array(exp_full, dtype=np.float32), chars_full)

    def check_real(self, df_real, expected, start_idx=0, layout="qwerty"):
        if df_real is None or df_real.empty:
            pytest.skip(f"Sequence not found in dataset. Skipping real data test.")
            return

        enriched, keys = ENGINE.enrich(df_real, n_pads=0, **LAYOUTS[layout])
        
        if start_idx + len(expected) > len(enriched[self.feature_name]):
            raise AssertionError(f"start_idx {start_idx} + len(expected) {len(expected)} exceeds total sequence length {len(enriched[self.feature_name])}")

        actual = enriched[self.feature_name][start_idx:start_idx+len(expected)]
        
        if self.is_one_hot:
            actual = np.where(actual[:, 0] == -1.0, -1, np.argmax(actual, axis=1))

        exp_full = np.array(self._parse_expected(expected), dtype=np.float32)
        chars_full =[chr(int(k)) if k != 0 else "PAD" for k in keys[start_idx:start_idx+len(expected)]]

        self._assert_match(np.array(actual, dtype=np.float32), exp_full, chars_full)

    def check_padding(self, n_pads=3, layout="qwerty"):
        df = pd.DataFrame([{
            "PARTICIPANT_ID": "SYNTH", "SEQUENCE_ID": "S_1", "ORIGINAL_SEQUENCE_ID": "S_1", 
            "KEY_ID": ord('a'), "TIME": 0, "iki": 0.0, "iki_z": 0.0, "iki_log_z": 0.0
        }])
        
        enriched, _ = ENGINE.enrich(df, n_pads=n_pads, **LAYOUTS[layout])
        actual = enriched[self.feature_name]
        
        expected_pad_val = PAD_VALUES.get(self.feature_name, np.nan)
        
        for i in range(n_pads):
            val = actual[i]
            if self.is_one_hot:
                assert np.all(val == -1.0), f"Padding mismatch at index {i}. Expected array of -1.0s."
            else:
                assert np.allclose(val, expected_pad_val, equal_nan=True), f"Padding mismatch at index {i}. Expected {expected_pad_val}, got {val}"

    def _assert_match(self, actual, expected, characters):
        if not np.allclose(actual, expected, equal_nan=True):
            def fmt(v): return " N " if np.isnan(v) else (f"{int(v):^3}" if v == int(v) else f"{v:^4.1f}")
            char_row = " ".join([f"{c:^3}" for c in characters])
            exp_row  = " ".join([fmt(e) for e in expected])
            act_row  = " ".join([fmt(a) for a in actual])
            raise AssertionError(f"\n\nFeature '{self.feature_name}' mismatch!\nKeys:     [{char_row}]\nExpected:[{exp_row}]\nActual:   [{act_row}]\n")