import numpy as np
import pandas as pd
from tests.enrichment.utils import FeatureTester, load_real_data, ENGINE, LAYOUTS

tester_movement = FeatureTester("movement")

def get_movement(text, layout="qwerty", n_pads=0):
    df = pd.DataFrame([
        {"PARTICIPANT_ID": "SYNTH", "SEQUENCE_ID": "S_1", "ORIGINAL_SEQUENCE_ID": "S_1", 
         "KEY_ID": ord(c), "TIME": i*100, "iki": 0.0, "iki_z": 0.0, "iki_log_z": 0.0}
        for i, c in enumerate(text)
    ])
    enriched, _ = ENGINE.enrich(df, n_pads=n_pads, **LAYOUTS[layout])
    return enriched["movement"]

def check_real_movement(df_real, expected, start_idx, layout):
    enriched, _ = ENGINE.enrich(df_real, n_pads=0, **LAYOUTS[layout])
    actual = enriched["movement"][start_idx : start_idx + len(expected)]
    assert np.allclose(actual, expected, atol=1e-5), f"Mismatch in {layout}\nActual:\n{actual}\nExpected:\n{expected}"

def test_movement_first_key_is_padding():
    # The first key of any sequence must be padding because there is no 'previous' key
    mov = get_movement("f")
    expected_pad = [-1.0, -2.0, -2.0]
    assert np.allclose(mov[0], expected_pad)

def test_synthetic_movement_stay_at_home():
    # 'ff' = 0 distance, -2.0 padding for angles
    mov = get_movement("ff")
    assert np.allclose(mov[1],[0.0, -2.0, -2.0])

def test_synthetic_movement_repetition():
    # 'tt' = 0 distance, -2.0 padding for angles
    mov = get_movement("tt")
    assert np.allclose(mov[1], [0.0, -2.0, -2.0])

def test_movement_padding():
    expected_pad =[-1.0, -2.0, -2.0]
    mov = get_movement("a", n_pads=3)
    for i in range(3):
        assert np.allclose(mov[i], expected_pad), f"Padding mismatch at index {i}"


def test_real_data_movement_the():
    df_real = load_real_data(['100001_1091076_0_0'])
    check_real_movement(df_real, [[1.25, 0.8, 0.6], [1.0, 0.0, 1.0],[1.0307764, 0.9701425, -0.24253562]], 15, 'qwerty')
    check_real_movement(df_real, [[0.0, -2.0, -2.0],[0.0, -2.0, -2.0], [0.0, -2.0, -2.0]], 15, 'dvorak')

    df_real_2 = load_real_data(['100020_1091346_0_0'])
    check_real_movement(df_real_2, [[1.25, 0.8, 0.6], [1.0, 0.0, 1.0],[1.0307764, 0.9701425, -0.24253562]], 15, 'qwerty')
    check_real_movement(df_real_2, [[0.0, -2.0, -2.0],[0.0, -2.0, -2.0], [0.0, -2.0, -2.0]], 15, 'dvorak')

def test_real_data_movement_rt():
    df_real = load_real_data(['10027_106999_0_0'])
    check_real_movement(df_real, [[1.0307764, 0.9701425, -0.24253562],[1.0, 0.0, 1.0]], 4, 'qwerty')
    check_real_movement(df_real, [[1.0307764, 0.9701425, 0.24253562],[0.0, -2.0, -2.0]], 4, 'dvorak')

def test_real_data_movement_cat():
    df_real = load_real_data(['100446_1095949_0_1'])
    check_real_movement(df_real, [[1.118034, -0.8944272, 0.4472136], [0.0, -2.0, -2.0],[1.25, 0.8, 0.6]], 4, 'qwerty')
    check_real_movement(df_real, [[1.0307764, 0.9701425, 0.24253562],[0.0, -2.0, -2.0], [0.0, -2.0, -2.0]], 4, 'dvorak')