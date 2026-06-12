import numpy as np
import pandas as pd
from tests.enrichment.utils import FeatureTester, load_real_data, ENGINE, LAYOUTS

tester_word_frequency = FeatureTester("word_frequency", is_one_hot=False)

def get_word_freq(text, layout="qwerty"):
    df = pd.DataFrame([
        {"PARTICIPANT_ID": "SYNTH", "SEQUENCE_ID": "S_1", "ORIGINAL_SEQUENCE_ID": "S_1", 
         "KEY_ID": ord(c), "TIME": i*100, "iki": 0.0, "iki_z": 0.0, "iki_log_z": 0.0}
        for i, c in enumerate(text)
    ])
    enriched, _ = ENGINE.enrich(df, n_pads=0, **LAYOUTS[layout])
    return enriched["word_frequency"]

def test_synthetic_word_frequency():
    freq_qwerty = get_word_freq(" the ")
    freq_dvorak = get_word_freq(" the ", layout="dvorak")
    np.testing.assert_array_equal(freq_qwerty, freq_dvorak)
    
    freq_space = get_word_freq(" word ")
    freq_dot = get_word_freq(" word.")
    freq_comma = get_word_freq(",word!")
    
    assert freq_space[1] == freq_dot[1] == freq_comma[1], "Whitespace and punctuation should be treated equally as delimiters"
    
    assert np.isnan(freq_space[0]) and np.isnan(freq_space[-1])
    assert np.isnan(freq_dot[0]) and np.isnan(freq_dot[-1])

def test_padding():
    tester_word_frequency.check_padding(n_pads=3)

def test_real_data_word_frequency():
    #  the 
    df_real = load_real_data(['100001_1091076_0_0'])
    tester_word_frequency.check_real(df_real, expected=[np.nan, 7.73, 7.73, 7.73, np.nan], start_idx=14, layout='qwerty')
    tester_word_frequency.check_real(df_real, expected=[np.nan, 7.73, 7.73, 7.73, np.nan], start_idx=14, layout='dvorak')

    #  the 
    df_real = load_real_data(['100020_1091346_0_0'])
    tester_word_frequency.check_real(df_real, expected=[np.nan, 7.73, 7.73, 7.73, np.nan], start_idx=14, layout='qwerty')
    tester_word_frequency.check_real(df_real, expected=[np.nan, 7.73, 7.73, 7.73, np.nan], start_idx=14, layout='dvorak')
