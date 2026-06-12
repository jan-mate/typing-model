import numpy as np
import pandas as pd
from tests.enrichment.utils import FeatureTester, ENGINE, LAYOUTS

test_uni = FeatureTester("unigram_frequency")
test_bi = FeatureTester("bigram_frequency")

def get_freqs(text, layout="qwerty"):
    df = pd.DataFrame([
        {"PARTICIPANT_ID": "SYNTH", "SEQUENCE_ID": "S_1", "ORIGINAL_SEQUENCE_ID": "S_1", 
         "KEY_ID": ord(c), "TIME": i*100, "iki": 0.0, "iki_z": 0.0, "iki_log_z": 0.0}
        for i, c in enumerate(text)
    ])
    enriched, _ = ENGINE.enrich(df, n_pads=0, **LAYOUTS[layout])
    return enriched["unigram_frequency"], enriched["bigram_frequency"]

def test_unigram_relative_frequency():
    uni_freqs, _ = get_freqs("ez")
    assert uni_freqs[0] > uni_freqs[1], f"Expected 'e' > 'z'"

def test_unigram_consistency():
    uni_freqs, _ = get_freqs("ee")
    assert np.isclose(uni_freqs[0], uni_freqs[1]), "Frequency of 'e' is not consistent"

def test_bigram_relative_frequency():
    _, bi_freqs_th = get_freqs("th")
    _, bi_freqs_ca = get_freqs("ca")
    
    # Bigram frequencies are registered on the second character
    freq_th = bi_freqs_th[1]
    freq_ca = bi_freqs_ca[1]
    
    assert freq_th > freq_ca, f"Expected 'th' > 'ca'"

def test_bigram_consistency():
    _, bi_freqs = get_freqs("thoth")
    
    # th
    assert np.isclose(bi_freqs[1], bi_freqs[4]), "Frequency of 'th' bigram is not consistent"

def test_bigram_initial_padding():
    _, bi_freqs = get_freqs("th")
    assert bi_freqs[0] == -1.0, f"Expected padding at index 0, got {bi_freqs[0]}"

def test_padding():
    test_uni.check_padding(n_pads=3)
    test_bi.check_padding(n_pads=3)