import numpy as np
from tests.enrichment.utils import FeatureTester, load_real_data

tester_idx = FeatureTester("word_index", is_one_hot=False)
tester_len = FeatureTester("word_length", is_one_hot=False)
tester_rel = FeatureTester("word_relative_pos", is_one_hot=False)
tester_start = FeatureTester("is_word_start", is_one_hot=False)
tester_end = FeatureTester("is_word_end", is_one_hot=False)

def test_synthetic_word_boundaries():
    tester_idx.check("a bc", "0P01", layout="qwerty")
    tester_len.check("a bc", "1P22", layout="qwerty")
    tester_rel.check("a bc", [0.0, -1.0, 0.0, 1.0], layout="qwerty")
    
    tester_start.check("a bc", "1010", layout="qwerty")
    tester_end.check("a bc", "1001", layout="qwerty")

    tester_idx.check("cat", "012", layout="qwerty")
    tester_len.check("cat", "333", layout="qwerty")
    tester_rel.check("cat", [0.0, 0.5, 1.0], layout="qwerty")
    tester_start.check("cat", "100", layout="qwerty")
    tester_end.check("cat", "001", layout="qwerty")

def test_padding():
    tester_idx.check_padding(n_pads=3)
    tester_len.check_padding(n_pads=3)
    tester_rel.check_padding(n_pads=3)
    tester_start.check_padding(n_pads=3)
    tester_end.check_padding(n_pads=3)

def test_real_data_word_position():
    #  the 
    df_real = load_real_data(['100001_1091076_0_0'])
    
    tester_idx.check_real(df_real, expected="P012P", start_idx=14, layout='qwerty')
    tester_len.check_real(df_real, expected="P333P", start_idx=14, layout='qwerty')
    tester_rel.check_real(df_real, expected=[-1.0, 0.0, 0.5, 1.0, -1.0], start_idx=14, layout='qwerty')
    tester_start.check_real(df_real, expected="01000", start_idx=14, layout='qwerty')
    tester_end.check_real(df_real, expected="00010", start_idx=14, layout='qwerty')