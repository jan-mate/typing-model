from tests.enrichment.utils import FeatureTester, load_real_data

tester_word_index = FeatureTester("word_index", is_one_hot=False)

def test_synthetic_word_index():
    tester_word_index.check(" word ", "P0123P", layout="qwerty")
    tester_word_index.check(" word ", "P0123P", layout="dvorak")
    
    tester_word_index.check(" word.", "P0123P", layout="qwerty")
    tester_word_index.check(" word.", "P0123P", layout="dvorak")
    
    tester_word_index.check("a, b!", "0PP0P", layout="qwerty")
    tester_word_index.check("a, b!", "0PP0P", layout="dvorak")

def test_padding():
    tester_word_index.check_padding(n_pads=3)

def test_real_data_word_index():
    #  the 
    df_real = load_real_data(['100001_1091076_0_0'])
    tester_word_index.check_real(df_real, expected="P012P", start_idx=14, layout='qwerty')
    tester_word_index.check_real(df_real, expected="P012P", start_idx=14, layout='dvorak')

    #  the 
    df_real = load_real_data(['100020_1091346_0_0'])
    tester_word_index.check_real(df_real, expected="P012P", start_idx=14, layout='qwerty')
    tester_word_index.check_real(df_real, expected="P012P", start_idx=14, layout='dvorak')

    # e t
    df_real = load_real_data(['100156_1092917_2_0'])
    tester_word_index.check_real(df_real, expected="3P0", start_idx=9, layout='qwerty')
    tester_word_index.check_real(df_real, expected="3P0", start_idx=9, layout='dvorak')

    #  the 
    df_real = load_real_data(['100446_1095936_1_1'])
    tester_word_index.check_real(df_real, expected="P012P", start_idx=24, layout='qwerty')
    tester_word_index.check_real(df_real, expected="P012P", start_idx=24, layout='dvorak')