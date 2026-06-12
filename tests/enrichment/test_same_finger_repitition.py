from tests.enrichment.utils import FeatureTester, load_real_data

tester_skipgram_rep = FeatureTester("skipgram_repetition", is_one_hot=False)

def test_synthetic_skipgram_repetition():
    tester_skipgram_rep.check("level", "PP010", layout="qwerty")
    tester_skipgram_rep.check("radar", "PP010", layout="qwerty") 
    tester_skipgram_rep.check("iki", "PP1", layout="qwerty")
    tester_skipgram_rep.check("cat", "PP0", layout="qwerty")
    tester_skipgram_rep.check("UwU", "PP1", layout="qwerty")

def test_padding():
    tester_skipgram_rep.check_padding(n_pads=3)

def test_real_data_skipgram_repetition():
    # level
    df_real = load_real_data(['100560_1097182_0_0'])
    tester_skipgram_rep.check_real(df_real, expected="00010", start_idx=22, layout='qwerty')
    tester_skipgram_rep.check_real(df_real, expected="00010", start_idx=22, layout='dvorak')

    # level
    df_real = load_real_data(['104886_1146085_0_0'])
    tester_skipgram_rep.check_real(df_real, expected="00010", start_idx=14, layout='qwerty')
    tester_skipgram_rep.check_real(df_real, expected="00010", start_idx=14, layout='dvorak')