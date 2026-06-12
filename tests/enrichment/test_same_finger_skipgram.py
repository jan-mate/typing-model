from tests.enrichment.utils import FeatureTester, load_real_data

tester_sfs = FeatureTester("same_finger_skipgram", is_one_hot=False)

def test_synthetic_same_finger_skipgram():
    tester_sfs.check("f t", "PP1", layout="qwerty")
    tester_sfs.check("d e", "PP1", layout="qwerty")

    tester_sfs.check("iki", "PP0", layout="qwerty")
    
    tester_sfs.check("dec", "PP0", layout="qwerty")
    
    tester_sfs.check("aza", "PP0", layout="qwerty")

    tester_sfs.check("UwU", "PP0", layout="qwerty")

def test_padding():
    tester_sfs.check_padding(n_pads=3)

def test_real_data_same_finger_skipgram():
    # these
    df_real = load_real_data(['101030_1102308_0_0'])
    tester_sfs.check_real(df_real, expected="10000", start_idx=21, layout='qwerty')
    tester_sfs.check_real(df_real, expected="00000", start_idx=21, layout='dvorak')

    # These
    df_real = load_real_data(['115597_1256424_0_0'])
    tester_sfs.check_real(df_real, expected="PP000", start_idx=0, layout='qwerty')
    tester_sfs.check_real(df_real, expected="PP000", start_idx=0, layout='dvorak')