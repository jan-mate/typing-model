from tests.enrichment.utils import FeatureTester, load_real_data

tester_sft = FeatureTester("same_finger_trigram", is_one_hot=False)

def test_synthetic_same_finger_trigram():
    tester_sft.check("dec", "PP1", layout="qwerty")

    tester_sft.check("xsw", "PP1", layout="qwerty")

    tester_sft.check("iki", "PP1", layout="qwerty")
    
    tester_sft.check("dad", "PP0", layout="qwerty")

def test_padding():
    tester_sft.check_padding(n_pads=3)

def test_real_data_same_finger_trigram():
    # level
    df_real = load_real_data(['104886_1146085_0_0'])
    tester_sft.check_real(df_real, expected="00000", start_idx=14, layout='qwerty')
    tester_sft.check_real(df_real, expected="00000", start_idx=14, layout='dvorak')

    # These
    df_real = load_real_data(['115597_1256424_0_0'])
    tester_sft.check_real(df_real, expected="PP000", start_idx=0, layout='qwerty')
    tester_sft.check_real(df_real, expected="PP000", start_idx=0, layout='dvorak')