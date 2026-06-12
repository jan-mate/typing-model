from tests.enrichment.utils import FeatureTester, load_real_data

tester_is_syllable_start = FeatureTester("is_syllable_start", is_one_hot=False)
tester_is_syllable_end = FeatureTester("is_syllable_end", is_one_hot=False)

def test_synthetic_syllables():
    tester_is_syllable_start.check("cat", "100", layout="qwerty")
    tester_is_syllable_end.check("cat", "001", layout="qwerty")

    tester_is_syllable_start.check("tomorrow", "10100100", layout="qwerty")
    tester_is_syllable_end.check("tomorrow", "01001001", layout="qwerty")

    tester_is_syllable_start.check("these", "10000", layout="qwerty")
    tester_is_syllable_end.check("these", "00001", layout="qwerty")

    tester_is_syllable_start.check("a cat!", "1P100P", layout="qwerty")
    tester_is_syllable_end.check("a cat!", "1P001P", layout="qwerty")
    
    tester_is_syllable_start.check("a cat!", "1P100P", layout="dvorak")
    tester_is_syllable_end.check("a cat!", "1P001P", layout="dvorak")

def test_padding():
    tester_is_syllable_start.check_padding(n_pads=3)
    tester_is_syllable_end.check_padding(n_pads=3)

def test_real_data_syllable_position():
    # cat
    df_real = load_real_data(['100446_1095949_0_1'])
    tester_is_syllable_start.check_real(df_real, expected="100", start_idx=4, layout='qwerty')
    tester_is_syllable_start.check_real(df_real, expected="100", start_idx=4, layout='dvorak')

    tester_is_syllable_end.check_real(df_real, expected="000", start_idx=4, layout='qwerty')
    tester_is_syllable_end.check_real(df_real, expected="000", start_idx=4, layout='dvorak')

    # tomorrow
    df_real = load_real_data(['100581_1097630_2_0'])
    tester_is_syllable_start.check_real(df_real, expected="10100100", start_idx=2, layout='qwerty')
    tester_is_syllable_start.check_real(df_real, expected="10100100", start_idx=2, layout='dvorak')

    tester_is_syllable_end.check_real(df_real, expected="01001001", start_idx=2, layout='qwerty')
    tester_is_syllable_end.check_real(df_real, expected="01001001", start_idx=2, layout='dvorak')

    # these
    df_real = load_real_data(['101030_1102308_0_0'])
    tester_is_syllable_start.check_real(df_real, expected="10000", start_idx=21, layout='qwerty')
    tester_is_syllable_start.check_real(df_real, expected="10000", start_idx=21, layout='dvorak')

    tester_is_syllable_end.check_real(df_real, expected="00001", start_idx=21, layout='qwerty')
    tester_is_syllable_end.check_real(df_real, expected="00001", start_idx=21, layout='dvorak')