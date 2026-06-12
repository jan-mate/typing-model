from tests.enrichment.utils import FeatureTester, load_real_data

tester_repetition = FeatureTester("repetition", is_one_hot=False)

def test_synthetic_repetition():
    tester_repetition.check("call", "P001", layout="qwerty")
    tester_repetition.check("squirrel", "P0000100", layout="qwerty") 
    tester_repetition.check("poppy", "P0010", layout="qwerty")
    tester_repetition.check("cat", "P00", layout="qwerty")
    tester_repetition.check("rt", "P0", layout="qwerty")

def test_padding():
    tester_repetition.check_padding(n_pads=3)

def test_real_data_repetition():
    # rt
    df_real = load_real_data(['10027_106999_0_0'])
    tester_repetition.check_real(df_real, expected="00", start_idx=4, layout='qwerty')
    tester_repetition.check_real(df_real, expected="00", start_idx=4, layout='dvorak')

    # cat
    df_real = load_real_data(['100446_1095949_0_1'])
    tester_repetition.check_real(df_real, expected="000", start_idx=4, layout='qwerty')
    tester_repetition.check_real(df_real, expected="000", start_idx=4, layout='dvorak')

    # call
    df_real = load_real_data(['100446_1096017_0_1'])
    tester_repetition.check_real(df_real, expected="0001", start_idx=1, layout='qwerty')
    tester_repetition.check_real(df_real, expected="0001", start_idx=1, layout='dvorak')