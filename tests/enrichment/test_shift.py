from tests.enrichment.utils import FeatureTester, load_real_data

tester_shift = FeatureTester("shift", is_one_hot=False)

def test_synthetic_shift():
    tester_shift.check("abc", "000", layout="qwerty")
    tester_shift.check("A Bc", "1010", layout="qwerty")
    
    tester_shift.check("1!", "01", layout="qwerty")

def test_padding():
    tester_shift.check_padding(n_pads=3)

def test_real_data_shift():
    # the
    df_real = load_real_data(['100001_1091076_0_0'])
    tester_shift.check_real(df_real, expected="000", start_idx=15, layout='qwerty')
    tester_shift.check_real(df_real, expected="000", start_idx=15, layout='dvorak')

    #  i 
    df_real = load_real_data(['100001_1091153_0_0'])
    tester_shift.check_real(df_real, expected="010", start_idx=8, layout='qwerty')
    tester_shift.check_real(df_real, expected="010", start_idx=8, layout='dvorak')

    # cat
    df_real = load_real_data(['100446_1095949_0_1'])
    tester_shift.check_real(df_real, expected="000", start_idx=4, layout='qwerty')
    tester_shift.check_real(df_real, expected="000", start_idx=4, layout='dvorak')