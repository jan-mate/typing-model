from tests.enrichment.utils import FeatureTester, load_real_data

tester_finger = FeatureTester("finger", is_one_hot=True)

def test_synthetic_fingers():
    tester_finger.check("asdfg", "01233")
    tester_finger.check("hjkl", "6678")
    tester_finger.check("squirrels", "106733281")

def test_padding():
    tester_finger.check_padding(n_pads=3)

def test_real_data_fingers():
    # cat
    df_real = load_real_data(['175221_1903083_3_1'])
    tester_finger.check_real(df_real, expected="203", start_idx=17, layout='qwerty')
    tester_finger.check_real(df_real, expected="707", start_idx=17, layout='dvorak')

    # the
    df_real = load_real_data(['100545_1097134_0_0'])
    tester_finger.check_real(df_real, expected="362", start_idx=3, layout='qwerty')
    tester_finger.check_real(df_real, expected="762", start_idx=3, layout='dvorak')