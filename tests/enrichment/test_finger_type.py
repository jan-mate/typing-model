from tests.enrichment.utils import FeatureTester, load_real_data

tester_finger_type = FeatureTester("finger_type", is_one_hot=True)

def test_synthetic_finger_types():
    tester_finger_type.check("asdfg", "01233")
    tester_finger_type.check("hjkl", "3321")
    tester_finger_type.check("squirrels", "103233211")

def test_padding():
    tester_finger_type.check_padding(n_pads=3)

def test_real_data_finger_types():
    # cat
    df_real = load_real_data(['175221_1903083_3_1'])
    tester_finger_type.check_real(df_real, expected="203", start_idx=17, layout='qwerty')
    tester_finger_type.check_real(df_real, expected="202", start_idx=17, layout='dvorak')

    # the
    df_real = load_real_data(['100545_1097134_0_0'])
    tester_finger_type.check_real(df_real, expected="332", start_idx=3, layout='qwerty')
    tester_finger_type.check_real(df_real, expected="232", start_idx=3, layout='dvorak')