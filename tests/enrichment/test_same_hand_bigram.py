from tests.enrichment.utils import FeatureTester, load_real_data

tester_same_hand = FeatureTester("same_hand", is_one_hot=False)

def test_synthetic_same_hand_bigram():
    tester_same_hand.check("xD", "P1", layout="qwerty")
    tester_same_hand.check(":3", "P0", layout="qwerty")
    
    tester_same_hand.check("Meow", "P000", layout="qwerty")
    tester_same_hand.check("Meow", "P010", layout="dvorak")

    tester_same_hand.check("cat", "P11", layout="qwerty")

def test_padding():
    tester_same_hand.check_padding(n_pads=3)

def test_real_data_same_hand_bigram():
    # ce
    df_real = load_real_data(['100001_1091076_0_0'])
    tester_same_hand.check_real(df_real, expected="01", start_idx=45, layout='qwerty')
    tester_same_hand.check_real(df_real, expected="00", start_idx=45, layout='dvorak')

    # mo
    df_real = load_real_data(['100001_1091153_1_0'])
    tester_same_hand.check_real(df_real, expected="01", start_idx=10, layout='qwerty')
    tester_same_hand.check_real(df_real, expected="00", start_idx=10, layout='dvorak')

    # ce
    df_real = load_real_data(['100020_1091346_0_0'])
    tester_same_hand.check_real(df_real, expected="01", start_idx=30, layout='qwerty')
    tester_same_hand.check_real(df_real, expected="10", start_idx=30, layout='dvorak')

    # mo
    df_real = load_real_data(['100030_1091489_2_0'])
    tester_same_hand.check_real(df_real, expected="01", start_idx=8, layout='qwerty')
    tester_same_hand.check_real(df_real, expected="00", start_idx=8, layout='dvorak')