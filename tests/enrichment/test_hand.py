from tests.enrichment.utils import FeatureTester, load_real_data

tester_hand = FeatureTester("hand", is_one_hot=True)

def test_qwerty_hands():
    tester_hand.check("asdf", "0000")
    tester_hand.check("jkl;", "1111")
    tester_hand.check("cat",  "000")
    tester_hand.check("meow",  "1010")

def test_dvorak_hands():
    tester_hand.check("aoeu", "0000", layout="dvorak")
    tester_hand.check("htns", "1111", layout="dvorak")
    tester_hand.check("cat",  "101", layout="dvorak")

def test_spacebar():
    tester_hand.check("a a", "020")

def test_padding():
    tester_hand.check_padding(n_pads=3)

def test_real_data_hands():
    # the
    df_real = load_real_data(['100545_1097134_0_0'])
    tester_hand.check_real(df_real, expected="010", start_idx=3, layout='qwerty')
    tester_hand.check_real(df_real, expected="110", start_idx=3, layout='dvorak')

    # cat
    df_real = load_real_data(['130479_1417468_3_0'])
    tester_hand.check_real(df_real, expected="000", start_idx=5, layout='qwerty')
    tester_hand.check_real(df_real, expected="101", start_idx=5, layout='dvorak')