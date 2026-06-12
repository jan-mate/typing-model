from tests.enrichment.utils import FeatureTester, load_real_data

tester_sfb = FeatureTester("same_finger", is_one_hot=False)

def test_synthetic_same_finger_bigram():
    # 'r' -> 't' is same finger (left index) in qwerty, but different fingers in dvorak
    tester_sfb.check("rt", "P1", layout="qwerty")
    tester_sfb.check("rt", "P0", layout="dvorak")
    
    # 'e' -> 'c' is same finger (left middle) in qwerty
    tester_sfb.check("ec", "P1", layout="qwerty")
    
    # 'c' -> 't' is same finger in dvorak
    tester_sfb.check("ct", "P1", layout="dvorak")
    
    # repitition does not count as a same_finger_bigram
    tester_sfb.check("aa", "P0", layout="qwerty")
    tester_sfb.check("call", "P000", layout="qwerty")
    tester_sfb.check("call", "P000", layout="dvorak")
    
    tester_sfb.check("cat", "P00", layout="qwerty")
    tester_sfb.check("cat", "P00", layout="dvorak")

def test_padding():
    tester_sfb.check_padding(n_pads=3)

def test_real_data_same_finger_bigram():
    # rt
    df_real = load_real_data(['10027_106999_0_0'])
    tester_sfb.check_real(df_real, expected="01", start_idx=4, layout='qwerty')
    tester_sfb.check_real(df_real, expected="00", start_idx=4, layout='dvorak')

    # cat
    df_real = load_real_data(['100446_1095949_0_1'])
    tester_sfb.check_real(df_real, expected="000", start_idx=4, layout='qwerty')
    tester_sfb.check_real(df_real, expected="000", start_idx=4, layout='dvorak')

    # call
    df_real = load_real_data(['100446_1096017_0_1'])
    tester_sfb.check_real(df_real, expected="0000", start_idx=1, layout='qwerty')
    tester_sfb.check_real(df_real, expected="0000", start_idx=1, layout='dvorak')