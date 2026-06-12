from tests.enrichment.utils import FeatureTester, load_real_data

test = FeatureTester("in_triroll")

def test_qwerty_in_trirolls():
    # Left hand: pinky -> ring -> middle
    test.check("asd", "PP1")
    # Right hand: pinky -> ring -> middle
    test.check("poi", "PP1")
    # These are out_trirolls
    test.check("dsa", "PP0")
    test.check("iop", "PP0")

def test_dvorak_in_trirolls():
    # Left hand home row
    test.check("aoe", "PP1", layout="dvorak")
    # Right hand home row
    test.check("snt", "PP1", layout="dvorak")

def test_padding_boundary():
    # Requires two P's per sequence start
    test.check(["asd", "poi"], 
        ["PP1", "PP1"]
    )

def test_padding():
    test.check_padding(n_pads=3)

def test_real_data_in_triroll():
    # the
    df_real = load_real_data(['100001_1091076_0_0'])
    test.check_real(df_real, expected="000", start_idx=15, layout='qwerty')
    test.check_real(df_real, expected="000", start_idx=15, layout='dvorak')

    # the
    df_real = load_real_data(['100020_1091346_0_0'])
    test.check_real(df_real, expected="000", start_idx=15, layout='qwerty')
    test.check_real(df_real, expected="000", start_idx=15, layout='dvorak')