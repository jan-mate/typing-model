from tests.enrichment.utils import FeatureTester, load_real_data

test = FeatureTester("out_triroll")

def test_qwerty_out_trirolls():
    # Left hand: middle -> ring -> pinky
    test.check("dsa", "PP1")
    # Right hand: middle -> ring -> pinky
    test.check("iop", "PP1")
    # These are in_trirolls
    test.check("asd", "PP0")
    test.check("poi", "PP0")

def test_dvorak_out_trirolls():
    # Left hand home row
    test.check("eoa", "PP1", layout="dvorak")
    # Right hand home row
    test.check("tns", "PP1", layout="dvorak")

def test_padding_boundary():
    test.check(
        ["dsa", "iop"],["PP1", "PP1"]
    )

def test_padding():
    test.check_padding(n_pads=3)

def test_real_data_out_triroll():
    # the
    df_real = load_real_data(['100001_1091076_0_0'])
    test.check_real(df_real, expected="000", start_idx=15, layout='qwerty')
    test.check_real(df_real, expected="000", start_idx=15, layout='dvorak')

    # the
    df_real = load_real_data(['100020_1091346_0_0'])
    test.check_real(df_real, expected="000", start_idx=15, layout='qwerty')
    test.check_real(df_real, expected="000", start_idx=15, layout='dvorak')