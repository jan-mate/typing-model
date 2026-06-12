from tests.enrichment.utils import FeatureTester, load_real_data

test = FeatureTester("out_roll")

def test_qwerty_out_rolls():
    test.check("sa", "P1")
    test.check("op", "P1")
    test.check("as", "P0")
    test.check("po", "P0")
    test.check("dsa", "P11")

def test_dvorak_out_rolls():
    test.check("oa", "P1", layout="dvorak")
    test.check("ns", "P1", layout="dvorak")

def test_padding_boundary():
    test.check(
        ["sa", "op"], 
        ["P1", "P1"]
    )

def test_padding():
    test.check_padding(n_pads=3)

def test_real_data_out_roll():
    # the
    df_real = load_real_data(['100001_1091076_0_0'])
    test.check_real(df_real, expected="000", start_idx=15, layout='qwerty')
    test.check_real(df_real, expected="000", start_idx=15, layout='dvorak')

    # the
    df_real = load_real_data(['100020_1091346_0_0'])
    test.check_real(df_real, expected="000", start_idx=15, layout='qwerty')
    test.check_real(df_real, expected="000", start_idx=15, layout='dvorak')