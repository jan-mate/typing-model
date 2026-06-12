from tests.enrichment.utils import FeatureTester, load_real_data

test = FeatureTester("double_row_jump")

def test_qwerty_jumps():
    test.check("foxes", "P0010")
    test.check("nuts",  "P100")
    test.check("taxonomy", "P0001111")
    test.check("plants", "P00000")

def test_padding_boundary():
    test.check(["foxes", "nuts"],["P0010", "P100"]
    )
    test.check(
        ["nut", "cat"],["P10", "P00"]
    )

def test_dvorak_jumps():
    test.check("xyrid", "P1000", layout="dvorak")
    test.check("bract", "P1000", layout="dvorak")
    test.check("squirrel", "P0000000", layout="dvorak")

def test_padding():
    test.check_padding(n_pads=3)

def test_real_data_jumps():
    # ce
    df_real = load_real_data(['100416_1095623_0_0'])
    test.check_real(df_real, expected="01", start_idx=11, layout='qwerty')
    test.check_real(df_real, expected="00", start_idx=11, layout='dvorak')

    # ce
    df_real = load_real_data(['100774_1099670_0_1'])
    test.check_real(df_real, expected="11", start_idx=6, layout='qwerty')
    test.check_real(df_real, expected="00", start_idx=6, layout='dvorak')