from tests.enrichment.utils import FeatureTester, load_real_data

test = FeatureTester("in_roll")

def test_qwerty_in_rolls():
    # Left hand: moving pinky to index (a -> s)
    test.check("as", "P1")
    # Right hand: moving pinky to index (p -> o)
    test.check("po", "P1")
    # Reverse directions (these are out-rolls)
    test.check("sa", "P0")
    test.check("op", "P0")
    # Multiple rolls
    test.check("asd", "P11")

def test_dvorak_in_rolls():
    # Left hand home row (a -> o)
    test.check("ao", "P1", layout="dvorak")
    # Right hand home row (s -> n)
    test.check("sn", "P1", layout="dvorak")

def test_padding_boundary():
    # Tests that the engine correctly pads boundaries between separate sequences
    test.check(
        ["as", "po"], 
        ["P1", "P1"]
    )

def test_padding():
    test.check_padding(n_pads=3)

def test_real_data_in_roll():
    # the
    df_real = load_real_data(['100001_1091076_0_0'])
    test.check_real(df_real, expected="000", start_idx=15, layout='qwerty')
    test.check_real(df_real, expected="010", start_idx=15, layout='dvorak')

    # the
    df_real = load_real_data(['100020_1091346_0_0'])
    test.check_real(df_real, expected="000", start_idx=15, layout='qwerty')
    test.check_real(df_real, expected="010", start_idx=15, layout='dvorak')