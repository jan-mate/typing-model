from tests.enrichment.utils import FeatureTester, load_real_data

tester_redirects = FeatureTester("redirects", is_one_hot=False)

def test_synthetic_redirects():
    tester_redirects.check("ccc", "PP0", layout="qwerty")
    
    tester_redirects.check("pop", "PP1", layout="qwerty")

    tester_redirects.check("cat", "PP1", layout="qwerty")
    
    tester_redirects.check("lol", "PP0", layout="qwerty")
    
    tester_redirects.check("fact", "PP10", layout="qwerty")

def test_padding():
    tester_redirects.check_padding(n_pads=3)

def test_real_data_redirects():
    # these
    df_real = load_real_data(['101030_1102308_0_0'])
    tester_redirects.check_real(df_real, expected="00001", start_idx=21, layout='qwerty')

    # level
    df_real = load_real_data(['104886_1146085_0_0'])
    tester_redirects.check_real(df_real, expected="00010", start_idx=14, layout='qwerty')