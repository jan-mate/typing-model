from tests.enrichment.utils import FeatureTester, load_real_data

tester_scissors = FeatureTester("scissors", is_one_hot=False)

def test_synthetic_scissors():
    tester_scissors.check("ex", "P1", layout="qwerty")
    
    tester_scissors.check("zw", "P1", layout="qwerty")
    
    tester_scissors.check("p.", "P1", layout="qwerty")
    
    tester_scissors.check("ce", "P0", layout="qwerty")
    
    tester_scissors.check("qa", "P0", layout="qwerty")
    
    tester_scissors.check("cr", "P0", layout="qwerty")

def test_padding():
    tester_scissors.check_padding(n_pads=3)

def test_real_data_scissors():

    # ex
    df_real = load_real_data(['100446_1095936_1_1'])
    tester_scissors.check_real(df_real, expected="P1", start_idx=0, layout='qwerty')
    tester_scissors.check_real(df_real, expected="P0", start_idx=0, layout='dvorak')

    # mo
    df_real = load_real_data(['100001_1091153_1_0'])
    tester_scissors.check_real(df_real, expected="00", start_idx=10, layout='qwerty')
    tester_scissors.check_real(df_real, expected="00", start_idx=10, layout='dvorak')

    # ce
    df_real = load_real_data(['100020_1091346_0_0'])
    tester_scissors.check_real(df_real, expected="00", start_idx=30, layout='qwerty')
    tester_scissors.check_real(df_real, expected="00", start_idx=30, layout='dvorak')