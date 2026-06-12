from tests.enrichment.utils import FeatureTester, load_real_data

tester_x = FeatureTester("x", is_one_hot=False)
tester_y = FeatureTester("y", is_one_hot=False)

def test_synthetic_space_coords():
    tester_x.check("cat",[2.5, 0.0, 3.75], layout="qwerty", n_pads=0)
    tester_y.check("cat", [-1.0, 0.0, 1.0], layout="qwerty", n_pads=0)
    
    tester_x.check("cat",[2.25, 0.0, 2.0], layout="dvorak", n_pads=0)
    tester_y.check("cat",[1.0, 0.0, 0.0], layout="dvorak", n_pads=0)

    tester_x.check("meow",[2.5, 1.75, 1.25, 0.75], layout="qwerty", n_pads=0)
    tester_y.check("meow", [-1.0, 1.0, 1.0, 1.0], layout="qwerty", n_pads=0)
    
    tester_x.check("meow",[2.5, 2.0, 1.0, 1.5], layout="dvorak", n_pads=0)
    tester_y.check("meow",[-1.0, 0.0, 0.0, -1.0], layout="dvorak", n_pads=0)

    tester_x.check(":3", [0.0, 1.25], layout="qwerty", n_pads=0)
    tester_y.check(":3",[0.0, 2.0], layout="qwerty", n_pads=0)
    
    tester_x.check(":3", [0.5, 1.25], layout="dvorak", n_pads=0)
    tester_y.check(":3", [-1.0, 2.0], layout="dvorak", n_pads=0)

    tester_x.check("OwO",[1.25, 0.75, 1.25], layout="qwerty", n_pads=0)
    tester_y.check("OwO",[1.0, 1.0, 1.0], layout="qwerty", n_pads=0)
    
    tester_x.check("OwO", [1.0, 1.5, 1.0], layout="dvorak", n_pads=0)
    tester_y.check("OwO",[0.0, -1.0, 0.0], layout="dvorak", n_pads=0)

    tester_x.check("420",[2.25, 0.25, 0.75], layout="qwerty", n_pads=0)
    tester_y.check("420", [2.0, 2.0, 2.0], layout="qwerty", n_pads=0)
    
    tester_x.check("420",[2.25, 0.25, 0.75], layout="dvorak", n_pads=0)
    tester_y.check("420",[2.0, 2.0, 2.0], layout="dvorak", n_pads=0)

def test_padding():
    tester_x.check_padding(n_pads=3)
    tester_y.check_padding(n_pads=3)

def test_real_data_space_coords():
    # cat
    df_real = load_real_data(['100446_1095949_0_1'])
    
    tester_x.check_real(df_real, expected=[2.5, 0.0, 3.75], start_idx=4, layout='qwerty')
    tester_y.check_real(df_real, expected=[-1.0, 0.0, 1.0], start_idx=4, layout='qwerty')
    
    tester_x.check_real(df_real, expected=[2.25, 0.0, 2.0], start_idx=4, layout='dvorak')
    tester_y.check_real(df_real, expected=[1.0, 0.0, 0.0], start_idx=4, layout='dvorak')

    # cat
    df_real = load_real_data(['102515_1120554_0_2'])
    
    tester_x.check_real(df_real, expected=[2.5, 0.0, 3.75], start_idx=17, layout='qwerty')
    tester_y.check_real(df_real, expected=[-1.0, 0.0, 1.0], start_idx=17, layout='qwerty')
    
    tester_x.check_real(df_real, expected=[2.25, 0.0, 2.0], start_idx=17, layout='dvorak')
    tester_y.check_real(df_real, expected=[1.0, 0.0, 0.0], start_idx=17, layout='dvorak')