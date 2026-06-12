import pytest
from src.abbr_dict.intuitiveness import compute_heuristic_score

def test_score_range():
    words = ["information", "the", "classification", "see", "cat"]
    abbrs = ["in", "info", "t", "th", "clas", "c", "si", "kat", "x", ""]

    for word in words:
        for abbr in abbrs:
            score = compute_heuristic_score(word, abbr)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {word}/{abbr}"

def test_obvious_rankings():
    # information: in > info > informatin > ifotn
    s_in = compute_heuristic_score("information", "in")
    s_info = compute_heuristic_score("information", "info")
    s_informatin = compute_heuristic_score("information", "informatin")
    s_ifotn = compute_heuristic_score("information", "ifotn")

    assert s_in > s_informatin
    assert s_info > s_ifotn
    assert s_in > s_ifotn

    # and: n > a > an > ad
    s_n = compute_heuristic_score("and", "n")
    s_a = compute_heuristic_score("and", "a")
    s_an = compute_heuristic_score("and", "an")
    s_ad = compute_heuristic_score("and", "ad")

    assert s_n > s_ad
    assert s_a > s_ad
    # 'n' is a canonical sound match (n->and), so it should be very high
    assert s_n >= 0.9

def test_phonetic_matches():
    # kat for cat should be reasonably high, certainly better than random noise
    s_kat = compute_heuristic_score("cat", "kat")
    s_sat = compute_heuristic_score("cat", "sat")
    s_random = compute_heuristic_score("cat", "xyz")

    assert s_kat > s_sat
    assert s_kat > s_random
    assert s_kat > 0.6

    assert compute_heuristic_score("see", "s") >= 0.8
    assert compute_heuristic_score("see", "c") >= 0.9  # LETTER_SOUNDS_LIKE short-circuit

def test_prefix_utility():
    # for "information", "info" (4 chars) is strong; "i" (1 char) is penalized for being too short
    s_info = compute_heuristic_score("information", "info")
    s_i = compute_heuristic_score("information", "i")
    s_informatio = compute_heuristic_score("information", "informatio")

    assert s_info > s_i
    assert s_info > s_informatio

def test_letter_name_matches():
    assert compute_heuristic_score("you", "u") == 0.95
    assert compute_heuristic_score("why", "y") == 0.95
    assert compute_heuristic_score("are", "r") == 0.95
    assert compute_heuristic_score("and", "n") == 0.95

def test_contiguity():
    # manager: man > mngr (internal deletions penalized)
    s_man = compute_heuristic_score("manager", "man")
    s_mngr = compute_heuristic_score("manager", "mngr")
    assert s_man > s_mngr

def test_empty_input():
    assert compute_heuristic_score("word", "") >= 0.0
    assert compute_heuristic_score("", "abbr") >= 0.0