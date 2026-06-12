
import pytest
from src.abbr_dict.candidates import _is_valid_candidate, _expand_trigger_forms
from src.abbr_dict.speed import expand_with_rpt, build_abbr_keystream, TRG_CHAR

def get_word_cost(word, rpt_key=True):
    return len(expand_with_rpt(word, rpt_key)) + 1

def get_trg_cost(abbr, trigger_form, rpt_key=True):
    return len(build_abbr_keystream(abbr, trigger_form, rpt_key=rpt_key))

def test_strictly_shorter_rule():
    # Word 'to' has 3 keys: t, o, TRG
    word = "to"
    word_cost = get_word_cost(word)
    assert word_cost == 3
    
    # Abbr 'tt' triggers:
    # trg: t, RPT, TRG (3 keys) -> NOT SHORTER
    # doubletap: t, t (2 keys) -> SHORTER
    # rpt_trg: t, RPT, TRG (3 keys) -> NOT SHORTER
    
    expansion = _expand_trigger_forms("tt", word, rpt_key=True)
    forms = [e["trigger_form"] for e in expansion]
    
    assert "doubletap" in forms
    assert "trg" not in forms
    assert "rpt_trg" not in forms

def test_internal_and_multiple_rpt():
    # Abbreviation 'nni' for 'tinnitus'
    # Keystream: n, RPT, i, TRG (4 keys)
    cost = get_trg_cost("nni", "trg", rpt_key=True)
    assert cost == 4
    
    # Abbreviation 'aaabbb'
    # Keystream: a, RPT, RPT, b, RPT, RPT, TRG (7 keys)
    cost = get_trg_cost("aaabbb", "trg", rpt_key=True)
    assert cost == 7

def test_tz_for_to_disallowed():
    # 'tz' for 'to' is not shorter (3 vs 3) and not a doublechar exception
    assert _is_valid_candidate("tz", "to") is False
    
    # Even if we forced it into expand:
    expansion = _expand_trigger_forms("tz", "to", rpt_key=True)
    assert len(expansion) == 0

def test_t_rpt_for_tt_disallowed():
    # Word 'tt' has 3 keys: t, RPT, TRG
    word = "tt"
    word_cost = get_word_cost(word)
    assert word_cost == 3
    
    # Abbr 'tt' trigger rpt_trg: t, RPT, TRG (3 keys) -> NOT SHORTER
    expansion = _expand_trigger_forms("tt", word, rpt_key=True)
    forms = [e["trigger_form"] for e in expansion]
    
    assert "rpt_trg" not in forms
    assert "doubletap" in forms # 2 keys < 3 keys

def test_real_doublechar_for_long_word():
    # 'letter' -> l, e, t, RPT, e, r, TRG (7 keys)
    word = "letter"
    word_cost = get_word_cost(word)
    assert word_cost == 7
    
    # Abbr 'tt'
    expansion = _expand_trigger_forms("tt", word, rpt_key=True)
    forms = [e["trigger_form"] for e in expansion]

    # With rpt on, the plain "tt"+TRG form is dropped (keystroke-identical to
    # rpt_trg [t, RPT, TRG]); only doubletap and rpt_trg remain.
    assert "trg" not in forms
    assert "doubletap" in forms
    assert "rpt_trg" in forms

def test_doublechar_at_least_once():
    # 'tt' for 'to' should be valid now (char appears once)
    assert _is_valid_candidate("tt", "to") is True
    
    # 'xx' for 'to' should NOT be valid (char does not appear)
    assert _is_valid_candidate("xx", "to") is False

def test_three_letter_word_strictly_shorter():
    # Word 'the' has 4 keys: t, h, e, TRG
    word = "the"
    word_cost = get_word_cost(word)
    assert word_cost == 4
    
    # Abbr 'tt' (rpt on): the plain trg form is dropped as a duplicate of rpt_trg.
    # doubletap: 2 keys < 4. SHORTER.
    # rpt_trg: 3 keys < 4. SHORTER.

    expansion = _expand_trigger_forms("tt", word, rpt_key=True)
    forms = [e["trigger_form"] for e in expansion]
    assert "trg" not in forms
    assert "doubletap" in forms
    assert "rpt_trg" in forms
