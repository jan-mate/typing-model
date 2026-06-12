import pytest
import random
from src.abbr_dict.candidates import (
    get_first_syllable,
    generate_candidates,
    generate_all_candidates,
    _generate_deletions,
    _generate_replacements,
    _generate_phonetic_replacements,
    _generate_letter_name_candidates,
    _generate_syllable_candidates,
    _is_doublechar,
    _is_valid_candidate,
)


# ---------------------------------------------------------------------------
# get_first_syllable
# ---------------------------------------------------------------------------

def test_get_first_syllable_computer():
    assert get_first_syllable("computer") == "com"


def test_get_first_syllable_information():
    assert get_first_syllable("information") == "in"


def test_get_first_syllable_short_word_returns_none():
    assert get_first_syllable("to") is None
    assert get_first_syllable("a") is None


def test_get_first_syllable_the():
    result = get_first_syllable("the")
    assert result is None or result == "the"


def test_get_first_syllable_does_not_return_full_word():
    result = get_first_syllable("computer")
    assert result != "computer"


# ---------------------------------------------------------------------------
# _generate_deletions
# ---------------------------------------------------------------------------

def test_deletions_shorter_than_original():
    results = _generate_deletions("and", min_len=1)
    for r in results:
        assert len(r) < len("and")


def test_deletions_min_len_respected():
    results = _generate_deletions("and", min_len=2)
    for r in results:
        assert len(r) >= 2


def test_deletions_original_not_in_results():
    results = _generate_deletions("test", min_len=1)
    assert "test" not in results


def test_deletions_known_results():
    results = _generate_deletions("the", min_len=1)
    assert "th" in results
    assert "te" in results
    assert "he" in results


# ---------------------------------------------------------------------------
# _generate_replacements
# ---------------------------------------------------------------------------

def test_replacements_validity():
    results = _generate_replacements("and", max_replacements=1)
    for r in results:
        assert len(r) < len("and") or _is_doublechar(r)


def test_replacements_differ_from_original():
    results = _generate_replacements("and", max_replacements=1)
    for r in results:
        r != "and"


# ---------------------------------------------------------------------------
# generate_candidates
# ---------------------------------------------------------------------------

def test_generate_candidates_returns_list():
    result = generate_candidates("computer", n=5)
    assert isinstance(result, list)


def test_generate_candidates_n_limit():
    result = generate_candidates("information", n=5)
    assert len(result) <= 5


def test_generate_candidates_min_len():
    result = generate_candidates("computer", n=10, min_len=2)
    for c in result:
        assert len(c) >= 2


def test_generate_candidates_no_original():
    word = "computer"
    result = generate_candidates(word, n=10)
    assert word not in result


def test_generate_candidates_all_strings():
    result = generate_candidates("information", n=5)
    for c in result:
        assert isinstance(c, str)
        assert len(c) > 0


def test_generate_candidates_short_word_constraint():
    # 'to' can only abbreviate to a doublechar (tt/oo) or something shorter
    result = generate_candidates("to", n=10)
    for c in result:
        if len(c) == 2:
            assert _is_doublechar(c)
        else:
            assert len(c) < 2

def test_generate_candidates_three_letter_word_constraint():
    # 3-letter word: max abbr len = int(3*0.75) = 2
    result = generate_candidates("the", n=20)
    for c in result:
        assert len(c) <= 2


# ---------------------------------------------------------------------------
# _is_valid_candidate — 25% shortening rule
# ---------------------------------------------------------------------------

def test_valid_candidate_25_percent_rule_4_letter_word():
    # int(4*0.75)=3, so a 3-char abbr is exactly at the threshold
    assert _is_valid_candidate("abc", "abcd") is True   # 3 <= 3
    assert _is_valid_candidate("abcd", "abcd") is False  # 4 > 3


def test_valid_candidate_5_letter_word_rejects_4_chars():
    # int(5*0.75)=3, so a 4-char abbr (only 20% shorter) is rejected
    assert _is_valid_candidate("abcd", "abcde") is False  # 4 > 3
    assert _is_valid_candidate("abc", "abcde") is True    # 3 <= 3


def test_valid_candidate_6_letter_word():
    # int(6*0.75)=4
    assert _is_valid_candidate("abcd", "abcdef") is True   # 4 <= 4
    assert _is_valid_candidate("abcde", "abcdef") is False  # 5 > 4


def test_valid_candidate_3_letter_word():
    # int(3*0.75)=2
    assert _is_valid_candidate("ab", "abc") is True   # 2 <= 2
    assert _is_valid_candidate("abc", "abc") is False  # 3 > 2


def test_valid_candidate_doublechar_exception():
    # doublechars are always valid regardless of the 25% rule
    assert _is_valid_candidate("tt", "to") is True
    assert _is_valid_candidate("oo", "to") is True


def test_valid_candidate_2_letter_word_non_doublechar():
    # non-doublechar must obey the 25% rule: max len = int(2*0.75)=1
    assert _is_valid_candidate("t", "to") is True   # 1 <= 1
    assert _is_valid_candidate("to", "to") is False  # 2 > 1 (and not doublechar)


def test_generate_candidates_reproducible_with_seed():
    random.seed(42)
    r1 = generate_candidates("computer", n=5)
    random.seed(42)
    r2 = generate_candidates("computer", n=5)
    assert r1 == r2


# ---------------------------------------------------------------------------
# generate_all_candidates
# ---------------------------------------------------------------------------

SAMPLE_VOCAB = {
    "items": [
        {"text": "computer", "type": "singleword", "frequency": 100},
        {"text": "ing", "type": "suffix", "frequency": 500},
        {"text": "i", "type": "singleword", "frequency": 1000},  # single char, skipped
    ]
}


def test_generate_all_candidates_structure():
    result = generate_all_candidates(SAMPLE_VOCAB, n_per_item=5)
    assert "items" in result
    assert isinstance(result["items"], list)


def test_generate_all_candidates_item_fields():
    result = generate_all_candidates(SAMPLE_VOCAB, n_per_item=5)
    for item in result["items"]:
        assert "text" in item
        assert "type" in item
        assert "frequency" in item
        assert "candidates" in item
        assert isinstance(item["candidates"], list)


def test_generate_all_candidates_candidate_fields():
    result = generate_all_candidates(SAMPLE_VOCAB, n_per_item=5)
    for item in result["items"]:
        for cand in item["candidates"]:
            assert "abbr" in cand


def test_generate_all_candidates_enable_flags():
    result_no_suffixes = generate_all_candidates(SAMPLE_VOCAB, n_per_item=5, enable_suffixes=False)
    types = [i["type"] for i in result_no_suffixes["items"]]
    assert "suffix" not in types

    result_no_words = generate_all_candidates(SAMPLE_VOCAB, n_per_item=5, enable_words=False)
    types = [i["type"] for i in result_no_words["items"]]
    assert "singleword" not in types


def test_generate_all_candidates_skips_single_char():
    result = generate_all_candidates(SAMPLE_VOCAB, n_per_item=5)
    texts = [i["text"] for i in result["items"]]
    assert "i" not in texts


# ---------------------------------------------------------------------------
# _generate_letter_name_candidates
# ---------------------------------------------------------------------------

def test_letter_name_see():
    assert "c" in _generate_letter_name_candidates("see")

def test_letter_name_are():
    assert "r" in _generate_letter_name_candidates("are")

def test_letter_name_you():
    assert "u" in _generate_letter_name_candidates("you")

def test_letter_name_unknown_word():
    assert _generate_letter_name_candidates("computer") == set()


# ---------------------------------------------------------------------------
# _generate_syllable_candidates
# ---------------------------------------------------------------------------

def test_syllable_acronym_without():
    result = _generate_syllable_candidates("without")
    assert "wo" in result

def test_syllable_acronym_information():
    result = _generate_syllable_candidates("information")
    assert len(result) > 0
    for c in result:
        assert _is_valid_candidate(c, "information")

def test_syllable_candidates_short_word_skipped():
    assert _generate_syllable_candidates("to") == set()

def test_syllable_candidates_all_valid():
    for word in ["computer", "manager", "without", "people"]:
        for c in _generate_syllable_candidates(word):
            assert _is_valid_candidate(c, word), f"{c!r} invalid for {word!r}"


# ---------------------------------------------------------------------------
# _generate_phonetic_replacements
# ---------------------------------------------------------------------------

def test_phonetic_replacements_all_valid():
    for c in _generate_phonetic_replacements("computer"):
        assert _is_valid_candidate(c, "computer"), f"{c!r} invalid for 'computer'"

def test_phonetic_replacements_differ_from_original():
    for c in _generate_phonetic_replacements("computer"):
        assert c != "computer"

def test_phonetic_replacements_produces_results():
    # 'computer' has chars with phonetic subs (c->k, o->u, etc.)
    result = _generate_phonetic_replacements("computer")
    assert len(result) > 0


# ---------------------------------------------------------------------------
# generate_candidates — integration with new sources
# ---------------------------------------------------------------------------

def test_generate_candidates_includes_letter_name():
    result = generate_candidates("see", n=100)
    assert "c" in result

def test_generate_all_candidates_no_multiword():
    vocab_with_phrase = {
        "items": [
            {"text": "computer", "type": "singleword", "frequency": 100},
            {"text": "by the way", "type": "multiword", "frequency": 50},
        ]
    }
    result = generate_all_candidates(vocab_with_phrase, n_per_item=5)
    types = [i["type"] for i in result["items"]]
    assert "multiword" not in types
