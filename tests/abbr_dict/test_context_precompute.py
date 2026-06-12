import numpy as np
import pytest

from src.abbr_dict.context_strips import (
    Occurrence,
    ContextStrip,
    build_word_index,
    build_suffix_index,
    find_occurrences,
    build_context_strip,
    weighted_mean_savings,
    compute_costs_sliced,
)


SENTENCES = [
    "the committee approved the budget",
    "a committee of experts reviewed it",
    "the department announced new rules",
    "each department submitted a report",
    "different views on the government policy",
    "the government released a statement",
    "this is just a filler sentence here",
]

WORD_FREQ_MAP = {
    "committee": 4.5,
    "department": 4.8,
    "different":  5.2,
    "government": 5.0,
}

SINGLEWORD_ITEM = {"text": "committee", "type": "singleword"}
SINGLEWORD_CAND = {"abbr": "comi", "trigger_form": "trg"}

SUFFIX_ITEM = {"text": "ment", "type": "suffix"}
SUFFIX_CAND = {"abbr": "mt",   "trigger_form": "trg"}


@pytest.fixture
def word_index():
    return build_word_index(SENTENCES, max_per_word=50)


@pytest.fixture
def suffix_idx(word_index):
    return build_suffix_index(word_index)


def test_word_index_contains_known_words(word_index):
    assert "committee" in word_index
    assert "department" in word_index


def test_word_index_respects_max_per_word():
    many = [f"the committee did thing {i}" for i in range(300)]
    idx = build_word_index(many, max_per_word=50)
    assert len(idx["committee"]) <= 50


def test_suffix_index_maps_ment(suffix_idx):
    # "department", "government" end in "ment"
    host_words = suffix_idx.get("ment", [])
    assert "department" in host_words
    assert "government" in host_words


def test_suffix_index_requires_word_longer_than_suffix(suffix_idx):
    # "ment" itself should not be a host word for suffix "ment"
    assert "ment" not in suffix_idx.get("ment", [])


def test_find_occurrences_singleword(word_index, suffix_idx):
    occs = find_occurrences("committee", "singleword", word_index, suffix_idx, N=5)
    assert len(occs) == 2   # exactly 2 sentences contain "committee"
    assert all(occ.body.lower() == "committee" for occ in occs)


def test_find_occurrences_suffix(word_index, suffix_idx):
    occs = find_occurrences("ment", "suffix", word_index, suffix_idx, N=5)
    bodies = [occ.body.lower() for occ in occs]
    assert any(b.endswith("ment") and len(b) > len("ment") for b in bodies)


def test_find_occurrences_fewer_than_n_returns_all(word_index, suffix_idx):
    # only 2 sentences with "committee" — N=10 should return 2, not error
    occs = find_occurrences("committee", "singleword", word_index, suffix_idx, N=10)
    assert len(occs) == 2


def test_find_occurrences_unknown_word_returns_empty(word_index, suffix_idx):
    occs = find_occurrences("zzzyyyxxx", "singleword", word_index, suffix_idx, N=3)
    assert occs == []


def test_find_occurrences_honours_n_cap(word_index, suffix_idx):
    occs = find_occurrences("ment", "suffix", word_index, suffix_idx, N=1)
    assert len(occs) <= 1


def _make_occ(sent, word):
    idx = sent.lower().find(word)
    assert idx != -1
    return Occurrence(sent=sent, start=idx, end=idx + len(word), body=sent[idx:idx + len(word)])


def test_strip_lengths_singleword():
    occ = _make_occ("the committee approved", "committee")
    strip = build_context_strip(
        occ, "committee", "singleword", "comi", "trg",
        left_len=3, right_len=1, rpt_key=False, word_freq_map=WORD_FREQ_MAP,
    )
    assert len(strip.word_ks) == 3 + strip.word_len + 1
    assert len(strip.abbr_ks) == 3 + strip.abbr_len + 1


def test_word_starts_at_left_len_singleword():
    occ = _make_occ("the committee approved", "committee")
    strip = build_context_strip(
        occ, "committee", "singleword", "comi", "trg",
        left_len=3, right_len=1, rpt_key=False, word_freq_map=WORD_FREQ_MAP,
    )
    assert strip.word_ks[3:3 + strip.word_len] == "committee"


def test_suffix_strip_preserves_stem():
    occ = _make_occ("each department submitted", "department")
    strip = build_context_strip(
        occ, "ment", "suffix", "mt", "trg",
        left_len=3, right_len=1, rpt_key=False, word_freq_map=WORD_FREQ_MAP,
    )
    assert "department" in strip.word_ks
    assert "depart" in strip.abbr_ks
    assert strip.abbr_ks[3:3 + 6] == "depart"   # stem starts at left_len


def test_singleword_weight_is_one():
    occ = _make_occ("the committee approved", "committee")
    strip = build_context_strip(
        occ, "committee", "singleword", "comi", "trg",
        left_len=3, right_len=1, rpt_key=False, word_freq_map=WORD_FREQ_MAP,
    )
    assert strip.weight == 1.0


def test_suffix_weight_is_zipf_based():
    occ = _make_occ("each department submitted", "department")
    strip = build_context_strip(
        occ, "ment", "suffix", "mt", "trg",
        left_len=3, right_len=1, rpt_key=False, word_freq_map=WORD_FREQ_MAP,
    )
    expected = 10.0 ** WORD_FREQ_MAP["department"]
    assert abs(strip.weight - expected) < 1e-6


def test_left_context_padded_near_sentence_start():
    occ = _make_occ("committee approved", "committee")  # no preceding chars
    strip = build_context_strip(
        occ, "committee", "singleword", "comi", "trg",
        left_len=3, right_len=1, rpt_key=False, word_freq_map=WORD_FREQ_MAP,
    )
    assert len(strip.word_ks[:3]) == 3   # always exactly left_len chars


def test_weighted_mean_uniform():
    assert weighted_mean_savings([1.0, 3.0], [1.0, 1.0]) == pytest.approx(2.0)


def test_weighted_mean_skewed():
    result = weighted_mean_savings([0.0, 4.0], [3.0, 1.0])
    assert result == pytest.approx(1.0)


def test_weighted_mean_empty():
    assert weighted_mean_savings([], []) == 0.0


class _MockModel:
    # constant predictions regardless of input, to test the slice logic
    w_back  = 3
    w_ahead = 1
    features = ["f0", "f1"]

    def __init__(self, constant=1.0):
        self.constant = constant

    def predict(self, X):
        n = len(X)
        return np.full(n, self.constant), np.zeros(n)


def _make_feats_and_offsets(strips):
    import pandas as pd
    all_chars = []
    offsets = []
    w_back, w_ahead = 3, 1
    for ks in strips:
        pad = [0] * w_back
        chars = [ord(c) for c in ks]
        padend = [0] * w_ahead
        start = len(all_chars) + w_back
        all_chars.extend(pad + chars + padend)
        offsets.append((start, len(ks)))

    n = len(all_chars)
    df = pd.DataFrame({"f0": np.zeros(n, dtype=np.float32),
                       "f1": np.zeros(n, dtype=np.float32)})
    return df, offsets


def test_sliced_sums_only_word_portion():
    # strip: "abc" (left=3) + "xy" (word, len=2) + "d" (right=1) = "abcxyd"
    strip = "abcxyd"
    df, offsets = _make_feats_and_offsets([strip])
    model = _MockModel(constant=1.0)

    # full compute_costs would sum positions 1..5; sliced sums only positions 3 and 4
    costs = compute_costs_sliced(df, offsets, model, left_len=3, word_lens=[2])
    assert costs[0][0] == pytest.approx(2.0)   # 2 positions × 1.0 each


def test_sliced_excludes_left_context():
    strip = "abcxyd"
    df, offsets = _make_feats_and_offsets([strip])

    model_high = _MockModel(constant=5.0)
    costs = compute_costs_sliced(df, offsets, model_high, left_len=3, word_lens=[2])
    # context included would give 25 (5 positions); slice gives 10
    assert costs[0][0] == pytest.approx(10.0)


def test_sliced_multiple_strips():
    strips = ["abcxyd", "abcxyzt"]   # word_lens = [2, 3]
    df, offsets = _make_feats_and_offsets(strips)
    model = _MockModel(constant=1.0)
    costs = compute_costs_sliced(df, offsets, model, left_len=3, word_lens=[2, 3])
    assert costs[0][0] == pytest.approx(2.0)
    assert costs[1][0] == pytest.approx(3.0)


def test_sliced_empty_offsets():
    import pandas as pd
    df = pd.DataFrame({"f0": [], "f1": []})
    costs = compute_costs_sliced(df, [], _MockModel(), left_len=3, word_lens=[])
    assert costs == []