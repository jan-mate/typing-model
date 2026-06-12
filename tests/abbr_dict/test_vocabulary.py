import pytest
from src.abbr_dict.vocabulary import extract_words, extract_suffixes, tokenize


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_apostrophe():
    assert tokenize("don't can't") == ["don't", "can't"]


def test_tokenize_strips_punctuation():
    tokens = tokenize("hello, world! foo.")
    assert tokens == ["hello", "world", "foo"]


def test_tokenize_empty():
    assert tokenize("") == []


# ---------------------------------------------------------------------------
# extract_words
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "the cat sat on the mat",
    "the dog chased the cat",
    "a cat and a dog",
]


def test_extract_words_returns_list_of_dicts():
    result = extract_words(SAMPLE_TEXTS, top_k=10)
    assert isinstance(result, list)
    for item in result:
        assert "text" in item
        assert "type" in item
        assert "frequency" in item
        assert item["type"] == "singleword"


def test_extract_words_top_k():
    result = extract_words(SAMPLE_TEXTS, top_k=3)
    assert len(result) <= 3


def test_extract_words_frequency_order():
    result = extract_words(SAMPLE_TEXTS, top_k=100)
    freqs = [item["frequency"] for item in result]
    assert freqs == sorted(freqs, reverse=True)


def test_extract_words_most_common_first():
    result = extract_words(SAMPLE_TEXTS, top_k=100)
    assert result[0]["text"] == "the"


def test_extract_words_frequency_values():
    result = extract_words(SAMPLE_TEXTS, top_k=100)
    freq_map = {item["text"]: item["frequency"] for item in result}
    assert freq_map["the"] == 4
    assert freq_map["cat"] == 3


def test_extract_words_empty_corpus():
    result = extract_words([], top_k=10)
    assert result == []


def test_extract_words_combined_corpus_format():
    texts = ["hello world hello", "foo bar baz foo"]
    result = extract_words(texts, top_k=5)
    freq_map = {item["text"]: item["frequency"] for item in result}
    assert freq_map["hello"] == 2
    assert freq_map["foo"] == 2


# ---------------------------------------------------------------------------
# extract_suffixes
# ---------------------------------------------------------------------------

SUFFIX_TEXTS = [
    "running jumping walking talking singing",
    "running fast jumping high walking slow",
    "talking loudly singing softly running wild",
    "formation nation station ration action",
    "connection section direction election",
]


def test_extract_suffixes_returns_list_of_dicts():
    result = extract_suffixes(SUFFIX_TEXTS, top_k=100, min_distinct_words=1)
    assert isinstance(result, list)
    for item in result:
        assert "text" in item
        assert "type" in item
        assert "frequency" in item
        assert item["type"] == "suffix"


def test_extract_suffixes_common_suffix_found():
    result = extract_suffixes(SUFFIX_TEXTS, top_k=100, min_distinct_words=1)
    suffix_texts = [item["text"] for item in result]
    assert "ng" in suffix_texts or "ing" in suffix_texts


def test_extract_suffixes_top_k():
    result = extract_suffixes(SUFFIX_TEXTS, top_k=5, min_distinct_words=1)
    assert len(result) <= 5


def test_extract_suffixes_min_len():
    result = extract_suffixes(SUFFIX_TEXTS, top_k=100, min_len=3, min_distinct_words=1)
    for item in result:
        assert len(item["text"]) >= 3


def test_extract_suffixes_max_len():
    result = extract_suffixes(SUFFIX_TEXTS, top_k=100, max_len=3, min_distinct_words=1)
    for item in result:
        assert len(item["text"]) <= 3


def test_extract_suffixes_frequency_order():
    result = extract_suffixes(SUFFIX_TEXTS, top_k=100, min_distinct_words=1)
    freqs = [item["frequency"] for item in result]
    assert freqs == sorted(freqs, reverse=True)


def test_extract_suffixes_empty_corpus():
    result = extract_suffixes([], top_k=100, min_distinct_words=1)
    assert result == []


def test_extract_suffixes_min_distinct_words_filters():
    result_loose = extract_suffixes(SUFFIX_TEXTS, top_k=100, min_distinct_words=1)
    result_strict = extract_suffixes(SUFFIX_TEXTS, top_k=100, min_distinct_words=10)
    assert len(result_strict) <= len(result_loose)


def test_extract_suffixes_excludes_words():
    result = extract_suffixes(SUFFIX_TEXTS, top_k=100, min_distinct_words=1,
                              exclude_words={"ing", "running"})
    texts = [i["text"] for i in result]
    assert "ing" not in texts
    assert "running" not in texts