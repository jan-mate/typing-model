import pytest

from src.preprocess_data.download_expanded_corpus import (
    iter_sentences,
    is_real_sentence,
    is_bot_or_spam,
    MIN_LEN,
    MAX_LEN,
)


@pytest.mark.parametrize("text", [
    "Very true",
    "hello world",
    "I am happy today",
    "2 km road here now",          # majority alphabetic, >= 2 words
])
def test_is_real_sentence_accepts_sentence_like(text):
    assert is_real_sentence(text) is True


@pytest.mark.parametrize("text", [
    "Thx",                         # single token
    "Beautiful",                   # single word
    "6 2 H",                       # only one alphabetic token
    "the unk word",                # WikiText <unk> placeholder
    "UNK here now",                # unk case-insensitive
])
def test_is_real_sentence_rejects_junk(text):
    assert is_real_sentence(text) is False


@pytest.mark.parametrize("raw", [
    "I am a bot, and this action was performed automatically",
    "Please contact the moderators of this subreddit",
    "beep boop I am here",
    "Chat with one of these girls now",
    "Use my promo code for a discount",
])
def test_is_bot_or_spam_flags_known_signatures(raw):
    assert is_bot_or_spam(raw) is True


@pytest.mark.parametrize("raw", [
    "Just a normal comment about cats",
    "I love squirrels",
    "I think learning a new keyboard layout is a waste of time",
])
def test_is_bot_or_spam_passes_normal_text(raw):
    assert is_bot_or_spam(raw) is False


@pytest.mark.parametrize("raw", [None, 123, 4.5, ["list"]])
def test_is_bot_or_spam_rejects_non_strings(raw):
    assert is_bot_or_spam(raw) is True


def test_iter_sentences_splits_on_sentence_boundaries():
    assert list(iter_sentences("Hello world. Second one!")) == ["Hello world", "Second one"]


def test_iter_sentences_strips_urls():
    out = list(iter_sentences("check http://example.com now please"))
    assert out == ["check now please"]
    assert all("http" not in s for s in out)


def test_iter_sentences_rejects_disallowed_chars():
    # emoji and accented/symbol chars cause the whole piece to be dropped
    assert list(iter_sentences("I love this thing")) == ["I love this thing"]
    assert list(iter_sentences("I love this 😀 thing")) == []
    assert list(iter_sentences("Héllò")) == []


def test_iter_sentences_drops_long_repeats():
    # a 4+ char repeat run disqualifies the whole piece
    assert list(iter_sentences("aaaa is loud")) == []


def test_iter_sentences_respects_length_bounds():
    too_short = "ok"                       # 2 chars, below MIN_LEN
    assert list(iter_sentences(too_short)) == []
    too_long = " ".join(["word"] * 30)     # > MAX_LEN chars
    assert len(too_long) > MAX_LEN
    assert list(iter_sentences(too_long)) == []
    just_right = "meow meow"
    assert MIN_LEN <= len(just_right) <= MAX_LEN
    assert list(iter_sentences(just_right)) == ["meow meow"]


def test_iter_sentences_drops_single_word_fragments():
    # "Hello." splits to one word -> rejected by is_real_sentence
    assert list(iter_sentences("Hello.")) == []


@pytest.mark.parametrize("raw", [None, 123, 4.5])
def test_iter_sentences_handles_non_strings(raw):
    assert list(iter_sentences(raw)) == []


def test_iter_sentences_output_within_bounds_and_sentence_like():
    text = ("The cute fox. tiny. http://meow.com gone! "
            "aaaa noise. A normal clean sentence here please")
    for s in iter_sentences(text):
        assert MIN_LEN <= len(s) <= MAX_LEN
        assert is_real_sentence(s)
        assert "http" not in s