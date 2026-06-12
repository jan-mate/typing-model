import json
import os
import pytest

DICT_PATH = "data/abbr_dict/optimization_results/optimized_dict.json"

@pytest.fixture
def dictionary_data():
    if not os.path.exists(DICT_PATH):
        pytest.skip(f"Dictionary file not found at {DICT_PATH}")
    with open(DICT_PATH) as f:
        return json.load(f)

def test_uniqueness(dictionary_data):
    seen = set()
    for entry in dictionary_data["dictionary"]:
        key = (entry["abbr"], entry["trigger_form"])
        assert key not in seen, f"Duplicate abbreviation {key} for '{entry['text']}'"
        seen.add(key)

def test_max_size(dictionary_data):
    max_size = dictionary_data["metadata"]["max_dict_size"]
    assert len(dictionary_data["dictionary"]) <= max_size

def is_doublechar(abbr):
    return len(abbr) == 2 and abbr[0] == abbr[1]

def test_validity_length_rule(dictionary_data):
    for entry in dictionary_data["dictionary"]:
        word = entry["text"]
        abbr = entry["abbr"]

        if is_doublechar(abbr):  # exempt from the 25% rule
            continue

        max_len = int(len(word) * 0.75)
        assert len(abbr) <= max_len, f"Abbreviation '{abbr}' for '{word}' violates the 25% length rule (max allowed: {max_len})"

def test_rpt_key_consistency(dictionary_data):
    rpt_key_enabled = dictionary_data["metadata"]["rpt_key"]
    if not rpt_key_enabled:
        # rpt off: only 'trg' is allowed
        for entry in dictionary_data["dictionary"]:
            assert entry["trigger_form"] == "trg"
    else:
        # rpt on: doublechars must be doubletap or rpt_trg
        for entry in dictionary_data["dictionary"]:
            if is_doublechar(entry["abbr"]):
                assert entry["trigger_form"] in ["doubletap", "rpt_trg"], \
                    f"Doublechar abbr '{entry['abbr']}' for '{entry['text']}' uses 'trg' form while RPT_KEY is enabled."

def test_linked_text_consistency(dictionary_data):
    seen_text = {}
    for entry in dictionary_data["dictionary"]:
        text = entry["text"]
        seen_text[text] = entry

def test_positive_savings(dictionary_data):
    for entry in dictionary_data["dictionary"]:
        score = entry["savings_z_eff"]
        assert score > 0, f"Entry '{entry['text']}' has non-positive score: {score}"

def test_intuitiveness_floor(dictionary_data):
    for entry in dictionary_data["dictionary"]:
        assert entry["intuitiveness"] >= 0.1, f"Entry '{entry['text']}' below intuitiveness floor: {entry['intuitiveness']}"