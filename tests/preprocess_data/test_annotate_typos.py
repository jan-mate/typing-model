import pytest
import pandas as pd
from src.preprocess_data.annotate_typos import get_edit_operations, reconstruct_final_text, label_keystrokes

@pytest.mark.parametrize("target, typed, expected_tags", [
    ("hello", "hello", ["M", "M", "M", "M", "M"]),
    ("cat", "bat", ["S", "M", "M"]),
    ("cat", "cart", ["M", "M", "I", "M"]),
    ("cart", "cat", ["M", "M", "D", "M"]),
    ("form", "from", ["M", "T", "M"]),
    ("hello", "ello", ["D", "M", "M", "M", "M"]),
    ("hello", "hell", ["M", "M", "M", "M", "D"]),
    ("cat", "acat", ["I", "M", "M", "M"]),
    ("cat", "cata", ["M", "M", "M", "I"])
])
def test_get_edit_operations(target, typed, expected_tags):
    actions = get_edit_operations(list(target), list(typed))
    tags = [a[0] for a in actions]
    assert tags == expected_tags

def test_reconstruct_final_text():
    df = pd.DataFrame({
        "LETTER": ["a", "b", "c", "BKSP", "d"],
        "PRESS_TIME": [10, 20, 30, 40, 50]
    })
    
    stack = reconstruct_final_text(df)
    letters = [x["LETTER"] for x in stack]
    
    assert letters == ["a", "b", "d"]
    assert stack[0]["_CORR"] is False
    assert stack[1]["_CORR"] is False
    assert stack[2]["_CORR"] is True

def test_label_keystrokes():
    stack = [
        {"LETTER": "c", "PRESS_TIME": 100, "_CORR": False},
        {"LETTER": "a", "PRESS_TIME": 200, "_CORR": False},
        {"LETTER": "r", "PRESS_TIME": 300, "_CORR": False},
        {"LETTER": "t", "PRESS_TIME": 400, "_CORR": False}
    ]
    target = "cat"
    
    labels = label_keystrokes("p1", 1, stack, target)
    
    assert len(labels) == 4
    assert labels[0]["IS_TYPO"] is False
    assert labels[1]["IS_TYPO"] is False
    assert labels[2]["IS_TYPO"] is True
    assert labels[2]["TYPO_TYPE"] == "Insertion"
    assert labels[3]["IS_TYPO"] is False
    assert labels[0]["TIME"] == 0
    assert labels[3]["TIME"] == 300

def test_label_keystrokes_deletion_at_start():
    stack = [
        {"LETTER": "e", "PRESS_TIME": 100, "_CORR": False},
        {"LETTER": "l", "PRESS_TIME": 200, "_CORR": False},
        {"LETTER": "l", "PRESS_TIME": 300, "_CORR": False},
        {"LETTER": "o", "PRESS_TIME": 400, "_CORR": False}
    ]
    target = "hello"
    
    labels = label_keystrokes("p1", 2, stack, target)
    
    assert labels[0]["IS_TYPO"] is True
    assert labels[0]["TYPO_TYPE"] == "Deletion"
    assert labels[1]["IS_TYPO"] is False

def test_label_keystrokes_deletion_at_end():
    stack = [
        {"LETTER": "h", "PRESS_TIME": 100, "_CORR": False},
        {"LETTER": "e", "PRESS_TIME": 200, "_CORR": False},
        {"LETTER": "l", "PRESS_TIME": 300, "_CORR": False},
        {"LETTER": "l", "PRESS_TIME": 400, "_COR_R": False}
    ]
    target = "hello"
    
    labels = label_keystrokes("p1", 3, stack, target)
    
    assert labels[2]["IS_TYPO"] is False
    assert labels[3]["IS_TYPO"] is True
    assert labels[3]["TYPO_TYPE"] == "Deletion"