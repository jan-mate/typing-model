import pandas as pd
import pytest
from src.preprocess_data.filter_typos import run

@pytest.fixture
def base_config():
    return {
        "Substitution": False, 
        "Insertion": False, 
        "Deletion": False, 
        "Transposition": False, 
        "Proofreading": False
    }

@pytest.mark.parametrize("typo_type",[
    "Substitution", 
    "Insertion", 
    "Deletion", 
    "Transposition", 
    "Proofreading"
])
def test_filter_typos_splits_sequence_on_all_error_types(tmp_path, base_config, typo_type):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    
    df = pd.DataFrame({
        "SEQUENCE_ID":["seq1", "seq1", "seq1"],
        "TIME":[0, 100, 200],
        "TYPO_TYPE": [None, typo_type, None]
    })
    df.to_parquet(input_path)
    
    run(str(input_path), str(output_path), base_config)
    res_df = pd.read_parquet(output_path)
    
    assert len(res_df) == 2
    assert pd.isna(res_df["TYPO_TYPE"].iloc[0])
    assert pd.isna(res_df["TYPO_TYPE"].iloc[1])
    assert list(res_df["SEQUENCE_ID"]) ==["seq1_0", "seq1_1"]
    assert list(res_df["TIME"]) ==[0, 0]

def test_filter_typos_foxes_example(tmp_path, base_config):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    
    df = pd.DataFrame({
        "SEQUENCE_ID": ["seq_foxes"] * 5,
        "LETTER": ["f", "o", "z", "e", "s"],
        "TIME":[0, 110, 260, 380, 485], 
        "TYPO_TYPE": [None, None, "Substitution", None, None]
    })
    df.to_parquet(input_path)
    
    run(str(input_path), str(output_path), base_config)
    res_df = pd.read_parquet(output_path)
    
    assert len(res_df) == 4
    assert list(res_df["LETTER"]) == ["f", "o", "e", "s"]
    assert list(res_df["SEQUENCE_ID"]) ==["seq_foxes_0", "seq_foxes_0", "seq_foxes_1", "seq_foxes_1"]
    assert list(res_df["TIME"]) == [0, 110, 0, 105]

def test_filter_typos_keeps_configured_errors_without_splitting(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    
    df = pd.DataFrame({
        "SEQUENCE_ID":["seq2", "seq2", "seq2"],
        "TIME":[0, 150, 300],
        "TYPO_TYPE": [None, "Insertion", None]
    })
    df.to_parquet(input_path)
    
    config = {"Insertion": True}
    
    run(str(input_path), str(output_path), config)
    res_df = pd.read_parquet(output_path)
    
    assert len(res_df) == 3
    assert res_df["TYPO_TYPE"].iloc[1] == "Insertion"
    assert list(res_df["SEQUENCE_ID"]) ==["seq2_0", "seq2_0", "seq2_0"]
    assert list(res_df["TIME"]) ==[0, 150, 300]


def test_filter_typos_proofreading_split(tmp_path, base_config):
    # typing 'cat' as c,b,BKSP,a,t: the proofreading b/BKSP drop, splitting into 'c' and 'at'
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    
    df = pd.DataFrame({
        "SEQUENCE_ID": ["seq_proof"] * 5,
        "LETTER":["c", "b", "BKSP", "a", "t"],
        "TIME":[0, 120, 250, 400, 510],
        # 'b' and 'BKSP' are flagged as Proofreading
        "TYPO_TYPE": [None, "Proofreading", "Proofreading", None, None]
    })
    df.to_parquet(input_path)
    
    run(str(input_path), str(output_path), base_config)
    res_df = pd.read_parquet(output_path)
    
    assert len(res_df) == 3
    assert list(res_df["LETTER"]) == ["c", "a", "t"]
    
    # jumps 0 -> 2 because 2 rows were skipped (sub_id += 1 happened twice)
    assert list(res_df["SEQUENCE_ID"]) ==["seq_proof_0", "seq_proof_2", "seq_proof_2"]

    # times reset to 0 at the start of each new segment
    assert list(res_df["TIME"]) == [0, 0, 110] # 510 - 400 = 110


def test_filter_typos_deletion_split(tmp_path, base_config):
    # typing 'cats' as c,t,s (skipped 'a'): the deletion-boundary t drops, splitting into 'c' and 's'
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    
    df = pd.DataFrame({
        "SEQUENCE_ID": ["seq_del"] * 3,
        "LETTER": ["c", "t", "s"],
        "TIME":[0, 150, 270],
        "TYPO_TYPE":[None, "Deletion", None]
    })
    df.to_parquet(input_path)
    
    run(str(input_path), str(output_path), base_config)
    res_df = pd.read_parquet(output_path)
    
    assert len(res_df) == 2
    assert list(res_df["LETTER"]) == ["c", "s"]
    assert list(res_df["SEQUENCE_ID"]) == ["seq_del_0", "seq_del_1"]
    
    assert list(res_df["TIME"]) ==[0, 0]