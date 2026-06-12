import json
import numpy as np

def load_unigram_map(path):
    arr = np.full(128, np.nan, dtype=np.float32)
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            if len(k) == 1:
                idx = ord(k)
                if idx < 128:
                    arr[idx] = v
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        pass
    return arr

def load_bigram_map(path):
    matrix = np.full((128, 128), np.nan, dtype=np.float32)
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            if len(k) == 2:
                idx1, idx2 = ord(k[0]), ord(k[1])
                if idx1 < 128 and idx2 < 128:
                    matrix[idx1, idx2] = v
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        pass
    return matrix