import json
import numpy as np

def load_map(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_word_frequencies(ids, word_map):
    freqs = np.full(len(ids), np.nan, dtype=np.float32)
    delimiters = {32, 46, 44, 33, 63}
    
    i = 0
    while i < len(ids):
        if ids[i] == 0 or ids[i] in delimiters:
            i += 1
            continue
            
        start = i
        while i < len(ids) and ids[i] != 0 and ids[i] not in delimiters:
            i += 1
        end = i
        
        word = "".join(chr(ids[j]).lower() for j in range(start, end))
        val = word_map.get(word, 0.0)
        freqs[start:end] = val
        
    return freqs