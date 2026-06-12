import numpy as np

def get_features(seq_len):
    pos = np.arange(seq_len, dtype=np.float32)
    length = np.full(seq_len, seq_len, dtype=np.float32)
    if seq_len > 1:
        rel_pos = pos / (seq_len - 1)
    else:
        rel_pos = np.zeros(seq_len, dtype=np.float32)
    return pos, length, rel_pos