import numpy as np

def get_features(ids):
    seq_len = len(ids)
    index = np.full(seq_len, -1.0, dtype=np.float32)
    length = np.full(seq_len, -1.0, dtype=np.float32)
    rel_pos = np.full(seq_len, -1.0, dtype=np.float32)
    is_start = np.zeros(seq_len, dtype=np.float32)
    is_end = np.zeros(seq_len, dtype=np.float32)

    delimiters = {
    0,    # NUL / pad
    9,    # tab
    10,   # newline
    13,   # carriage return
    32,   # space
    33,   # !
    34,   # "
    40,   # (
    41,   # )
    44,   # ,
    46,   # .
    58,   # :
    59,   # ;
    63,   # ?
    91,   # [
    93,   # ]
    123,  # {
    125   # }
}
    i = 0
    while i < seq_len:
        if ids[i] in delimiters:
            i += 1
            continue
        start = i
        while i < seq_len and ids[i] not in delimiters:
            i += 1
        end = i
        w_len = end - start
        
        index[start:end] = np.arange(w_len, dtype=np.float32)
        length[start:end] = w_len
        if w_len > 1:
            rel_pos[start:end] = np.arange(w_len, dtype=np.float32) / (w_len - 1)
        else:
            rel_pos[start:end] = 0.0
        
        is_start[start] = 1.0
        is_end[end - 1] = 1.0

    return index, length, rel_pos, is_start, is_end