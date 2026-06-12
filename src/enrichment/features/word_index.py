import numpy as np

def get_indices(ids):
    indices = np.full(len(ids), -1.0, dtype=np.float32)
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
    
    curr = -1
    for i in range(len(ids)):
        if ids[i] in delimiters:
            curr = -1
        else:
            curr += 1
            indices[i] = float(curr)
    return indices