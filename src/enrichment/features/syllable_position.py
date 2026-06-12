import numpy as np
import pyphen

_PYPHEN = pyphen.Pyphen(lang='en_US')

def get_features(ids):
    seq_len = len(ids)
    is_start = np.full(seq_len, -1.0, dtype=np.float32)
    is_end   = np.full(seq_len, -1.0, dtype=np.float32)

    delimiters = {0, 9, 10, 13, 32, 33, 34, 40, 41, 44, 46, 58, 59, 63, 91, 93, 123, 125}

    i = 0
    while i < seq_len:
        if ids[i] in delimiters:
            i += 1
            continue

        start = i
        while i < seq_len and ids[i] not in delimiters:
            i += 1
        end = i

        is_start[start:end] = 0.0
        is_end[start:end]   = 0.0

        word = "".join(chr(c).lower() for c in ids[start:end] if c < 128)
        if not word:
            continue

        parts = _PYPHEN.inserted(word).split('-')
        pos = start
        for part in parts:
            if part:
                is_start[pos] = 1.0
                is_end[pos + len(part) - 1] = 1.0
                pos += len(part)

    return is_start, is_end