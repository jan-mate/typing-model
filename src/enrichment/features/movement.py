import numpy as np

def get_matrix(keycode_to_slot, bigram_json):
    matrix = np.full((128, 128, 3), np.nan, dtype=np.float32)
    slot_to_chars = {}
    for char, slot in keycode_to_slot.items():
        if len(char) == 1:
            slot_to_chars.setdefault(slot, []).append(char)
            
    for bigram_str, values in bigram_json.items():
        if len(bigram_str) < 2: continue
        s1, s2 = bigram_str[:-1], bigram_str[-1:]
        if s1 in slot_to_chars and s2 in slot_to_chars:
            for c1 in slot_to_chars[s1]:
                for c2 in slot_to_chars[s2]:
                    o1, o2 = ord(c1), ord(c2)
                    if o1 < 128 and o2 < 128:
                        matrix[o1, o2] = values
    return matrix