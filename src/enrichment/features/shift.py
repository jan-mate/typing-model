import numpy as np

def get_map(keycode_to_slot, layout_map, shift_map):
    arr = np.full(128, 0.0, dtype=np.float32)
    for s_char in shift_map.keys():
        if ord(s_char) < 128:
            arr[ord(s_char)] = 1.0
    return arr