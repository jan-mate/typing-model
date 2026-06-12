import numpy as np

def get_map(keycode_to_slot, reach_json):
    arr = np.full((128, 3), np.nan, dtype=np.float32)
    for keycode, slot in keycode_to_slot.items():
        if len(keycode) == 1 and slot in reach_json:
            o = ord(keycode)
            if o < 128:
                arr[o] = reach_json[slot]
    return arr