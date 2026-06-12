import numpy as np

def get_map(keycode_to_slot, layout_map, shift_map):
    slot_to_val = {item['slot']: (item['x'], item['y']) for item in layout_map}
    arr = np.full((128, 2), np.nan, dtype=np.float32)
    for keycode, slot in keycode_to_slot.items():
        if len(keycode) == 1 and slot in slot_to_val:
            val = slot_to_val[slot]
            arr[ord(keycode)] = val
    for s_char, b_char in shift_map.items():
        if ord(b_char) < 128 and ord(s_char) < 128:
            arr[ord(s_char)] = arr[ord(b_char)]
    return arr