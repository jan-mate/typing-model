import numpy as np

def _get_base_map(keycode_to_slot, layout_map, shift_map):
    slot_to_val = {item['slot']: item['finger'] for item in layout_map}
    arr = np.full(128, -1.0, dtype=np.float32)
    for keycode, slot in keycode_to_slot.items():
        if len(keycode) == 1 and slot in slot_to_val:
            val = slot_to_val[slot]
            arr[ord(keycode)] = val
    for s_char, b_char in shift_map.items():
        if ord(b_char) < 128 and ord(s_char) < 128:
            arr[ord(s_char)] = arr[ord(b_char)]
    return arr

def get_map(keycode_to_slot, layout_map, shift_map):
    arr = _get_base_map(keycode_to_slot, layout_map, shift_map)
    one_hot = np.zeros((128, 10), dtype=np.float32)
    valid = (arr >= 0) & (arr < 10)
    one_hot[valid, arr[valid].astype(int)] = 1.0
    return one_hot