import numpy as np
from src.enrichment.features import finger

def get_map(keycode_to_slot, layout_map, shift_map):
    arr = finger._get_base_map(keycode_to_slot, layout_map, shift_map)
    transformed = np.where(arr >= 5, 9 - arr, arr)
    one_hot = np.zeros((128, 5), dtype=np.float32)
    valid = (transformed >= 0) & (transformed < 5)
    one_hot[valid, transformed[valid].astype(int)] = 1.0
    return one_hot