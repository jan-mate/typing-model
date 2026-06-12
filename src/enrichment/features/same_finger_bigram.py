import numpy as np

def get_matrix(finger_map):
    matrix = np.zeros((128, 128), dtype=np.float32)
    valid_fingers = (finger_map >= 0)
    
    P, C = np.ix_(np.arange(128), np.arange(128))
    
    valid_mask = valid_fingers[P] & valid_fingers[C]
    not_thumb = (finger_map[P] != 4) & (finger_map[P] != 5)
    same_finger = finger_map[P] == finger_map[C]
    diff_key = P != C
    
    condition = valid_mask & same_finger & diff_key & not_thumb
    matrix[condition] = 1.0
    return matrix