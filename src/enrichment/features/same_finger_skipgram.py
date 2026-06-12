import numpy as np

def get_matrix(finger_map):
    matrix = np.zeros((128, 128, 128), dtype=np.float32)
    valid_fingers = (finger_map >= 0)
    
    PP, P, C = np.ix_(np.arange(128), np.arange(128), np.arange(128))
    
    valid_mask = valid_fingers[PP] & valid_fingers[P] & valid_fingers[C]
    

    same_finger_ends = finger_map[PP] == finger_map[C]
    
    diff_finger_mid = finger_map[PP] != finger_map[P]
    
    diff_key_ends = PP != C
    
    condition = valid_mask & same_finger_ends & diff_finger_mid & diff_key_ends
    matrix[condition] = 1.0
    return matrix