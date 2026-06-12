import numpy as np

def get_matrix(finger_map, hand_map):
    matrix = np.zeros((128, 128, 128), dtype=np.float32)
    valid = (finger_map >= 0) & (hand_map >= 0)
    
    PP, P, C = np.ix_(np.arange(128), np.arange(128), np.arange(128))
    
    same_hand = (hand_map[PP] == hand_map[P]) & (hand_map[P] == hand_map[C]) & (hand_map[P] < 2)
    valid_mask = valid[PP] & valid[P] & valid[C] & same_hand
    
    h = hand_map[PP]
    f1, f2, f3 = finger_map[PP], finger_map[P], finger_map[C]
    
    left_out = (h == 0) & (f2 < f1) & (f3 < f2)
    right_out = (h == 1) & (f2 > f1) & (f3 > f2)
    
    condition = valid_mask & (left_out | right_out)
    matrix[condition] = 1.0
    return matrix