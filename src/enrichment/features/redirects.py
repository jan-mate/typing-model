import numpy as np

def get_matrix(finger_map, hand_map):
    matrix = np.zeros((128, 128, 128), dtype=np.float32)
    valid = (finger_map >= 0) & (hand_map >= 0)
    
    PP, P, C = np.ix_(np.arange(128), np.arange(128), np.arange(128))
    
    same_hand = (hand_map[PP] == hand_map[P]) & (hand_map[P] == hand_map[C]) & (hand_map[P] < 2)
    valid_mask = valid[PP] & valid[P] & valid[C] & same_hand
    
    f1 = finger_map[PP]
    f2 = finger_map[P]
    f3 = finger_map[C]
    
    dir1 = f2 - f1
    dir2 = f3 - f2
    
    redirect = ((dir1 > 0) & (dir2 < 0)) | ((dir1 < 0) & (dir2 > 0))
    
    condition = valid_mask & redirect
    matrix[condition] = 1.0
    return matrix