import numpy as np

def get_matrix(finger_map, hand_map, coords_map):
    matrix = np.zeros((128, 128), dtype=np.float32)
    y_coords = coords_map[:, 1]
    
    valid = (finger_map >= 0) & (hand_map >= 0) & ~np.isnan(y_coords)
    
    P, C = np.ix_(np.arange(128), np.arange(128))
    
    valid_mask = valid[P] & valid[C]
    same_hand = hand_map[P] == hand_map[C]
    diff_key = P != C
    
    row_jump = np.abs(y_coords[P] - y_coords[C]) >= 2
    
    ft = np.where(finger_map >= 5, 9 - finger_map, finger_map)
    ft_P = ft[P]
    ft_C = ft[C]
    
    adjacent = np.abs(ft_P - ft_C) == 1
    no_index_thumb = (ft_P < 3) & (ft_C < 3)
    
    condition = valid_mask & same_hand & diff_key & row_jump & adjacent & no_index_thumb
    matrix[condition] = 1.0
    return matrix