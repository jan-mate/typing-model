import numpy as np

def get_matrix(coords_map, hand_map):
    matrix = np.zeros((128, 128), dtype=np.float32)
    
    y_coords = coords_map[:, 1]
    
    valid = (hand_map >= 0) & ~np.isnan(y_coords)
    
    P, C = np.ix_(np.arange(128), np.arange(128))
    
    valid_mask = valid[P] & valid[C]
    same_hand = hand_map[P] == hand_map[C]
    
    y1 = y_coords[P]
    y2 = y_coords[C]
    
    jump_up = (y1 == -1) & ((y2 == 1) | (y2 == 2))
    jump_down = ((y1 == 1) | (y1 == 2)) & (y2 == -1)
    
    condition = valid_mask & same_hand & (jump_up | jump_down)
    matrix[condition] = 1.0
    return matrix