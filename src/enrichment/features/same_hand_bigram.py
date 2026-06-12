import numpy as np

def get_matrix(hand_map):
    matrix = np.full((128, 128), 0.0, dtype=np.float32)
    for i in range(128):
        for j in range(128):
            if not np.isnan(hand_map[i]) and not np.isnan(hand_map[j]):
                if hand_map[i] == hand_map[j] and hand_map[i] not in (-1, 2):
                    matrix[i, j] = 1.0
    return matrix