import numpy as np

def get_matrix(finger_map, hand_map):
    matrix = np.full((128, 128), 0.0, dtype=np.float32)
    for i in range(128):
        for j in range(128):
            f1, f2 = finger_map[i], finger_map[j]
            h1, h2 = hand_map[i], hand_map[j]
            
            if h1 == -1 or h1 == 2 or h1 != h2 or f1 == -1 or f2 == -1 or f1 == f2:
                continue
            
            if h1 == 0:
                if f2 < f1:
                    matrix[i, j] = 1.0
            elif h1 == 1:
                if f2 > f1:
                    matrix[i, j] = 1.0
    return matrix