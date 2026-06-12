import numpy as np

def get_matrix(finger_map):
    matrix = np.zeros((128, 128, 128), dtype=np.float32)
    for i in range(128):
        f = finger_map[i]
        if f == -1 or np.isnan(f):
            continue
        same_indices = np.where(finger_map == f)[0]
        for j in same_indices:
            for k in same_indices:
                matrix[i, j, k] = 1.0
    return matrix