import numpy as np

def get_matrix():
    matrix = np.zeros((128, 128), dtype=np.float32)
    P, C = np.ix_(np.arange(128), np.arange(128))
    
    valid = (P > 0) & (C > 0)
    same_key = P == C
    
    condition = valid & same_key
    matrix[condition] = 1.0
    return matrix