import numpy as np

def get_matrix():
    matrix = np.zeros((128, 128, 128), dtype=np.float32)
    PP, P, C = np.ix_(np.arange(128), np.arange(128), np.arange(128))
    
    valid = (PP > 0) & (P > 0) & (C > 0)
    
    same_key_ends = PP == C
    diff_key_mid = PP != P
    
    condition = valid & same_key_ends & diff_key_mid
    matrix[condition] = 1.0
    return matrix