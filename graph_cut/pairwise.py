

import numpy as np


# Compute Pairwise Term Using Potts Model
def compute_pairwise_term(image, K, lambda_val=1.0):
    h, w, _ = image.shape
    pairwise = np.zeros((h, w, K, K))
    
    for i in range(h - 1):
        for j in range(w - 1):
            for k1 in range(K):
                for k2 in range(K):
                    pairwise[i, j, k1, k2] = lambda_val * (k1 != k2)  # Potts model penalty
    
    return pairwise