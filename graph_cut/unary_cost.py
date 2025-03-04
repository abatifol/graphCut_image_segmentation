import numpy as np 
# Compute Color Histograms for Each Label
def compute_histograms(image, labels, K, bins=16):
    histograms = []
    for k in range(K):
        mask = (labels == k)
        pixels = image[mask]
        hist = np.histogramdd(pixels, bins=bins, range=[(0, 256), (0, 256), (0, 256)])[0]
        histograms.append(hist / np.sum(hist))  # Normalize histogram
    return histograms
# Compute Unary Term Using Histograms
def compute_unary_term(image, labels, histograms, K, bins=16):
    h, w, c = image.shape
    unary = np.zeros((h, w, K))
    
    for k in range(K):
        hist = histograms[k]
        for i in range(h):
            for j in range(w):
                pixel = image[i, j]
                bin_idx = np.floor(pixel / bins).astype(int)  # Map pixel to histogram bin
                unary[i, j, k] = -np.log(hist[bin_idx[0], bin_idx[1], bin_idx[2]] + 1e-8)  # Avoid log(0)
    
    return unary