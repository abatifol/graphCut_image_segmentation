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


def compute_unary_term_L2(image,labels_color):
    h, w, _ = image.shape
    K = labels_color.shape[0]
    unary = np.zeros((h, w, K))

    for k in range(K):
        unary[:, :, k] = np.linalg.norm(image - labels_color[k],axis=2)

    return unary
import numpy as np
import cv2
def unary_term_L2_sCIELAB(image, labels_colors):
    """
    Computes the L2 norm unary term in the CIELAB color space.
    
    Parameters:
    - image: Input image in sRGB format (H, W, 3), dtype=np.uint8.
    - labels_colors: Array of label colors in sRGB (N, 3), dtype=np.uint8.
    
    Returns:
    - Unary term: L2 distance matrix (H, W, N), dtype=np.float32.
    """
    # Convert image and label colors to CIELAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    labels_colors_lab = cv2.cvtColor(np.uint8(labels_colors[None, :, :]), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    
    # Compute L2 norm
    unary_term = np.linalg.norm(image_lab[:, :, None, :] - labels_colors_lab[None, None, :, :], axis=-1)
    
    return unary_term