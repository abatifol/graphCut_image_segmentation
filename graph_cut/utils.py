
# Load Image
import cv2
from sklearn.cluster import KMeans

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return image

# Initialize Labels using K-Means
def initialize_labels(image, K=3):
    h, w, c = image.shape
    pixels = image.reshape((-1, 3))  # Flatten image to (num_pixels, 3)
    
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)  # Assign labels to pixels
    
    return labels.reshape((h, w))  # Reshape back to image dimensions

def initialize_labels_bis(image, K, method='kmeans'):
    """
    Initialize the segmentation labels.
    
    Args:
        image: Input image (h, w, channels)
        K: Number of labels
        method: 'kmeans' or 'random'
    
    Returns:
        Initial labels (h, w)
    """
    h, w, c = image.shape
    pixels = image.reshape(-1, c)
    
    if method == 'random':
        return np.random.randint(0, K, (h, w))
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=K, random_state=0).fit(pixels)
        return kmeans.labels_.reshape(h, w)
    else:
        raise ValueError(f"Unknown initialization method: {method}")


import numpy as np

def get_dominant_colors(labels, image, K):
    dominant_colors = np.zeros((K, 3))
    for k in range(K):
        mask = (labels == k)
        dominant_colors[k] = np.mean(image[mask], axis=0)
    return dominant_colors