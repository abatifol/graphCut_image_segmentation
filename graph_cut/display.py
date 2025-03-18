
import numpy as np
import matplotlib.pyplot as plt
# Display Initial Segmentation with Unique Colors per Label
def show_segmentation(image, labels, K=3,title=""):
    h, w = labels.shape
    segmented_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Assign a unique random color to each label
    np.random.seed(42)  # For consistent colors
    colors = np.random.randint(0, 255, size=(K, 3), dtype=np.uint8)

    for k in range(K):
        segmented_image[labels == k] = colors[k]

    # Display images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("Segmented Image")
    plt.axis("off")
    # plt.suptitle(title)
    plt.suptitle(title)
    plt.show()

from graph_cut.utils import get_dominant_colors
def show_dominant_colors(labels,image,K=3,title="Dominant Colors"):
    dominant_colors=get_dominant_colors(labels,image,K)
    K=dominant_colors.shape[0]
    # plt.subplot(1,K,2)
    plt.imshow(dominant_colors.astype(np.uint8).reshape(1, K, 3))
    # add in the x labels the corresponding integers
    plt.xticks(np.arange(K), np.arange(K))
    plt.title(title)
    plt.show()