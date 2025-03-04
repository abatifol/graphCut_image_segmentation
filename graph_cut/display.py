
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

