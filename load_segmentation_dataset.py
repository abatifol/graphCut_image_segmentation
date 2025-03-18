import fiftyone.zoo as foz
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from typing import List

from typing import Union, Tuple, List, Dict


def resize_mask_and_img(mask, img, scale_factor=0.5):
    w, h, _ = img.shape
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return resized_mask, resized_img


class SegmentationDataset:
    def __init__(
        self,
        classes: List[str] = ["sheep", "cow", "elephant"],
        nb_samples: int = 25,
        resize_factor=0.5,
    ):
        self.classes = classes
        self.dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            classes=classes,
            label_types=["segmentations"],
            max_samples=nb_samples,
        )
        self.images = []
        self.masks = []
        self.labels_in_img = []
        self.LABEL_COLORS = {}
        self.resize_factor = resize_factor

        self.prepare_dataset()

    def prepare_dataset(self):
        for sample in self.dataset.iter_samples():
            # Load image and convert to RGB

            img = cv2.cvtColor(cv2.imread(sample.filepath), cv2.COLOR_BGR2RGB)

            # Extract labels and annotations and define unique colors if it does not exist already
            labels = sorted(
                set([d["label"] for d in sample.ground_truth["detections"]])
            )
            annotations = sample.ground_truth["detections"]
            for label in labels:
                if label not in self.LABEL_COLORS:
                    self.LABEL_COLORS[label] = np.random.randint(
                        0, 255, (3,), dtype=np.uint8
                    )

            # Initialize blank mask
            h, w, _ = img.shape
            mask = np.zeros((h, w, 3), dtype=np.uint8)

            for annotation in annotations:
                # Get bounding box coordinates
                x_1, y_1, bb_width, bb_height = annotation["bounding_box"]

                # Rescale the coordinates relatively to image scale
                x_1 = floor(x_1 * w)
                y_1 = floor(y_1 * h)
                x_2 = x_1 + floor(bb_width * w)
                y_2 = y_1 + floor(bb_height * h)

                # Get segmentation mask and assign color
                annotation_mask = np.array(annotation["mask"], dtype=np.uint8).T
                annotation_mask = np.moveaxis(
                    annotation_mask[:, :, np.newaxis].repeat(3, axis=2), 0, 1
                )
                mask_color = self.LABEL_COLORS[annotation["label"]]
                colored_mask = annotation_mask * mask_color

                # Ensure we update only non-zero regions of the mask with the new annotation to avoid displaying bounding box
                mask_region = mask[y_1:y_2, x_1:x_2, :]
                mask_nonzero = np.any(annotation_mask, axis=-1)

                # Resize colored_mask and mask_nonzero to match the region size
                mask_region_height, mask_region_width = mask_region.shape[:2]

                # Resize colored_mask to match mask_region
                colored_mask_resized = cv2.resize(
                    colored_mask,
                    (mask_region_width, mask_region_height),
                    interpolation=cv2.INTER_NEAREST,
                )

                # Resize mask_nonzero to match mask_region's height and width
                mask_nonzero_resized = cv2.resize(
                    mask_nonzero.astype(np.uint8),
                    (mask_region_width, mask_region_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

                mask_region[mask_nonzero_resized] = colored_mask_resized[
                    mask_nonzero_resized
                ]

            mask[y_1:y_2, x_1:x_2, :] = mask_region

            # resize the image
            mask, img = resize_mask_and_img(mask, img, scale_factor=self.resize_factor)

            self.images.append(img)
            self.masks.append(mask)
            self.labels_in_img.append(labels)

    def __len__(self):
        return len(self.dataset)

    def get_segmentation_mask(
        self, image: np.ndarray, annotations: List[Dict], n_labels: int, COLORS
    ) -> np.ndarray:
        """

        :param image: np.ndarray (current image)
        :param annotations: dictionary of annotations (from COCO dataset)
        It namely contains the segmentation mask and the bounding boxes
        to locate the mask on the full image
        :param n_labels: number of labels in the image
        :return: Segmentation mask
        """
        height, width, _ = image.shape
        mask = np.zeros((height, width, 3))
        for annotation in annotations:
            bounding_box = annotation["bounding_box"]
            x_1, y_1, bb_width, bb_height = bounding_box
            x_1 = floor(x_1 * width)
            y_1 = floor(y_1 * height)

            current_mask = np.array(annotation["mask"], dtype=np.int32).T
            current_mask = current_mask[:, :, np.newaxis].repeat(3, 2)
            x_2 = current_mask.shape[0] + x_1
            y_2 = current_mask.shape[1] + y_1
            if n_labels == 1:
                mask_color = COLORS[0]
            else:
                mask_color = COLORS[self.segmentation_labels[annotation["label"]]]
            current_mask = current_mask * mask_color
            mask[y_1:y_2, x_1:x_2, :] = np.swapaxes(current_mask, 0, 1)
        return mask

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx], self.labels_in_img[idx]

    def display_sample(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        labels = self.labels_in_img[idx]

        # **Blend the original image with the mask**
        overlayed_image = cv2.addWeighted(img, 1, mask, 1, 0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(mask)
        axes[1].set_title("Mask")
        axes[1].axis("off")
        axes[2].imshow(overlayed_image)
        axes[2].set_title("Overlayed Image")
        axes[2].axis("off")

        # # Adding legend with label colors
        # handles = []
        # labels_legend = []
        # for label in labels:
        #     color = self.LABEL_COLORS[label]
        #     handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
        #     handles.append(handle)
        #     labels_legend.append(label)

        # # Add the legend
        # fig.legend(handles, labels_legend, loc="center", bbox_to_anchor=(0.5, 0.02), ncol=3)

        plt.show()
