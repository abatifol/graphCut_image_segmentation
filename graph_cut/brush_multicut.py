import cv2
import numpy as np
import maxflow
from typing import Tuple, List, Dict
import os


class MultiLabelGraphCut:
    DEFAULT_VAL = -1
    SEEDS = 0
    SEGMENTED = 1

    def __init__(self, filename: str, num_labels=3):
        self.image = cv2.imread(filename)
        if self.image is None:
            raise ValueError("Unable to load image.")

        self.graph = np.full(self.image.shape[:2], self.DEFAULT_VAL, dtype=np.int32)
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = np.full(self.image.shape[:2], -1, dtype=np.int32)

        self.labels = list(range(1, num_labels + 1))
        self.seed_points: Dict[int, List[Tuple[int, int]]] = {
            label: [] for label in self.labels
        }
        self.current_overlay = self.SEEDS
        self.current_label = 1

        self.label_colors = {
            1: (0, 255, 0),
            2: (0, 0, 255),
            3: (255, 0, 0),
            4: (255, 255, 0),
            5: (0, 255, 255),
        }

    def add_seed(self, x: int, y: int, label: int):
        if label not in self.labels:
            return

        coord = (x, y)
        if coord not in self.seed_points[label]:
            self.seed_points[label].append(coord)
            color = self.label_colors.get(label, (255, 255, 255))
            cv2.rectangle(self.seed_overlay, (x - 2, y - 2), (x + 2, y + 2), color, -1)
            self.mask[y, x] = label

    def reset_seeds(self):
        self.seed_points = {label: [] for label in self.labels}
        self.seed_overlay.fill(0)
        self.mask.fill(-1)

    def get_node_index(self, x: int, y: int, width: int) -> int:
        return y * width + x

    # def pairwise_cost(self, pixel1, pixel2):
    #     """ Compute a smoothness constraint based on color similarity """
    #     x1, y1 = pixel1
    #     x2, y2 = pixel2
    #     color_diff = np.linalg.norm(self.image[y1, x1] - self.image[y2, x2])
    #     return np.exp(-color_diff / 10.0)  # Reduce smoothness penalty for similar colors

    def unary_cost(self, x: int, y: int, label: int):
        """Unary term based on color similarity to label seeds"""
        if not self.seed_points[label]:
            # No seeds for this label, assign based on closest mean seed color
            mean_colors = {
                lbl: self.get_mean_seed_color(lbl)
                for lbl in self.labels
                if self.seed_points[lbl]
            }
            if not mean_colors:  # No labels have seeds, return uniform cost
                return 1000

            # Assign pixel to the closest mean color
            pixel_color = self.image[y, x]
            closest_label = min(
                mean_colors,
                key=lambda lbl: np.linalg.norm(pixel_color - mean_colors[lbl]),
            )
            return np.linalg.norm(pixel_color - mean_colors[closest_label]) ** 2 / 100

        # Regular unary cost
        color_dist = np.linalg.norm(
            self.image[y, x]
            - np.mean(
                [self.image[py, px] for px, py in self.seed_points[label]], axis=0
            )
        )
        return min(1e6, (color_dist**2) / 100)

    def pairwise_cost(self, pixel1, pixel2):
        """Compute a smoothness constraint based on color similarity"""
        x1, y1 = pixel1
        x2, y2 = pixel2
        return 1.0 / (
            1
            + np.sum(
                (
                    self.image[y1, x1].astype(np.float32)
                    - self.image[y2, x2].astype(np.float32)
                )
                ** 2
            )
        )

        # works worse
        # return np.exp(-np.linalg.norm(self.image[y1, x1] - self.image[y2, x2]) / 50.0)

    def alpha_expansion(self):
        """Perform multi-label segmentation using alpha expansion"""
        print("Performing Alpha Expansion...")
        height, width = self.image.shape[:2]

        for alpha in self.labels:
            g = maxflow.Graph[float]()
            node_ids = g.add_nodes(height * width)

            # Add terminal edges (data term)
            for i in range(height):
                for j in range(width):
                    pixel_index = i * width + j
                    if self.mask[i, j] == alpha:
                        g.add_tedge(node_ids[pixel_index], 1e9, 0)  # Hard constraint
                    elif self.mask[i, j] > 0:
                        g.add_tedge(node_ids[pixel_index], 0, 1e9)

                    # Add pairwise edges (smoothness term)
                    if i < height - 1:
                        neighbor_index_down = pixel_index + width
                        weight_down = self.pairwise_cost((j, i), (j, i + 1))
                        g.add_edge(
                            node_ids[pixel_index],
                            node_ids[neighbor_index_down],
                            weight_down,
                            weight_down,
                        )

                    if j < width - 1:
                        neighbor_index_right = pixel_index + 1
                        weight = self.pairwise_cost((j, i), (j + 1, i))
                        g.add_edge(
                            node_ids[pixel_index],
                            node_ids[neighbor_index_right],
                            weight,
                            weight,
                        )

                    # weight_x = self.pairwise_cost((x, y), (x + 1, y))
                    # weight_y = self.pairwise_cost((x, y), (x, y + 1))
                    # g.add_edge(node_ids[y, x], node_ids[y, x + 1], weight_x, weight_x)
                    # g.add_edge(node_ids[y, x], node_ids[y + 1, x], weight_y, weight_y)

            g.maxflow()

            for i in range(height):
                for j in range(width):
                    if (
                        g.get_segment(node_ids[i * width + j]) == 0
                    ):  # Belongs to 'alpha'
                        self.mask[i, j] = alpha

        # Apply segmentation result
        self.segment_overlay.fill(0)
        for y in range(height):
            for x in range(width):
                if self.mask[y, x] > 0:
                    self.segment_overlay[y, x] = self.label_colors.get(
                        self.mask[y, x], (255, 255, 255)
                    )


class BrushCut:
    def __init__(self, filename: str, num_labels=3):
        if not os.path.exists(filename):
            raise ValueError("Image File not found.")

        self.graphcut = MultiLabelGraphCut(filename, num_labels)
        self.base_image = self.graphcut.image.copy()
        self.mode = 1  # Default to first label
        self.started_click = False

    def run(self):
        window_name = "Multi-Label Segmentation with Graph Cut"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.handle_users_seed)

        print(
            "Press 'c' to clear seeds, 'g' to segment, '1-5' to change label, 'Esc' to exit."
        )

        while True:
            if self.graphcut.current_overlay == self.graphcut.SEEDS:
                overlay = self.graphcut.seed_overlay
            else:
                overlay = self.graphcut.segment_overlay

            display = cv2.addWeighted(self.base_image, 0.7, overlay, 0.5, 0.1)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # Esc key
                break

            elif ord("1") <= key <= ord("5"):
                self.mode = int(chr(key))
                print(f"Switched to label {self.mode}")

            elif key == ord("g"):
                self.graphcut.alpha_expansion()
                self.graphcut.current_overlay = self.graphcut.SEGMENTED

            elif key == ord("c"):
                self.graphcut.reset_seeds()
                self.graphcut.current_overlay = self.graphcut.SEEDS

        cv2.destroyAllWindows()

    def handle_users_seed(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.started_click = True
            self.graphcut.add_seed(x, y, self.mode)

        elif event == cv2.EVENT_LBUTTONUP:
            self.started_click = False

        elif event == cv2.EVENT_MOUSEMOVE and self.started_click:
            self.graphcut.add_seed(x, y, self.mode)


if __name__ == "__main__":
    brushcut = BrushCut("images/cow.ppm", num_labels=4)
    brushcut.run()
