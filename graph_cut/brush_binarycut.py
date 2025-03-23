import cv2
import numpy as np
import maxflow
from typing import Tuple, List
import os

class BinaryGraphCut:
    # Constants for seed types
    FOREGROUND = 1
    BACKGROUND = 0
    DEFAULT_VAL = 0.5

    # Constants to know which overlay to display
    SEEDS = 0
    SEGMENTED = 1

    def __init__(self, filename: str):
        self.image = cv2.imread(filename)
        if self.image is None:
            raise ValueError("Unable to load image.")
        self.graph = np.zeros(self.image.shape[:2], dtype=np.float32)
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = None

        self.background_seeds: List[Tuple[int,int]] = []
        self.foreground_seeds: List[Tuple[int,int]] = []
        self.nodes = []
        self.edges = []
        self.current_overlay = self.SEEDS

    def add_seed(self, x: int, y: int, seed_type: int):
        # Use coordinates as given (or adjust them once here if needed)
        # It draws a rectangle of of 4x4 pixels around the seed point
        coord = (x, y)
        if seed_type == self.BACKGROUND and coord not in self.background_seeds:
            self.background_seeds.append(coord)
            cv2.rectangle(self.seed_overlay, (x-2, y-2), (x+2, y+2), (0, 0, 255), -1)
        elif seed_type == self.FOREGROUND and coord not in self.foreground_seeds:
            self.foreground_seeds.append(coord)
            cv2.rectangle(self.seed_overlay, (x-2, y-2), (x+2, y+2), (0, 255, 0), -1)

    def reset_seeds(self):
        self.background_seeds = []
        self.foreground_seeds = []
        self.seed_overlay.fill(0)

    def get_node_index(self, x: int, y: int, width: int) -> int:
        return y * width + x
    
    def pairwise_cost(self, pixel1, pixel2):
        x1,y1 = pixel1
        x2,y2 = pixel2
        # sigma = 10.0
        # diff = np.linalg.norm(self.image[y1, x1].astype(np.float32) - self.image[y2, x2].astype(np.float32))  # Euclidean distance
        # weight = np.exp(-diff**2 / (2 * sigma**2))  # Gaussian function
        # return weight
        return 1.0 / (1 + np.sum((self.image[y1, x1].astype(np.float32) - self.image[y2, x2].astype(np.float32))**2))

    
    def create_graph(self):
        if not self.background_seeds or not self.foreground_seeds:
            print("Please enter at least one foreground and background seed.")
            return
        print("Creating graph")
        # Initialize graph to default value and update seed pixels
        self.graph = np.full(self.image.shape[:2], self.DEFAULT_VAL, dtype=np.float32)
        for x, y in self.background_seeds:
            x = min(max(x, 0), self.image.shape[1]-1)
            y = min(max(y, 0), self.image.shape[0]-1)
            self.graph[y, x] = 0.0
        for x, y in self.foreground_seeds:
            x = min(max(x, 0), self.image.shape[1]-1)
            y = min(max(y, 0), self.image.shape[0]-1)
            self.graph[y, x] = 1.0

        self.nodes = []
        self.edges = []
        height, width = self.graph.shape
        # Build node list
        for (y, x), value in np.ndenumerate(self.graph):
            node_idx = self.get_node_index(x, y, width)
            # background nodes
            if value == 0.0:
                self.nodes.append((node_idx, 1e9, 0))
            # foreground nodes
            elif value == 1.0:
                self.nodes.append((node_idx, 0, 1e9))
            # unsegmented nodes
            else:
                self.nodes.append((node_idx, 0, 0))

        # Build edge list for 4-connected grid
        for y in range(height):
            for x in range(width):
                if x < width - 1:
                    node1 = self.get_node_index(x, y, width)
                    node2 = self.get_node_index(x+1, y, width)
                    self.edges.append((node1, node2, self.pairwise_cost((x, y), (x+1, y))))
                if y < height - 1:
                    node1 = self.get_node_index(x, y, width)
                    node2 = self.get_node_index(x, y+1, width)
                    self.edges.append((node1, node2, self.pairwise_cost((x, y), (x, y+1))))
    
    def cut_graph(self):
        height, width = self.image.shape[:2]
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = np.zeros_like(self.image, dtype=bool)
        g = maxflow.Graph[float](len(self.nodes), len(self.edges))
        node_ids = g.add_nodes(len(self.nodes))
        for node in self.nodes:
            g.add_tedge(node_ids[node[0]], node[1], node[2])
        for edge in self.edges:
            g.add_edge(edge[0], edge[1], edge[2], edge[2])
        g.maxflow()
        for index in range(len(self.nodes)):
            if g.get_segment(index) == 1:
                x, y = index % width, index // width
                self.segment_overlay[y, x] = self.image[y, x] #(255, 0, 255)
                self.mask[y, x] = True

class BrushCut:
    def __init__(self, filename: str):
        if not os.path.exists(filename):
            raise ValueError("Image File not found.")
        
        self.graphcut = BinaryGraphCut(filename)
        self.base_image = self.graphcut.image.copy()
        self.mode = self.graphcut.FOREGROUND
        self.started_click = False

    def run(self):
        window_name = 'Binary Segmentation with graph cut'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.handle_users_seed)
        print("Press 'c' to clear seeds, 'g' to segment, 't' to toggle between label, 'Esc' to exit.")
        while True:
            if self.graphcut.current_overlay == self.graphcut.SEEDS:
                overlay = self.graphcut.seed_overlay
            else:
                overlay = self.graphcut.segment_overlay
            display_image = overlay if self.graphcut.current_overlay == self.graphcut.SEGMENTED else \
                    cv2.addWeighted(self.base_image, 0.9, overlay, 0.6, 0.1)
            cv2.imshow(window_name, display_image)
            # cv2.imshow(window_name, cv2.addWeighted(self.base_image, 0.9, overlay, 0.6, 0.1))
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # Esc to exit
                break

            # switch between foreground and background
            elif key == ord('t'):
                self.mode = 1 - self.mode
                self.graphcut.current_overlay = self.graphcut.SEEDS

            # segment the image
            elif key == ord('g'):
                self.graphcut.create_graph()
                self.graphcut.cut_graph()
                self.graphcut.current_overlay = self.graphcut.SEGMENTED

            # clear seeds
            elif key == ord('c'):
                self.graphcut.current_overlay = self.graphcut.SEEDS
                self.graphcut.reset_seeds()
        cv2.destroyAllWindows()

    def handle_users_seed(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.started_click = True
            self.graphcut.add_seed(x, y, self.mode)
        elif event == cv2.EVENT_LBUTTONUP:
            self.started_click = False
        elif event == cv2.EVENT_MOUSEMOVE and self.started_click:
            self.graphcut.add_seed(x, y, self.mode)

if __name__ == '__main__':
    brushcut = BrushCut("images/flower.jpg")
    brushcut.run()
