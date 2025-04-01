## This version only work for Potts Model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
from sklearn.cluster import KMeans
import maxflow


from graph_cut.display import show_segmentation
from graph_cut.base_algorithm import Runner


def unary_recycle(unary, label, K):
    """Constructs a new unary term for the recycle algorithm."""
    W, H, k = unary.shape
    unary_term = np.zeros((W, H, 2))

    for i in range(unary.shape[0]):
        for j in range(unary.shape[1]):
            partition = np.partition(unary[i, j, :], 1)  # Get two smallest values
            min_unary = (
                partition[1] if unary[i, j, label] == partition[0] else partition[0]
            )
            unary_term[i, j, 0] = unary[i, j, label]
            unary_term[i, j, 1] = min_unary

    return unary_term


def binary_recycle(pairwise_term, K):
    # Construct the new binary term used for the recycle alogorithm for label lm
    pairwise_term_recycled = np.zeros(
        (pairwise_term.shape[0], pairwise_term.shape[1], 2, 2)
    )
    for i in range(pairwise_term.shape[0]):
        for j in range(pairwise_term.shape[1]):
            gamma = pairwise_term[i, j, 0, 1]
            pairwise_term_recycled[i, j, 0, 1] = pairwise_term_recycled[i, j, 1, 0] = (
                gamma
            )

    return pairwise_term_recycled


class Recycle(Runner):
    # to remove the self.assigned labels , just remove the if condition in the construct graph
    def __init__(self, image, unary, pairwise, K):
        super(Recycle, self).__init__(image, unary, pairwise, K)
        self.cst = 0  #! Cstant term in the energy

    def construct_graph(self, assigned_labels):
        graph = maxflow.Graph[float]()
        h = self.h
        w = self.w
        nodes = graph.add_nodes(h * w)

        pairwise = self.pairwise
        unary = self.unary

        for i in range(h):
            for j in range(w):
                # Add unary terms
                pixel_index = i * w + j
                if assigned_labels[i, j] == self.epsilon:
                    graph.add_tedge(
                        nodes[pixel_index], unary[i, j, 0], unary[i, j, 1]
                    )  # Keep alpha pixels fixed

                # Add pairwise terms
                if i < h - 1:
                    neighbor_index_down = pixel_index + w

                    weight_down = pairwise[i, j, 0, 1]  # ?

                    if (
                        assigned_labels[i, j] == self.epsilon
                        and assigned_labels[i + 1, j] == self.epsilon
                    ):
                        graph.add_edge(
                            nodes[pixel_index],
                            nodes[neighbor_index_down],
                            weight_down,
                            weight_down,
                        )

                if j < w - 1:
                    neighbor_index_right = pixel_index + 1
                    weight_right = pairwise[i, j, 0, 1]
                    if (
                        assigned_labels[i, j] == self.epsilon
                        and assigned_labels[i, j + 1] == self.epsilon
                    ):
                        graph.add_edge(
                            nodes[pixel_index],
                            nodes[neighbor_index_right],
                            weight_right,
                            weight_right,
                        )
        return graph, nodes

    def run(self, image, init_labels=None, assigned_labels=None):
        # init_labels is the initial labels for the image

        h, w, _ = image.shape
        print("---start Projection---")
        # Unary/pairwise of the true graph
        unary = self.unary.copy()
        pairwise = self.pairwise
        K = self.K
        if init_labels is not None:
            print("you provided the initial labels")
            labels = init_labels
        else:
            labels = np.ones((h, w), dtype=np.int32) * self.epsilon
        if assigned_labels is not None:
            print("you provided the assigned labels")
            self.assigned_labels = assigned_labels

        # project to int
        labels = labels.astype(np.int32)
        show_segmentation(
            image, labels, title="initialization of the Recycle algorithm"
        )
        assigned_labels = (
            self.assigned_labels
        )  # Will be True iif the pixel is assigned to a label and will not move
        for alpha in range(K):
            print("alpha", alpha)

            unary_term_recycled = unary_recycle(unary, alpha, K)
            pairwise_term_recycled = binary_recycle(pairwise, K)
            self.unary = unary_term_recycled
            self.pairwise = pairwise_term_recycled
            # Construct graph
            graph, nodes = self.construct_graph(assigned_labels=assigned_labels)
            # Compute min-cut
            graph.maxflow()

            # Update labels and get the array of assigned labels
            nv_labels, assigned_labels = self.update_labels(
                graph, nodes, alpha, labels, assigned_labels=assigned_labels
            )
            # Convert the unary and pairwise term Based on the assignment
            unary, pairwise = self.project(nv_labels, assigned_labels, unary, pairwise)
            labels = nv_labels

            show_segmentation(image, labels, K, title=f" Projection: alpha {alpha}")

        return labels, assigned_labels, self.cst, unary, pairwise

    def project(self, labels, assigned_labels, unary_term, pairwise_term):
        return super(Recycle, self).project(
            labels, assigned_labels, unary_term, pairwise_term
        )

    def update_labels(self, graph, nodes, alpha, labels, assigned_labels):
        h, w = self.h, self.w
        nv_labels = labels.copy()
        for i in range(h):
            for j in range(w):
                pixel_index = i * w + j
                if graph.get_segment(nodes[pixel_index]) == 1:  #!
                    nv_labels[i, j] = alpha
                    if assigned_labels[i, j] == self.epsilon:
                        assigned_labels[i, j] = 1
                    else:
                        print(
                            "we have already assigned this pixel to a label, is it normal?",
                            assigned_labels[i, j],
                            i,
                            j,
                        )
        return nv_labels, assigned_labels
