from graph_cut.energy import compute_energy
import numpy as np


class Runner:
    def __init__(self, image, unary, pairwise, K):
        """Base class of Alpha expansion and Alpha recycle"""
        self.unary = unary
        self.pairwise = pairwise

        # Hyperparameters
        self.K = K
        self.h, self.w, _ = image.shape
        # related to the energy computation
        self.cst = 0
        self.l_energy = []

        # for the assignment of labels
        self.epsilon = -1
        self.assigned_labels = np.ones((self.h, self.w)) * self.epsilon

    def compute_energy(self, labels, unary, pairwise):
        # Compute the energy of the current labeling
        energy = compute_energy(labels, unary, pairwise, self.cst)
        # self.l_energy.append(energy)
        return energy

    def construct_graph(self, alpha):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def run(self, img):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def update_labels(self, graph, nodes, alpha, labels):
        h, w = self.h, self.w
        nv_labels = labels.copy()
        for i in range(h):
            for j in range(w):
                pixel_index = i * w + j
                if graph.get_segment(nodes[pixel_index]) == 1:
                    nv_labels[i, j] = alpha
        return nv_labels

    def project(self, labels, assigned_labels, unary_term, pairwise_term):
        #!  May be used for the graph cut
        # Project the function Using the initial code.
        # Slide the pairwise term and update the corresponding pairwise term
        for i in range(self.h - 1):
            for j in range(self.w - 1):
                idx1 = []
                idx2 = []
                if i < self.h - 1:
                    idx1.append((i, j))
                    idx2.append((i + 1, j))
                if j < self.w - 1:
                    idx1.append((i, j))
                    idx2.append((i, j + 1))
                for l in range(len(idx1)):
                    node = idx1[l]
                    neighbor = idx2[l]
                    # print("node",node,"neighbor",neighbor)

                    if (
                        assigned_labels[node] == self.epsilon
                        and assigned_labels[neighbor] != self.epsilon
                    ):
                        for k in range(self.K):
                            unary_term[node[0], node[1], k] += pairwise_term[
                                node[0], node[1], k, labels[neighbor[0], neighbor[1]]
                            ]
                    elif (
                        assigned_labels[node[0], node[1]] != self.epsilon
                        and assigned_labels[neighbor[0], neighbor[1]] == self.epsilon
                    ):
                        for k in range(self.K):
                            unary_term[neighbor[0], neighbor[1], k] += pairwise_term[
                                node[0], node[1], labels[node[0], node[1]], k
                            ]
                    elif (
                        assigned_labels[node[0], node[1]] != self.epsilon
                        and assigned_labels[neighbor[0], neighbor[1]] != self.epsilon
                    ):
                        self.cst += pairwise_term[
                            node[0],
                            node[1],
                            labels[node[0], node[1]],
                            labels[neighbor[0], neighbor[1]],
                        ]

        # Update the pairwise term
        for i in range(self.h):
            for j in range(self.w):
                if assigned_labels[i, j] != self.epsilon:
                    self.cst += unary_term[i, j, labels[i, j]]
        return unary_term, pairwise_term
