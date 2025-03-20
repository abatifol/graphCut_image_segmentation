from graph_cut.display import show_segmentation
import numpy as np
from graph_cut.energy import compute_energy
import maxflow


def alpha_expansion1(
    image,
    unary,
    pairwise,
    K,
    method="kmeans",
    max_iterations=20,
    return_energy: bool = False,
):
    l_energy = []
    h, w, _ = image.shape
    # labels = np.argmin(unary, axis=2)  # Initialize labels using unary term
    # labels = initialize_labels_bis(image,method=method, K=K)
    labels = np.argmin(unary, axis=2)
    show_segmentation(image, labels, title="initialization")
    energy = compute_energy(labels, unary, pairwise)
    l_energy.append(energy)
    print("first energy", compute_energy(labels, unary, pairwise))
    for _ in range(max_iterations):
        print(f"iterations nb: {_}")
        for alpha in range(K):
            print(f"alpha: {alpha}")
            graph = maxflow.Graph[float]()
            nodes = graph.add_nodes(h * w)

            for i in range(h):
                for j in range(w):
                    # Add unary terms
                    pixel_index = i * w + j
                    if labels[i, j] == alpha:
                        graph.add_tedge(
                            nodes[pixel_index], unary[i, j, alpha], np.inf
                        )  # Keep alpha pixels fixed
                    else:
                        graph.add_tedge(
                            nodes[pixel_index],
                            unary[i, j, alpha],
                            unary[i, j, labels[i, j]],
                        )

                    # Add pairwise terms
                    if i < h - 1:
                        neighbor_index_down = pixel_index + w

                        weight_down = pairwise[i, j, labels[i, j], alpha]

                        if labels[i, j] != labels[i + 1, j]:
                            aux_node = graph.add_nodes(1)
                            graph.add_edge(
                                nodes[pixel_index], aux_node, weight_down, weight_down
                            )
                            graph.add_edge(
                                nodes[neighbor_index_down],
                                aux_node,
                                pairwise[i + 1, j, labels[i + 1, j], alpha],
                                pairwise[i + 1, j, labels[i + 1, j], alpha],
                            )

                            graph.add_tedge(aux_node, 0, weight_down)

                        else:
                            graph.add_edge(
                                nodes[pixel_index],
                                nodes[neighbor_index_down],
                                weight_down,
                                weight_down,
                            )

                    if j < w - 1:
                        neighbor_index_right = pixel_index + 1
                        weight_right = pairwise[i, j, labels[i, j], alpha]
                        if labels[i, j] != labels[i, j + 1]:
                            aux_node = graph.add_nodes(1)
                            graph.add_edge(
                                nodes[pixel_index], aux_node, weight_right, weight_right
                            )
                            graph.add_edge(
                                nodes[neighbor_index_right],
                                aux_node,
                                pairwise[i, j + 1, labels[i, j + 1], alpha],
                                pairwise[i, j + 1, labels[i, j + 1], alpha],
                            )
                            graph.add_tedge(aux_node, 0, weight_right)
                        else:
                            graph.add_edge(
                                nodes[pixel_index],
                                nodes[neighbor_index_right],
                                weight_right,
                                weight_right,
                            )

            # Compute min-cut
            graph.maxflow()

            # Update labels

            nv_labels = labels.copy()
            for i in range(h):
                for j in range(w):
                    pixel_index = i * w + j
                    if graph.get_segment(nodes[pixel_index]) == 1:
                        nv_labels[i, j] = alpha  # Expand α-region
            nv_energy = compute_energy(nv_labels, unary, pairwise)
            print(
                "computed energy", nv_energy, "will we swap labels?", nv_energy < energy
            )
            if nv_energy < energy:
                labels = nv_labels
                energy = nv_energy
                l_energy.append(energy)

            show_segmentation(image, labels, K, title=f"iteration {_} alpha {alpha}")
            print("energy", compute_energy(labels, unary, pairwise))
        if _ % 5 == 0:
            show_segmentation(image, labels, K)

    if return_energy:
        return labels, l_energy
    return labels


import numpy as np
from graph_cut.base_algorithm import Runner

class Alpha_expansion2(Runner):
    def __init__(self, image, unary, pairwise, K, max_iterations):
        super().__init__(image, unary, pairwise, K)

        self.max_iterations = max_iterations
        # labels = np.argmin(unary, axis=2)  # Initialize labels using unary term
        # labels = initialize_labels_bis(image,method=method, K=K)

    def construct_graph(self, alpha, labels):
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
                if self.assigned_labels[i,j]!=self.epsilon: # if the pixel is already assigned
                    continue 
                if labels[i, j] == alpha:
                    graph.add_tedge(
                        nodes[pixel_index], unary[i, j, alpha], np.inf
                    )  # Keep alpha pixels fixed
                else:
                    graph.add_tedge(
                        nodes[pixel_index],
                        unary[i, j, alpha],
                        unary[i, j, labels[i, j]],
                    )

                # Add pairwise terms
                if i < h - 1:
                    neighbor_index_down = pixel_index + w
                    if self.assigned_labels[i,j]==self.epsilon and self.assigned_labels[i+1,j]==self.epsilon: # if the pixel is already assigned
                        weight_down = pairwise[i, j, labels[i, j], alpha]

                        if labels[i, j] != labels[i + 1, j]:
                            aux_node = graph.add_nodes(1)
                            graph.add_edge(
                                nodes[pixel_index], aux_node, weight_down, weight_down
                            )
                            graph.add_edge(
                                nodes[neighbor_index_down],
                                aux_node,
                                pairwise[i + 1, j, labels[i + 1, j], alpha],
                                pairwise[i + 1, j, labels[i + 1, j], alpha],
                            )

                            graph.add_tedge(aux_node, 0, pairwise[i,j,labels[i,j],labels[i+1,j]])

                        else:
                            graph.add_edge(
                                nodes[pixel_index],
                                nodes[neighbor_index_down],
                                weight_down,
                                weight_down,
                            )

                if j < w - 1:
                    if self.assigned_labels[i,j]==self.epsilon and self.assigned_labels[i,j+1]==self.epsilon: # if the pixel is already assigned
                        neighbor_index_right = pixel_index + 1
                        weight_right = pairwise[i, j, labels[i, j], alpha]
                        if labels[i, j] != labels[i, j + 1]:
                            aux_node = graph.add_nodes(1)
                            graph.add_edge(
                                nodes[pixel_index], aux_node, weight_right, weight_right
                            )
                            graph.add_edge(
                                nodes[neighbor_index_right],
                                aux_node,
                                pairwise[i, j + 1, labels[i, j + 1], alpha],
                                pairwise[i, j + 1, labels[i, j + 1], alpha],
                            )
                            graph.add_tedge(aux_node, 0, pairwise[i,j,labels[i,j],labels[i,j+1]])
                        else:
                            graph.add_edge(
                                nodes[pixel_index],
                                nodes[neighbor_index_right],
                                weight_right,
                                weight_right,
                            )
        return graph, nodes

    def run(self, image,assigned_labels=None,init_labels=None):
        if assigned_labels is not None:
            self.assigned_labels = assigned_labels
        h, w, _ = image.shape
        unary = self.unary
        pairwise = self.pairwise
        l_energy = self.l_energy
        K = self.K
        max_iterations = self.max_iterations
        if init_labels is not None:
            labels= init_labels
        else:
            labels = np.argmin(unary, axis=2)
        show_segmentation(image, labels, title="initialization of the Alpha expansion")
        energy = self.compute_energy(labels, unary, pairwise)
        l_energy.append(energy)

        print("first energy", self.compute_energy(labels, unary, pairwise))
        for _ in range(max_iterations):
            self.node_changed=False
            print(f"iterations nb: {_}")
            for alpha in range(K):
                graph, nodes = self.construct_graph(alpha, labels)

                # Compute min-cut
                graph.maxflow()

                # Update labels

                nv_labels = self.update_labels(graph, nodes, alpha, labels)
                nv_energy = self.compute_energy(nv_labels, unary, pairwise)
                print(
                    "computed energy",
                    nv_energy,
                    "is it greater than initial energy?",
                    nv_energy > energy,
                )
                if nv_energy < energy:
                    labels = nv_labels
                    energy = nv_energy
                    l_energy.append(energy)

                    show_segmentation(
                        image, labels, K, title=f"Alpha_exp: iteration {_} alpha {alpha}"
                    )
                    print("energy", self.compute_energy(labels, unary, pairwise))
                if self.node_changed==False:
                    print("no node have changed during the whole iteration, we stop here")
                    break
            # if _ % 5 == 0:
            #     show_segmentation(image, labels, K)
        return labels

    def update_labels(self, graph, nodes, alpha, labels):
        h, w = self.h, self.w
        nv_labels = labels.copy()
        for i in range(h):
            for j in range(w):
                pixel_index = i * w + j
                if graph.get_segment(nodes[pixel_index]) == 1:
                    nv_labels[i, j] = alpha
                    self.node_changed=True
        return nv_labels
