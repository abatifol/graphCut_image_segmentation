from graph_cut.base_algorithm import Runner
from graph_cut.alpha_expansion import Alpha_expansion2
from graph_cut.recycle import Recycle
import numpy as np
from graph_cut.display import show_segmentation
from graph_cut.energy import compute_energy


class Alpha_and_recycle(Runner):
    def __init__(self, image, unary, pairwise, K, max_iterations):
        """
        Alpha Expansion with the recycling step
        Image: the input image
        unary: np.array of the unary term
        pairwise: np.array of the pairwise term
        K: number of labels
        max_iterations: number of iterations
        """
        super().__init__(image, unary, pairwise, K)

        self.max_iterations = max_iterations
        self.Alpha_exp = Alpha_expansion2(
            image=image,
            unary=unary,
            pairwise=pairwise,
            K=K,
            max_iterations=max_iterations,
        )

        self.recycle = Recycle(image=image, unary=unary, pairwise=pairwise, K=K)

    def init_unassigned_labels(self, labels, assigned_labels, unary):
        for i in range(self.h):
            for j in range(self.w):
                if assigned_labels[i, j] == self.epsilon:
                    labels[i, j] = np.argmin(unary[i, j, :])
        return labels

    def run(self, image):
        unary = self.unary
        pairwise = self.pairwise
        l_energy = self.l_energy
        K = self.K
        max_iterations = self.max_iterations
        assigned_labels = self.assigned_labels
        labels = (
            np.ones((self.h, self.w), dtype=np.int32) * self.epsilon
        )  #! Initially no labels are assigned
        show_segmentation(image, labels, title="initialization")
        energy = compute_energy(labels, unary, pairwise)
        l_energy.append(energy)

        print("first energy", compute_energy(labels, unary, pairwise))
        # first project the function
        self.recycle = Recycle(image=image, unary=unary, pairwise=pairwise, K=K)
        # now run the recycling
        labels, assigned_labels, self.cst, unary, pairwise = self.recycle.run(
            image=image
        )
        print(
            "after recycling",
            show_segmentation(image, labels, K, title="after recycling"),
        )

        # now we have the labels, we need to assign the unassigned labels
        labels = self.init_unassigned_labels(labels, assigned_labels, unary)
        print(
            "after init unassigned labels",
            show_segmentation(image, labels, K, title="after init unassigned labels"),
        )

        # Run the Alpha expansion algorihtm
        self.Alpha_exp = Alpha_expansion2(
            image=image,
            unary=unary,
            pairwise=pairwise,
            K=K,
            max_iterations=max_iterations,
        )
        # now run the Alpha expansion
        labels = self.Alpha_exp.run(
            image=image, assigned_labels=assigned_labels, init_labels=labels
        )

        # Display the results
        print("final energy", compute_energy(labels, unary, pairwise))
        print("final segmentation", show_segmentation(image, labels, K))
        return labels, assigned_labels

    def update_labels(self, graph, nodes, alpha, labels):
        h, w = self.h, self.w
        nv_labels = labels.copy()
        for i in range(h):
            for j in range(w):
                pixel_index = i * w + j
                if graph.get_segment(nodes[pixel_index]) == 1:
                    nv_labels[i, j] = alpha
        return nv_labels
