

from graphCut_image_segmentation.graph_cut.display import show_segmentation
import numpy as np
from graphCut_image_segmentation.graph_cut.energy import compute_energy
import maxflow

# Alpha-Expansion Graph Cut using PyMaxflow
def alpha_expansion(image,unary, pairwise, K, method='kmeans', max_iterations=20):
    h, w, _ = image.shape
    # labels = np.argmin(unary, axis=2)  # Initialize labels using unary term
    # labels = initialize_labels_bis(image,method=method, K=K)
    labels = np.argmin(unary, axis=2)
    
    show_segmentation(image, labels,title="initialization")
    energy=compute_energy(labels,unary,pairwise)
    print("first energy",compute_energy(labels,unary,pairwise))
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
                        graph.add_tedge(nodes[pixel_index], unary[i, j, alpha], np.inf)  # Keep alpha pixels fixed
                    else:
                        graph.add_tedge(nodes[pixel_index], unary[i, j, alpha], unary[i, j, labels[i, j]])
            
            # Add pairwise terms
                    if i < h-1:
                        neighbor_index_down = pixel_index + w
                        
                        weight_down = pairwise[i, j, labels[i, j], alpha]

                        if labels[i,j] != labels[i+1,j]:
                            aux_node = graph.add_nodes(1)
                            graph.add_edge(nodes[pixel_index], aux_node, weight_down, weight_down)
                            graph.add_edge(nodes[neighbor_index_down], aux_node, pairwise[i+1,j,labels[i+1,j],alpha], pairwise[i+1,j,labels[i+1,j],alpha])
                        
                            graph.add_tedge(aux_node, 0, weight_down)

                        else:
                            graph.add_edge(nodes[pixel_index], nodes[neighbor_index_down], weight_down, weight_down)
                        
                    if j < w-1:
                        neighbor_index_right = pixel_index + 1
                        weight_right = pairwise[i, j, labels[i, j], alpha]
                        if labels[i,j] != labels[i,j+1]:
                            aux_node = graph.add_nodes(1)
                            graph.add_edge(nodes[pixel_index], aux_node, weight_right, weight_right)
                            graph.add_edge(nodes[neighbor_index_right],aux_node, pairwise[i,j+1,labels[i,j+1],alpha], pairwise[i,j+1,labels[i,j+1],alpha])
                            graph.add_tedge(aux_node, 0, weight_right)
                        else:
                            graph.add_edge(nodes[pixel_index], nodes[neighbor_index_right], weight_right, weight_right)

        
            # Compute min-cut
            graph.maxflow()
            
            # Update labels
            print("whole graph:",graph.get_grid_segments(nodes))
            print("whole graph:",graph.get_grid_segments(nodes).sum())
            print("len of nodes",len(nodes))

            nv_labels=labels.copy()
            for i in range(h):
                for j in range(w):
                    pixel_index = i * w + j
                    if graph.get_segment(nodes[pixel_index]) == 1:
                        nv_labels[i, j] = alpha  # Expand α-region
            nv_energy=compute_energy(nv_labels,unary,pairwise)
            print("computed energy",nv_energy,"is it greater than initial energy?",nv_energy>energy)
            if nv_energy<energy:
                labels=nv_labels
                energy=nv_energy
                

            show_segmentation(image, labels, K,title=f"iteration {_} alpha {alpha}")
            print("energy",compute_energy(labels,unary,pairwise))
        if _ % 5 == 0:
            show_segmentation(image, labels, K)
    return labels