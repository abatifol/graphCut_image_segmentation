import numpy as np
def compute_energy(assignment,unary_term,pairwise_term):
    energy=0
    
    for i,j in np.ndindex(assignment.shape):
        energy+=unary_term[i,j,assignment[i,j]]
        if i+1<assignment.shape[0]:
            energy+=pairwise_term[i,j,assignment[i,j],assignment[i+1,j]]
        if j+1<assignment.shape[1]:
            energy+=pairwise_term[i,j,assignment[i,j],assignment[i,j+1]]
    return energy