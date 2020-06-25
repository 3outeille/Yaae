import numpy as np 

def topo_sort(v, L, visited):
    """
    """
    if v not in visited:
        visited.add(v)
        for child in v.children:
            topo_sort(child, L, visited)
        L.append(v)
    return L

def compress_gradient(grad, other_tensor):
    """

    """
    ndims_added = grad.ndim - other_tensor.ndim
    for _ in range(ndims_added): 
        grad = grad.sum(axis=0)         
    for i, dim in enumerate(other_tensor.shape):
        if dim == 1: 
            grad = grad.sum(axis=i, keepdims=True) 
    return grad