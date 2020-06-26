import numpy as np 

def topo_sort(v, L, visited):
    """
        Returns list of all of the children in the graph in topological order.

        Parameters:
        - v: last node in the graph.
        - L: list of all of the children in the graph in topological order (empty at the beginning).
        - visited: set of visited children.
    """
    if v not in visited:
        visited.add(v)
        for child in v.children:
            topo_sort(child, L, visited)
        L.append(v)
    return L

def compress_gradient(grad, other_tensor_shape):
    """
        Returns the gradient but compressed (needed when gradient shape mismatch during reverse mode).

        Paramaters:
        - grad: gradient.
        - other_tensor_shape: shape of target tensor.
    """
    ndims_added = grad.ndim - len(other_tensor_shape)
    for _ in range(ndims_added): 
        grad = grad.sum(axis=0)         
    for i, dim in enumerate(other_tensor_shape):
        if dim == 1: 
            grad = grad.sum(axis=i, keepdims=True) 
    return grad