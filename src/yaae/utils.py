import numpy as np 

def topo_sort(v, topo, visited):
    """
    """
    if v not in visited:
        visited.add(v)
        for child in v.children:
            topo_sort(child, topo, visited)
        topo.append(v)
    return topo


# def broadcast_gradient(grad, shape):
#     '''
#     Broadcast a gradient to a shape with more dimension (broadcast 1 to non-1 size)
#     :param grad:
#     :param shape:
#     :return:
#     '''
#     assert len(grad.shape) == len(shape)
#     n_dim = len(grad.shape)
#     dims = []
#     for i in range(n_dim):
#         if shape[i] == 1:
#             dims.append(i)
#     return np.sum(grad, axis = tuple(dims), keepdims=True)


# def broadcast_gradient(grad, other_tensor): # Compress grad w.r.t a tensor
#     ndims_added = grad.ndim - other_tensor.ndim
#     for _ in range(ndims_added): 
#         grad = grad.sum(axis=0)         
#     for i, dim in enumerate(other_tensor.shape):
#         if dim == 1: 
#             grad = grad.sum(axis=i, keepdims=True) 
#     return grad


# shape = np.array([
#                     [1],
#                     [2]
#                 ])
# grad = np.array([
#                     [3,4],
#                     [5,6]
#                 ])

# print(broadcast_gradient(grad, shape.shape))
# print(_compress_grad(grad, shape))