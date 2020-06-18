import torch
from src.yaae.engine import Node

def test_scalar():
    w1 = Node(2.)
    w2 = Node(3.)
    w3 = w2 * w1
    w4 = w1.sin()
    w5 = w3 + w4
    z = w5
    z.backward()
    w1_yaae, w2_yaae, z_yaae = w1, w2, z


    w1 = torch.Tensor([2.]).double()
    w1.requires_grad = True
    w2 = torch.Tensor([3.]).double()
    w2.requires_grad = True
    w3 = w2 * w1
    w4 = w1.sin()
    w5 = w3 + w4
    z = w5
    z.backward()
    w1_torch, w2_torch, z_torch = w1, w2, z

    # Forward pass.
    assert z_yaae.val == z_torch.data.item()
    # Backward pass.
    assert w1_yaae.grad ==  w1_torch.grad.item()
    assert w2_yaae.grad == w2_torch.grad.item()

def test_tensor_add():
    W = Node([
                [
                    [1, 2, 3],
                    [1, 2, 3]
                ]
            ])
    X = Node([
                [
                    [4, 5, 6], 
                    [4, 5, 6]
                ]
            ])
    D = W + X
    dD = [
            [
                [-1, -2, -3],
                [-1, -2, -3]
            ]
        ]
    D.backward(dD)
    D_yaae, W_yaae, X_yaae = D, W, X

    W = torch.Tensor([
                        [
                            [1, 2, 3],
                            [1, 2, 3]
                        ]
                    ])
    W.requires_grad=True
    X = torch.Tensor([
                        [
                            [4, 5, 6], 
                            [4, 5, 6]
                        ]
                    ])
    X.requires_grad=True
    D = W + X
    dD = torch.Tensor([
                        [
                            [-1, -2, -3],
                            [-1, -2, -3]
                        ]
                    ])
    D.backward(dD)
    D_torch, W_torch, X_torch = D, W, X

    # Forward pass.
    assert (D_yaae.val == D_torch.data.tolist()).all()
    # Backward pass.
    assert (W_yaae.grad ==  W_torch.grad.tolist()).all()
    assert (X_yaae.grad == X_torch.grad.tolist()).all()

def test_tensor_mul():

    W = Node([
                [
                    [1, 2, 3],
                    [1, 2, 3]
                ]
            ])
    X = Node([
                [
                    [4, 4], 
                    [5, 5],
                    [6, 6]
                ]
            ])
    D = W.matmul(X)
    dD = [
            [
                [-1, -2],
                [-3, -4]
            ]
        ]
    D.backward(dD)
    D_yaae, W_yaae, X_yaae = D, W, X

    W = torch.Tensor([
                        [
                            [1, 2, 3],
                            [1, 2, 3]
                        ]
                    ])
    W.requires_grad=True
    X = torch.Tensor([
                        [
                            [4, 4], 
                            [5, 5],
                            [6, 6]
                        ]
                    ])
    X.requires_grad=True
    D = torch.matmul(W, X)
    dD = torch.Tensor([
                        [
                            [-1, -2],
                            [-3, -4]
                        ]
                    ])
    D.backward(dD)
    D_torch, W_torch, X_torch = D, W, X
    
    # Forward pass.
    assert (D_yaae.val == D_torch.data.tolist()).all()
    # Backward pass.
    assert (W_yaae.grad ==  W_torch.grad.tolist()).all()
    assert (X_yaae.grad == X_torch.grad.tolist()).all()

test_scalar()
test_tensor_add()
test_tensor_mul()