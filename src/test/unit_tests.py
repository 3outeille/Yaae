import torch
import numpy as np
from src.yaae.engine import Node

test_registry = {}
def register_test(func):
    test_registry[func.__name__] = func
    return func

@register_test
def test_add():
    # Yaae.
    n1 = Node([
                [
                    [1, 2, 3],
                    [1, 2, 3]
                ]
            ])
    n2 = Node([
                [
                    [4, 5, 6], 
                    [4, 5, 6]
                ]
            ])
    n3 = n1 + n2
    n3.backward([
                    [
                        [-1, -2, -3],
                        [-1, -2, -3]
                    ]
                ])
    n1_yaae, n2_yaae, n3_yaae = n1, n2, n3

    # Pytorch.
    n1 = torch.Tensor([
                        [
                            [1, 2, 3],
                            [1, 2, 3]
                        ]
                    ])
    n1.requires_grad = True
    n2 = torch.Tensor([
                        [
                            [4, 5, 6], 
                            [4, 5, 6]
                        ]
                    ])
    n2.requires_grad = True
    n3 = n1 + n2
    n3.retain_grad()
    n3.backward(torch.Tensor([
                        [
                            [-1, -2, -3],
                            [-1, -2, -3]
                        ]
                    ])
                )
    n1_torch, n2_torch, n3_torch = n1, n2, n3

    # Forward pass.
    assert (n3_yaae.val == n3_torch.data.numpy()).all()
    # Backward pass.
    assert (n3_yaae.grad == n3_torch.grad.numpy()).all()
    assert (n2_yaae.grad == n2_torch.grad.numpy()).all()
    assert (n1_yaae.grad == n1_torch.grad.numpy()).all()

@register_test
def test_add_broadcast():
    print("Not Implemented")
    return

@register_test
def test_sum():
    # Yaae.
    n1 = Node([
                [1,2,3],
                [1,2,3]
            ])
    n2 = n1.sum(axis=1)
    n2.backward([3, 3])
    n1_yaae, n2_yaae = n1, n2

    # Pytorch.
    n1 = torch.Tensor([
                        [1,2,3],
                        [1,2,3]
                    ])
    n1.requires_grad = True
    n2 = n1.sum(axis=1)
    n2.retain_grad()
    n2.backward(torch.Tensor([3,3]))
    n1_torch, n2_torch = n1, n2

    # Forward pass.
    assert (n2_yaae.val == n2_torch.data.numpy()).all()
    # Backward pass.
    assert (n2_yaae.grad == n2_torch.grad.numpy()).all()
    assert (n1_yaae.grad == n1_torch.grad.numpy()).all()

@register_test
def test_mul():
    # Yaae.
    n1 = Node([1, 2, 3])
    n2 = Node([4, 5, 6])
    n3 = n1 * n2
    n3.backward([-1, -2, -3])
    n1_yaae, n2_yaae, n3_yaae = n1, n2, n3

    # Pytorch.
    n1 = torch.Tensor([1, 2, 3])
    n1.requires_grad = True
    n2 = torch.Tensor([4, 5, 6])
    n2.requires_grad = True
    n3 = n1 * n2
    n3.retain_grad()
    n3.backward(torch.Tensor([-1, -2, -3]))
    n1_torch, n2_torch, n3_torch = n1, n2, n3
    
    # Forward pass.
    assert (n3_yaae.val == n3_torch.data.numpy()).all()
    # Backward pass.
    assert (n3_yaae.grad == n3_torch.grad.numpy()).all()
    assert (n2_yaae.grad == n2_torch.grad.numpy()).all()
    assert (n1_yaae.grad == n1_torch.grad.numpy()).all()

@register_test
def test_mul_broadcast():
    print("Not Implemented")
    return

@register_test
def test_matmul():
    # Yaae.
    n1 = Node([
                [
                    [1, 2, 3],
                    [1, 2, 3]
                ]
            ])
    n2 = Node([
                [
                    [4, 4], 
                    [5, 5],
                    [6, 6]
                ]
            ])
    n3 = n1.matmul(n2)
    n3.backward([
                    [
                        [-1, -2],
                        [-3, -4]
                    ]
                ])
    n1_yaae, n2_yaae, n3_yaae = n1, n2, n3

    # Pytorch.
    n1 = torch.Tensor([
                        [
                            [1, 2, 3],
                            [1, 2, 3]
                        ]
                    ])
    n1.requires_grad = True
    n2 = torch.Tensor([
                        [
                            [4, 4], 
                            [5, 5],
                            [6, 6]
                        ]
                    ])
    n2.requires_grad = True
    n3 = torch.matmul(n1, n2)
    n3.retain_grad()
    n3.backward(torch.Tensor([
                                [
                                    [-1, -2],
                                    [-3, -4]
                                ]
                            ]))
    n1_torch, n2_torch, n3_torch = n1, n2, n3
    
    # Forward pass.
    assert (n3_yaae.val == n3_torch.data.numpy()).all()
    # Backward pass.
    assert (n3_yaae.grad == n3_torch.grad.numpy()).all()
    assert (n2_yaae.grad == n2_torch.grad.numpy()).all()
    assert (n1_yaae.grad == n1_torch.grad.numpy()).all()

@register_test
def test_neg():
    # Yaae.
    n1 = Node([
                [
                    [1, 2, 3],
                    [1, 2, 3]
                ]
            ])
    n2 = -n1
    n2.backward(np.zeros_like(n1.val))
    n1_yaae, n2_yaae = n1, n2

    # Pytorch.
    n1 = torch.Tensor([
                        [
                            [1, 2, 3],
                            [1, 2, 3]
                        ]
                    ])
    n1.requires_grad = True
    n2 = -n1
    n2.backward(torch.zeros_like(n1.data))
    n1_torch, n2_torch = n1, n2

    # Forward pass.
    assert (n2_yaae.val == n2_torch.data.numpy()).all()
    # Backward pass.
    assert (n1_yaae.grad == n1_torch.grad.numpy()).all()

@register_test
def test_sin():
    # Yaae.
    n1 = Node([2, 3, 4])
    n2 = n1.sin()
    n2.backward([-1, -2, -3])
    n1_yaae, n2_yaae = n1, n2

    # Pytorch.
    n1 = torch.Tensor([2, 3, 4])
    n1.requires_grad = True
    n2 = n1.sin()
    n2.retain_grad()
    n2.backward(torch.Tensor([-1, -2, -3]))
    n1_torch, n2_torch  = n1, n2

    # Forward pass.
    assert np.allclose(n2_yaae.val, n2_torch.data.numpy())
    # Backward pass.
    assert np.allclose(n2_yaae.grad, n2_torch.grad.numpy())
    assert np.allclose(n1_yaae.grad, n1_torch.grad.numpy())

@register_test
def test_relu():
    # Yaae.
    n1 = Node([1, 2, 3])
    n2 = n1.relu()
    n2.backward([-1, -2, -3])
    n1_yaae, n2_yaae = n1, n2

    # Pytorch.
    n1 = torch.Tensor([1, 2, 3])
    n1.requires_grad = True
    n2 = n1.relu()
    n2.retain_grad()
    n2.backward(torch.Tensor([-1, -2, -3]))
    n1_torch, n2_torch  = n1, n2

    # Forward pass.
    assert (n2_yaae.val == n2_torch.data.numpy()).all()
    # Backward pass.
    assert (n2_yaae.grad == n2_torch.grad.numpy()).all()
    assert (n1_yaae.grad == n1_torch.grad.numpy()).all()

@register_test
def test_exp():
    # Yaae.
    n1 = Node([1, 2, 3])
    n2 = n1.relu()
    n2.backward([-1, -2, -3])
    n1_yaae, n2_yaae = n1, n2

    # Pytorch.
    n1 = torch.Tensor([1, 2, 3])
    n1.requires_grad = True
    n2 = n1.relu()
    n2.retain_grad()
    n2.backward(torch.Tensor([-1, -2, -3]))
    n1_torch, n2_torch  = n1, n2

    # Forward pass.
    assert (n2_yaae.val == n2_torch.data.numpy()).all()
    # Backward pass.
    assert (n2_yaae.grad == n2_torch.grad.numpy()).all()
    assert (n1_yaae.grad == n1_torch.grad.numpy()).all()

@register_test
def test_log():
    # Yaae.
    n1 = Node([1, 2, 3])
    n2 = n1.log()
    n2.backward([-1, -2, -3])
    n1_yaae, n2_yaae = n1, n2

    # Pytorch.
    n1 = torch.Tensor([1, 2, 3])
    n1.requires_grad = True
    n2 = n1.log()
    n2.retain_grad()
    n2.backward(torch.Tensor([-1, -2, -3]))
    n1_torch, n2_torch  = n1, n2

    # Forward pass.
    assert np.allclose(n2_yaae.val, n2_torch.data.numpy())
    # Backward pass.
    assert np.allclose(n2_yaae.grad, n2_torch.grad.numpy())
    assert np.allclose(n1_yaae.grad, n1_torch.grad.numpy())

for name, test in test_registry.items():
    print(f'Running {name}:', end=" ")
    test()
    print("OK")