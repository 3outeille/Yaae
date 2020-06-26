import torch
import numpy as np
from sklearn.datasets import make_regression
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
            ], requires_grad=True)
    n2 = Node([
                [
                    [4, 5, 6], 
                    [4, 5, 6]
                ]
            ], requires_grad=True)
    n3 = n1 + n2
    n3.backward(Node([
                    [
                        [-1, -2, -3],
                        [-1, -2, -3]
                    ]
                ], requires_grad=False))
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
    assert (n3_yaae.data == n3_torch.data.numpy()).all()
    # Backward pass.
    assert (n3_yaae.grad.data == n3_torch.grad.data.numpy()).all()
    assert (n2_yaae.grad.data == n2_torch.grad.data.numpy()).all()
    assert (n1_yaae.grad.data == n1_torch.grad.data.numpy()).all()

@register_test
def test_sum():
    # Yaae.
    n1 = Node([
                [1.,2.,3.],
                [1.,2.,3.]
            ], requires_grad=True)
    n2 = n1.sum(axis=1, keepdims=False)
    n2.backward(Node([10.,20.], requires_grad=False))
    n1_yaae, n2_yaae = n1, n2

    # Pytorch.
    n1 = torch.Tensor([
                        [1.,2.,3.],
                        [1.,2.,3.]
                    ])
    n1.requires_grad = True
    n2 = n1.sum(axis=1)
    n2.retain_grad()
    n2.backward(torch.Tensor([10.,20.]))
    n1_torch, n2_torch = n1, n2

    # Forward pass.
    assert (n2_yaae.data == n2_torch.data.numpy()).all()
    # Backward pass.
    assert (n2_yaae.grad.data == n2_torch.grad.data.numpy()).all()
    assert (n1_yaae.grad.data == n1_torch.grad.data.numpy()).all()
    
@register_test
def test_mul():
    # Yaae.
    n1 = Node([1, 2, 3], requires_grad=True)
    n2 = Node([4, 5, 6], requires_grad=True)
    n3 = n1 * n2
    n3.backward(Node([-1, -2, -3], requires_grad=False))
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
    assert (n3_yaae.data == n3_torch.data.numpy()).all()
    # Backward pass.
    assert (n3_yaae.grad.data == n3_torch.grad.data.numpy()).all()
    assert (n2_yaae.grad.data == n2_torch.grad.data.numpy()).all()
    assert (n1_yaae.grad.data == n1_torch.grad.data.numpy()).all()

@register_test
def test_matmul():
    # Yaae.
    n1 = Node([
                [
                    [1, 2, 3],
                    [1, 2, 3]
                ]
            ], requires_grad=True)
    n2 = Node([
                [
                    [4, 4], 
                    [5, 5],
                    [6, 6]
                ]
            ], requires_grad=True)
    n3 = n1.matmul(n2)
    n3.backward(Node([
                        [
                            [-1, -2],
                            [-3, -4]
                        ]
                    ], requires_grad=False))
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
    assert (n3_yaae.data == n3_torch.data.numpy()).all()
    # Backward pass.
    assert (n3_yaae.grad.data == n3_torch.grad.data.numpy()).all()
    assert (n2_yaae.grad.data == n2_torch.grad.data.numpy()).all()
    assert (n1_yaae.grad.data == n1_torch.grad.data.numpy()).all()

@register_test
def test_neg():
    # Yaae.
    n1 = Node([
                [
                    [1., 2., 3.],
                    [1., 2., 3.]
                ]
            ], requires_grad=True)
    n2 = -n1
    n2.backward(np.ones_like(n1.data))
    n1_yaae, n2_yaae = n1, n2

    # Pytorch.
    n1 = torch.Tensor([
                        [
                            [1., 2., 3.],
                            [1., 2., 3.]
                        ]
                    ])
    n1.requires_grad = True
    n2 = -n1
    n2.retain_grad()
    n2.backward(torch.ones_like(n1.data))
    n1_torch, n2_torch = n1, n2

    # Forward pass.
    assert (n2_yaae.data == n2_torch.data.numpy()).all()
    # Backward pass.
    assert (n2_yaae.grad.data == n2_torch.grad.data.numpy()).all()
    assert (n1_yaae.grad.data == n1_torch.grad.data.numpy()).all()

@register_test
def test_div():
    # Yaae.
    n1 = Node([1, 2, 3], requires_grad=True)
    n2 = Node([4, 5, 6], requires_grad=True)
    n3 = n1 / n2
    n3.backward(Node([-1, -2, -3], requires_grad=False))
    n1_yaae, n2_yaae, n3_yaae = n1, n2, n3

    # Pytorch.
    n1 = torch.Tensor([1, 2, 3])
    n1.requires_grad = True
    n2 = torch.Tensor([4, 5, 6])
    n2.requires_grad = True
    n3 = n1 / n2
    n3.retain_grad()
    n3.backward(torch.Tensor([-1, -2, -3]))
    n1_torch, n2_torch, n3_torch = n1, n2, n3

    # Forward pass.
    assert np.isclose(n3_yaae.data, n3_torch.data.numpy()).all()
    # Backward pass.
    assert np.isclose(n3_yaae.grad.data, n3_torch.grad.data.numpy()).all()
    assert np.isclose(n2_yaae.grad.data, n2_torch.grad.data.numpy()).all()
    assert np.isclose(n1_yaae.grad.data, n1_torch.grad.data.numpy()).all()

@register_test
def test_sin():
    # Yaae.
    n1 = Node([2, 3, 4], requires_grad=True)
    n2 = n1.sin()
    n2.backward(Node([-1, -2, -3], requires_grad=False))
    n1_yaae, n2_yaae = n1, n2

    # Pytorch.
    n1 = torch.Tensor([2, 3, 4])
    n1.requires_grad = True
    n2 = n1.sin()
    n2.retain_grad()
    n2.backward(torch.Tensor([-1, -2, -3]))
    n1_torch, n2_torch  = n1, n2

    # Forward pass.
    assert np.isclose(n2_yaae.data, n2_torch.data.numpy()).all()
    # Backward pass.
    assert np.isclose(n2_yaae.grad.data, n2_torch.grad.data.numpy()).all()
    assert np.isclose(n1_yaae.grad.data, n1_torch.grad.data.numpy()).all()

@register_test
def test_relu():
    # Yaae.
    n1 = Node([1, 2, 3], requires_grad=True)
    n2 = n1.relu()
    n2.backward(Node([-1, -2, -3], requires_grad=False))
    n1_yaae, n2_yaae = n1, n2
    
    # Pytorch.
    n1 = torch.Tensor([1, 2, 3])
    n1.requires_grad = True
    n2 = n1.relu()
    n2.retain_grad()
    n2.backward(torch.Tensor([-1, -2, -3]))
    n1_torch, n2_torch  = n1, n2

    # Forward pass.
    assert (n2_yaae.data == n2_torch.data.numpy()).all()
    # Backward pass.
    assert (n2_yaae.grad.data == n2_torch.grad.data.numpy()).all()
    assert (n1_yaae.grad.data == n1_torch.grad.data.numpy()).all()

@register_test
def test_exp():
    # Yaae.
    n1 = Node([1, 2, 3], requires_grad=True)
    n2 = n1.relu()
    n2.backward(Node([-1, -2, -3], requires_grad=False))
    n1_yaae, n2_yaae = n1, n2

    # Pytorch.
    n1 = torch.Tensor([1, 2, 3])
    n1.requires_grad = True
    n2 = n1.relu()
    n2.retain_grad()
    n2.backward(torch.Tensor([-1, -2, -3]))
    n1_torch, n2_torch  = n1, n2

    # Forward pass.
    assert np.isclose(n2_yaae.data, n2_torch.data.numpy()).all
    # Backward pass.
    assert np.isclose(n2_yaae.grad.data, n2_torch.grad.data.numpy()).all
    assert np.isclose(n1_yaae.grad.data, n1_torch.grad.data.numpy()).all

@register_test
def test_log():
    # Yaae.
    n1 = Node([10, 20, 30], requires_grad=True)
    n2 = n1.log()
    n2.backward(Node([1, 2, 3], requires_grad=False))
    n1_yaae, n2_yaae = n1, n2

    # Pytorch.
    n1 = torch.Tensor([10, 20, 30])
    n1.requires_grad = True
    n2 = n1.log()
    n2.retain_grad()
    n2.backward(torch.Tensor([1, 2, 3]))
    n1_torch, n2_torch  = n1, n2

    # Forward pass.
    assert np.isclose(n2_yaae.data, n2_torch.data.numpy()).all
    # Backward pass.
    assert np.isclose(n2_yaae.grad.data, n2_torch.grad.data.numpy()).all
    assert np.isclose(n1_yaae.grad.data, n1_torch.grad.data.numpy()).all

@register_test
def test_sigmoid():
    # Yaae.
    n1 = Node([1, 2, 3], requires_grad=True)
    n2 = n1.sigmoid()
    n2.backward(Node([-1, -2, -3], requires_grad=False))
    n1_yaae, n2_yaae = n1, n2

    # Pytorch.
    n1 = torch.Tensor([1, 2, 3])
    n1.requires_grad = True
    n2 = n1.sigmoid()
    n2.retain_grad()
    n2.backward(torch.Tensor([-1, -2, -3]))
    n1_torch, n2_torch  = n1, n2

    # Forward pass.
    assert np.isclose(n2_yaae.data, n2_torch.data.numpy()).all
    # Backward pass.
    assert np.isclose(n2_yaae.grad.data, n2_torch.grad.data.numpy()).all
    assert np.isclose(n1_yaae.grad.data, n1_torch.grad.data.numpy()).all

@register_test
def test_linear_regression():

    # Generate data.
    x, y = make_regression(n_samples=100,
                           n_features=3,
                           noise=10,
                           random_state=42)     

    X = Node(x, requires_grad=False)
    y_true = Node(y, requires_grad=False)
    W = Node(np.random.randn(3), requires_grad=True)
    b = Node(np.random.randn(), requires_grad=True)

    lr = 0.01

    for epoch in range(100):
        W.zero_grad()
        b.zero_grad()
        y_pred = X.matmul(W) + b
        errors = y_pred - y_true
        loss = (errors * errors).sum(keepdims=False) / 100
        loss.backward()
        W -= lr * W.grad.data 
        b -= lr * b.grad.data
    
    # Compute R^2.
    y_bar = np.average(y_true.data)
    SStot = np.sum((y_true.data - y_bar)**2)
    SSres = np.sum((y_true.data - y_pred.data)**2)
    r2 = 1 - (SSres/SStot)
    
    assert r2 > 0.9

for name, test in test_registry.items():
    print(f'Running {name}:', end=" ")
    test()
    print("OK")