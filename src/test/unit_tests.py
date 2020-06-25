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
                [1,2,3],
                [1,2,3]
            ], requires_grad=True)
    n2 = n1.sum(axis=1)
    n2.backward(Node([[3], [3]], requires_grad=False))
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
    print(n2_torch.grad.data.numpy())
    print(n1_torch.grad.data.numpy())
    
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
                    [1, 2, 3],
                    [1, 2, 3]
                ]
            ], requires_grad=True)
    n2 = -n1
    n2.backward(np.zeros_like(n1.data))
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
    assert (n2_yaae.data == n2_torch.data.numpy()).all()
    # Backward pass.
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
def test_scalar():
    # Yaae.
    w1 = Node(2, requires_grad=True)
    w2 = Node(3, requires_grad=True)
    w3 = w2 * w1
    w4 = w1.sin()
    w5 = w3 + w4
    z = w5
    z.backward()
    w1_yaae, w2_yaae, z_yaae = w1, w2, z

    # Pytorch.
    w1 = torch.Tensor([2]).double()
    w1.requires_grad = True
    w2 = torch.Tensor([3]).double()
    w2.requires_grad = True
    w3 = w2 * w1
    w4 = w1.sin()
    w5 = w3 + w4
    z = w5
    z.backward()
    w1_torch, w2_torch, z_torch = w1, w2, z
    
    # Forward pass.
    assert z_yaae.data == z_torch.data.item()
    # Backward pass.
    assert w1_yaae.grad.data ==  w1_torch.grad.item()
    assert w2_yaae.grad.data == w2_torch.grad.item()

# @register_test
# def test_minimize_sos():
#     x = Node([10, -10, 10, -5, 6, 3, 1], requires_grad=True)
#     # we want to minimize the sum of squares
    
#     for i in range(100):
#         sum_of_squares = (x*x).sum(keepdims=False) # 0-Tensor
#         sum_of_squares.backward()
#         delta_x = x.grad.data * 0.1 
#         x = Node(x.data - delta_x, requires_grad=True)

#     assert sum_of_squares.data < 1e-12

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


# @register_test
# def test_cross_entropy():
#     # Yaae.
#     X = Node([
#                 [1, 2], 
#                 [3, 4]
#             ],requires_grad=True)
#     a = X.exp().sum(axis=1, keepdims=False)
#     b = a.log()
#     c = -X
#     d = b + c
#     dD = [
#             [-1, -2],
#             [-3, -4] 
#     ]
#     d.backward(dD)
#     a_yaae, b_yaae, c_yaae, d_yaae = a, b, c, d

#     # Pytorch.
#     X = torch.Tensor([
#                         [1, 2], 
#                         [3, 4]
#                 ])
#     X.requires_grad = True

#     a = X.exp().sum(axis=1)
#     a.retain_grad()
#     b = a.log()
#     b.retain_grad()
#     c = -X
#     c.retain_grad()
#     d = b + c
#     d.retain_grad()
#     dD = torch.Tensor([
#                         [-1, -2],
#                         [-3, -4] 
#                 ])
#     d.backward(dD)
#     a_torch, b_torch, c_torch, d_torch = a, b, c, d

#     # Forward pass.
#     # Make sur that the numbers are the same within about 6 decimal digits.
#     assert np.isclose(a_yaae.data, a_torch.data.numpy(), rtol=1e-06, atol=0).all()
#     assert np.isclose(b_yaae.data, b_torch.data.numpy(), rtol=1e-06, atol=0).all()
#     assert np.isclose(c_yaae.data, c_torch.data.numpy(), rtol=1e-06, atol=0).all()
#     assert np.isclose(d_yaae.data, d_torch.data.numpy(), rtol=1e-06, atol=0).all()

#     # Backward pass.
#     assert np.isclose(a_yaae.grad.data, a_torch.grad.data.numpy(), rtol=1e-06, atol=0).all()
#     assert np.isclose(b_yaae.grad.data, b_torch.grad.data.numpy(), rtol=1e-06, atol=0).all()
#     assert np.isclose(c_yaae.grad.data, c_torch.grad.data.numpy(), rtol=1e-06, atol=0).all()
#     assert np.isclose(d_yaae.grad.data, d_torch.grad.data.numpy(), rtol=1e-06, atol=0).all()

for name, test in test_registry.items():
    print(f'Running {name}:', end=" ")
    test()
    print("OK")