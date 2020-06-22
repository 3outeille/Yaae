
def test_scalar_sin():
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

def test_scalar_relu():
    w1 = Node(2.)
    w2 = Node(3.)
    w3 = w2 * w1
    w4 = w1.relu()
    w5 = w3 + w4
    z = w5
    z.backward()
    w1_yaae, w2_yaae, z_yaae = w1, w2, z


    w1 = torch.Tensor([2.]).double()
    w1.requires_grad = True
    w2 = torch.Tensor([3.]).double()
    w2.requires_grad = True
    w3 = w2 * w1
    w4 = w1.relu()
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
    D.retain_grad()

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
    assert (D_yaae.grad == D_torch.grad.tolist()).all()
    assert (W_yaae.grad ==  W_torch.grad.tolist()).all()
    assert (X_yaae.grad == X_torch.grad.tolist()).all()

def test_tensor_add_relu():
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
    C = W + X
    D = C.relu()
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
    C = W + X
    D = C.relu()
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

def test_tensor_mul_relu():
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
    C = W.matmul(X)
    D = C.relu()
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
    C = torch.matmul(W, X)
    D = C.relu()
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

def test_tensor_cross_entropy_loss():
    X = Node([
                [1, 2], 
                [3, 4]
            ])

    a = X.exp().sum(axis=1)
    b = a.log()
    c = -X
    d = b + c

    dD = [
            [-1, -2],
            [-3, -4] 
    ]

    print('X', X, end='\n\n')
    print('a', a, end='\n\n')
    print('b', b, end='\n\n')
    print('c', c, end='\n\n')
    print('d', d, end='\n\n')

    print('---BACKWARD---\n')
    d.backward(dD)
    a_yaae, b_yaae, c_yaae, d_yaae = a, b, c, d
    
    print('d.grad', d.grad, end='\n\n')
    print('c.grad', c.grad, end='\n\n')
    print('b.grad', b.grad, end='\n\n')
    print('a.grad', a.grad, end='\n\n')
    print('X.grad', X.grad, end='\n\n')

    return

    X = torch.Tensor([
                        [1, 2], 
                        [3, 4]
                ])
    X.requires_grad = True

    a = X.exp().sum(axis=1)
    a.retain_grad()

    b = a.log()
    b.retain_grad()
    
    c = -X
    c.retain_grad()
    
    d = b + c
    d.retain_grad()

    dD = torch.Tensor([
                        [-1, -2],
                        [-3, -4] 
                ])

    print('X', X, end='\n\n')
    print('a', a, end='\n\n')
    print('b', b, end='\n\n')
    print('c', c, end='\n\n')
    print('d', d, end='\n\n')

    print('---BACKWARD---\n')
    d.backward(dD)
    a_torch, b_torch, c_torch, d_torch = a, b, c, d
    
    print('d.grad', d.grad, end='\n\n')
    print('c.grad', c.grad, end='\n\n')
    print('b.grad', b.grad, end='\n\n')
    print('a.grad', a.grad, end='\n\n')
    print('X.grad', X.grad, end='\n\n')
    
    # # Forward pass.
    # assert (a_yaae.val == a_torch.data.tolist()).all()
    # assert (b_yaae.val == b_torch.data.tolist()).all()
    # assert (c_yaae.val == c_torch.data.tolist()).all()
    # assert (d_yaae.val == d_torch.data.tolist()).all()

    # # Backward pass.
    # assert (a_yaae.grad == a_torch.grad.tolist()).all()
    # assert (b_yaae.grad == b_torch.grad.tolist()).all()
    # assert (c_yaae.grad == c_torch.grad.tolist()).all()
    # assert (d_yaae.grad == d_torch.grad.tolist()).all()

# test_scalar_sin()
# test_scalar_relu()
# test_tensor_add()
# test_tensor_add_relu()
# test_tensor_mul()
# test_tensor_mul_relu()
# test_tensor_cross_entropy_loss()
