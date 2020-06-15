from src.yaae_scalar.engine import Node

def example():
    w1 = Node(2.)
    w2 = Node(3.)
    w3 = w2 * w1
    w4 = w1.sin()
    w5 = w3 + w4
    z = w5
    print('w1:', w1, w1.children)
    print('w2:', w2, w2.children)
    print('w3:', w3, w3.children)
    print('w4:', w4, w4.children)
    print('w5:', w5, w5.children)
    print('z:', z, z.children)
    z.backward()
    print('\n---backward---')
    print('z:', z)
    print('w5:', w5)
    print('w4:', w4)
    print('w3:', w3)
    print('w2:', w2)
    print('w1:', w1)
    
example()
