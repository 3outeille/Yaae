from src.yaae_scalar.utils import topo_sort
import numpy as np

class Node:
    
    def __init__(self, value, children=[]):
        self.val = value
        self.children = children
        self.grad = 0
        # Stores function.
        self.compute_derivatives = lambda: None
    
    def __repr__(self):
        return f"Node(val={self.val}, grad={self.grad})"

    def backward(self):
        topo = topo_sort(self)
        self.grad = 1
        for v in reversed(topo):
            v.compute_derivatives()

    # Operators.
    def __add__(self, other):
        op = Add(self, other)
        out = op.forward_pass()
        out.compute_derivatives = op.compute_derivatives
        return out
    
    def __mul__(self, other):
        op = Mul(self, other)
        out = op.forward_pass()
        out.compute_derivatives = op.compute_derivatives
        return out
    
    def __neg__(self): 
        # -self
        return self * -1

    def __radd__(self, other): 
        # other + self
        return self + other

    def __sub__(self, other):
        # self - other
        return self + (-other)

    def __rsub__(self, other): 
        # other - self
        return other + (-self)

    def __rmul__(self, other): 
        # other * self
        return self * other

    def __truediv__(self, other): 
        # self / other
        return self * other**-1

    def __rtruediv__(self, other): 
        # other / self
        return other * self**-1

    # Functions.
    def sin(self):
        op = Sin(self)
        out = op.forward_pass()
        out.compute_derivatives = op.compute_derivatives
        return out

# ----------------------------------------------------------------------------
# Operators & Functions.

class Add():
    
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2 if isinstance(node2, Node) else Node(node2)
    
    def forward_pass(self):
        self.out = Node(self.node1.val + self.node2.val, children=[self.node1, self.node2])
        return self.out

    def compute_derivatives(self):
        self.node1.grad += self.out.grad
        self.node2.grad += self.out.grad

class Mul():

    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2 if isinstance(node2, Node) else Node(node2)
    
    def forward_pass(self):
        self.out = Node(self.node1.val * self.node2.val, children=[self.node1, self.node2])
        return self.out

    def compute_derivatives(self):
        self.node1.grad += self.node2.val * self.out.grad
        self.node2.grad += self.node1.val * self.out.grad  

class Sin():

    def __init__(self, node1):
        self.node1 = node1
    
    def forward_pass(self):
        self.out = Node(np.sin(self.node1.val), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        self.node1.grad += np.cos(self.node1.val) * self.out.grad
