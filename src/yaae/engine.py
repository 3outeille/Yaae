from src.yaae.utils import topo_sort#, broadcast_gradient
import numpy as np

class Node:
    
    def __init__(self, value, children=[]):
        self.val = value if isinstance(value, np.ndarray) else np.array(value)
        self.children = children
        self.grad = np.zeros_like(self.val, dtype=np.float64) if isinstance(self.val, np.ndarray) else 0
        # Stores function.
        self._compute_derivatives = lambda: None

    def __repr__(self):
        return f"Value(val={self.val},\n grad={self.grad})\n"

    def backward(self, grad=None):
        topo = topo_sort(self, topo=[], visited=set())
        if grad is None:
            self.grad = 1
        else:
            self.grad = np.array(grad)

        for v in reversed(topo):
            v._compute_derivatives()

    # Operators.
    def __add__(self, other):
        op = Add(self, other)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out
    
    def sum(self, axis=None):
        op = Sum(self, axis)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def __mul__(self, other):
        op = Mul(self, other)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def matmul(self, other):
        op = Matmul(self, other)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def __neg__(self): 
        # -self
        op = Neg(self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

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
        raise NotImplementedError

    def __rtruediv__(self, other): 
        raise NotImplementedError
    
    def __pow__(self, other):
        raise NotImplementedError

    # Functions.
    def sin(self):
        op = Sin(self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def relu(self):
        op = ReLU(self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def exp(self):
        op = Exp(self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def log(self):
        op = Log(self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

# ----------------------------------------------------------------------------
# Operators & Functions.

class Add():
    
    def __init__(self, node1, node2):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
        self.node2 = node2 if isinstance(node2, Node) else Node(node2)
    
    def forward_pass(self):
        self.out = Node(np.add(self.node1.val, self.node2.val), children=[self.node1, self.node2])
        return self.out

    def compute_derivatives(self):
        self.node1.grad += self.out.grad
        self.node2.grad += self.out.grad
        # self.node1.grad += broadcast_gradient(self.out.grad, self.node1.grad)
        # self.node2.grad += broadcast_gradient(self.out.grad, self.node2.grad)

class Sum():
    def __init__(self, node1, axis=None):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
        self.axis = axis

    def forward_pass(self):
        self.out = Node(np.sum(self.node1.val, self.axis, keepdims=True), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        if self.axis != None:
            self.out.grad = np.expand_dims(self.out.grad, self.axis)
        self.node1.grad += self.out.grad * np.ones_like(self.node1.val)

class Mul():

    def __init__(self, node1, node2):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
        self.node2 = node2 if isinstance(node2, Node) else Node(node2)
    
    def forward_pass(self):
        self.out = Node(np.multiply(self.node1.val, self.node2.val), children=[self.node1, self.node2])
        return self.out

    def compute_derivatives(self):
        # Broadcast gradient ?
        # self.node1.grad += broadcast_gradient(self.out.grad, self.node2.val)
        # self.node2.grad += broadcast_gradient(self.out.grad, self.node1.val)
        self.node1.grad += self.out.grad * self.node2.val
        self.node2.grad += self.out.grad * self.node1.val

class Matmul():

    def __init__(self, node1, node2):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
        self.node2 = node2 if isinstance(node2, Node) else Node(node2)

    def forward_pass(self):
        self.out = Node(self.node1.val @ self.node2.val, children=[self.node1, self.node2])
        return self.out

    def compute_derivatives(self):
        dim = [i for i in range(len(self.node1.val.shape))]
        if len(dim) > 1:
            dim[-1], dim[-2] = dim[-2], dim[-1]
        self.node1.grad = self.out.grad @ self.node2.val.transpose(dim)
        self.node2.grad = self.node1.val.transpose(dim) @ self.out.grad

class Neg():

    def __init__(self, node1):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
   
    def forward_pass(self):
        return Node(-self.node1.val, children=[self.node1])

    def compute_derivatives(self):
        return lambda node: -node.val

class Sin():

    def __init__(self, node1):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
    
    def forward_pass(self):
        self.out = Node(np.sin(self.node1.val), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        self.node1.grad += np.cos(self.node1.val) * self.out.grad

class ReLU():

    def __init__(self, node1):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)

    def forward_pass(self):
        self.out = Node(self.node1.val * (self.node1.val > 0), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        self.node1.grad += (self.out.val > 0) * self.out.grad

class Exp():

    def __init__(self, node1):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)

    def forward_pass(self):
        self.out = Node(np.exp(self.node1.val), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        self.node1.grad += self.out.grad * np.exp(self.node1.val)

class Log():

    def __init__(self, node1):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)

    def forward_pass(self):
        self.out = Node(np.log(self.node1.val), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        self.node1.grad += self.out.grad * (1/self.node1.val)