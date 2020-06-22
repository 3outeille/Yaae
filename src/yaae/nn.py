from src.yaae.engine import Node
import numpy as np

class Linear():
    
    def __init__(self, name, row, column, isLastLayer=False):
        self.name = name
        self.row = row
        self.col = column
        self.isLastLayer = isLastLayer

        bound = 1 / np.sqrt(self.row)
        self.W = Node(np.random.uniform(-bound, bound, size=(row, column)))
        self.b = Node(0)

    def __call__(self, X):
        act = X.matmul(self.W) + self.b
        return act.relu() if not self.isLastLayer else act
        
    def __repr__(self):
        return f"({self.name}): Linear(row={self.row}, column={self.col}, isLastLayer={self.isLastLayer})\n"
        

class NN():

    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Linear(f'Linear{i}', sizes[i], sizes[i+1], isLastLayer=(i == len(nouts)-1)) for i in range(len(nouts))]
    
    def __call__(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def __repr__(self):
        s = "model(\n"
        for layer in self.layers:
            s += "   " + str(layer)
        s += ")"
        return s

class Optimizer():
    
    def __init__(self):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass