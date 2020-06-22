import numpy as np
from sklearn.datasets import make_circles
from src.yaae.engine import Node
from src.yaae.nn import NN

X, y = make_circles(n_samples=200, noise=0.05, factor=0.7)
model = NN(nin=2, nouts=[16,16,1])
print(model)

inputs = Node(X)
y_pred = model(inputs)

def cross_entropy_loss(y_pred, y):
    # 1. Compute training loss.
    # a. Softmax
    # b. NLL.
    
    #y_pred.val = np.exp(y_pred.val) / np.sum(np.exp(y_pred.val), axis=0)
    #train_loss = y * np.log(y_pred.val)
    #train_loss = np.sum(train_loss) / len(train_loss)
    
    # 2. Compute training accuracy.
    train_acc = [(yi > 0) == (y_predi > 0)  for yi, y_predi in zip(y, y_pred.val[...,0])]
    train_acc = np.sum(train_acc) / len(train_acc)

    return train_loss, train_acc

train_loss, train_acc = cross_entropy_loss(y_pred, y)
print(train_loss)
print(train_acc)