import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score
from src.yaae.engine import Node
from src.yaae.nn import NN, Optimizer

np.random.seed(0)
samples = 100
X, y = make_circles(n_samples=samples, noise=0.05, factor=0.7)
# X, y = make_moons(n_samples=100, noise=0.1)
inputs = Node(X, requires_grad=False)
labels = Node(y[:, np.newaxis], requires_grad=False)

model = NN(nin=2, nouts=[16,16,1])
print(model)
optimizer = Optimizer(params=model.parameters(), lr=0.1)

def cross_entropy_loss(y_pred, labels):
    # 1. Compute binary cross entropy loss.
    loss = labels * (y_pred.log()) + (1. - labels) * ((1. - y_pred).log())
    train_loss = loss.sum(keepdims=False) / -samples

    # 2. Compute training accuracy.
    y_pred_class = np.where(y_pred.data<0.5, 0, 1)
    train_acc = np.sum(labels.data == y_pred_class.data) / samples
    #train_acc = accuracy_score(labels.data, y_pred_class)

    return train_loss, train_acc

EPOCHS = 500

for epoch in range(EPOCHS):
    # Empty weights/biases gradient.
    optimizer.zero_grad()
    # Forward pass.
    y_pred = model(inputs)
    train_loss, train_acc = cross_entropy_loss(y_pred, labels)
    #train_loss.grad.data = 1.
    # Backward pass.
    train_loss.backward()
    # Parameters update.
    optimizer.step()
    print(f"Epoch {epoch+1}: train-loss: {train_loss.data} | train-acc: {train_acc}")

import matplotlib.pyplot as plt
# Set min and max values and give it some padding
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = 0.25

# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the function value for the whole grid
Xmesh = np.c_[xx.ravel(), yy.ravel()]
Z = model(Node(Xmesh, requires_grad=False))
pred_class = np.where(Z.data<0.5, 0, 1)
pred_class = pred_class.reshape(xx.shape)

# Plot the contour and training examples
fig = plt.figure()
plt.contourf(xx, yy, pred_class, cmap=plt.cm.Spectral, alpha=0.8)
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()