import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score
from src.yaae.engine import Node
from src.yaae.nn import NN, Optimizer

np.random.seed(0)
# X, y = make_circles(n_samples=2, noise=0.05, factor=0.7
samples = 100
X, y = make_moons(n_samples=samples, noise=0.01)
inputs = Node(X, requires_grad=False)
labels = Node(y[:, np.newaxis], requires_grad=False)

model = NN(nin=2, nouts=[16,16,1])
print(model)
optimizer = Optimizer(params=model.parameters(), lr=0.01)

def cross_entropy_loss(y_pred, labels):
    # 1. Compute binary cross entropy loss.
    loss = labels * (y_pred.log()) + (1. - labels) * ((1. - y_pred).log())
    train_loss = loss.sum(keepdims=False) / -samples

    # 2. Compute training accuracy.
    y_pred_class = np.where(y_pred.data<0.5, 0, 1)
    train_acc = accuracy_score(labels.data, y_pred_class)

    return train_loss, train_acc

EPOCHS = 10

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


