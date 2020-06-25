import numpy as np
from sklearn.datasets import make_circles, make_moons
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(0)
torch.manual_seed(0)

# Load dataset
X, y = make_moons(n_samples=200, noise=0.2)

# Create model
class NN(nn.Module):
    
    def __init__(self):
        super(NN, self).__init__()    
        self.FC1 = nn.Linear(in_features=2, out_features=16)
        self.FC2 = nn.Linear(in_features=16, out_features=16)
        self.FC3 = nn.Linear(in_features=16, out_features=1)  
 
    def forward(self, x):
        x = torch.relu(self.FC1(x))
        x = torch.relu(self.FC2(x))
        x = self.FC3(x)
        return torch.sigmoid(x)

model = NN()
print(model)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

EPOCHS = 200
inputs = torch.FloatTensor(X)
labels = torch.FloatTensor(y).reshape(-1, 1)
nb_examples = len(labels)

#Training phase.
for epoch in range(EPOCHS):
    train_loss = 0
    correct_train  = 0
    
    optimizer.zero_grad()
    # Forward pass.
    prediction = model(inputs)
    # Compute the loss.
    loss = criterion(prediction, labels)
    # Backward pass.
    loss.backward()
    # Optimize.
    optimizer.step()
    # Compute training accuracy.
    #print(prediction.detach().numpy())
    #print(np.where(prediction.detach().numpy()<0.5, 0, 1))
    pred_class = np.where(prediction.detach().numpy()<0.5, 0., 1.)
    correct_train = np.sum(labels.detach().numpy() == pred_class) 

    # Compute batch loss.
    train_loss = loss.data.item() / nb_examples
    train_acc =  correct_train / nb_examples

    info = "[Epoch {}/{}]: train-loss = {:0.6f} | train-acc = {:0.3f}"
    print(info.format(epoch+1, EPOCHS, train_loss, train_acc))