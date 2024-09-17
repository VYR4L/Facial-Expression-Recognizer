import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import warnings
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import itertools
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


x, y = make_circles(n_samples=10000,
                    noise=0.05,
                    random_state=26)

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=26)

fig, (train_ax, test_ax) = plt.subplots(ncols=2,
                                        sharex=True,
                                        sharey=True,
                                        figsize=(10, 5)
                                    )

train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
train_ax.set_title('Training Data')
train_ax.set_xlabel('Feature #0')
train_ax.set_ylabel('Feature #1')

test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
test_ax.set_xlabel('Feature #0')
test_ax.set_title('Testing Data')
plt.show()

# Using GPU:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pythorch Dataset and DataLoader:

warnings.filterwarnings('ignore')

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = self.X.shape[0]

        self.X = self.X.to(device)
        self.y = self.y.to(device)

        # also can be used:
        # self.X = torch.from_numpy(X.astype(np.float32))
        # self.y = torch.from_numpy(y.astype(np.float32))
        # self.len = self.X.shape[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
batch_size = 64

train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

for batch, (X, y) in enumerate(train_dataloader):
    print(f'Batch: {batch+1}, X: {X.shape}, y: {y.shape}')
    break

# Neural Network Model with ReLU Activation Function:

input_dim = 2
hidden_dim = 10
output_dim = 1


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity='relu')
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))
        return x

model = NeuralNetwork(input_dim, hidden_dim, output_dim)
model.to(device)
print(model)

# Initially, the loss starts at 0.7 and gradually decreases 

learning_rate = 0.01
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.7)

# Training the Model:

num_epochs = 100
loss_values = []


for epoch in range(num_epochs):
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

print("Training Complete")

# Plotting the Loss:

step = np.linspace(0, 100, 10500)

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Predictions and Evaluation:

y_pred = []
y_test = []
total = 0.0
correct = 0.0

with torch.no_grad():
    for X, y in test_dataloader:
        outpus = model(X)
        predicted = np.where(outpus < 0.5, 0, 1)
        predicted = list(itertools.chain(*predicted))
        y_pred.append(predicted)
        y_test.append(y)
        total += y.size(0)
        correct += (predicted == y.numpy()).sum().item()
print(f'Accuracy of the network on the 3300 test instances: {100 * correct // total}%')

y_pred = list(itertools.chain(*y_pred))
y_test = list(itertools.chain(*y_test))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.subplots(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt='g', cbar=False)
plt.show()