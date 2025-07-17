import glob
import torch
import numpy as np
from torch import nn
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from torchvision.models import Inception3
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.nn.functional import softmax
from utils.dataset import Data, output_type, OUTPUT_DIR, count_type


training_data = Data(is_train=True)

for key, val in output_type.items():
    print(f'{key}: {count_type[val]}')

validation_data = Data(is_train=False)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)


# Inception3 Convolutional Neural Network
class Inception(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 64, 1),
            nn.Conv2d(64, 64, 3, padding=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 64, 1),
            nn.Conv2d(64, 64, 5, padding=2)
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(1, 64, 1)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(1, 64, 1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


# Inception Convolutional Neural Network Training
class TrainModuleInception(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_network = Inception()
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.neural_network(x)
        print(f"Output shape after the Inception block: {x.shape}")  # Print the shape to check H and W
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

    

learning_rate = 2.5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TrainModuleInception().to(device)
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


def train(data_loader, model, loss_function, optimizer):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        predict = model(X.to(device))
        loss = loss_function(predict, y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss}  [{current}/{size}]')


def test(data_loader, model):
    size = len(data_loader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            predict = model(X.to(device))
            test_loss += loss_fn(predict, y.to(device)).item()
            correct += (predict.argmax(1) == y.to(device)).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f'Accuracy: {(100 * correct)}%, Avg loss: {test_loss}')


epochs = 50
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    test(validation_dataloader, model)

y_pred = np.array([])
predictions = np.array([])
with torch.no_grad():
    for (X_, y_) in validation_dataloader:
        preds = model(X_.to(device)).cpu().numpy().argmax(1)
        labels = y_.cpu().numpy()

        predictions = np.concatenate((predictions, preds))
        y = np.concatenate((y_pred, labels))

# Confusion Matrix
cm = confusion_matrix(y, predictions)
disp = ConfusionMatrixDisplay(cm)
disp.plot()

plt.savefig(OUTPUT_DIR / 'confusion_matrix_inception.png')
print(classification_report(y, predictions, target_names=list(output_type.keys()))) 

torch.save(model, OUTPUT_DIR / 'inception_model.pth')
