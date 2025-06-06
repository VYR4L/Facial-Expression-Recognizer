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


ROOT_DIR = Path(__file__).parent.parent
DATA_SET = ROOT_DIR / 'dataset' / 'images'
TRAIN_IMAGES = ROOT_DIR / 'dataset' / 'images' / 'train'
VALIDATION_IMAGES = ROOT_DIR / 'dataset' / 'images' / 'validation'

output_type = {}
count_type = [0 for i in range(7)]

for index, val in enumerate(glob.glob(str(TRAIN_IMAGES / '*'))):
    output_type[Path(val).parts[-1]] = index


class Data(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        data_type = 'train' if is_train else 'validation'
        
        self.path_file = glob.glob(str(DATA_SET / data_type / '*' / '*'))

        # Inbalanceing dataset:
        if is_train:
            temp_data = []
            for i in range(9):
                self.path_file += glob.glob(str(DATA_SET / data_type / 'disgust' / '*'))
            
            for type in output_type:
                counter = 0
                for path in self.path_file:
                    path_parts = Path(path).parts
                    if len(path_parts) > 6 and path_parts[-2] == type:
                        counter += 1
                        temp_data.append(path)
                        count_type[output_type[type]] += 1
            self.path_file = temp_data

        self.transform = ToTensor()
        self.data = nn.Sequential(
                                RandomResizedCrop((48, 48),
                                    scale=(0.8, 1),
                                    ratio=(0.5, 1)),
                                RandomHorizontalFlip(),
                                RandomVerticalFlip(),
        )

    def __len__(self):
        return len(self.path_file)

    def __getitem__(self, idx):
        image_path = self.path_file[idx]
        image = Image.open(image_path)
        label = output_type[Path(image_path).parts[-2]]
        image = self.transform(image)
        if self.is_train:
            image = self.data(image)
        
        return (image, label)
    

training_data = Data(is_train=True)

for key, val in output_type.items():
    print(f'{key}: {count_type[val]}')

validation_data = Data(is_train=False)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)


# Residual Convolutional Neural Network (ResNet)
class ResidualCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(),  # Trying to improve accuracy
            nn.Dropout(p=0.25),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
        )

        self.resize = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x):
        return nn.Dropout(p=0.25)(nn.ReLU()(self.block(x) + self.resize(x)))
    



# ResNet Convolutional Neural Network Training
class TrainModuleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_network = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(p=0.25),

            ResidualCNN(64, 128),
            ResidualCNN(128, 256),
            ResidualCNN(256, 512),
            ResidualCNN(512, 512),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.neural_network(x)
        x = self.fc_layers(x)
        return x



    


learning_rate = 2.5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models
resnet_model = TrainModuleCNN().to(device)

model = TrainModuleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(data_loader, model, loss_function, optimizer):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        predict = model(X.to(device))
        loss = loss_function(predict, y.to(device)) 

        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss}  [{current}/{size}]')


def test(data_loader, model, loss_function):
    size = len(data_loader.dataset)
    model.eval()
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    test_loss, correct = 0, 0 

    with torch.no_grad():
        for X, y in data_loader:
            predict = model(X.to(device))
            test_loss += loss_function(predict, y.to(device)).item()
            correct += (predict.argmax(1) == y.to(device)).type(torch.float).sum().item() 

        test_loss /= num_batches
        correct /= size
        print(f'Test Error: \n Accuracy: {(100 * correct)}%, Avg loss: {test_loss} \n')


epochs = 50
for t in range(epochs):
    print(f'Epoch {t + 1}\n-------------------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    test(validation_dataloader, model, loss_fn)


y = np.array([])
predict = np.array([])
with torch.no_grad():
    for (X_, y_) in validation_dataloader:
        preds = model(X_.to(device)).cpu().numpy().argmax(1)
        labels = y_.cpu().numpy() 

        predict = np.concatenate([predict, preds])
        y = np.concatenate([y, labels]) 

# Matriz de confusão
cm = confusion_matrix(y, predict)
disp = ConfusionMatrixDisplay(cm)
disp.plot() 

# Salvar a matriz de confusão em um arquivo
plt.savefig(ROOT_DIR / 'confusion_matrix.png')
print(classification_report(y, predict, target_names=list(output_type.keys()))) 

torch.save(model, ROOT_DIR / 'model.pth')
