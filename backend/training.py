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
from sklearn.model_selection import train_test_split
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


# Visual Geometry Group Convolutional Neural Network
class VGGCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.neural_network = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.neural_network(x)


# Xception Block (used to build the Xception Convolutional Neural Network)
class SeparableConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding, groups=input_channels)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class XceptionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, repetitions, stride=1):
        super(XceptionBlock, self).__init__()

        layers = []
        for i in range(repetitions):
            # First layer keeps the stride, other ones use stride 1
            if i == 0:
                layers.append(SeparableConv2d(input_channels, output_channels, stride=stride))
            else:
                layers.append(SeparableConv2d(output_channels, output_channels))
            layers.append(nn.BatchNorm2d(output_channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)

        # Shortcut to match the number of channels
        self.shortcut = nn.Sequential()
        if input_channels != output_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size= 1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.block(x)
        out += shortcut
        return out

# Xception Convolutional Neural Network
class XceptionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_network = nn.Sequential(
            # Entry Flow
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),

            XceptionBlock(64, 128, 2),
            XceptionBlock(128, 256, 2),
            XceptionBlock(256, 728, 2),

            # Middle Flow (repeating XceptionBlock)
            *[XceptionBlock(728, 728, 3) for _ in range(8)],

            # Exit Flow
            XceptionBlock(728, 1024, 2),
            nn.Conv2d(1024, 1536, 3, stride=2, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            nn.Conv2d(1536, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 7)
        )

    def forward(self, x):
        return self.neural_network(x)


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
            nn.Flatten(),
            nn.Linear(512, 7),
            nn.Softmax(dim=1),
            )

    def forward(self, x):
        x = self.neural_network(x)
        x = self.fc_layers(x)
        return x
    

# VGG Convolutional Neural Network Training
class TrainVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_network = nn.Sequential(
            VGGCNN(),
            nn.Linear(7, 7),
            nn.Softmax(dim=1),
            )

    def forward(self, x):
        return self.neural_network(x)
    

# Xception Convolutional Neural Network Training
class TrainXception(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_network = XceptionCNN()

    def forward(self, x):
        return self.neural_network(x)
    

learning_rate = 2.5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models
resnet_model = TrainModuleCNN().to(device)
vgg_model = TrainVGG().to(device)
xception_model = TrainXception().to(device)

list_models = [resnet_model, vgg_model, xception_model]

loss_fn = nn.CrossEntropyLoss()

# Training function
def train(data_loader, list_models, loss_function, learning_rate):
    size = len(data_loader.dataset)
    
    for model in list_models:
        model.train()
        
        # Create an optimizer to current model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            predictions = model(X)
            loss = loss_function(predictions, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss_value, current = loss.item(), (batch + 1) * len(X)
                print(f'Model: {model.__class__.__name__}, Loss: {loss_value:.4f}  [{current}/{size}]')


def test(data_loader, list_models, loss_function):
    for model in list_models:
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


def training_module():
    # Training
    epochs = 50
    for t in range(epochs):
        print(f'Epoch {t + 1}\n-------------------------------')
        
        for model in list_models:
            # Training
            train(train_dataloader, [model], loss_fn, learning_rate)
            # Validation
            test(validation_dataloader, [model], loss_fn)
        
    # Save metrics and models
    for model in list_models:
        print(f'Saving metrics and model for {model.__class__.__name__}...')

        # Confusion Matrix
        y_true = np.array([])
        y_pred = np.array([])

        with torch.no_grad():
            for (X_, y_) in validation_dataloader:
                preds = model(X_.to(device)).cpu().numpy().argmax(1)
                labels = y_.cpu().numpy()

                y_pred = np.concatenate([y_pred, preds])
                y_true = np.concatenate([y_true, labels])

        # Create and save the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.savefig(ROOT_DIR / f'confusion_matrix_{model.__class__.__name__}.png')

        # Create and save the classification report
        with open(f'{ROOT_DIR}/classification_report_{model.__class__.__name__}.txt', 'w') as f:
            f.write(classification_report(y_true, y_pred, target_names=list(output_type.keys())))

        # Save trained model
        torch.save(model.state_dict(), ROOT_DIR / f'{model.__class__.__name__}_model.pth')
