import glob
import torch
import argparse
import numpy as np
from torch import nn
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from utils.focal_loss import FocalLoss
from torchvision.transforms import ToTensor, RandomResizedCrop, RandomHorizontalFlip, RandomRotation, GaussianBlur, RandomVerticalFlip
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from utils.dataset import Data, output_type, OUTPUT_DIR


# Carregar os dados de treinamento e validação    
training_data = Data(is_train=True)

# Carregar os dados de validação
validation_data = Data(is_train=False)
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=True)


# Bloco residual usado na Darknet-53 (YOLOv3)
class ResidualBlock(nn.Module):
    '''
    Bloco residual usado na Darknet-53 (YOLOv3).
    Este bloco é composto por duas camadas convolucionais com normalização em lote e ativação LeakyReLU.
    Ele recebe um número de canais como parâmetro e aplica uma convolução 1x1
    seguida por uma convolução 3x3, mantendo a mesma dimensão de entrada e saída.
    A saída do bloco é a soma da entrada original e a saída do bloco convolucional.
    '''
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels // 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return x + self.block(x)


class Darknet53(nn.Module):
    '''
    Esta classe define a arquitetura Darknet-53 com blocos residuais,
    composta por várias camadas convolucionais seguidas por blocos residuais.
    A arquitetura é projetada para extrair características de imagens de entrada,
    mantendo a eficiência e a capacidade de detecção de objetos.
    '''
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1)
            )
        self.stage1 = self._make_stage(32, 64, 1)
        self.stage2 = self._make_stage(64, 128, 2)
        self.stage3 = self._make_stage(128, 256, 8)
        self.stage4 = self._make_stage(256, 512, 8)
        self.stage5 = self._make_stage(512, 1024, 4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        return x


class Yolo(nn.Module):
    '''
    Implementação da arquitetura Yolo inspirada no YOLOv3 (Darknet-53) para detecção de emoções faciais.
    '''
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = Darknet53()
        self.flatten = nn.Flatten()
        self.fc_layer = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        '''
        Método de passagem direta da rede neural Yolo com backbone Darknet-53 (YOLOv3).
        params:
            x (torch.Tensor): tensor de entrada com as imagens.
        returns:
            torch.Tensor: tensor de saída com as previsões de classes.
        '''
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc_layer(x)
        return x


class TrainYolo(nn.Module):
    '''
    Classe para treinar o modelo Yolo.
    Esta classe herda de nn.Module e implementa o método forward para treinar o modelo.
    '''
    def __init__(self, model, criterion, optimizer):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, x, y):
        '''
        Método de treinamento do modelo Yolo.
        params:
            x (torch.Tensor): tensor de entrada com as imagens.
            y (torch.Tensor): tensor de rótulos correspondentes.
        returns:
            torch.Tensor: perda calculada pelo critério.
        '''
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    

def train(train_loader, model, loss_function, optimizer, device):
    size = len(train_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        preds = model(X, y)
        loss = preds  # TrainYolo já retorna o loss
        if batch % 100 == 0:
            loss_val = loss.item()
            current = (batch + 1) * len(X)
            print(f'loss: {loss_val}  [{current}/{size}]')


def test(val_loader, model, loss_function, device):
    size = len(val_loader.dataset)
    num_batches = len(val_loader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model.model(X)  # model é TrainYolo, model.model é Yolo
            loss = loss_function(outputs, y)
            test_loss += loss.item()
            correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100 * correct):.2f}%, Avg loss: {test_loss:.4f}\n')


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv3 on facial emotion dataset')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--criterion', type=str, default='cross_entropy', choices=['cross_entropy', 'focal_loss'], help='Loss function to use')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Yolo(num_classes=len(output_type)).to(device)
    if args.criterion == 'focal_loss':
        criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean').to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_yolo = TrainYolo(model, criterion, optimizer).to(device)

    for t in range(args.num_epochs):
        print(f'Epoch {t + 1}\n-------------------------------')
        train(train_dataloader, train_yolo, criterion, optimizer, device)
        test(validation_dataloader, train_yolo, criterion, device)

    # Avaliação final e matriz de confusão
    y_true = np.array([])
    y_pred = np.array([])
    model.eval()
    with torch.no_grad():
        for X, y in validation_dataloader:
            X = X.to(device)
            outputs = model(X)
            preds = outputs.cpu().numpy().argmax(1)
            labels = y.cpu().numpy()
            y_pred = np.concatenate([y_pred, preds])
            y_true = np.concatenate([y_true, labels])

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(output_type.keys()))
    disp.plot()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix_yolo.png')
    print(classification_report(y_true, y_pred, target_names=list(output_type.keys())))

    torch.save(model.state_dict(), OUTPUT_DIR / 'model_yolo.pth')

if __name__ == '__main__':
    main()
