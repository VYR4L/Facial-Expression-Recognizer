import time
from torch import nn
from PIL import Image
from pathlib import Path
from utils.focal_loss import FocalLoss
from torchvision.transforms import ToTensor, RandomResizedCrop, RandomHorizontalFlip, RandomRotation, GaussianBlur, RandomVerticalFlip
from torch.utils.data import DataLoader, Dataset
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

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
    

class YoloFullDN53(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Darknet53()
        self.flatten = nn.Flatten()  # Adicionar o flatten
        self.fc_layer = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)  # Usar self.flatten
        x = self.fc_layer(x)
        return x


def train(train_loader, model, loss_function, optimizer, device):
    size = len(train_loader.dataset)
    model.train()
    total_loss = 0
    correct = 0
    
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        outputs = model(X)
        loss = loss_function(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
        
        if batch % 100 == 0:
            loss_val = loss.item()
            current = (batch + 1) * len(X)
            # print(f'loss: {loss_val}  [{current}/{size}]')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / size
    return avg_loss, accuracy


def test(val_loader, model, loss_function, device):
    size = len(val_loader.dataset)
    num_batches = len(val_loader)
    test_loss, correct = 0, 0
    model.eval()
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_function(outputs, y)
            test_loss += loss.item()
            correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    return test_loss, correct


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Train YOLO Full DN53 on facial emotion dataset')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--criterion', type=str, default='cross_entropy_loss', 
                        choices=['cross_entropy_loss', 'focal_loss'], 
                        help='Loss function to use: cross_entropy_loss or focal_loss')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Balanceamento de Classes
    labels = [output_type[Path(path).parts[-2]] for path in training_data.path_file]
    counts = Counter(labels)
    total = sum(counts.values())
    class_weights = [total / counts[i] for i in range(len(output_type))]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = YoloFullDN53(num_classes=len(output_type)).to(device)
    
    # Escolha da função de perda
    if args.criterion == 'focal_loss':
        criterion = FocalLoss(alpha=1.0, gamma=2.0).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # Early Stopping
    best_loss = float('inf')
    patience = args.patience
    counter = 0
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    training_start_time = time.time()


    for t in range(args.num_epochs):
        # print(f'Epoch {t + 1}\n-------------------------------')
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
        val_loss, val_acc = test(validation_dataloader, model, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model_yolo_full.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break

    training_time = time.time() - training_start_time

    # Plot monitoring
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.savefig(OUTPUT_DIR / 'training_plots_yolo_full.png')

    eval_start = time.time()


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

    report = classification_report(y_true, y_pred, target_names=list(output_type.keys()))

    with open(OUTPUT_DIR / 'classification_report_full_yolo.txt', 'w') as f:
        f.write(report)

    torch.save(model.state_dict(), OUTPUT_DIR / 'model_yolo.pth')

    eval_time = time.time() - eval_start
    total_time = time.time() - start_time
    
    # Exibir tempos no terminal
    print("\n" + "="*50)
    print("TEMPO DE EXECUÇÃO - YOLO FULL DN53 MODEL")
    print("="*50)
    print(f"Critério usado: {args.criterion}")
    print(f"Tempo de treinamento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)")
    print(f"Tempo de avaliação: {eval_time:.2f} segundos")
    print(f"Tempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    print(f"Classification report salvo em: {OUTPUT_DIR / 'classification_report_full_yolo.txt'}")
    print("="*50)

if __name__ == '__main__':
    main()
