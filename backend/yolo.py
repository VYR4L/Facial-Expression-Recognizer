import glob
import torch
import argparse
from torch import nn
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Lambda, RandomVerticalFlip
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


# Definindo o diretório raiz e os caminhos para os conjuntos de dados de treinamento e validação
ROOT_DIR = Path(__file__).parent.parent
DATA_SET = ROOT_DIR / 'dataset' / 'images'
TRAIN_IMAGES = ROOT_DIR / 'dataset' / 'images' / 'train'
VALIDATION_IMAGES = ROOT_DIR / 'dataset' / 'images' / 'validation'

# Definindo o caminho para o conjunto de dados
output_type = {}
count_type = [0 for i in range(7)]

# Definindo os tipos de emoções e suas respectivas contagens
for index, val in enumerate(glob.glob(str(TRAIN_IMAGES / '*'))):
    output_type[Path(val).parts[-1]] = index


class Data(Dataset):
    '''
    Classe para carregar os dados de treinamento e validação.

    A classe herda de Dataset do PyTorch e implementa os métodos __len__ e __getitem__.
    '''
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
                                RandomRotation(10),
                                RandomVerticalFlip(),
        )

    def __len__(self):
        '''
        Retorna o número total de imagens no conjunto de dados.
        O número total de imagens é obtido contando o número de arquivos de imagem no diretório especificado.
        '''
        return len(self.path_file)

    def __getitem__(self, idx):
        '''
        Retorna uma imagem e seu rótulo correspondente com base no índice fornecido.
        O método carrega a imagem correspondente ao índice, aplica as transformações necessárias e retorna a imagem e seu rótulo.

        params:
            idx (int): índice da imagem a ser carregada.

        returns:
            tuple: imagem transformada e rótulo correspondente.
        '''
        image_path = self.path_file[idx]
        image = Image.open(image_path)
        label = output_type[Path(image_path).parts[-2]]
        image = self.transform(image)
        if self.is_train:
            image = self.data(image)
        
        return (image, label)
    
# Carregar os dados de treinamento e validação    
training_data = Data(is_train=True)

# Carregar os dados de validação
validation_data = Data(is_train=False)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)


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


class Darknet53Small(nn.Module):
    '''
    Darknet-53 simplificado para imagens pequenas (YOLOv3 backbone).
    Esta classe define a arquitetura Darknet-53 com blocos residuais,
    composta por várias camadas convolucionais seguidas por blocos residuais.
    A arquitetura é projetada para extrair características de imagens de entrada,
    mantendo a eficiência e a capacidade de detecção de objetos.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1)
            ),
            ResidualBlock(64),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1)
            ),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1)
            ),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Sequential(
                nn.Conv2d(256, 512, 3, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1)
            ),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Sequential(
                nn.Conv2d(512, 1024, 3, 2, 1, bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1)
            ),
            ResidualBlock(1024),
            ResidualBlock(1024)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Yolo(nn.Module):
    '''
    Implementação da arquitetura Yolo inspirada no YOLOv3 (Darknet-53) para detecção de emoções faciais.
    '''
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = Darknet53Small()
        self.flatten = nn.Flatten()
        self.fc_layer = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 1024),
            nn.ReLU(),
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
    

def train(model, train_loader, criterion, optimizer, num_epochs=10):
    '''
    Função para treinar o modelo Yolo.
    params:
        model (nn.Module): modelo Yolo a ser treinado.
        train_loader (DataLoader): DataLoader com os dados de treinamento.
        criterion (nn.Module): função de perda a ser usada.
        optimizer (torch.optim.Optimizer): otimizador a ser usado.
        num_epochs (int): número de épocas para treinamento.
    '''
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)
    optimizer.to(device)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = model(images, labels)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


def evaluate(model, validation_loader):
    '''
    Função para avaliar o modelo Yolo.
    params:
        model (nn.Module): modelo Yolo a ser avaliado.
        validation_loader (DataLoader): DataLoader com os dados de validação.
    '''
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=list(output_type.keys())))
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(output_type.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv3 on facial emotion dataset')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()

    model = Yolo(num_classes=len(output_type))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_yolo = TrainYolo(model, criterion, optimizer)

    train(train_yolo, train_dataloader, criterion, optimizer, num_epochs=args.num_epochs)
    evaluate(model, validation_dataloader)

