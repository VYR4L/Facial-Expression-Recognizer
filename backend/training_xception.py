import glob
import torch
import argparse
import numpy as np
from torch import nn
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation, Lambda
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from utils.focal_loss import FocalLoss


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
                                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                Lambda(lambda x: x + 0.02 * torch.randn_like(x)),
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

# Comentado para evitar prints desnecessários durante a execução do aplicativo final
# for key, val in output_type.items():
#     print(f'{key}: {count_type[val]}')

# Carregar os dados de validação
validation_data = Data(is_train=False)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)


class SeparableConv2d(nn.Module):
    '''
    Construção de um bloco de convolução separável, que é uma combinação de convolução de profundidade
    e convolução ponto a ponto.
    A convolução de profundidade aplica um filtro a cada canal de entrada separadamente,
    enquanto a convolução ponto a ponto combina os resultados da convolução de profundidade usando uma convolução 1x1.
    Isso reduz o número de parâmetros e melhora a eficiência computacional.

    A classe herda de nn.Module do PyTorch e implementa os métodos __init__ e forward.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        '''
        Inicializa a classe SeparableConv2d.
        Define os parâmetros de entrada e saída, o tamanho do kernel, o passo e o preenchimento.

        params:
            in_channels (int): número de canais de entrada.
            out_channels (int): número de canais de saída.
            kernel_size (int): tamanho do kernel.
            stride (int): passo da convolução.
            padding (int): preenchimento da convolução.
        '''
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        '''
        Realiza a passagem da entrada pela camada de convolução separável.
        Aplica a convolução de profundidade seguida pela convolução ponto a ponto.

        params:
            x (Tensor): entrada da camada.
        returns:
            Tensor: saida da camada.
        '''
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    

class XceptionCNN(nn.Module):
    '''
    Classe para a arquitetura XceptionCNN.
    A classe herda de nn.Module do PyTorch e implementa os métodos __init__ e forward.
    A arquitetura Xception é uma rede neural convolucional profunda que utiliza convoluções separáveis
    para melhorar a eficiência computacional e reduzir o número de parâmetros.

    A arquitetura é composta por três partes principais: Entry Flow, Middle Flow e Exit Flow.
    A Entry Flow é responsável por extrair características iniciais da imagem de entrada.
    O Middle Flow é responsável por refinar as características extraídas e o Exit Flow é responsável
    por gerar a saída final da rede.

    Hiperparâmetros:
        - Tamanho do kernel: 3x3
        - Tamanho da imagem de entrada: 48x48
        - Tamanho do passo: 1
        - Preenchimento: 1
        - Número de canais de entrada: 1 (imagem em escala de cinza)
        - Número de canais de saída: 7 (emoções)
        - Número de filtros: 32, 64, 128, 256, 728, 1024, 1536, 2048
        - Número de filtros na camada totalmente conectada: 2048
        - Número de filtros na camada totalmente conectada: 7 (emoções)
        - Função de ativação: ReLU
        - Função de perda: Focal Loss
        - Otimizador: Adam
        - Taxa de aprendizado: 2.5e-4
        - Número de épocas: 50
        - Tamanho do lote: 64
        - Tamanho do lote de validação: 64
        - Tamanho do lote de treinamento: 64
        - Tamanho do lote de teste: 64

    Número de camadas ocultas:
        - Entry Flow: 5 camadas convolucionais
        - Middle Flow: 8 blocos de convolução separável (16 camadas convolucionais)
        - Exit Flow: 4 camadas convolucionais

    Número de camadas totalmente conectadas:
        - 1 camada totalmente conectada

    Número de classes de saída:
        - 7 classes (emoções)


    '''
    def __init__(self, num_classes=7):
        super().__init__()
        
        # Entry Flow
        self.entry_flow = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),  # 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SeparableConv2d(64, 128),  # 3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1),
            SeparableConv2d(128, 256),  # 4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SeparableConv2d(256, 256),  # 5
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1),
        )
        
        # Middle Flow
        middle_flow = []
        for _ in range(8):  # Repetir o bloco 8 vezes
            middle_flow.extend([
                nn.ReLU(),
                SeparableConv2d(256, 256),  # 1
                nn.BatchNorm2d(256),
                nn.ReLU(),
                SeparableConv2d(256, 256),  # 2
                nn.BatchNorm2d(256),
            ])
        self.middle_flow = nn.Sequential(*middle_flow)
        
        # Exit Flow
        self.exit_flow = nn.Sequential(
            SeparableConv2d(256, 728),  # 1
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeparableConv2d(728, 1024),  # 2
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1),
            SeparableConv2d(1024, 1536), # 3
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            SeparableConv2d(1536, 2048),  # 4
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Fully Connected Layer
        self.fc = nn.Linear(2048, num_classes)  # 1

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class TrainXception(nn.Module):
    '''
    Classe para treinar a arquitetura XceptionCNN.
    A classe herda de nn.Module do PyTorch e implementa os métodos __init__ e forward.
    '''
    def __init__(self):
        '''
        Inicializa a classe TrainXception.
        Define a arquitetura da rede neural XceptionCNN e uma camada totalmente conectada.
        A camada totalmente conectada aplica a função Softmax à saída da rede neural.   
        '''
        super().__init__()
        self.neural_network = XceptionCNN()
        self.fc_layers = nn.Sequential(
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        '''
        Função de passagem para frente da classe TrainXception.
        A entrada passa pela arquitetura XceptionCNN e pela camada totalmente conectada.
        A saída é a classificação final.

        params:
            x (Tensor): entrada da rede neural.
        returns:
            Tensor: saída da rede neural.
        '''
        x = self.neural_network(x)
        x = self.fc_layers(x)
        return x


if __name__ == '__main__':
    # Passagem de argumentos para o script
    parser = argparse.ArgumentParser(description='Xception Convolutional Neural Network Training')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas para treinamento')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4, help='Taxa de aprendizado')
    args = parser.parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate

    # Xception Convolutional Neural Network Training
    # Definir o dispositivo para GPU se disponível, caso contrário, usar CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Definir o número de classes de saída
    model = TrainXception().to(device)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25).to(device) # Focal Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    def train(data_loader, model, loss_function, optimizer):
        '''
        Função para treinar o modelo.
        A função itera sobre os lotes de dados, calcula a perda e atualiza os pesos do modelo.
        A função também imprime a perda atual a cada 100 lotes.

        params:
            data_loader (DataLoader): carregador de dados para o conjunto de treinamento.
            model (nn.Module): modelo a ser treinado.
            loss_function (nn.Module): função de perda a ser usada.
            optimizer (torch.optim.Optimizer): otimizador a ser usado.
        '''
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
        '''
        Função para testar o modelo.
        A função itera sobre os lotes de dados, calcula a perda e a precisão do modelo.
        A função também imprime a perda média e a precisão do modelo.

        params:
            data_loader (DataLoader): carregador de dados para o conjunto de validação.
            model (nn.Module): modelo a ser testado.
            loss_function (nn.Module): função de perda a ser usada.
        '''
        size = len(data_loader.dataset)
        model.eval()
        size = len(data_loader.dataset)
        num_batches = len(data_loader)
        test_loss, correct = 0, 0 

        # Desativar o cálculo do gradiente para economizar memória e acelerar a computação
        # durante a avaliação
        with torch.no_grad():
            for X, y in data_loader:
                predict = model(X.to(device))
                test_loss += loss_function(predict, y.to(device)).item()
                correct += (predict.argmax(1) == y.to(device)).type(torch.float).sum().item() 

            test_loss /= num_batches
            correct /= size
            print(f'Test Error: \n Accuracy: {(100 * correct)}%, Avg loss: {test_loss} \n')

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
    plt.savefig(ROOT_DIR / 'confusion_matrix_xception.png')
    print(classification_report(y, predict, target_names=list(output_type.keys()))) 

    torch.save(model.state_dict(), ROOT_DIR / 'model_xception.pth')