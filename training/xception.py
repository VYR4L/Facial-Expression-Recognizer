import glob
import torch
import argparse
import time
import numpy as np
from torch import nn
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from utils.dataset import Data, output_type, OUTPUT_DIR

# Carregar os dados de treinamento e validação    
training_data = Data(is_train=True)

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
        self.neural_network = nn.Sequential(
            # Entry Flow
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SeparableConv2d(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            SeparableConv2d(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SeparableConv2d(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Middle Flow (repetido 8x)
            *[layer for _ in range(8) for layer in [
                nn.ReLU(),
                SeparableConv2d(256, 256),
                nn.BatchNorm2d(256),
            ]],
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Exit Flow
            SeparableConv2d(256, 728),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeparableConv2d(728, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            SeparableConv2d(1024, 1536),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            SeparableConv2d(1536, 2048),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.neural_network(x)
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
    learning_rate = 2.5e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TrainXception().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # def train(data_loader, model, loss_function, optimizer):
    #     size = len(data_loader.dataset)
    #     model.train()
    #     for batch, (X, y) in enumerate(data_loader):
    #         predict = model(X.to(device))
    #         loss = loss_function(predict, y.to(device)) 

    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad() 

    #         if batch % 100 == 0:
    #             loss, current = loss.item(), (batch + 1) * len(X)
    #             # print(f'loss: {loss}  [{current}/{size}]')


def train(data_loader, model, loss_function, optimizer, device):
        '''
        Função para treinar o modelo.
        A função itera sobre os lotes de dados, calcula a perda e atualiza os pesos do modelo.
        A função também imprime a perda atual a cada 100 lotes.

        params:
            data_loader (DataLoader): carregador de dados para o conjunto de treinamento.
            model (nn.Module): modelo a ser treinado.
            loss_function (nn.Module): função de perda a ser usada.
            optimizer (torch.optim.Optimizer): otimizador a ser usado.
            device (torch.device): dispositivo a ser usado (CPU ou GPU).
        '''
        size = len(data_loader.dataset)
        model.train()
        total_loss = 0
        correct = 0
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            predict = model(X)
            loss = loss_function(predict, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (predict.argmax(1) == y).type(torch.float).sum().item()
            # if batch % 100 == 0:
            #     print(f'loss: {loss.item():.4f}  [{(batch + 1) * len(X)}/{size}]')
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / size
        return avg_loss, accuracy

def test(data_loader, model, loss_function, device):
    '''
    Função para testar o modelo.
    A função itera sobre os lotes de dados, calcula a perda e a precisão do modelo.
    A função também imprime a perda média e a precisão do modelo.

    params:
        data_loader (DataLoader): carregador de dados para o conjunto de validação.
        model (nn.Module): modelo a ser testado.
        loss_function (nn.Module): função de perda a ser usada.
        device (torch.device): dispositivo a ser usado (CPU ou GPU).
    '''
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            predict = model(X)
            loss = loss_function(predict, y)
            test_loss += loss.item()
            correct += (predict.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    # print(f'Test Error: \n Accuracy: {(100 * correct):.4f}%, Avg loss: {test_loss:.4f} \n')
    return test_loss, correct


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Xception Facial Expression Recognition')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4, help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--criterion', type=str, default='cross_entropy_loss', choices=['cross_entropy_loss'], help='Loss function to use')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrainXception().to(device)
    if args.criterion == 'cross_entropy_loss':
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss() # Placeholder for outros critérios
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, device)
        val_loss, val_acc = test(validation_dataloader, model, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model_xception.pth')
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping!')
                break

    training_time = time.time() - training_start_time

    # Plots de monitoramento
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
    plt.savefig(OUTPUT_DIR / 'training_plots_xception.png')

    eval_start = time.time()
    # Avaliação final
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
    plt.savefig(OUTPUT_DIR / 'confusion_matrix_xception.png')

    report = classification_report(y_true, y_pred, target_names=list(output_type.keys()))
    with open(OUTPUT_DIR / 'classification_report_xception.txt', 'w') as f:
        f.write(report)

    torch.save(model.state_dict(), OUTPUT_DIR / 'model_xception.pth')

    eval_time = time.time() - eval_start
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("TEMPO DE EXECUÇÃO - XCEPTION MODEL")
    print("="*50)
    print(f"Critério usado: {args.criterion}")
    print(f"Tempo de treinamento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)")
    print(f"Tempo de avaliação: {eval_time:.2f} segundos")
    print(f"Tempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    print(f"Classification report salvo em: {OUTPUT_DIR / 'classification_report_xception.txt'}")
    print("="*50)

if __name__ == '__main__':
    main()