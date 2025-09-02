import time
import torch
import argparse
import numpy as np
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from utils.focal_loss import FocalLoss
from utils.dataset import Data, output_type, OUTPUT_DIR


class Conv(nn.Module):
    """
    Bloco Convolução padrão
    Este bloco realiza uma convolução 2D seguida de normalização em lote e
    uma função de ativação SiLU (Swish).

    A clase herda de nn.Module do PyTorch e implementa os métodos __init__ e forward.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Inicializa o bloco Conv.
        params:
            c1 (int): número de canais de entrada.
            c2 (int): número de canais de saída.
            k (int): tamanho do kernel da convolução.
            s (int): stride da convolução.
            p (int): padding da convolução. Se None, será calculado automaticamente.
            g (int): número de grupos na convolução.
            d (int): dilatação da convolução.
            act (bool ou nn.Module): se True, usa SiLU; se False, não usa ativação; se nn.Module, usa a ativação fornecida.
        returns:
            None
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        '''
        Realiza a passagem para frente do bloco Conv.
        params:
            x (Tensor): entrada da rede neural.
        returns:
            Tensor: saída da rede neural.
        '''
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        '''
        Realiza a passagem para frente do bloco Conv sem a normalização em lote.
        Usado para inferência após a fusão da normalização em lote.
        params:
            x (Tensor): entrada da rede neural. 
        returns:
            Tensor: saída da rede neural.
        '''
        return self.act(self.conv(x))

    @staticmethod
    def autopad(k, p=None, d=1):
        """Calcula o padding automático para manter o tamanho da entrada.
        params:
            k (int): tamanho do kernel.
            p (int): padding. Se None, será calculado automaticamente.
            d (int): dilatação.
        returns:
            int: padding calculado.
        """
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p


class Bottleneck(nn.Module):
    """
    Boloco Bottleneck padrão
    Este bloco implementa um bloco residual com duas convoluções.
    O bloco pode incluir uma conexão de atalho (skip connection) se o número de canais de entrada
    for igual ao número de canais de saída e o parâmetro shortcut for True.
    params:
        c1 (int): número de canais de entrada.
        c2 (int): número de canais de saída.
        shortcut (bool): se True, utiliza conexão de atalho.
        g (int): número de grupos na convolução.
        k (tuple): tamanho dos kernels das convoluções.
        e (float): fator de expansão para calcular os canais ocultos.
    returns:
        None
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # canais ocultos
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Bloco C2f da YOLOv8/v11 - Cross Stage Partial Bottleneck com 2 convolutions"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # canais ocultos
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2(nn.Module):
    """
    YOLOv11 C3k2 block simplificado
    Este bloco é uma variação do bloco C3, utilizando convoluções 1x1 para reduzir a dimensionalidade
    e convoluções 3x3 para extração de características.
    A estrutura do bloco é composta por duas ramificações que processam a entrada de forma paralela,
    seguidas por uma concatenação e uma convolução final para combinar as características extraídas.
    params:
        c1 (int): número de canais de entrada.
        c2 (int): número de canais de saída.
        n (int): número de blocos Bottleneck a serem empilhados na primeira ram
        shortcut (bool): se True, utiliza conexões de atalho nos blocos Bottleneck.
        g (int): número de grupos na convolução.
        e (float): fator de expansão para calcular os canais ocultos.

    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.ModuleList([Bottleneck(c_, c_, shortcut, g) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        for m in self.m:
            y1 = m(y1)
        return self.cv3(torch.cat((y1, y2), 1))


class C2PSA(nn.Module):
    """
    YOLOv11 C2PSA block simplificado
    Este bloco é uma variação do bloco C2f, incorporando um mecanismo de atenção espacial para melhorar
    a capacidade de extração de características.
    A estrutura do bloco é composta por duas ramificações que processam a entrada de forma paralela,
    seguidas por uma concatenação e uma convolução final para combinar as características extraídas.
    O mecanismo de atenção espacial é aplicado a uma das ramificações para enfatizar as regiões
    mais relevantes da imagem.
    params:
        c1 (int): número de canais de entrada.
        c2 (int): número de canais de saída.
        n (int): número de blocos Bottleneck a serem empilhados na primeira ramificação.
        shortcut (bool): se True, utiliza conexões de atalho nos blocos Bottleneck.
        g (int): número de grupos na convolução.
        e (float): fator de expansão para calcular os canais ocultos.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.attn = nn.Sequential(
            nn.Conv2d(c_, c_, 1),
            nn.Sigmoid()
        )
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.ModuleList([Bottleneck(c_, c_, shortcut, g) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        attn = self.attn(y1)
        y1 = y1 * attn
        for m in self.m:
            y1 = m(y1)
        return self.cv3(torch.cat((y1, y2), 1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer
    Este bloco implementa o mecanismo de pooling espacial piramidal rápido,
    conforme descrito na arquitetura YOLOv11.
    O bloco utiliza múltiplas operações de max pooling com diferentes tamanhos de kernel
    para capturar informações em diferentes escalas, seguido por convoluções para combinar
    as características extraídas.
    params:
        c1 (int): número de canais de entrada.
        c2 (int): número de canais de saída.
        k (int): tamanho do kernel para as operações de max pooling.
    returns:
        None
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # canais ocultos
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class YOLOv11Backbone(nn.Module):
    """
    Backbone da YOLOv11 com C3k2 e C2PSA para classificação
    Esta classe define a arquitetura backbone da YOLOv11, utilizando blocos C3k2 e C2PSA
    para extração de características. A arquitetura é composta por várias camadas convolucionais seguidas
    por blocos residuais, culminando em uma camada de pooling adaptativo.
    params:
        ch (int): número de canais de entrada (1 para grayscale, 3 para RGB).
        nc (int): número de classes para classificação.
    returns:
        None
    """
    def __init__(self, ch=1, nc=7):
        super().__init__()
        self.stem = Conv(ch, 32, 3, 2)
        self.stage1 = nn.Sequential(
            Conv(32, 64, 3, 2),
            C3k2(64, 64, 3, True)
        )
        self.stage2 = nn.Sequential(
            Conv(64, 128, 3, 2),
            C3k2(128, 128, 6, True)
        )
        self.stage3 = nn.Sequential(
            Conv(128, 256, 3, 2),
            C3k2(256, 256, 6, True)
        )
        self.stage4 = nn.Sequential(
            Conv(256, 512, 3, 2),
            C2PSA(512, 512, 3, True), 
            SPPF(512, 512, 5)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        return x


class YOLOv11Classifier(nn.Module):
    """
    YOLOv11 para classificação de emoções faciais
    Esta classe define a arquitetura completa da YOLOv11 para a tarefa de classificação de emoções faciais.
    A arquitetura utiliza a YOLOv11Backbone para extração de características, seguida por uma camada de flatten
    e um classificador totalmente conectado com dropout para evitar overfitting.
    
    returns:
    """
    def __init__(self, num_classes=7, ch=1):
        super().__init__()
        self.backbone = YOLOv11Backbone(ch=ch, nc=num_classes)
        self.flatten = nn.Flatten()
        
        # Classificador com dropout
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.SiLU(),
            # nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# Carregar dados
training_data = Data(is_train=True)
validation_data = Data(is_train=False)
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=128, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=128, shuffle=True)


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
            print(f'loss: {loss_val:.4f}  [{current}/{size}]')
    
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

    print(f'Test Error: \n Accuracy: {(100 * correct):.4f}%, Avg loss: {test_loss:.4f} \n')

    return test_loss, correct


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='YOLOv11 Facial Expression Recognition')
    parser.add_argument('--num_epochs', type=int, default=250, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=15 , help='Early stopping patience')
    parser.add_argument('--criterion', type=str, default='cross_entropy_loss', 
                        choices=['cross_entropy_loss', 'focal_loss'], 
                        help='Loss function to use: cross_entropy_loss or focal_loss')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Balanceamento de Classes
    labels = [output_type[Path(path).parts[-2]] for path in training_data.path_file]
    counts = Counter(labels)
    total = sum(counts.values())
    class_weights = [total / counts[i] for i in range(len(output_type))]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = YOLOv11Classifier(num_classes=len(output_type), ch=1).to(device)
    
    # Escolha da função de perda
    if args.criterion == 'focal_loss':
        criterion = FocalLoss(alpha=1.0, gamma=2.0).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    
    # Otimizador com configurações da YOLOv11
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

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
        print(f'Epoch {t + 1}\n-------------------------------')
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
        val_loss, val_acc = test(validation_dataloader, model, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model_yolov11.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
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
    plt.savefig(OUTPUT_DIR / 'training_plots_yolov11.png')

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
    plt.savefig(OUTPUT_DIR / 'confusion_matrix_yolov11.png')

    report = classification_report(y_true, y_pred, target_names=list(output_type.keys()))

    # Salvar classification report
    with open(OUTPUT_DIR / 'classification_report_yolov11.txt', 'w') as f:
        f.write(report)

    torch.save(model.state_dict(), OUTPUT_DIR / 'model_yolov11.pth')

    eval_time = time.time() - eval_start
    total_time = time.time() - start_time
    
    # Exibir tempos no terminal
    print("\n" + "="*50)
    print("TEMPO DE EXECUÇÃO - YOLOv11 MODEL")
    print("="*50)
    print(f"Critério usado: {args.criterion}")
    print(f"Tempo de treinamento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)")
    print(f"Tempo de avaliação: {eval_time:.2f} segundos")
    print(f"Tempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    print(f"Classification report salvo em: {OUTPUT_DIR / 'classification_report_yolov11.txt'}")
    print("="*50)


if __name__ == '__main__':
    main()