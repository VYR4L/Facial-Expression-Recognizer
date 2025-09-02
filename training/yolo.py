from torch import nn


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
                nn.Conv2d(1, 32, 3, 1, 1, bias=False),
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
            ResidualBlock(1024),
            nn.AdaptiveAvgPool2d((1, 1)),
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
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        '''
        Função de passagem para frente da classe TrainYolo.
        A entrada passa pela arquitetura Yolo e retorna a saída do modelo.
        params:
            x (Tensor): entrada da rede neural.
        returns:
            Tensor: saída da rede neural.
        '''
        x = self.model(x)
        return x
