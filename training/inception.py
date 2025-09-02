from torch import nn


class InceptionModule(nn.Module):
    '''
    Módulo Inception conforme descrito no artigo "Going Deeper with Convolutions" (Szegedy et al., 2015).
    Este módulo aplica múltiplas convoluções paralelas com diferentes tamanhos de kernel e concatena os resultados.
    A arquitetura Inception é projetada para capturar características em múltiplas escalas, melhorando a capacidade
    de extração de características da rede neural.
    A classe herda de nn.Module do PyTorch e implementa os métodos __init__ e forward.
    '''
    def __init__(self, input_channels, output_channels):
        '''
        Inicializa a classe InceptionModule.
        params:
            input_channels (int): número de canais de entrada.
            output_channels (int): número de canais de saída para cada ramo convolucional.
        returns:
            None
        '''
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
        )

        self.resize = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x):
        '''
        Realiza a passagem da entrada pelo módulo Inception.
        Aplica as convoluções paralelas e concatena os resultados.
        params:
            x (Tensor): entrada da camada.
        returns:
            Tensor: saida da camada.
        '''
        return nn.Dropout(p=0.25)(nn.ReLU()(self.block(x) + self.resize(x)))
    

class TrainInception(nn.Module):
    '''
    Classe para treinar o modelo Inception.
    Esta classe herda de nn.Module e implementa o método forward para treinar o modelo.
    '''
    def __init__(self):
        '''
        Inicializa a classe TrainInception.
        Define a arquitetura da rede neural Inception para classificação de emoções faciais.
        A arquitetura é composta por múltiplos módulos Inception seguidos por camadas totalmente conectadas.
        params:
            None
        returns:
            None
        '''
        super().__init__()
        self.neutral_network = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(p=0.25),

            InceptionModule(64, 128),
            InceptionModule(128, 256),
            InceptionModule(256, 512),
            InceptionModule(512, 512),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 7),
            nn.Softmax(dim=1),
            )

    def forward(self, x):
        '''
        Função de passagem para frente da classe TrainInception.
        A entrada passa pela arquitetura Inception e pela camada totalmente conectada.
        A saída é a classificação final.
        params:
            x (Tensor): entrada da rede neural.
        returns:
            Tensor: saída da rede neural.
        '''
        return self.neutral_network(x)
