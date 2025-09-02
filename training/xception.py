from torch import nn


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
    '''
    def __init__(self, num_classes=7):
        '''
        Inicializa a classe XceptionCNN.
        Define a arquitetura da rede neural XceptionCNN e uma camada totalmente conectada.
        A camada totalmente conectada aplica a função Softmax à saída da rede neural.
        params:
            num_classes (int): número de classes de saída.
        returns:
            None
        '''
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
        '''
        Função de passagem para frente da classe XceptionCNN.
        A entrada passa pela arquitetura XceptionCNN e pela camada totalmente conectada.
        A saída é a classificação final.
        params:
            x (Tensor): entrada da rede neural.
        returns:
            Tensor: saída da rede neural.
        '''
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
