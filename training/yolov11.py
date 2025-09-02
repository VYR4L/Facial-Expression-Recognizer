import torch
from torch import nn

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
