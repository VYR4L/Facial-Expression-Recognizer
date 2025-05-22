import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''
    Função de perda focal para lidar com o desbalanceamento de classes.
    A função de perda focal é uma modificação da função de perda cruzada
    que dá mais peso a exemplos difíceis e menos peso a exemplos fáceis.
    Isso é útil em tarefas de classificação onde algumas classes são
    muito mais frequentes do que outras.
    
    Params:
        alpha (float): Peso para a classe positiva.
        gamma (float): Fator de modulação.
        reduction (str): Método de redução da perda. Pode ser 'none', 'mean' ou 'sum'.

    Returns:
        loss (torch.Tensor): Valor da perda calculada.
    '''

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)

        focal_loss = -self.alpha * (1 - pt) ** self.gamma
        loss = focal_loss * torch.log(pt + 1e-6) 

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    