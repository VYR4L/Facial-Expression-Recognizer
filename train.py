import torch


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