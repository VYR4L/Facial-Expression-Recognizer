import torch


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