import glob
import torch
import numpy as np
import argparse
import time
from torch import nn
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from utils.dataset import Data, output_type, OUTPUT_DIR

# Carregar os dados de treinamento e validação    
training_data = Data(is_train=True)

# Carregar os dados de validação
validation_data = Data(is_train=False)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)


class ResidualCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
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
        return nn.Dropout(p=0.25)(nn.ReLU()(self.block(x) + self.resize(x)))
    

class TrainModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.neutral_network = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(p=0.25),

            ResidualCNN(64, 128),
            ResidualCNN(128, 256),
            ResidualCNN(256, 512),
            ResidualCNN(512, 512),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 7),
            nn.Softmax(dim=1),
            )

    def forward(self, x):
        return self.neutral_network(x)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TrainModule().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)


def train(data_loader, model, loss_function, optimizer, device):
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
        if batch % 100 == 0:
            print(f'loss: {loss.item():.4f}  [{(batch + 1) * len(X)}/{size}]')
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / size
    return avg_loss, accuracy


def test(data_loader, model, loss_function, device):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            predict = model(X.to(device))
            test_loss += loss_function(predict, y.to(device)).item()
            correct += (predict.argmax(1) == y.to(device)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100 * correct):.2f}%, Avg loss: {test_loss:.4f} \n')
    return test_loss, correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inception Facial Expression Recognition')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4, help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    args = parser.parse_args()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    start_time = time.time()
    training_start_time = time.time()

    for t in range(args.num_epochs):
        print(f'Epoch {t + 1}\n-------------------------------')
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, device)
        val_loss, val_acc = test(validation_dataloader, model, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model_inception.pth')
        else:
            counter += 1
            if counter >= args.patience:
                print("Early stopping!")
                break

    training_time = time.time() - training_start_time

    # Plots
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
    plt.savefig(OUTPUT_DIR / 'training_plots_inception.png')

    eval_start = time.time()
    # Avaliação final
    y = np.array([])
    predict = np.array([])
    model.eval()
    with torch.no_grad():
        for (X_, y_) in validation_dataloader:
            preds = model(X_.to(device)).cpu().numpy().argmax(1)
            labels = y_.cpu().numpy()
            predict = np.concatenate([predict, preds])
            y = np.concatenate([y, labels])
    cm = confusion_matrix(y, predict)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(output_type.keys()))
    disp.plot()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix_inception.png')
    report = classification_report(y, predict, target_names=list(output_type.keys()))
    with open(OUTPUT_DIR / 'classification_report_inception.txt', 'w') as f:
        f.write(report)
    torch.save(model.state_dict(), OUTPUT_DIR / 'model_inception.pth')
    eval_time = time.time() - eval_start
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("TEMPO DE EXECUÇÃO - INCEPTION MODEL")
    print("="*50)
    print(f"Tempo de treinamento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)")
    print(f"Tempo de avaliação: {eval_time:.2f} segundos")
    print(f"Tempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    print(f"Classification report salvo em: {OUTPUT_DIR / 'classification_report_inception.txt'}")
    print("="*50)