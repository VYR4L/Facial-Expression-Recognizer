import argparse
import time
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from training.utils.dataset import Data, output_type, OUTPUT_DIR
from train import train
from test import test
from training.xception import TrainXception
from training.yolov11 import YOLOv11Classifier
from training.inception import TrainInception
from training.yolo import TrainYolo
from pathlib import Path
from collections import Counter




def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Facial Expression Recognition')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4, help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--architecture', type=str, default='inception', choices=['xception', 'inception', 'yolo', 'yolov11'], help='Model architecture to use')
    parser.add_argument('--criterion', type=str, default='cross_entropy_loss', choices=['cross_entropy_loss'], help='Loss function to use')
    parser.add_argument('--balance_classes', action='store_true', help='Whether to balance classes during training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    args = parser.parse_args()

    training_data = Data(is_train=True)

    validation_data = Data(is_train=False)
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labels = [output_type[Path(path).parts[-2]] for path in training_data.path_file]
    counts = Counter(labels)
    total = sum(counts.values())
    class_weights = [total / counts[i] for i in range(len(output_type))]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    match args.architecture:
        case 'xception':
            if args.balance_classes:
                model = TrainXception(num_classes=len(output_type), in_channels=1, class_weights=class_weights).to(device)
            else:
                model = TrainXception(num_classes=len(output_type), in_channels=1).to(device)
        case 'inception':
            if args.balance_classes:
                model = TrainInception(num_classes=len(output_type), in_channels=1, class_weights=class_weights).to(device)
            else:
                model = TrainInception(num_classes=len(output_type), in_channels=1).to(device)
        case 'yolo':
            if args.balance_classes:
                model = TrainYolo(num_classes=len(output_type), ch=1, class_weights=class_weights).to(device)
            else:
                model = TrainYolo(num_classes=len(output_type), ch=1).to(device)
        case 'yolov11':
            if args.balance_classes:
                model = YOLOv11Classifier(num_classes=len(output_type), ch=1, class_weights=class_weights).to(device)
            else:
                model = YOLOv11Classifier(num_classes=len(output_type), ch=1).to(device)

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