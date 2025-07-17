1. **Balanceamento de Classes**
   * **Usando pesos na função de perda:** Calcular os pesos das classes e passar para o CrossEntropyLoss:
    ```python
    from collections import Counter
    labels = [output_type[Path(path).parts[-2]] for path in training_data.path_file]
    counts = Counter(labels)
    total = sum(counts.values())
    class_weights = [total / counts[i] for i in range(len(output_type))]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    ```

2. **Aprimorar Arquitetura**
   *  **Adicionando Dropout:** Incluir camadas de Dropout após as ativações da fc_layer:
  ```python
    self.fc_layer = nn.Sequential(
        nn.Linear(1024 * 1 * 1, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),  # Dropout para regularização
        nn.Linear(1024, num_classes)
    )
  ```
   * **Aumentando a capacidade:**  Adicionar mais blocos residuais ou aumentar o número de filtros nas camadas convolucionais do backbone.

3. **Early Stopping e Regularização**
    * **Regularização L2 (weight_decay):** Adicione o parâmetro weight_decay ao otimizador:
    ```python
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    ```

    * **Early Stopping:** Implementar um controle simples no loop de treino:
    ```python
        # ...existing code...
    best_loss = float('inf')
    patience = 5
    counter = 0
    
    for t in range(args.num_epochs):
        # ...train and test...
        val_loss = ... # calcule a loss de validação
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break
    # ...existing code...
    ```

4. **Cálculo de over & under fitting**:
    ```python
        # ...existing code...
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for t in range(args.num_epochs):
        print(f'Epoch {t + 1}\n-------------------------------')
        train_loss, train_acc = train(train_dataloader, train_yolo, criterion, optimizer, device)
        val_loss, val_acc = test(validation_dataloader, train_yolo, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
    
    # Depois do treinamento, plote os gráficos:
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show()
    
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.show()
    ```