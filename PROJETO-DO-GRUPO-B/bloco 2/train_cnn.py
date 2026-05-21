import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
import os

# ==============================================================================
# DEFINIÇÃO DA CNN COM PYTORCH
# Conv2d + BatchNorm + MaxPool (conforme especificação do Bloco 2)
# ==============================================================================
class FastCNN(nn.Module):
    def __init__(self):
        super(FastCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 28x28 -> 14x14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 14x14 -> 7x7
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    print("=" * 60)
    print("  BLOCO 2 - CNN (Conv2d+BatchNorm+MaxPool) com PyTorch")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. CARREGAR DADOS (do bloco 1)
    # ------------------------------------------------------------------
    data_path = os.path.join('..', 'bloco 1', 'fashion-mnist_test.csv')
    if not os.path.exists(data_path):
        data_path = 'fashion-mnist_test.csv'
    print(f"\n[1/6] Carregando dados de: {data_path}")
    data = pd.read_csv(data_path)

    y = data['label'].values
    X = data.drop('label', axis=1).values

    # ------------------------------------------------------------------
    # 2. PRÉ-PROCESSAMENTO
    # ------------------------------------------------------------------
    print("[2/6] Normalizando e reshaping para (N, 1, 28, 28)...")
    X = (X / 255.0).reshape(-1, 1, 28, 28)

    # ------------------------------------------------------------------
    # 3. DIVISÃO: 70% Treino, 15% Validação, 15% Teste
    # ------------------------------------------------------------------
    print("[3/6] Dividindo dados...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    print(f"       Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")

    # Tensores
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val,   dtype=torch.float32), torch.tensor(y_val,   dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(X_test,  dtype=torch.float32), torch.tensor(y_test,  dtype=torch.long))

    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # ------------------------------------------------------------------
    # 4. MODELO E HIPERPARÂMETROS
    # ------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[4/6] Device: {device}")
    model = FastCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    epochs = 10  # Treino rápido (poucos segundos)

    # ------------------------------------------------------------------
    # 5. TREINO COM MLFLOW
    # ------------------------------------------------------------------
    # Garante que o MLflow salva tudo DENTRO desta pasta
    mlflow.set_tracking_uri("file:///" + os.path.abspath("mlruns").replace("\\", "/"))
    mlflow.set_experiment("Fashion_MNIST_CNN")

    with mlflow.start_run(run_name="CNN_bloco2_rapida"):
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 0.002)
        mlflow.log_param("architecture", "Conv2d+BatchNorm+MaxPool x2 -> FC128 -> 10")

        print(f"[5/6] Treinando por {epochs} épocas...\n")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            # Validação rápida
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    preds = model(inputs).argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
            val_acc = val_correct / len(val_loader.dataset)

            print(f"  Época {epoch+1:2d}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc*100:.1f}%")
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # --------------------------------------------------------------
        # 6. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
        # --------------------------------------------------------------
        print(f"\n[6/6] Avaliando no conjunto de teste ({len(test_ds)} amostras)...")
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='weighted')
        rec  = recall_score(all_labels, all_preds, average='weighted')
        f1   = f1_score(all_labels, all_preds, average='weighted')

        mlflow.log_metric("test_accuracy",  acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall",    rec)
        mlflow.log_metric("test_f1",        f1)

        print("\n" + "=" * 50)
        print("  MÉTRICAS DE EFICIÊNCIA (TESTE)")
        print("=" * 50)
        print(f"  1. Acurácia  : {acc*100:.2f}%")
        print(f"  2. Precisão  : {prec*100:.2f}%")
        print(f"  3. Revocação : {rec*100:.2f}%")
        print(f"  4. F1-Score  : {f1*100:.2f}%")
        print("=" * 50)

        # Classification Report -> arquivo
        report = classification_report(all_labels, all_preds)
        report_path = 'classification_report_cnn.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BLOCO 2 - CNN (Conv2d+BatchNorm+MaxPool) com PyTorch\n")
            f.write("=" * 50 + "\n\n")
            f.write("MÉTRICAS DE EFICIÊNCIA\n")
            f.write("-" * 50 + "\n")
            f.write(f"Acurácia  : {acc*100:.2f}%\n")
            f.write(f"Precisão  : {prec*100:.2f}%\n")
            f.write(f"Revocação : {rec*100:.2f}%\n")
            f.write(f"F1-Score  : {f1*100:.2f}%\n")
            f.write("-" * 50 + "\n\n")
            f.write("Relatório por Classe:\n")
            f.write(report)
        print(f"\n  -> Relatório salvo: {report_path}")

        # Confusion Matrix -> imagem
        cm = confusion_matrix(all_labels, all_preds)
        class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusão - CNN (PyTorch) - Bloco 2')
        plt.xlabel('Previsão')
        plt.ylabel('Real')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        cm_path = 'confusion_matrix_cnn.png'
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"  -> Matriz salva: {cm_path}")

        # Log artefatos no MLflow
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)
        mlflow.pytorch.log_model(model, "pytorch_cnn_model")

        print("\n  -> Modelo e artefatos registrados no MLflow!")
        print("\nConcluído com sucesso!")

if __name__ == '__main__':
    main()
