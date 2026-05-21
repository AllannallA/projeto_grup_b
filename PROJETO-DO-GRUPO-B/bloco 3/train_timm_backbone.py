import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
import timm


def resolve_data_path() -> str:
    data_path = os.path.join("..", "bloco 1", "fashion-mnist_test.csv")
    if os.path.exists(data_path):
        return data_path
    return "fashion-mnist_test.csv"


def preprocess_batch(inputs: torch.Tensor) -> torch.Tensor:
    # inputs: (B, 1, 28, 28) -> (B, 3, 224, 224) for EfficientNet-B0
    if inputs.ndim != 4 or inputs.size(1) != 1:
        raise ValueError(f"Esperado (B, 1, H, W). Recebido: {tuple(inputs.shape)}")
    inputs = inputs.repeat(1, 3, 1, 1)
    inputs = F.interpolate(inputs, size=(224, 224), mode="bilinear", align_corners=False)
    return inputs


def main():
    print("=" * 70)
    print("  BLOCO 3 - Backbones via timm | EfficientNet-B0 (treino rápido)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. CARREGAR DADOS (do bloco 1)
    # ------------------------------------------------------------------
    data_path = resolve_data_path()
    print(f"\n[1/6] Carregando dados de: {data_path}")
    data = pd.read_csv(data_path)

    y = data["label"].values
    X = data.drop("label", axis=1).values

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

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # ------------------------------------------------------------------
    # 4. MODELO E HIPERPARÂMETROS
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[4/6] Device: {device}")

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=10,
        in_chans=3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    epochs = 3

    # ------------------------------------------------------------------
    # 5. TREINO COM MLFLOW
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri("file:///" + os.path.abspath("mlruns").replace("\\", "/"))
    mlflow.set_experiment("Fashion_MNIST_TIMM_Backbones")

    with mlflow.start_run(run_name="EfficientNetB0_bloco3_rapido"):
        mlflow.log_param("backbone", "timm.efficientnet_b0")
        mlflow.log_param("pretrained", True)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 0.0005)
        mlflow.log_param("input_resize", "1x28x28 -> 3x224x224 (repeat+bilinear)")

        print(f"[5/6] Treinando por {epochs} épocas...\n")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = preprocess_batch(inputs)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            model.eval()
            val_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = preprocess_batch(inputs)
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
                inputs = preprocess_batch(inputs)
                preds = model(inputs).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="weighted")
        rec = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1", f1)

        print("\n" + "=" * 50)
        print("  MÉTRICAS DE EFICIÊNCIA (TESTE)")
        print("=" * 50)
        print(f"  1. Acurácia  : {acc*100:.2f}%")
        print(f"  2. Precisão  : {prec*100:.2f}%")
        print(f"  3. Revocação : {rec*100:.2f}%")
        print(f"  4. F1-Score  : {f1*100:.2f}%")
        print("=" * 50)

        report = classification_report(all_labels, all_preds)
        report_path = "classification_report_timm_effnetb0.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("BLOCO 3 - timm backbone: EfficientNet-B0 (pretrained)\n")
            f.write("=" * 60 + "\n\n")
            f.write("MÉTRICAS DE EFICIÊNCIA\n")
            f.write("-" * 60 + "\n")
            f.write(f"Acurácia  : {acc*100:.2f}%\n")
            f.write(f"Precisão  : {prec*100:.2f}%\n")
            f.write(f"Revocação : {rec*100:.2f}%\n")
            f.write(f"F1-Score  : {f1*100:.2f}%\n")
            f.write("-" * 60 + "\n\n")
            f.write("Relatório por Classe:\n")
            f.write(report)
        print(f"\n  -> Relatório salvo: {report_path}")

        cm = confusion_matrix(all_labels, all_preds)
        class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
        plt.title("Matriz de Confusão - EfficientNet-B0 (timm) - Bloco 3")
        plt.xlabel("Previsão")
        plt.ylabel("Real")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        cm_path = "confusion_matrix_timm_effnetb0.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"  -> Matriz salva: {cm_path}")

        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)
        mlflow.pytorch.log_model(model, "timm_effnetb0_model")

        print("\n  -> Modelo e artefatos registrados no MLflow!")
        print("\nConcluído com sucesso!")


if __name__ == "__main__":
    main()

