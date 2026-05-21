import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# AMBIENTE E SETUP
# ==============================================================================
# Para rodar este código, certifique-se de usar um ambiente virtual (venv).
# Comandos para criar e ativar o ambiente (no terminal/cmd):
# 1. python -m venv venv
# 2. venv\Scripts\activate  (no Windows)
# 3. pip install pandas numpy scikit-learn
# ==============================================================================

def main():
    print("Iniciando pipeline de treinamento do modelo Fashion MNIST...\n")

    # 1. CARREGAMENTO DOS DADOS
    # Usaremos o arquivo disponível. Na prática, você teria um arquivo de treino separado.
    print("[1/5] Carregando o arquivo de dados...")
    data = pd.read_csv('fashion-mnist_test.csv')
    
    # Separando Features (X) e Target (y)
    y = data['label'].values
    X = data.drop('label', axis=1).values

    # ==============================================================================
    # REGRAS E PRÉ-PROCESSAMENTO
    # ==============================================================================
    print("[2/5] Aplicando regras de pré-processamento (Normalização)...")
    # Regra: Normalizar os valores dos pixels (0 a 255) para a escala (0.0 a 1.0)
    # Isso acelera e melhora a eficiência do treinamento de qualquer modelo.
    X_normalized = X / 255.0

    # ==============================================================================
    # DIVISÃO DE GRUPOS (SPLITS)
    # ==============================================================================
    print("[3/5] Realizando a divisão dos grupos (Treino, Validação e Teste)...")
    # Vamos dividir nossos dados em 3 grupos:
    # 70% para Treino (ensinar o modelo)
    # 15% para Validação (ajustar o modelo durante o desenvolvimento)
    # 15% para Teste (avaliar a eficiência final em dados nunca vistos)

    # Primeira divisão: separa Treino (70%) do resto (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y, test_size=0.30, random_state=42)
    
    # Segunda divisão: divide o resto em Validação (15%) e Teste (15%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    print(f"  -> Dados de Treino:    {X_train.shape[0]} amostras")
    print(f"  -> Dados de Validação: {X_val.shape[0]} amostras")
    print(f"  -> Dados de Teste:     {X_test.shape[0]} amostras\n")

    # ==============================================================================
    # TREINAMENTO DO MODELO
    # ==============================================================================
    print("[4/5] Treinando o modelo (XGBoost com vetor 784-dim flat)...")
    # Estamos utilizando XGBoost para lidar com as 784 dimensões achatadas
    model = XGBClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    # ==============================================================================
    # MEDIÇÃO DE EFICIÊNCIA DO MODELO
    # ==============================================================================
    print("[5/5] Realizando medições de eficiência nos dados de Teste...\n")
    
    # Fazendo predições com o grupo de Teste
    y_pred = model.predict(X_test)

    # Métodos de Medição:
    # 1. Acurácia (Accuracy): Porcentagem total de acertos.
    # 2. Precisão (Precision): Dos que o modelo previu como classe X, quantos realmente eram X?
    # 3. Revocação (Recall): De todos os que eram classe X, quantos o modelo conseguiu prever corretamente?
    # 4. F1-Score: Média harmônica entre Precisão e Revocação (útil se as classes fossem desbalanceadas).

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("-" * 50)
    print("MÉTRICAS DE EFICIÊNCIA (MÉDIA PONDERADA)")
    print("-" * 50)
    print(f"1. Acurácia  : {acc * 100:.2f}%")
    print(f"2. Precisão  : {prec * 100:.2f}%")
    print(f"3. Revocação : {rec * 100:.2f}%")
    print(f"4. F1-Score  : {f1 * 100:.2f}%")
    print("-" * 50)

    print("\nRelatório de Classificação Completo por Classe:")
    report = classification_report(y_test, y_pred)
    print(report)

    # Salvando o relatório em um arquivo de texto
    with open('classification_report.txt', 'w', encoding='utf-8') as f:
        f.write("MÉTRICAS DE EFICIÊNCIA (MÉDIA PONDERADA)\n")
        f.write("-" * 50 + "\n")
        f.write(f"1. Acurácia  : {acc * 100:.2f}%\n")
        f.write(f"2. Precisão  : {prec * 100:.2f}%\n")
        f.write(f"3. Revocação : {rec * 100:.2f}%\n")
        f.write(f"4. F1-Score  : {f1 * 100:.2f}%\n")
        f.write("-" * 50 + "\n\n")
        f.write("Relatório de Classificação Completo por Classe:\n")
        f.write(report)
    print("Relatório salvo em 'classification_report.txt' com sucesso!")

    # ==============================================================================
    # GERANDO A MATRIZ DE CONFUSÃO EM IMAGEM
    # ==============================================================================
    print("\nGerando Matriz de Confusão e salvando em 'confusion_matrix.png'...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)])
    
    plt.xlabel('Classe Prevista pelo Modelo')
    plt.ylabel('Classe Real (Verdadeira)')
    plt.title('Matriz de Confusão - Fashion MNIST (XGBoost)')
    
    # Salvando a imagem no diretório
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Arquivo 'confusion_matrix.png' gerado com sucesso!")

if __name__ == "__main__":
    main()
