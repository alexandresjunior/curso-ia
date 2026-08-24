"""
Solução: Análise de Sensibilidade Comparativa
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

# 1. Carregamento dos dados
# Nota: Assumimos que o script é rodado de dentro da pasta exercicios/
df = pd.read_csv('../../dados/clientes.csv')

# Definindo a variável alvo (ajuste conforme a estrutura real do seu clientes.csv)
target_col = 'alvo' 

if target_col in df.columns:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Pré-processamento
    # Codificação de variáveis categóricas
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Padronização (Crucial para algoritmos baseados em distância/margem como SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Treinamento dos Modelos
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train_scaled, y_train)
    
    svm_clf = SVC(random_state=42)
    svm_clf.fit(X_train_scaled, y_train)

    # 4. Análise de Sensibilidade (Permutation Importance)
    print("Calculando sensibilidade para a Árvore de Decisão...")
    tree_result = permutation_importance(tree_clf, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    print("Calculando sensibilidade para o SVM...")
    svm_result = permutation_importance(svm_clf, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    # 5. Visualização Comparativa
    feature_names = X.columns
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Árvore de Decisão
    sorted_idx_tree = tree_result.importances_mean.argsort()
    ax1.boxplot(tree_result.importances[sorted_idx_tree].T, vert=False, labels=feature_names[sorted_idx_tree])
    ax1.set_title("Permutation Importance - Árvore de Decisão")
    ax1.set_xlabel("Queda na Acurácia")
    
    # Plot SVM
    sorted_idx_svm = svm_result.importances_mean.argsort()
    ax2.boxplot(svm_result.importances[sorted_idx_svm].T, vert=False, labels=feature_names[sorted_idx_svm])
    ax2.set_title("Permutation Importance - SVM")
    ax2.set_xlabel("Queda na Acurácia")
    
    plt.tight_layout()
    plt.show()

else:
    print(f"Para executar o gabarito, altere 'target_col' para a coluna alvo do dataset. Colunas: {df.columns.tolist()}")