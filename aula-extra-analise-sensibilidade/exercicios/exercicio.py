"""
Exercício Prático: Análise de Sensibilidade Comparativa

Objetivo: Treinar dois modelos diferentes (uma Árvore de Decisão e um SVM) 
usando a base 'clientes.csv' e comparar a sensibilidade de ambos utilizando 
a técnica de Permutation Importance.

Siga as instruções nos comentários marcados com 'TODO'.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

# 1. Carregue o dataset clientes.csv
# TODO: Leia o arquivo '../../dados/clientes.csv' com o Pandas
df = # SEU CÓDIGO AQUI

# Substitua 'alvo' pelo nome correto da coluna que deseja prever
target_col = 'alvo' 

if target_col in df.columns:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Pré-processamento
    # TODO: Converta colunas categóricas usando LabelEncoder
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # TODO: Padronize os dados usando StandardScaler (importante para o SVM)
    scaler = # SEU CÓDIGO AQUI
    X_train_scaled = # SEU CÓDIGO AQUI
    X_test_scaled = # SEU CÓDIGO AQUI

    # 3. Treinamento dos Modelos
    # TODO: Instancie e treine um DecisionTreeClassifier
    tree_clf = # SEU CÓDIGO AQUI
    
    # TODO: Instancie e treine um SVC (Support Vector Classifier)
    svm_clf = # SEU CÓDIGO AQUI

    # 4. Análise de Sensibilidade (Permutation Importance)
    # TODO: Calcule o permutation_importance para a Árvore de Decisão
    # Dica: use os dados de teste (X_test_scaled, y_test)
    tree_importances = # SEU CÓDIGO AQUI
    
    # TODO: Calcule o permutation_importance para o SVM
    svm_importances = # SEU CÓDIGO AQUI

    # 5. Visualização
    # TODO: Crie gráficos (ex: boxplot ou bar plot) para comparar a 
    # importância das features calculadas para a Árvore e para o SVM.
    
    
else:
    print(f"Verifique o nome da coluna alvo. Colunas encontradas: {df.columns.tolist()}")