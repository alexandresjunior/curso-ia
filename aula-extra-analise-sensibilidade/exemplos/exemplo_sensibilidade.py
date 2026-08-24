import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder

# 1. Carregamento dos dados
df = pd.read_csv('../../dados/clientes.csv')

print("Visualizando as primeiras linhas do dataset:")
print(df.head())

# OBS: Defina a variável alvo (target) de acordo com o escopo de negócio discutido em sala. 
# Para este exemplo, assumimos que existe uma coluna 'churn' ou 'comprou'.
# Altere 'alvo' para o nome exato da coluna no seu clientes.csv.
target_col = 'alvo' 

if target_col in df.columns:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Pré-processamento rápido (Label Encoding para variáveis categóricas)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 2. Treinamento do Modelo (Baseline)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # ==========================================
    # ANÁLISE DE SENSIBILIDADE E EXPLICABILIDADE
    # ==========================================

    # 3. Feature Importance (Métrica nativa da árvore baseada em impureza)
    importances = clf.feature_importances_
    feature_names = X.columns
    
    plt.figure(figsize=(10, 5))
    plt.barh(feature_names, importances, color='steelblue')
    plt.title('Importância das Variáveis (Gini Importance)')
    plt.xlabel('Importância Relativa')
    plt.tight_layout()
    plt.show()

    # 4. Permutation Importance (Sensibilidade do modelo a dados não vistos)
    # Testa o quanto o modelo "sofre" se embaralharmos os dados de uma coluna
    result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    plt.figure(figsize=(10, 5))
    plt.boxplot(result.importances.T, labels=feature_names, vert=False)
    plt.title('Permutation Importance (Sensibilidade)')
    plt.xlabel('Queda na Performance (Acurácia)')
    plt.tight_layout()
    plt.show()

    # 5. Partial Dependence Plot (Dependência Parcial)
    # Mostra como a variação progressiva de uma feature afeta a predição.
    # Vamos plotar o efeito para as duas variáveis mais importantes (índices 0 e 1 como exemplo)
    fig, ax = plt.subplots(figsize=(12, 6))
    display = PartialDependenceDisplay.from_estimator(
        clf, X_train, features=[0, 1], feature_names=feature_names, ax=ax
    )
    plt.suptitle('Partial Dependence Plots (Efeito Marginal)')
    plt.tight_layout()
    plt.show()

else:
    print(f"Aviso: Defina a coluna '{target_col}' corretamente. Colunas disponíveis: {df.columns.tolist()}")