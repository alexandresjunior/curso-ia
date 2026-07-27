"""
Aula 05 - Exemplo prático
Objetivo: treinar k-NN, Árvore de Decisão e SVM para o mesmo problema
(previsão de churn) e comparar seus resultados lado a lado.

Como rodar:
    python exemplo_knn_arvore_svm.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("../../dados/clientes.csv").dropna()

features = ["idade", "renda_mensal", "tempo_de_casa_meses",
            "qtd_acessos_mes", "qtd_chamados_suporte", "atraso_pagamento"]
X = df[features]
y = df["churn"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# k-NN e SVM precisam de escala; a árvore de decisão não precisa, mas usar os
# dados escalados não prejudica em nada (ajuda a manter o código único aqui).
scaler = StandardScaler()
X_treino_esc = scaler.fit_transform(X_treino)
X_teste_esc = scaler.transform(X_teste)

resultados = {}

# ---------------------------------------------------------------------------
# 1) k-NN
# ---------------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_treino_esc, y_treino)
y_pred_knn = knn.predict(X_teste_esc)
resultados["k-NN (k=7)"] = {
    "acuracia": accuracy_score(y_teste, y_pred_knn),
    "f1": f1_score(y_teste, y_pred_knn),
}

# ---------------------------------------------------------------------------
# 2) Árvore de Decisão
# ---------------------------------------------------------------------------
arvore = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
arvore.fit(X_treino, y_treino)  # não precisa de escala
y_pred_arvore = arvore.predict(X_teste)
resultados["Árvore de Decisão (max_depth=4)"] = {
    "acuracia": accuracy_score(y_teste, y_pred_arvore),
    "f1": f1_score(y_teste, y_pred_arvore),
}

# ---------------------------------------------------------------------------
# 3) SVM
# ---------------------------------------------------------------------------
svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
svm.fit(X_treino_esc, y_treino)
y_pred_svm = svm.predict(X_teste_esc)
resultados["SVM (kernel RBF)"] = {
    "acuracia": accuracy_score(y_teste, y_pred_svm),
    "f1": f1_score(y_teste, y_pred_svm),
}

# ---------------------------------------------------------------------------
# Comparativo final
# ---------------------------------------------------------------------------
print("=== Comparativo de modelos para previsão de churn ===")
print(f"{'Modelo':35s} {'Acurácia':>10s} {'F1-score':>10s}")
for nome, metricas in resultados.items():
    print(f"{nome:35s} {metricas['acuracia']:>9.2%} {metricas['f1']:>9.2%}")

print(
    "\nNota: usamos F1-score (além da acurácia) porque a base é desbalanceada "
    "(poucos casos de churn) -- métricas serão detalhadas na Aula 06."
)

# ---------------------------------------------------------------------------
# Bônus: visualizar as regras aprendidas pela árvore de decisão (em texto)
# ---------------------------------------------------------------------------
print("\n=== Regras aprendidas pela Árvore de Decisão ===")
print(export_text(arvore, feature_names=features))
