"""
Aula 06 - Exemplo prático
Objetivo:
 1) Aplicar validação cruzada (k-fold estratificado) para comparar modelos
    de classificação de forma mais robusta que uma única divisão treino/teste.
 2) Calcular e interpretar MSE/MAE (regressão) e AUC/precisão/recall/F1
    (classificação).

Como rodar:
    python exemplo_avaliacao.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

df = pd.read_csv("../../dados/clientes.csv").dropna()

# ---------------------------------------------------------------------------
# 1) VALIDAÇÃO CRUZADA: comparando Regressão Logística vs. Árvore de Decisão
#    para prever churn, de forma mais robusta que uma única divisão.
# ---------------------------------------------------------------------------
features = ["idade", "renda_mensal", "tempo_de_casa_meses", "qtd_acessos_mes",
            "qtd_chamados_suporte", "atraso_pagamento"]
X = df[features]
y = df["churn"]

scaler = StandardScaler()
X_esc = scaler.fit_transform(X)  # aqui, para fins didáticos de CV, ajustamos no X completo;
# em um pipeline de produção, o ideal é encapsular o scaler DENTRO do cross-validation
# (ex.: usando sklearn.pipeline.Pipeline) para evitar qualquer vazamento de dados.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

modelo_log = LogisticRegression(class_weight="balanced")
modelo_arvore = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)

scores_log = cross_val_score(modelo_log, X_esc, y, cv=cv, scoring="roc_auc")
scores_arvore = cross_val_score(modelo_arvore, X_esc, y, cv=cv, scoring="roc_auc")

print("=== Validação Cruzada (5-fold estratificada) -- métrica: AUC ===")
print(f"Regressão Logística: média={scores_log.mean():.3f} | desvio={scores_log.std():.3f} | folds={scores_log.round(3)}")
print(f"Árvore de Decisão:   média={scores_arvore.mean():.3f} | desvio={scores_arvore.std():.3f} | folds={scores_arvore.round(3)}\n")

# ---------------------------------------------------------------------------
# 2) Métricas de CLASSIFICAÇÃO em detalhe, no conjunto de teste final
# ---------------------------------------------------------------------------
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_treino_esc = scaler.fit_transform(X_treino)
X_teste_esc = scaler.transform(X_teste)

modelo_log.fit(X_treino_esc, y_treino)
y_pred = modelo_log.predict(X_teste_esc)
y_proba = modelo_log.predict_proba(X_teste_esc)[:, 1]

print("=== Métricas detalhadas de classificação (Regressão Logística, teste final) ===")
print("Matriz de confusão:")
print(confusion_matrix(y_teste, y_pred))
print(f"Acurácia:  {accuracy_score(y_teste, y_pred):.2%}")
print(f"Precisão:  {precision_score(y_teste, y_pred):.2%}")
print(f"Recall:    {recall_score(y_teste, y_pred):.2%}")
print(f"F1-score:  {f1_score(y_teste, y_pred):.2%}")
print(f"AUC:       {roc_auc_score(y_teste, y_proba):.3f}\n")

# ---------------------------------------------------------------------------
# 3) Métricas de REGRESSÃO: prevendo renda_mensal (como na Aula 04)
# ---------------------------------------------------------------------------
features_reg = ["idade", "tempo_de_casa_meses", "qtd_acessos_mes"]
Xr = df[features_reg]
yr = df["renda_mensal"]

Xr_treino, Xr_teste, yr_treino, yr_teste = train_test_split(Xr, yr, test_size=0.2, random_state=42)
scaler_r = StandardScaler()
Xr_treino_esc = scaler_r.fit_transform(Xr_treino)
Xr_teste_esc = scaler_r.transform(Xr_teste)

modelo_linear = LinearRegression()
modelo_linear.fit(Xr_treino_esc, yr_treino)
yr_pred = modelo_linear.predict(Xr_teste_esc)

print("=== Métricas de regressão (prevendo renda_mensal) ===")
print(f"MAE:  R$ {mean_absolute_error(yr_teste, yr_pred):.2f}")
print(f"MSE:  {mean_squared_error(yr_teste, yr_pred):.2f}")
print(f"RMSE: R$ {np.sqrt(mean_squared_error(yr_teste, yr_pred)):.2f}")
