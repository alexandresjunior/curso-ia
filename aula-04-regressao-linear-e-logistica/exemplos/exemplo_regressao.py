"""
Aula 04 - Exemplo prático
Objetivo:
 1) Regressão Linear: prever 'valor_mensalidade' (só para fins didáticos,
    já que na prática essa variável depende só do plano -- usaremos
    'renda_mensal' como alvo para simular um problema de regressão real:
    prever o "poder de compra" de um cliente a partir de outras variáveis).
 2) Regressão Logística: prever 'churn' (classificação binária).

Como rodar:
    python exemplo_regressao.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

df = pd.read_csv("../../dados/clientes.csv").dropna()

# ---------------------------------------------------------------------------
# 1) REGRESSÃO LINEAR: prever 'renda_mensal' a partir de idade, tempo de
#    casa e quantidade de acessos (cenário: estimar poder de compra quando a
#    renda declarada é desconhecida, com base no comportamento de uso).
# ---------------------------------------------------------------------------
features_reg = ["idade", "tempo_de_casa_meses", "qtd_acessos_mes"]
X = df[features_reg]
y = df["renda_mensal"]

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_treino_esc = scaler.fit_transform(X_treino)
X_teste_esc = scaler.transform(X_teste)

modelo_linear = LinearRegression()
modelo_linear.fit(X_treino_esc, y_treino)

y_pred = modelo_linear.predict(X_teste_esc)

print("=== Regressão Linear: prevendo renda_mensal ===")
print("Coeficientes (padronizados):")
for nome, coef in zip(features_reg, modelo_linear.coef_):
    print(f"  {nome}: {coef:.2f}")
print(f"Intercepto: {modelo_linear.intercept_:.2f}\n")

print(f"MAE  (Erro Absoluto Médio): R$ {mean_absolute_error(y_teste, y_pred):.2f}")
print(f"MSE  (Erro Quadrático Médio): {mean_squared_error(y_teste, y_pred):.2f}")
print(f"RMSE (Raiz do MSE): R$ {np.sqrt(mean_squared_error(y_teste, y_pred)):.2f}")
print(f"R²   (variância explicada): {r2_score(y_teste, y_pred):.3f}\n")
# Discutiremos essas métricas em detalhe na Aula 06.

# ---------------------------------------------------------------------------
# 2) REGRESSÃO LOGÍSTICA: prever 'churn'
# ---------------------------------------------------------------------------
features_clf = ["idade", "renda_mensal", "tempo_de_casa_meses", "qtd_acessos_mes",
                "qtd_chamados_suporte", "atraso_pagamento"]
X2 = df[features_clf]
y2 = df["churn"]

X2_treino, X2_teste, y2_treino, y2_teste = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)

scaler2 = StandardScaler()
X2_treino_esc = scaler2.fit_transform(X2_treino)
X2_teste_esc = scaler2.transform(X2_teste)

modelo_log = LogisticRegression(class_weight="balanced")  # ajuda com desbalanceamento
modelo_log.fit(X2_treino_esc, y2_treino)

y2_pred = modelo_log.predict(X2_teste_esc)
y2_proba = modelo_log.predict_proba(X2_teste_esc)[:, 1]

print("=== Regressão Logística: prevendo churn ===")
print("Coeficientes (padronizados) -- quanto maior/mais positivo, mais aumenta a chance de churn:")
for nome, coef in zip(features_clf, modelo_log.coef_[0]):
    print(f"  {nome}: {coef:+.3f}")

print(f"\nAcurácia no teste: {accuracy_score(y2_teste, y2_pred):.2%}")

print("\nProbabilidade de churn prevista para os 5 primeiros clientes de teste:")
for i in range(5):
    print(f"  Cliente {i+1}: P(churn)={y2_proba[i]:.2%} | valor real={y2_teste.values[i]}")

# Observação: usamos class_weight="balanced" porque a base tem poucos casos
# de churn (desbalanceada). Isso faz o modelo "prestar mais atenção" na
# classe minoritária, o que costuma reduzir a acurácia geral mas aumentar o
# recall da classe de interesse (cancelamento) -- um trade-off que
# discutiremos com mais profundidade na Aula 06.

