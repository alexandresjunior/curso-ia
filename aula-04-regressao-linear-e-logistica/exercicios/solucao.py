"""
Aula 04 - Solução comentada do exercício
"""

# ---------------------------------------------------------------------------
# PARTE A — Respostas conceituais
# ---------------------------------------------------------------------------
#
# 1. Significa que, mantendo as demais variáveis constantes, cada R$ 1 a mais
#    investido em marketing está associado, em média, a R$ 3,20 a mais em
#    vendas mensais (uma relação linear estimada, não necessariamente causal).
#
# 2. A regressão linear pode prever valores fora do intervalo [0,1] (ex.:
#    -0.3 ou 1.8), que não têm interpretação como probabilidade. Além disso,
#    ela assume uma relação linear contínua entre as variáveis e o alvo, o
#    que não reflete bem a natureza de uma decisão binária (que tem uma
#    "transição" mais abrupta entre as classes). A regressão logística
#    resolve isso ao mapear a saída para uma probabilidade válida entre 0 e 1.
#
# 3. Faturas em atraso (+1.8): quanto mais faturas em atraso recentemente,
#    maior a chance de inadimplência futura -- coeficiente positivo aumenta
#    a probabilidade prevista.
#    Tempo de relacionamento (-0.4): quanto mais tempo de relacionamento com
#    o banco, MENOR a chance de inadimplência -- coeficiente negativo reduz
#    a probabilidade prevista (clientes antigos tendem a ser mais estáveis).

# ---------------------------------------------------------------------------
# PARTE B — Prática
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

df = pd.read_csv("../../dados/clientes.csv").dropna()

# --- 1) Regressão Linear ---
features_reg = ["idade", "renda_mensal", "tempo_de_casa_meses"]
X = df[features_reg]
y = df["qtd_acessos_mes"]

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_treino_esc = scaler.fit_transform(X_treino)
X_teste_esc = scaler.transform(X_teste)

modelo_linear = LinearRegression()
modelo_linear.fit(X_treino_esc, y_treino)
y_pred = modelo_linear.predict(X_teste_esc)

print("=== Regressão Linear: qtd_acessos_mes ===")
print(f"MAE: {mean_absolute_error(y_teste, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_teste, y_pred):.2f}")
print(f"R²:  {r2_score(y_teste, y_pred):.3f}")

coef_renda = modelo_linear.coef_[features_reg.index("renda_mensal")]
print(f"\nCoeficiente de 'renda_mensal': {coef_renda:.3f}")
if coef_renda > 0:
    print("-> Renda mais alta está associada a MAIS acessos mensais (relação positiva).")
else:
    print("-> Renda mais alta está associada a MENOS acessos mensais (relação negativa).")

# --- 2) Regressão Logística ---
features_clf = ["idade", "renda_mensal", "tempo_de_casa_meses", "qtd_acessos_mes", "qtd_chamados_suporte"]
X2 = df[features_clf]
y2 = df["churn"]

X2_treino, X2_teste, y2_treino, y2_teste = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)

scaler2 = StandardScaler()
X2_treino_esc = scaler2.fit_transform(X2_treino)
X2_teste_esc = scaler2.transform(X2_teste)

modelo_log = LogisticRegression()
modelo_log.fit(X2_treino_esc, y2_treino)
y2_pred = modelo_log.predict(X2_teste_esc)
y2_proba = modelo_log.predict_proba(X2_teste_esc)[:, 1]

print("\n=== Regressão Logística: churn ===")
print(f"Acurácia no teste: {accuracy_score(y2_teste, y2_pred):.2%}")

idx_max = np.argmax(y2_proba)
cliente_maior_risco = X2_teste.iloc[idx_max]
print(f"\nCliente com MAIOR probabilidade prevista de churn ({y2_proba[idx_max]:.2%}):")
print(cliente_maior_risco)
print(f"Valor real de churn: {y2_teste.iloc[idx_max]}")
