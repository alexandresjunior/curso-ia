"""
Aula 06 - Solução comentada do exercício
"""

# ---------------------------------------------------------------------------
# PARTE A — Respostas conceituais
# ---------------------------------------------------------------------------
#
# 1. NÃO necessariamente. Com apenas 0,5% de fraudes, um modelo "preguiçoso"
#    que sempre prevê "não fraude" já atinge 99,5% de acurácia sem detectar
#    UMA fraude sequer. Antes de aprovar o modelo, eu pediria RECALL (quantas
#    fraudes reais ele conseguiu identificar), PRECISÃO (quantos alertas são
#    de fato fraude) e a AUC -- métricas que revelam se o modelo realmente
#    aprendeu a distinguir as classes, algo que a acurácia sozinha esconde.
#
# 2. Priorizar RECALL: quando o custo de um FALSO NEGATIVO é alto -- ex.:
#    diagnóstico de uma doença grave (não podemos deixar passar um caso
#    real) ou detecção de fraude de alto valor.
#    Priorizar PRECISÃO: quando o custo de um FALSO POSITIVO é alto -- ex.:
#    bloquear preventivamente a conta de um cliente legítimo por suspeita de
#    fraude (gera atrito e insatisfação), ou uma campanha de retenção muito
#    cara que não deve ser disparada para clientes que não iam cancelar.
#
# 3. Uma única divisão treino/teste depende de UMA amostragem aleatória
#    específica -- por sorte ou azar, o conjunto de teste pode ficar "fácil"
#    ou "difícil" demais, distorcendo a avaliação. A validação cruzada roda
#    o treino/teste várias vezes com divisões diferentes dos dados, dando
#    uma média mais estável e um desvio padrão que revela o quão consistente
#    (ou instável) é o desempenho do modelo.

# ---------------------------------------------------------------------------
# PARTE B — Prática
# ---------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error,
)

df = pd.read_csv("../../dados/clientes.csv").dropna()

# --- 1) e 2): Árvore de decisão + validação cruzada ---
features = ["idade", "renda_mensal", "tempo_de_casa_meses", "qtd_acessos_mes",
            "qtd_chamados_suporte", "atraso_pagamento"]
X = df[features]
y = df["churn"]

modelo = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_cv = cross_val_score(modelo, X, y, cv=cv, scoring="roc_auc")

print("=== Validação cruzada (5-fold) -- AUC ===")
print(f"Média: {scores_cv.mean():.3f} | Desvio padrão: {scores_cv.std():.3f}")
print(f"Valores por dobra: {scores_cv.round(3)}\n")

# --- 3) Divisão única treino/teste ---
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
modelo.fit(X_treino, y_treino)
y_pred = modelo.predict(X_teste)
y_proba = modelo.predict_proba(X_teste)[:, 1]

print("=== Divisão única treino/teste ===")
print(f"Acurácia: {accuracy_score(y_teste, y_pred):.2%}")
print(f"Precisão: {precision_score(y_teste, y_pred):.2%}")
print(f"Recall:   {recall_score(y_teste, y_pred):.2%}")
print(f"F1-score: {f1_score(y_teste, y_pred):.2%}")
print(f"AUC:      {roc_auc_score(y_teste, y_proba):.3f}\n")

# --- 4) Comparação ---
print(
    "Comparando: o AUC da divisão única costuma ficar na faixa observada nas "
    "dobras da validação cruzada, mas pode variar sozinho mais do que a média "
    "de 5 dobras -- reforçando por que confiar em uma única divisão é mais "
    "arriscado que usar a validação cruzada para decisões importantes.\n"
)

# --- 5) Regressão linear de valor_mensalidade a partir de 'plano' ---
onehot = OneHotEncoder(sparse_output=False)
plano_encoded = onehot.fit_transform(df[["plano"]])

Xr_treino, Xr_teste, yr_treino, yr_teste = train_test_split(
    plano_encoded, df["valor_mensalidade"], test_size=0.2, random_state=42
)

modelo_linear = LinearRegression()
modelo_linear.fit(Xr_treino, yr_treino)
yr_pred = modelo_linear.predict(Xr_teste)

print("=== Regressão Linear: valor_mensalidade ~ plano ===")
print(f"MAE: {mean_absolute_error(yr_teste, yr_pred):.4f}")
print(f"MSE: {mean_squared_error(yr_teste, yr_pred):.4f}")
print(
    "\nEsperado: MAE e MSE próximos de ZERO. Isso faz total sentido, pois "
    "'valor_mensalidade' é DEFINIDO diretamente pelo plano contratado (não há "
    "ruído nessa relação no nosso dataset sintético) -- um ótimo lembrete de "
    "que, às vezes, a variável-alvo tem uma relação determinística (e não "
    "apenas estatística) com alguma feature, o que deve ser sempre "
    "investigado ao ver métricas 'boas demais para ser verdade'."
)
