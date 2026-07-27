"""
Aula 03 - Exemplo prático
Objetivo:
 1) Recalcular em código o exemplo clássico do exame médico (Teorema de
    Bayes "na mão").
 2) Aplicar o algoritmo Naive Bayes (GaussianNB) para prever churn de
    clientes a partir de suas características numéricas.

Como rodar:
    python exemplo_teorema_de_bayes.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------------------------
# 1) Teorema de Bayes "na mão": exemplo do exame médico
# ---------------------------------------------------------------------------
p_doente = 0.01
p_positivo_dado_doente = 0.95          # sensibilidade
p_positivo_dado_saudavel = 0.10        # 1 - especificidade (especificidade = 0.90)
p_saudavel = 1 - p_doente

p_positivo = (p_positivo_dado_doente * p_doente) + (p_positivo_dado_saudavel * p_saudavel)
p_doente_dado_positivo = (p_positivo_dado_doente * p_doente) / p_positivo

print("=== Teorema de Bayes: exemplo do exame médico ===")
print(f"P(doente)              = {p_doente:.2%}")
print(f"P(positivo | doente)   = {p_positivo_dado_doente:.2%}")
print(f"P(positivo | saudável) = {p_positivo_dado_saudavel:.2%}")
print(f"P(positivo) total      = {p_positivo:.4f}")
print(f"P(doente | positivo)   = {p_doente_dado_positivo:.2%}")
print(
    "\nOu seja: mesmo testando positivo, a chance real de estar doente é de "
    f"apenas {p_doente_dado_positivo:.1%}, porque a doença é rara. Este é o "
    "tipo de raciocínio que evita alarmismo/decisões equivocadas em negócio.\n"
)

# ---------------------------------------------------------------------------
# 2) Naive Bayes aplicado à previsão de churn
# ---------------------------------------------------------------------------
df = pd.read_csv("../../dados/clientes.csv")
df = df.dropna()  # para simplificar este exemplo, removemos linhas com NaN

features = ["idade", "renda_mensal", "tempo_de_casa_meses", "qtd_acessos_mes", "qtd_chamados_suporte"]
X = df[features]
y = df["churn"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Naive Bayes gaussiano não exige escala, mas escalar não prejudica e ajuda
# a comparar depois com outros algoritmos (aulas seguintes).
scaler = StandardScaler()
X_treino_esc = scaler.fit_transform(X_treino)
X_teste_esc = scaler.transform(X_teste)

modelo = GaussianNB()
modelo.fit(X_treino_esc, y_treino)

y_pred = modelo.predict(X_teste_esc)

print("=== Naive Bayes aplicado à previsão de churn ===")
print(f"Acurácia no teste: {accuracy_score(y_teste, y_pred):.2%}\n")
print("Matriz de confusão (linhas=real, colunas=previsto):")
print(confusion_matrix(y_teste, y_pred))
print("\nRelatório de classificação:")
print(classification_report(y_teste, y_pred, target_names=["não cancelou", "cancelou"]))

# Probabilidades previstas para os 5 primeiros clientes de teste
probas = modelo.predict_proba(X_teste_esc[:5])
print("Probabilidades previstas (não cancelou | cancelou) para 5 clientes de teste:")
for i, p in enumerate(probas):
    print(f"  Cliente {i+1}: P(não cancelar)={p[0]:.2%} | P(cancelar)={p[1]:.2%}")

# Observação importante: repare que a acurácia ficou alta (~90%), mas o
# modelo não acertou NENHUM caso de "cancelou" (recall = 0 para essa classe).
# Isso acontece porque a base é desbalanceada (poucos cancelamentos), e a
# acurácia sozinha "esconde" esse problema -- é exatamente o tipo de
# armadilha que discutiremos na Aula 06 (métricas de avaliação além da
# acurácia, como recall, precisão e AUC).

