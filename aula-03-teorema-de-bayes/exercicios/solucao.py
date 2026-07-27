"""
Aula 03 - Solução comentada do exercício
"""

# ---------------------------------------------------------------------------
# PARTE A — Cálculo de Bayes
# ---------------------------------------------------------------------------
p_fraude = 0.002
p_alerta_dado_fraude = 0.98
p_alerta_dado_legitima = 0.05
p_legitima = 1 - p_fraude

p_alerta = (p_alerta_dado_fraude * p_fraude) + (p_alerta_dado_legitima * p_legitima)
p_fraude_dado_alerta = (p_alerta_dado_fraude * p_fraude) / p_alerta

print("=== Parte A: Teorema de Bayes aplicado a antifraude ===")
print(f"P(fraude | alerta) = {p_fraude_dado_alerta:.2%}\n")
# Resposta esperada: aproximadamente 3.8%.
#
# Reflexão: mesmo com um sistema com 98% de sensibilidade, a maioria
# esmagadora dos ALERTAS (~96%) será de transações legítimas, simplesmente
# porque fraudes são muito raras. Isso não significa que o sistema é ruim --
# significa que a equipe de risco deve tratar cada alerta como um indício a
# ser investigado, não como uma confirmação de fraude, e que a análise de
# custo/benefício de investigar cada alerta deve levar essa proporção em
# conta (ex.: triagem adicional antes de bloquear a conta do cliente).

# ---------------------------------------------------------------------------
# PARTE B — Prática
# ---------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("../../dados/clientes.csv").dropna()

features_completas = [
    "idade", "valor_mensalidade", "tempo_de_casa_meses",
    "qtd_acessos_mes", "qtd_chamados_suporte", "atraso_pagamento",
]
y = df["churn"]

def treinar_e_avaliar(features, nome):
    X = df[features]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_treino_esc = scaler.fit_transform(X_treino)
    X_teste_esc = scaler.transform(X_teste)

    modelo = GaussianNB()
    modelo.fit(X_treino_esc, y_treino)
    y_pred = modelo.predict(X_teste_esc)
    acc = accuracy_score(y_teste, y_pred)
    print(f"[{nome}] Acurácia: {acc:.2%}")
    return modelo, scaler, X_teste, y_teste

print("=== Parte B: Naive Bayes com e sem 'atraso_pagamento' ===")
modelo_completo, scaler_completo, X_teste, y_teste = treinar_e_avaliar(
    features_completas, "COM atraso_pagamento"
)
treinar_e_avaliar(
    [f for f in features_completas if f != "atraso_pagamento"], "SEM atraso_pagamento"
)

# Probabilidades para 3 clientes de teste
X_teste_esc = scaler_completo.transform(X_teste)
probas = modelo_completo.predict_proba(X_teste_esc)[:3]
reais = y_teste.values[:3]

print("\n3 clientes de teste (modelo completo):")
for i, (p, real) in enumerate(zip(probas, reais)):
    print(f"  Cliente {i+1}: P(cancelar)={p[1]:.2%} | valor real de churn={real}")

# Observação esperada: a inclusão de 'atraso_pagamento' tende a melhorar (ou
# ao menos não piorar) a acurácia, pois essa variável tem forte relação com o
# cancelamento no processo gerador dos dados (ver dados/gerar_dataset_clientes.py).
