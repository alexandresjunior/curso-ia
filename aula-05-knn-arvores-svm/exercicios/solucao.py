"""
Aula 05 - Solução comentada do exercício
"""

# ---------------------------------------------------------------------------
# PARTE A — Respostas conceituais
# ---------------------------------------------------------------------------
#
# 1. k-NN e SVM dependem diretamente de cálculos de DISTÂNCIA (ou de margem
#    geométrica, no caso do SVM) entre pontos. Se as variáveis estão em
#    escalas muito diferentes, as de maior magnitude dominam o cálculo,
#    distorcendo os resultados. Já a árvore de decisão faz cortes baseados em
#    LIMIARES independentes por variável (ex.: "renda > 3000?"), então a
#    escala de cada variável não afeta a lógica das divisões.
#
# 2. Isso é um caso clássico de OVERFITTING: a árvore "decorou" os dados de
#    treino (inclusive seu ruído), criando folhas extremamente específicas
#    que não generalizam para dados novos. Soluções: limitar `max_depth`,
#    aumentar `min_samples_leaf`, usar poda (pruning), ou usar um ensemble
#    (Random Forest) que reduz a variância combinando várias árvores.
#
# 3. Árvores de decisão produzem regras do tipo "se X e Y, então Z", que
#    podem ser lidas e questionadas por qualquer pessoa não técnica -- um
#    comitê de crédito pode verificar exatamente por que um cliente foi
#    classificado como alto risco. Um SVM, por outro lado, é uma "caixa
#    relativamente preta": a decisão surge de uma combinação matemática no
#    espaço de kernel, difícil de traduzir em uma explicação simples e
#    auditável -- um ponto importante em contextos regulados.

# ---------------------------------------------------------------------------
# PARTE B — Prática
# ---------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("../../dados/clientes.csv").dropna()

features = ["idade", "renda_mensal", "tempo_de_casa_meses", "qtd_acessos_mes", "qtd_chamados_suporte"]
X = df[features]
y = df["churn"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_treino_esc = scaler.fit_transform(X_treino)
X_teste_esc = scaler.transform(X_teste)

resultados = []

# a) k-NN com diferentes valores de k
for k in [3, 7, 15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_treino_esc, y_treino)
    y_pred = knn.predict(X_teste_esc)
    resultados.append({
        "modelo": f"k-NN (k={k})",
        "acuracia": accuracy_score(y_teste, y_pred),
        "f1": f1_score(y_teste, y_pred),
    })

# b) Árvore de decisão
for profundidade in [3, None]:
    arvore = DecisionTreeClassifier(max_depth=profundidade, class_weight="balanced", random_state=42)
    arvore.fit(X_treino, y_treino)
    y_pred = arvore.predict(X_teste)
    resultados.append({
        "modelo": f"Árvore (max_depth={profundidade})",
        "acuracia": accuracy_score(y_teste, y_pred),
        "f1": f1_score(y_teste, y_pred),
    })

# c) SVM com diferentes kernels
for kernel in ["linear", "rbf"]:
    svm = SVC(kernel=kernel, class_weight="balanced", random_state=42)
    svm.fit(X_treino_esc, y_treino)
    y_pred = svm.predict(X_teste_esc)
    resultados.append({
        "modelo": f"SVM (kernel={kernel})",
        "acuracia": accuracy_score(y_teste, y_pred),
        "f1": f1_score(y_teste, y_pred),
    })

df_resultados = pd.DataFrame(resultados).sort_values("f1", ascending=False)
print("=== Comparativo de modelos ===")
print(df_resultados.to_string(index=False, formatters={
    "acuracia": "{:.2%}".format, "f1": "{:.2%}".format
}))

melhor = df_resultados.iloc[0]
print(f"\nMelhor F1-score: {melhor['modelo']} (F1={melhor['f1']:.2%})")
# Observação: em bases desbalanceadas como esta, é comum que árvores com
# class_weight="balanced" e profundidade limitada tenham F1 mais alto que
# modelos "genéricos", pois conseguem capturar melhor a classe minoritária
# sem overfitting -- confirmando a discussão conceitual da Parte A.
