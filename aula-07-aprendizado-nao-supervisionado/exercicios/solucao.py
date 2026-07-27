"""
Aula 07 - Solução comentada do exercício
"""

# ---------------------------------------------------------------------------
# PARTE A — Respostas conceituais
# ---------------------------------------------------------------------------
#
# 1. DBSCAN. Como não sabemos de antemão o número de grupos naturais e
#    suspeitamos de outliers, o DBSCAN é mais adequado: ele descobre o número
#    de clusters sozinho (baseado em densidade) e naturalmente isola pontos
#    atípicos como "ruído", em vez de forçá-los para dentro de algum grupo
#    (como o K-means faria).
#
# 2. K-means, DBSCAN e PCA dependem de cálculos de DISTÂNCIA ou de VARIÂNCIA
#    entre as variáveis. Se as variáveis estão em escalas muito diferentes
#    (ex.: renda em milhares vs. idade em dezenas), a variável de maior
#    magnitude dominará os cálculos, distorcendo os clusters/componentes
#    encontrados, mesmo que ela não seja, de fato, a mais importante para o
#    problema.
#
# 3. O Componente Principal 1 NÃO é igual a nenhuma variável original -- ele
#    é uma COMBINAÇÃO LINEAR de todas as variáveis originais (uma "mistura"
#    ponderada). Pode ser que "renda" tenha um peso alto nessa combinação,
#    mas o componente também carrega contribuições de outras variáveis. Tratar
#    o componente como sendo exatamente "a variável renda" é uma
#    simplificação equivocada.

# ---------------------------------------------------------------------------
# PARTE B — Prática
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv("../../dados/clientes.csv").dropna()

features = ["renda_mensal", "valor_mensalidade", "tempo_de_casa_meses", "qtd_acessos_mes"]
X = df[features]

scaler = StandardScaler()
X_esc = scaler.fit_transform(X)

# 1) e 2) K-means com k de 2 a 6, escolhendo o melhor via silhueta
resultados_k = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_esc)
    sil = silhouette_score(X_esc, labels)
    resultados_k.append((k, sil))
    print(f"k={k}: silhueta={sil:.3f}")

melhor_k = max(resultados_k, key=lambda t: t[1])[0]
print(f"\nMelhor k: {melhor_k}")

# 3) Perfil médio de cada cluster
kmeans_final = KMeans(n_clusters=melhor_k, n_init=10, random_state=42)
df["cluster"] = kmeans_final.fit_predict(X_esc)

perfil = df.groupby("cluster")[features + ["churn"]].mean().round(2)
print("\nPerfil médio de cada cluster:")
print(perfil)
print("\nTamanho de cada cluster:")
print(df["cluster"].value_counts().sort_index())

# Nomes de negócio (exemplo de interpretação -- os valores reais podem variar
# ligeiramente conforme a semente aleatória e execução):
# Ao observar o `perfil`, normalmente encontramos algo como:
#  - Cluster com alta renda + alto valor de mensalidade + baixo churn
#    -> "Clientes premium fiéis"
#  - Cluster com baixo tempo de casa + baixa renda + churn mais alto
#    -> "Clientes novos de baixo engajamento / risco de cancelamento"
# (Ajuste os nomes de acordo com os números reais impressos acima.)

# 4) PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_esc)
print(f"\nVariância explicada pelos 2 componentes do PCA: {pca.explained_variance_ratio_.sum():.1%}")

# 5) Desafio extra: DBSCAN
dbscan = DBSCAN(eps=1.0, min_samples=5)
df["cluster_dbscan"] = dbscan.fit_predict(X_esc)

n_clusters_dbscan = len(set(df["cluster_dbscan"])) - (1 if -1 in df["cluster_dbscan"].values else 0)
n_outliers = (df["cluster_dbscan"] == -1).sum()
print(f"\nDBSCAN encontrou {n_clusters_dbscan} clusters e {n_outliers} outliers "
      f"({n_outliers/len(df):.1%} da base).")

tabela_cruzada = pd.crosstab(df["cluster"], df["cluster_dbscan"])
print("\nTabela cruzada: cluster do K-means (linhas) x cluster/outlier do DBSCAN (colunas, -1 = outlier):")
print(tabela_cruzada)
