"""
Aula 07 - Exemplo prático
Objetivo:
 1) Segmentar clientes com K-means (usando o método do cotovelo para
    escolher k) e interpretar os clusters resultantes.
 2) Aplicar DBSCAN e comparar com o K-means, observando a detecção de
    outliers.
 3) Aplicar PCA para reduzir as variáveis a 2 dimensões e visualizar os
    clusters encontrados.

Como rodar:
    python exemplo_clusterizacao_pca.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv("../../dados/clientes.csv").dropna()

features = ["idade", "renda_mensal", "tempo_de_casa_meses", "qtd_acessos_mes", "qtd_chamados_suporte"]
X = df[features]

# Sempre padronizar antes de K-means, DBSCAN e PCA (Aula 02)
scaler = StandardScaler()
X_esc = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# 1) K-MEANS: método do cotovelo para escolher k
# ---------------------------------------------------------------------------
inercias = []
silhuetas = []
valores_k = range(2, 8)

for k in valores_k:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_esc)
    inercias.append(kmeans.inertia_)
    silhuetas.append(silhouette_score(X_esc, labels))

print("=== Método do cotovelo e coeficiente de silhueta ===")
for k, inercia, sil in zip(valores_k, inercias, silhuetas):
    print(f"k={k}: inércia={inercia:.1f} | silhueta={sil:.3f}")

melhor_k = valores_k[int(np.argmax(silhuetas))]
print(f"\nMelhor k pelo coeficiente de silhueta: {melhor_k}\n")

# Treinando o K-means final com o melhor k encontrado
kmeans_final = KMeans(n_clusters=melhor_k, n_init=10, random_state=42)
df["cluster_kmeans"] = kmeans_final.fit_predict(X_esc)

print(f"=== Perfil médio de cada cluster (k={melhor_k}) ===")
print(df.groupby("cluster_kmeans")[features + ["churn"]].mean().round(2))
print("\nTamanho de cada cluster:")
print(df["cluster_kmeans"].value_counts().sort_index())

# ---------------------------------------------------------------------------
# 2) DBSCAN
# ---------------------------------------------------------------------------
dbscan = DBSCAN(eps=1.2, min_samples=8)
df["cluster_dbscan"] = dbscan.fit_predict(X_esc)

n_clusters_dbscan = len(set(df["cluster_dbscan"])) - (1 if -1 in df["cluster_dbscan"].values else 0)
n_outliers = (df["cluster_dbscan"] == -1).sum()

print(f"\n=== DBSCAN ===")
print(f"Número de clusters encontrados: {n_clusters_dbscan}")
print(f"Número de pontos classificados como ruído/outlier: {n_outliers} ({n_outliers/len(df):.1%})")

# ---------------------------------------------------------------------------
# 3) PCA: reduzindo para 2 dimensões e visualizando os clusters do K-means
# ---------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_esc)

print(f"\n=== PCA ===")
print(f"Variância explicada pelo Componente 1: {pca.explained_variance_ratio_[0]:.1%}")
print(f"Variância explicada pelo Componente 2: {pca.explained_variance_ratio_[1]:.1%}")
print(f"Variância total explicada pelos 2 componentes: {pca.explained_variance_ratio_.sum():.1%}")

plt.figure(figsize=(7, 5))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster_kmeans"], cmap="viridis", alpha=0.7)
plt.title(f"Clientes visualizados em 2D via PCA (cor = cluster do K-means, k={melhor_k})")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("clusters_pca.png", dpi=120)
print("\nGráfico salvo em 'clusters_pca.png'")
