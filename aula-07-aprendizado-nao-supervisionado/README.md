# Aula 07 — Aprendizado Não Supervisionado: Clusterização e PCA

## Objetivos da aula

- Entender como agrupar dados sem rótulos usando K-means e DBSCAN.
- Entender o que é redução de dimensionalidade e como o PCA funciona.
- Saber aplicar essas técnicas para segmentação de clientes e visualização de dados complexos.

---

## 1. Por que agrupar dados sem rótulo?

Relembrando a Aula 01: no aprendizado não supervisionado, não temos uma variável-alvo conhecida. O objetivo é **descobrir estrutura escondida** nos dados — por exemplo, encontrar grupos ("clusters") de clientes com comportamento parecido, sem que ninguém tenha definido esses grupos previamente.

Aplicações de negócio comuns:
- **Segmentação de clientes** para campanhas de marketing direcionadas.
- **Agrupamento de produtos** com padrões de venda parecidos.
- **Detecção de perfis atípicos** (ex.: grupos de transações fora do padrão, possíveis fraudes).
- **Organização de documentos/tickets** por similaridade de conteúdo.

---

## 2. K-means

### 2.1 Ideia central

O K-means agrupa os dados em **k grupos** (você escolhe o número `k` de antemão), tentando minimizar a distância entre cada ponto e o **centro (centroide)** do seu grupo.

Algoritmo (resumido):
1. Escolhem-se `k` centroides iniciais (aleatoriamente).
2. Cada ponto é atribuído ao centroide mais próximo.
3. Os centroides são recalculados como a média dos pontos atribuídos a eles.
4. Repete-se os passos 2 e 3 até os centroides pararem de mudar significativamente (convergência).

### 2.2 Como escolher o número de clusters (k)?

- **Método do cotovelo (elbow method):** roda-se o K-means para vários valores de `k` e observa-se a soma das distâncias quadráticas dentro de cada cluster (inércia). Busca-se o ponto onde a curva "dobra" (o "cotovelo") — a partir dali, aumentar `k` traz pouco ganho adicional.
- **Coeficiente de silhueta (silhouette score):** mede o quão bem cada ponto se encaixa no seu cluster comparado a outros clusters (varia de -1 a 1; quanto maior, melhor).

### 2.3 Vantagens e limitações

**Vantagens:** simples, rápido, escala bem para bases grandes.
**Limitações:**
- Você precisa **escolher `k` a priori**.
- Assume clusters de formato aproximadamente **esférico** e de tamanho parecido — não lida bem com formatos irregulares.
- Sensível à escala das variáveis (**sempre padronize antes**, como vimos na Aula 02) e à posição inicial dos centroides (mitigado com múltiplas inicializações, `n_init`).
- Sensível a outliers.

---

## 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

### 3.1 Ideia central

Diferente do K-means, o DBSCAN agrupa pontos com base em **densidade**: regiões onde os pontos estão muito próximos uns dos outros formam um cluster; pontos isolados, em regiões de baixa densidade, são marcados como **ruído/outliers** (não pertencem a nenhum cluster).

Dois parâmetros principais:
- `eps`: raio máximo de distância para considerar dois pontos "vizinhos".
- `min_samples`: número mínimo de vizinhos para um ponto ser considerado um "ponto central" (core point) de um cluster.

### 3.2 Vantagens e limitações

**Vantagens:**
- **Não exige definir o número de clusters previamente** — o algoritmo descobre isso sozinho.
- Identifica naturalmente **outliers** como ruído, o que é útil para detecção de anomalias.
- Lida bem com clusters de formato irregular (não apenas esférico).

**Limitações:**
- Sensível à escolha de `eps` e `min_samples` (exige alguma experimentação).
- Tem dificuldade quando os clusters têm densidades muito diferentes entre si.

<div align="center">
  <img width="960" height="480" alt="db_scan_vs_k_means" src="https://github.com/user-attachments/assets/d269b51c-3e95-455c-a600-5d183baf3651" />
</div>

### 3.3 K-means vs. DBSCAN

| | K-means | DBSCAN |
|---|---|---|
| Define nº de clusters? | Você escolhe `k` | O algoritmo descobre |
| Lida com outliers? | Não naturalmente (todo ponto pertence a um cluster) | Sim (marca como ruído) |
| Formato dos clusters | Aproximadamente esférico | Qualquer formato (baseado em densidade) |
| Velocidade | Rápido, escala bem | Pode ser mais lento em bases muito grandes |

---

## 4. PCA (Principal Component Analysis / Análise de Componentes Principais)

### 4.1 O problema: muitas variáveis (alta dimensionalidade)

Quando temos muitas variáveis (dezenas ou centenas), fica difícil visualizar os dados, alguns algoritmos ficam mais lentos, e pode haver redundância (variáveis correlacionadas entre si carregando informação parecida). Isso é conhecido como a **"maldição da dimensionalidade"**.

### 4.2 Ideia central do PCA

O PCA busca **novas variáveis (componentes principais)**, que são **combinações lineares das variáveis originais**, ordenadas de forma que:
- O **1º componente principal** capture a maior variância possível dos dados.
- O **2º componente principal** capture a maior variância restante, **sendo perpendicular (ortogonal)** ao primeiro.
- E assim por diante.

Com isso, é possível **reduzir dezenas de variáveis para 2 ou 3 componentes principais**, mantendo a maior parte da informação (variância) original — o que permite, por exemplo, **visualizar** dados complexos em um gráfico 2D.

<div align="center">
  <img width="958" height="308" alt="pca" src="https://github.com/user-attachments/assets/24105fdc-7b2c-44c2-a30f-0d90e363b5d8" />
</div>

### 4.3 Quanto de variância é preservada?

Cada componente principal "explica" uma fração da variância total dos dados. É comum reportar o **"percentual de variância explicada"** por cada componente (e o acumulado), para decidir quantos componentes manter (ex.: "os 2 primeiros componentes explicam 80% da variância total").

### 4.4 Aplicações práticas

- **Visualização** de dados com muitas variáveis em 2D/3D.
- **Redução de ruído e redundância** antes de treinar outros modelos.
- **Pré-processamento** para acelerar algoritmos sensíveis a alta dimensionalidade.

### 4.5 Cuidados importantes

- O PCA é sensível à escala das variáveis — **sempre padronize os dados antes** (Aula 02).
- Os componentes principais **perdem interpretabilidade direta**: eles são combinações matemáticas das variáveis originais, não uma variável de negócio isolada (ex.: o "componente 1" pode ser uma mistura de renda + tempo de casa + acessos, sem um nome de negócio único e óbvio).

---

## 5. Juntando tudo: um fluxo comum na prática

1. Padronizar os dados (Aula 02).
2. (Opcional) Aplicar PCA para reduzir dimensionalidade, se houver muitas variáveis.
3. Aplicar K-means (definindo `k` via método do cotovelo/silhueta) ou DBSCAN.
4. Interpretar os clusters resultantes: analisar as médias/perfis de cada grupo para dar significado de negócio a eles (ex.: "Cluster 0 = clientes fiéis, alto uso, baixo risco"; "Cluster 1 = clientes novos, uso baixo, alto risco de churn").

**Próxima aula:** [Aula 08 — Introdução a Deep Learning](../aula-08-introducao-deep-learning/README.md)
