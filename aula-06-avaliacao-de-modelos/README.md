# Aula 06 — Avaliação de Modelos: Validação Cruzada e Métricas de Desempenho

## Objetivos da aula

- Entender por que uma única divisão treino/teste pode não ser suficiente.
- Aprender o conceito de validação cruzada (cross-validation).
- Dominar as principais métricas de avaliação: MSE, MAE (regressão) e AUC e outras métricas de classificação (acurácia, precisão, recall, F1, matriz de confusão).
- Saber escolher a métrica certa para o problema de negócio certo.

---

## 1. Por que uma divisão treino/teste não basta?

Até agora, usamos `train_test_split` para separar os dados uma única vez. Mas essa divisão é **aleatória**: se, por sorte (ou azar), o conjunto de teste ficar com uma amostra pouco representativa, nossa avaliação do modelo pode ficar enviesada — boa demais ou ruim demais, sem que isso reflita o desempenho real do modelo.

## 2. Validação Cruzada (Cross-Validation)

### 2.1 K-Fold Cross-Validation

A técnica mais comum é a **validação cruzada com k dobras (k-fold)**:

1. Os dados de treino são divididos em `k` partes (folds) de tamanho semelhante (ex.: k=5).
2. O modelo é treinado `k` vezes: em cada rodada, uma dobra diferente é usada como "teste" (validação), e as demais `k-1` dobras são usadas para treinar.
3. Ao final, temos `k` medições de desempenho — normalmente reportamos a **média** e o **desvio padrão** entre elas.

```
Rodada 1: treina em [2,3,4,5], valida em [1]
Rodada 2: treina em [1,3,4,5], valida em [2]
Rodada 3: treina em [1,2,4,5], valida em [3]
Rodada 4: treina em [1,2,3,5], valida em [4]
Rodada 5: treina em [1,2,3,4], valida em [5]
```

**Vantagens:**
- Usa todos os dados tanto para treino quanto para validação (em rodadas diferentes), aproveitando melhor a base.
- Fornece uma estimativa mais robusta e confiável do desempenho do modelo (média + variabilidade).
- Ajuda a identificar se o desempenho do modelo é instável (alto desvio padrão entre as dobras).

### 2.2 Validação cruzada estratificada

Para problemas de classificação com classes desbalanceadas (como nosso churn), usamos a **validação cruzada estratificada**, que garante que cada dobra mantenha a mesma proporção de classes que a base original — evitando dobras sem nenhum exemplo da classe minoritária.

### 2.3 Quando (e como) usar

- Normalmente, ainda se separa uma parte dos dados como **conjunto de teste final**, que **não participa** da validação cruzada nem de nenhum ajuste de hiperparâmetros — ele só é usado uma vez, ao final, para uma avaliação honesta do modelo escolhido.
- A validação cruzada é aplicada sobre o conjunto de **treino**, geralmente para comparar modelos/hiperparâmetros entre si (ex.: "qual valor de k no k-NN dá o melhor desempenho médio?").

---

## 3. Métricas para Regressão

Já usamos algumas na Aula 04 — vamos aprofundar:

### 3.1 MAE (Mean Absolute Error / Erro Absoluto Médio)

```
MAE = média( |y_real - y_previsto| )
```

- Interpretação direta: "em média, erramos X unidades" (na mesma unidade da variável-alvo).
- Não penaliza erros grandes desproporcionalmente — todos os erros pesam de forma linear.

### 3.2 MSE (Mean Squared Error / Erro Quadrático Médio)

```
MSE = média( (y_real - y_previsto)² )
```

- Penaliza mais fortemente erros grandes (por causa do quadrado) — útil quando erros grandes são particularmente indesejáveis.
- Não está na mesma unidade da variável-alvo (fica "ao quadrado"); por isso, é comum reportar o **RMSE** (raiz quadrada do MSE), que volta à unidade original.

### 3.3 Qual escolher?

- **MAE**: quando todos os erros importam de forma proporcional, e você quer uma métrica facilmente interpretável.
- **MSE/RMSE**: quando erros grandes são desproporcionalmente mais custosos (ex.: prever muito errado o estoque necessário pode gerar rupturas graves).

---

## 4. Métricas para Classificação

### 4.1 Matriz de confusão

Para um problema binário (ex.: churn = sim/não):

| | Previsto: Não | Previsto: Sim |
|---|---|---|
| **Real: Não** | Verdadeiro Negativo (VN) | Falso Positivo (FP) |
| **Real: Sim** | Falso Negativo (FN) | Verdadeiro Positivo (VP) |

### 4.2 Acurácia

```
Acurácia = (VP + VN) / Total
```

- Proporção geral de acertos. **Cuidado:** em bases desbalanceadas (como a nossa, com poucos casos de churn), a acurácia pode ser enganosa — um modelo que "chuta" sempre "não cancela" pode ter acurácia alta e ainda assim ser inútil (vimos isso nas Aulas 03 e 05).

### 4.3 Precisão (Precision)

```
Precisão = VP / (VP + FP)
```

- "Das vezes que o modelo previu 'vai cancelar', quantas vezes ele acertou?"
- Importante quando o **custo de um falso positivo é alto** (ex.: gastar uma campanha cara de retenção em clientes que não iam cancelar mesmo).

### 4.4 Recall (Sensibilidade / Revocação)

```
Recall = VP / (VP + FN)
```

- "Dos clientes que realmente cancelaram, quantos o modelo conseguiu identificar?"
- Importante quando o **custo de um falso negativo é alto** (ex.: deixar passar um cliente que ia cancelar e perder a chance de retê-lo; ou, em saúde, deixar passar um diagnóstico grave).

### 4.5 F1-score

```
F1 = 2 × (Precisão × Recall) / (Precisão + Recall)
```

- Média harmônica entre precisão e recall — útil quando você quer equilibrar as duas, especialmente em bases desbalanceadas.

### 4.6 AUC (Area Under the Curve) / Curva ROC

A **curva ROC** mostra a relação entre a Taxa de Verdadeiros Positivos (recall) e a Taxa de Falsos Positivos, à medida que variamos o **limiar de decisão** (threshold) do modelo (lembra do 0.5 padrão que mencionamos na Aula 04? Podemos variar esse valor).

A **AUC** (área sob essa curva) resume, em um único número entre 0 e 1, a capacidade do modelo de **separar as classes**, independentemente do limiar escolhido:

- AUC = 0.5 → o modelo é equivalente a "chutar" aleatoriamente.
- AUC = 1.0 → separação perfeita entre as classes.
- Na prática, valores acima de 0.7-0.8 já costumam ser considerados bons, dependendo do problema.

**Vantagem da AUC:** não depende da escolha de um limiar específico (0.5), nem é tão sensível ao desbalanceamento de classes quanto a acurácia — por isso é uma das métricas mais usadas para comparar modelos de classificação binária.

### 4.7 Como escolher a métrica certa? (pensando em negócio)

| Cenário de negócio | Métrica mais indicada |
|---|---|
| Toda campanha de retenção é barata, não quero perder nenhum cliente em risco | Priorizar **Recall** |
| Campanha de retenção é cara, só quero focar nos casos mais certos | Priorizar **Precisão** |
| Quero uma visão geral e equilibrada da capacidade de separação do modelo | **AUC** e **F1-score** |
| Prevendo um valor numérico (ex.: vendas, demanda) | **MAE** / **RMSE** |

**Próxima aula:** [Aula 07 — Aprendizado Não Supervisionado](../aula-07-aprendizado-nao-supervisionado/README.md)
