# Aula 04 — Modelos Supervisionados I: Regressão Linear e Logística

## Objetivos da aula

- Entender a regressão linear como ferramenta para prever valores numéricos contínuos.
- Entender a regressão logística como ferramenta para classificação binária.
- Saber interpretar os coeficientes de ambos os modelos.

---

## 1. Regressão Linear

A regressão linear é usada para prever uma **variável numérica contínua** (ex.: valor de venda, tempo de entrega, receita) a partir de uma ou mais variáveis explicativas.

### 1.1 Ideia central

O modelo tenta encontrar a **reta (ou hiperplano, com mais de uma variável) que melhor se ajusta aos dados**:

```
y = b0 + b1*x1 + b2*x2 + ... + bn*xn
```

- `y` → variável que queremos prever (ex.: valor da próxima compra).
- `x1, x2, ..., xn` → variáveis explicativas (ex.: idade, renda, tempo de casa).
- `b0` → intercepto (valor previsto quando todas as variáveis explicativas são 0).
- `b1, b2, ..., bn` → coeficientes: o quanto `y` muda para cada unidade de aumento em cada `x`, mantendo as outras variáveis constantes.

O ajuste é feito minimizando o **erro quadrático médio** entre os valores previstos e os valores reais (método dos mínimos quadrados) — voltaremos a essa métrica (MSE) na Aula 06.

### 1.2 Interpretação prática

Se um modelo de previsão de gasto mensal (`y`) resulta em `b_renda = 0.08`, isso significa: "a cada R$ 1 a mais de renda mensal, esperamos, em média, R$ 0,08 a mais de gasto mensal, mantendo as demais variáveis constantes".

### 1.3 Quando usar

- A variável-alvo é numérica e contínua.
- Existe (ao menos aproximadamente) uma relação linear entre as variáveis explicativas e o alvo.
- Você quer um modelo simples e interpretável.

### 1.4 Limitações

- Não captura relações não lineares complexas (a não ser que você crie manualmente termos polinomiais ou de interação).
- Sensível a outliers.
- Assume que as variáveis explicativas não são fortemente correlacionadas entre si (multicolinearidade).

---

## 2. Regressão Logística

Apesar do nome "regressão", a regressão logística é usada para **problemas de classificação** — mais comumente, classificação **binária** (duas classes, ex.: "vai cancelar" / "não vai cancelar").

### 2.1 Ideia central

Em vez de prever diretamente a classe, o modelo prevê a **probabilidade** de o exemplo pertencer à classe positiva, usando a função logística (sigmoide) para transformar uma combinação linear das variáveis em um valor entre 0 e 1:

```
z = b0 + b1*x1 + b2*x2 + ... + bn*xn
P(y=1 | x) = 1 / (1 + e^(-z))
```

- Se `P(y=1|x) > 0.5` (limiar padrão), classificamos como classe 1; caso contrário, classe 0. Esse limiar (*threshold*) pode ser ajustado conforme o problema de negócio (veremos isso na Aula 06, ao discutir AUC e trade-off entre precisão e recall).

### 2.2 Interpretação dos coeficientes

Diferente da regressão linear, os coeficientes da regressão logística afetam a **chance (odds)** de forma multiplicativa. Um coeficiente positivo aumenta a probabilidade da classe 1; um coeficiente negativo diminui.

- Exemplo: se o coeficiente de `atraso_pagamento` for positivo e grande, isso indica que clientes com atraso de pagamento têm chance bem maior de cancelar.

### 2.3 Quando usar

- A variável-alvo é categórica binária.
- Você quer não só a classificação, mas também uma **probabilidade interpretável** (muito valioso em negócio: "esse cliente tem 78% de chance de cancelar" é mais acionável do que apenas "vai cancelar").
- Quer um modelo rápido, interpretável e como bom ponto de partida (baseline).

### 2.4 Extensão para múltiplas classes

A regressão logística pode ser estendida para problemas com mais de duas classes (regressão logística multinomial), embora isso fuja do escopo introdutório desta aula.

---

## 3. Regressão Linear vs. Logística

<img width="1203" height="454" alt="reg_linear_vs_reg_log" src="https://github.com/user-attachments/assets/5b1e5856-86cb-4810-a114-f343456dc3cc" />


| | Regressão Linear | Regressão Logística |
|---|---|---|
| Tipo de problema | Regressão (valor contínuo) | Classificação (categorias) |
| Saída do modelo | Número real | Probabilidade entre 0 e 1 |
| Função usada | Combinação linear direta | Combinação linear + função sigmoide |
| Exemplo de uso | Prever valor de vendas do próximo mês | Prever se um cliente vai cancelar |

Ambos os modelos são a base de praticamente toda a área de ML supervisionado, e servem como **excelente ponto de partida (baseline)** antes de partir para modelos mais complexos, como os que veremos na Aula 05 (k-NN, árvores de decisão, SVM).

**Próxima aula:** [Aula 05 — k-NN, Árvore de Decisão e SVM](../aula-05-knn-arvores-svm/README.md)
