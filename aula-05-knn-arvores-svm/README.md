# Aula 05 — Modelos Supervisionados II: k-NN, Árvore de Decisão e SVM

## Objetivos da aula

- Entender a lógica do k-NN (k-vizinhos mais próximos).
- Entender como árvores de decisão tomam decisões baseadas em regras.
- Entender a ideia central do SVM (Máquinas de Vetores de Suporte).
- Saber comparar esses algoritmos entre si e com os vistos na Aula 04.

---

## 1. k-NN (k-Nearest Neighbors / k-Vizinhos Mais Próximos)

### 1.1 Ideia central

O k-NN é, talvez, o algoritmo mais intuitivo de todos: **"você é parecido com quem está por perto"**. Para classificar (ou prever) um novo exemplo, o algoritmo:

1. Calcula a **distância** entre o novo exemplo e todos os exemplos de treino.
2. Seleciona os **k exemplos mais próximos** (vizinhos).
3. Para classificação: usa a **classe majoritária** entre os k vizinhos.
4. Para regressão: usa a **média** dos valores dos k vizinhos.

### 1.2 Exemplo prático

Para prever se um novo cliente vai cancelar, o k-NN olha para os `k` clientes mais parecidos (em termos de idade, renda, tempo de casa etc.) e verifica: a maioria deles cancelou ou não?

### 1.3 Ponto crítico: escolha de `k`

- `k` muito pequeno (ex.: k=1): o modelo fica muito sensível a ruído (overfitting).
- `k` muito grande: o modelo fica "genérico demais", perdendo nuances (underfitting).
- Normalmente se testa vários valores de `k` usando validação cruzada (Aula 06).

### 1.4 Importância do pré-processamento

Como o k-NN se baseia inteiramente em **distância**, é fundamental **escalar as variáveis** (Aula 02) antes de usá-lo — variáveis em escalas maiores dominariam o cálculo de distância indevidamente.

### 1.5 Vantagens e limitações

**Vantagens:** simples, intuitivo, não faz suposições sobre a distribuição dos dados.
**Limitações:** lento para bases grandes (precisa calcular distância para todos os pontos a cada previsão), sensível a variáveis irrelevantes e à escala dos dados.

---

## 2. Árvore de Decisão

### 2.1 Ideia central

Uma árvore de decisão aprende uma sequência de **perguntas do tipo "sim/não"** sobre as variáveis, indo dividindo os dados em grupos cada vez mais "puros" (homogêneos em relação ao alvo).

Exemplo de árvore para prever churn:

```
tempo_de_casa_meses < 12?
├── Sim → qtd_chamados_suporte > 3?
│         ├── Sim → CANCELA
│         └── Não → NÃO CANCELA
└── Não → atraso_pagamento == 1?
          ├── Sim → CANCELA
          └── Não → NÃO CANCELA
```

Cada divisão (nó) é escolhida para **maximizar a separação entre as classes** (usando métricas como Gini ou entropia).

### 2.2 Vantagens e limitações

**Vantagens:**
- Fácil de visualizar e explicar para qualquer público (mesmo não técnico) — muito valioso em contextos regulatórios ou de negócio que exigem explicabilidade.
- Não exige escalonamento de variáveis (Aula 02).
- Lida naturalmente com variáveis numéricas e categóricas.

**Limitações:**
- Tende a **overfitting** (decorar os dados de treino) se crescer demais — controla-se isso limitando a profundidade máxima (`max_depth`) ou o número mínimo de exemplos por folha.
- Pode ser instável: pequenas mudanças nos dados podem gerar árvores bem diferentes.

> **Nota:** Random Forest e Gradient Boosting são extensões que combinam várias árvores para maior robustez — ficam como sugestão de aprofundamento após este curso introdutório.

---

## 3. SVM (Support Vector Machine / Máquina de Vetores de Suporte)

### 3.1 Ideia central

O SVM busca encontrar a **fronteira de decisão (hiperplano) que separa as classes com a MAIOR margem possível** — ou seja, a linha de separação que fica o mais distante possível dos pontos de ambas as classes mais próximos dela (os "vetores de suporte").

Imagine duas nuvens de pontos (clientes que cancelaram vs. não cancelaram): o SVM não busca só uma linha que separe as duas, mas a linha que deixa a "rua" mais larga possível entre elas — isso tende a gerar modelos que generalizam melhor para dados novos.

### 3.2 O "truque do kernel"

Quando os dados não são linearmente separáveis (não dá para separar as classes com uma linha reta), o SVM pode usar **funções de kernel** (ex.: RBF, polinomial) para projetar os dados em um espaço de maior dimensão, onde a separação linear se torna possível.

### 3.3 Vantagens e limitações

**Vantagens:** eficaz em espaços de alta dimensão, funciona bem mesmo com poucas amostras relativas ao número de variáveis.
**Limitações:** menos interpretável que árvores/regressão; pode ser custoso computacionalmente em bases muito grandes; exige escalonamento das variáveis (assim como o k-NN).

---

## 4. Comparativo geral (incluindo Aula 04)

| Algoritmo | Interpretabilidade | Precisa escalar dados? | Bom para bases grandes? | Ponto forte |
|---|---|---|---|---|
| Regressão Linear/Logística | Alta | Sim | Sim | Simplicidade e interpretação direta |
| k-NN | Média | Sim | Não (lento) | Simplicidade conceitual |
| Árvore de Decisão | Alta | Não | Sim | Explicabilidade (regras de negócio) |
| SVM | Baixa | Sim | Depende | Boa margem de separação, alta dimensão |

**Regra prática:** não existe "o melhor algoritmo" universal — a escolha depende do problema, do volume de dados, da necessidade de interpretabilidade e do tempo disponível para treinar/ajustar. Na prática, é comum testar múltiplos algoritmos e comparar seu desempenho de forma justa — é exatamente isso que aprenderemos a fazer com rigor na **Aula 06**.

**Próxima aula:** [Aula 06 — Avaliação de Modelos](../aula-06-avaliacao-de-modelos/README.md)
