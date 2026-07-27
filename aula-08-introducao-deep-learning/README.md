# Aula 08 — Introdução a Deep Learning e Redes Neurais

## Objetivos da aula

- Entender a estrutura básica de uma rede neural artificial (neurônio, camadas, pesos, função de ativação).
- Compreender o funcionamento do MLP (Perceptron Multicamadas).
- Ter uma visão geral de Redes Neurais Convolucionais (CNNs) e Redes Neurais Recorrentes (RNNs), e quando cada uma é usada.

---

## 1. O que é Deep Learning?

Deep Learning (Aprendizado Profundo) é um subcampo do Machine Learning baseado em **redes neurais artificiais com múltiplas camadas** ("profundas"). Diferente dos modelos vistos nas Aulas 04 e 05, redes neurais conseguem aprender **representações hierárquicas e não lineares** muito complexas dos dados — o que as torna especialmente poderosas para dados como imagens, áudio, texto e séries temporais longas.

> Deep Learning não substitui os modelos "clássicos" (regressão, árvores, SVM) — para muitos problemas de negócio com dados tabulares (planilhas, bancos de dados estruturados), os modelos das Aulas 04-05 continuam sendo excelentes escolhas, mais simples e interpretáveis. Deep Learning brilha principalmente quando há muitos dados não estruturados (imagens, texto, áudio, séries temporais complexas).

---

## 2. O neurônio artificial e o MLP (Multi-Layer Perceptron)

### 2.1 O neurônio artificial

Um neurônio artificial recebe entradas (`x1, x2, ..., xn`), cada uma multiplicada por um **peso** (`w1, w2, ..., wn`), soma tudo (mais um viés/bias `b`), e aplica uma **função de ativação**:

```
z = w1*x1 + w2*x2 + ... + wn*xn + b
saída = função_de_ativação(z)
```

Isso é, na verdade, muito parecido com a regressão logística que vimos na Aula 04 (que é, essencialmente, "um único neurônio" com ativação sigmoide)!

### 2.2 Funções de ativação comuns

- **Sigmoide**: comprime a saída entre 0 e 1 (útil na camada de saída de classificação binária).
- **ReLU (Rectified Linear Unit)**: `max(0, z)` — a mais usada em camadas internas hoje em dia, por ser simples e evitar alguns problemas de treinamento.
- **Softmax**: usada na camada de saída para classificação com múltiplas classes, transformando as saídas em probabilidades que somam 1.

### 2.3 MLP: empilhando neurônios em camadas

O **Perceptron Multicamadas (MLP)** organiza neurônios em **camadas**:

```
Camada de entrada → Camada(s) oculta(s) → Camada de saída
```

- **Camada de entrada:** recebe as variáveis (features) originais.
- **Camada(s) oculta(s):** cada neurônio combina as saídas da camada anterior de forma não linear — é aqui que a rede "aprende" representações complexas dos dados.
- **Camada de saída:** produz a previsão final (ex.: probabilidade de churn, valor previsto).

Ao "empilhar" várias camadas ocultas, a rede consegue aprender funções cada vez mais complexas — daí o nome "profundo" (deep).

### 2.4 Como a rede aprende? (intuição sobre backpropagation)

1. A rede faz uma previsão (passo "para frente"/*forward pass*).
2. Compara a previsão com o valor real, calculando um **erro** (função de perda/*loss*, ex.: erro quadrático para regressão, entropia cruzada para classificação).
3. O erro é propagado **de volta** pela rede (*backpropagation*), calculando o quanto cada peso contribuiu para o erro.
4. Os pesos são ajustados na direção que reduz o erro (usando um algoritmo de otimização, como o **gradiente descendente**).
5. Esse processo se repete por muitas iterações (*épocas*), até o erro se estabilizar.

Não é necessário implementar isso manualmente: bibliotecas como TensorFlow/Keras e PyTorch cuidam de todos esses cálculos automaticamente — o que precisamos entender é a **intuição** por trás do processo.

---

## 3. Redes Neurais Convolucionais (CNNs)

### 3.1 Para que servem

CNNs são especializadas em dados com **estrutura espacial**, principalmente **imagens** (mas também podem ser usadas em outros dados com padrões locais, como séries temporais ou até texto).

### 3.2 Ideia central: convolução

Em vez de conectar cada neurônio a **todos** os pixels da imagem (o que geraria um número gigantesco de parâmetros), a CNN usa **filtros (kernels)** pequenos que "deslizam" sobre a imagem, detectando padrões locais (bordas, texturas, formas) — e esses padrões vão se tornando mais complexos e abstratos à medida que passam por mais camadas (ex.: da detecção de bordas simples até o reconhecimento de "olho", "rosto", "gato").

Componentes típicos:
- **Camadas convolucionais:** aplicam os filtros e extraem características.
- **Camadas de pooling:** reduzem a dimensão espacial, resumindo a informação e tornando o modelo mais robusto a pequenas variações de posição.
- **Camadas densas (fully connected):** ao final, combinam as características extraídas para a decisão final (ex.: classificação da imagem).

### 3.3 Aplicações práticas

- Classificação de imagens (ex.: identificar defeitos em produtos numa linha de produção).
- Reconhecimento facial, leitura de documentos (OCR).
- Diagnóstico por imagem (ex.: raios-x, exames).

---

## 4. Redes Neurais Recorrentes (RNNs)

### 4.1 Para que servem

RNNs são especializadas em dados **sequenciais**, onde a ordem importa e há dependência entre elementos ao longo do tempo: texto, séries temporais (vendas mês a mês, sensores), áudio.

### 4.2 Ideia central: memória ao longo da sequência

Diferente do MLP (que trata cada entrada de forma independente), a RNN mantém um **estado interno (memória)** que é atualizado a cada novo elemento da sequência, permitindo que a rede "lembre" de informações de passos anteriores para influenciar a previsão atual.

```
entrada(t=1) → [RNN] → estado(1) → previsão(1)
                  ↓
entrada(t=2) → [RNN] → estado(2) → previsão(2)
                  ↓
entrada(t=3) → [RNN] → estado(3) → previsão(3)
```

### 4.3 Variantes mais robustas

RNNs simples têm dificuldade em "lembrar" de informações muito distantes no passado (problema do gradiente desvanecente). Por isso, criaram-se variantes mais sofisticadas:
- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**

Essas variantes têm mecanismos especiais ("portões") para decidir o que lembrar e o que esquecer ao longo da sequência — fica como sugestão de aprofundamento após este curso.

### 4.4 Aplicações práticas

- Previsão de séries temporais (vendas, demanda, consumo de energia).
- Processamento de linguagem natural (tradução, análise de sentimento, chatbots) — hoje, em grande parte substituído por arquiteturas mais modernas (Transformers), mas o conceito de RNN é a base histórica e conceitual desse campo.

---

## 5. Resumo comparativo

| Arquitetura | Tipo de dado ideal | Exemplo de aplicação |
|---|---|---|
| MLP | Dados tabulares/estruturados, vetores de features | Previsão de churn, classificação genérica |
| CNN | Dados com estrutura espacial (imagens) | Classificação de imagens, visão computacional |
| RNN (e variantes LSTM/GRU) | Dados sequenciais/temporais | Séries temporais, texto, áudio |

---

## 6. Quando vale a pena usar Deep Learning?

- Quando há **grande volume de dados** disponível (redes neurais profundas geralmente precisam de mais dados que modelos clássicos para não sofrerem overfitting).
- Quando os dados são **não estruturados** (imagens, áudio, texto) ou têm padrões muito complexos e não lineares.
- Quando a **interpretabilidade não é o requisito mais crítico** (redes neurais tendem a ser menos interpretáveis que árvores de decisão ou regressão).

Para dados tabulares "do dia a dia" (planilhas de vendas, RH, financeiro), frequentemente os modelos das Aulas 04 e 05 (ou ensembles como Random Forest/Gradient Boosting) já entregam ótimo desempenho com muito menos complexidade e mais interpretabilidade — vale sempre considerar essa relação custo-benefício antes de partir direto para Deep Learning.

**Este era o conteúdo final da ementa do curso.** Veja o [README principal](../README.md) para sugestões de próximos passos.
