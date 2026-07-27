"""
Aula 02 - Exercício

Parte A (conceitual):

1. Você está preparando dados de idade (18 a 75 anos) e salário (R$ 1.200 a
   R$ 20.000) para um algoritmo de k-NN (baseado em distância). Por que é
   importante escalar essas variáveis antes de treinar o modelo? O que
   aconteceria se você não escalasse?

2. A variável "nível de escolaridade" tem as categorias: Fundamental, Médio,
   Superior, Pós-graduação. Você usaria One-Hot Encoding ou Ordinal Encoding?
   Justifique.

3. A variável "cidade" tem 5 categorias sem nenhuma ordem natural. Você
   usaria One-Hot Encoding ou Ordinal Encoding? Justifique.

Parte B (prática):

Usando `dados/clientes.csv`:

1. Carregue o dataset e identifique quais colunas têm valores ausentes.
2. Trate os valores ausentes de `renda_mensal` usando a MEDIANA, e de
   `qtd_acessos_mes` também usando a MEDIANA (justifique por que a mediana é
   uma escolha mais segura que a média quando há outliers).
3. Separe o dataset em treino (80%) e teste (20%) usando `train_test_split`
   com `random_state=42`.
4. Padronize (Z-score) as colunas `idade`, `renda_mensal` e
   `tempo_de_casa_meses`, ajustando o `StandardScaler` SOMENTE nos dados de
   treino.
5. Aplique One-Hot Encoding na coluna `plano`.
6. Ao final, exiba o formato (shape) do conjunto de treino processado.

Desafio extra: refaça o item 4 usando `MinMaxScaler` em vez de
`StandardScaler` e compare os valores resultantes para os 5 primeiros
clientes do treino.
"""

# Escreva seu código abaixo
