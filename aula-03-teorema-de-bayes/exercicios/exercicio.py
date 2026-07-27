"""
Aula 03 - Exercício

Parte A (conceitual — aplique o Teorema de Bayes "na mão" ou em código):

Uma empresa de e-commerce usa um sistema antifraude que:
- Detecta corretamente 98% das transações realmente fraudulentas (sensibilidade).
- Classifica erroneamente como fraude 5% das transações legítimas (taxa de
  falso positivo).
- Apenas 0,2% de todas as transações são de fato fraudulentas.

Se uma transação for marcada como "fraude" pelo sistema, qual é a
probabilidade real de que ela seja, de fato, fraudulenta? Calcule usando o
Teorema de Bayes e reflita: esse resultado mudaria a forma como a equipe de
risco trata cada alerta?

Parte B (prática):

Usando `dados/clientes.csv`:

1. Treine um modelo `GaussianNB` para prever `churn`, usando como features:
   `idade`, `valor_mensalidade`, `tempo_de_casa_meses`, `qtd_acessos_mes`,
   `qtd_chamados_suporte` e `atraso_pagamento`.
2. Separe treino (80%) / teste (20%) com `random_state=42` e `stratify=y`.
3. Treine o modelo e calcule a acurácia no conjunto de teste.
4. Escolha 3 clientes do conjunto de teste e imprima a probabilidade prevista
   de cancelamento para cada um, junto com o valor real de `churn`.
5. Adicione a feature `atraso_pagamento` foi importante? Compare a acurácia
   do modelo COM e SEM essa variável.

Dica: reaproveite a estrutura do exemplo desta aula.
"""

# Escreva seu código abaixo
