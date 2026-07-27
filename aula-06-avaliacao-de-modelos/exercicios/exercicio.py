"""
Aula 06 - Exercício

Parte A (conceitual):

1. Um modelo de previsão de fraude tem 99% de acurácia, mas a base tem
   apenas 0,5% de casos de fraude. Isso é motivo para comemorar? Que outra(s)
   métrica(s) você pediria antes de aprovar esse modelo para produção?

2. Em qual situação você priorizaria RECALL sobre PRECISÃO, e em qual
   situação faria o oposto? Dê um exemplo de negócio para cada caso.

3. Por que a validação cruzada (k-fold) tende a fornecer uma estimativa mais
   confiável do desempenho de um modelo do que uma única divisão
   treino/teste?

Parte B (prática):

Usando `dados/clientes.csv`:

1. Treine um modelo de Árvore de Decisão (`max_depth=4`,
   `class_weight="balanced"`) para prever `churn`, usando as features
   `idade`, `renda_mensal`, `tempo_de_casa_meses`, `qtd_acessos_mes`,
   `qtd_chamados_suporte` e `atraso_pagamento`.
2. Rode uma validação cruzada estratificada com 5 dobras, reportando a
   média e o desvio padrão da métrica AUC.
3. Em seguida, faça uma divisão treino/teste única (80/20, random_state=42,
   stratify=y) e calcule, no conjunto de teste: acurácia, precisão, recall,
   F1-score e AUC.
4. Compare o AUC da validação cruzada com o AUC da divisão única. Eles são
   parecidos? O desvio padrão entre as dobras foi alto ou baixo?
5. Treine também um modelo de REGRESSÃO LINEAR para prever
   `valor_mensalidade` a partir de `plano` (dica: você precisará codificar
   essa variável categórica -- reveja a Aula 02!) e reporte MAE e MSE no
   teste. Esse resultado faz sentido, dado que `valor_mensalidade` é
   definido diretamente pelo plano contratado?
"""

# Escreva seu código abaixo
