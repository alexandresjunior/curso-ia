"""
Aula 05 - Exercício

Parte A (conceitual):

1. Por que é fundamental escalar as variáveis antes de usar k-NN ou SVM, mas
   isso não é necessário para uma árvore de decisão?

2. Uma árvore de decisão sem limite de profundidade (`max_depth=None`) atinge
   100% de acurácia no treino, mas apenas 62% no teste. O que está
   acontecendo, e como você resolveria isso?

3. Cite uma vantagem de negócio de se usar uma árvore de decisão em vez de um
   SVM em um contexto onde o modelo precisa ser explicado a um comitê de
   crédito ou a um órgão regulador.

Parte B (prática):

Usando `dados/clientes.csv`:

1. Treine três modelos para prever `churn`, usando as features `idade`,
   `renda_mensal`, `tempo_de_casa_meses`, `qtd_acessos_mes` e
   `qtd_chamados_suporte`:
   a) k-NN, testando 3 valores diferentes de `k` (ex.: 3, 7, 15).
   b) Árvore de decisão com `max_depth=3` e depois com `max_depth=None`.
   c) SVM com kernel `linear` e depois com kernel `rbf`.
2. Para cada modelo, calcule acurácia e F1-score no conjunto de teste
   (80/20, `random_state=42`, `stratify=y`).
3. Monte uma tabela comparativa (pode ser um `print` formatado ou um
   DataFrame) com todos os resultados.
4. Qual configuração teve o melhor F1-score? Isso bate com sua expectativa
   conceitual da Parte A?
"""

# Escreva seu código abaixo
