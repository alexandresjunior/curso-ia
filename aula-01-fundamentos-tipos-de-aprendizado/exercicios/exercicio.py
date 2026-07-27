"""
Aula 01 - Exercício

Parte A (conceitual — responda em comentários neste próprio arquivo ou em um
documento à parte):

Para cada situação abaixo, identifique se o problema é de aprendizado
SUPERVISIONADO, NÃO SUPERVISIONADO ou SEMI-SUPERVISIONADO, e justifique em
1 frase.

1. Uma rede de varejo tem o histórico de compras de todos os clientes (sem
   nenhuma categoria pré-definida) e quer descobrir "perfis de consumo"
   parecidos para criar campanhas direcionadas.

2. Um banco tem o histórico de transações dos últimos 5 anos, todas marcadas
   como "fraude" ou "não fraude", e quer treinar um sistema para identificar
   fraudes em transações novas.

3. Uma empresa de RH tem currículos de 10.000 candidatos, mas só 300 foram
   manualmente avaliados como "aderente" ou "não aderente" à vaga. Ela quer
   usar isso para triar automaticamente os outros 9.700.

4. Uma clínica quer prever, com base em exames anteriores (todos com
   diagnóstico confirmado), se um novo paciente tem risco de diabetes.


Parte B (prática):

Usando o dataset `dados/clientes.csv` (o mesmo usado no exemplo desta aula):

1. Carregue o dataset com pandas.
2. Imprima quantos clientes existem no total, e quantos são "churn = 1"
   (cancelaram) e "churn = 0" (não cancelaram).
3. Crie um gráfico de dispersão (scatter plot) entre `renda_mensal` (eixo X) e
   `qtd_chamados_suporte` (eixo Y):
   a) Uma versão SEM usar a coluna `churn` (visão não supervisionada).
   b) Uma versão colorindo os pontos pela coluna `churn` (visão supervisionada).
4. Escreva, em um comentário, o que você observa de diferente entre as duas
   visões: dá para "enxergar" visualmente algum padrão relacionado ao churn?

Dica: reaproveite a estrutura do arquivo de exemplo desta aula.
"""

# Escreva seu código abaixo
