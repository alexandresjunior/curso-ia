"""
Aula 04 - Exercício

Parte A (conceitual):

1. Um modelo de regressão linear para prever o valor de vendas mensais de uma
   loja resulta no coeficiente `b_investimento_marketing = 3.2`. O que isso
   significa, em termos de negócio?

2. Por que não devemos usar regressão LINEAR para prever se um cliente vai
   cancelar (0 ou 1)? O que pode dar errado?

3. Em um modelo de regressão logística para prever inadimplência, o
   coeficiente de "número de faturas em atraso nos últimos 12 meses" é
   +1.8, e o de "tempo de relacionamento com o banco (anos)" é -0.4. Explique,
   em palavras simples, o que cada um indica sobre o risco de inadimplência.

Parte B (prática):

Usando `dados/clientes.csv`:

1. Treine um modelo de REGRESSÃO LINEAR para prever `qtd_acessos_mes` a
   partir de `idade`, `renda_mensal` e `tempo_de_casa_meses`.
   - Separe treino/teste (80/20, random_state=42).
   - Reporte MAE, MSE e R² no conjunto de teste.
   - Interprete o coeficiente da variável `renda_mensal`: aumentar a renda
     está associado a mais ou menos acessos mensais?

2. Treine um modelo de REGRESSÃO LOGÍSTICA para prever `churn`, usando
   `idade`, `renda_mensal`, `tempo_de_casa_meses`, `qtd_acessos_mes` e
   `qtd_chamados_suporte` (sem incluir `atraso_pagamento` desta vez).
   - Separe treino/teste (80/20, random_state=42, stratify=y).
   - Reporte a acurácia no teste.
   - Escolha o cliente do conjunto de teste com MAIOR probabilidade prevista
     de churn e imprima seus dados originais (antes da padronização).
"""

# Escreva seu código abaixo
