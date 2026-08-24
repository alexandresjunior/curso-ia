# Aula 01 — Fundamentos de IA/ML e Tipos de Aprendizado

## Objetivos da aula

- Entender o que é Inteligência Artificial, Machine Learning (ML) e como eles se relacionam.
- Compreender os três grandes paradigmas de aprendizado: **supervisionado**, **não supervisionado** e **semi-supervisionado**.
- Reconhecer, em problemas do dia a dia profissional, qual tipo de aprendizado se aplica.

---

## 1. O que é Inteligência Artificial e Machine Learning?

**Inteligência Artificial (IA)** é a área da computação que busca fazer máquinas executarem tarefas que, tradicionalmente, exigiriam inteligência humana: reconhecer padrões, tomar decisões, entender linguagem, prever eventos.

**Machine Learning (ML)**, ou Aprendizado de Máquina, é o principal caminho usado hoje para se chegar à IA: em vez de programarmos regras explícitas ("se renda > X e idade < Y, então..."), damos **dados** para um algoritmo, e ele **aprende os padrões sozinho**.

> Programação tradicional: `Regras + Dados → Resultado`
> Machine Learning: `Dados + Resultado (exemplos) → Regras (modelo)`

Esse modelo aprendido pode depois ser usado para prever resultados em dados novos, nunca vistos antes.

<div align="center">
  <img width="640" height="320" alt="ia_versus_ml" src="https://github.com/user-attachments/assets/ee7c770a-21d8-4609-b618-5353a4726e74" />
</div>

### Por que isso importa para qualquer profissional?

Não é preciso ser cientista de dados para se beneficiar de ML. Alguns exemplos por área:

| Área | Aplicação de ML |
|---|---|
| Marketing | Prever quais clientes vão cancelar a assinatura (churn) |
| RH | Identificar padrões de rotatividade (turnover) de funcionários |
| Financeiro | Detectar transações fraudulentas, prever inadimplência |
| Saúde | Auxiliar diagnósticos a partir de exames |
| Operações | Prever demanda e otimizar estoque |
| Vendas | Segmentar clientes por perfil de compra |

---

## 2. Os três tipos de aprendizado

A diferença fundamental entre os tipos de aprendizado está em **o que os dados de treino contêm**: eles trazem "respostas certas" (rótulos) ou não?

### 2.1 Aprendizado Supervisionado

No aprendizado supervisionado, cada exemplo de treino tem um **rótulo** (a resposta certa). O algoritmo aprende a mapear entradas (variáveis explicativas, ou *features*) para saídas (variável-alvo, ou *target*).

- **Exemplo:** um histórico de clientes com dados (idade, renda, tempo de casa...) **e** a informação de se cada um cancelou ou não o serviço. O modelo aprende a prever o cancelamento de novos clientes.

Divide-se em dois grandes problemas:
- **Classificação**: a saída é uma categoria (ex.: "vai cancelar" / "não vai cancelar", "é spam" / "não é spam").
- **Regressão**: a saída é um número contínuo (ex.: prever o valor de uma venda, o preço de um imóvel).

Veremos os principais algoritmos supervisionados nas Aulas 4 e 5.

### 2.2 Aprendizado Não Supervisionado

Aqui, os dados **não têm rótulo**. O algoritmo tenta encontrar **estrutura ou padrões escondidos** nos dados sozinho, sem saber de antemão qual é a resposta "certa".

- **Exemplo:** você tem uma base de clientes, sem nenhuma classificação prévia, e quer descobrir se existem "grupos naturais" (perfis) de clientes com comportamentos parecidos — isso é **clusterização (agrupamento)**.

Principais tarefas:
- **Clusterização (clustering)**: agrupar exemplos parecidos (ex.: K-means, DBSCAN — Aula 7).
- **Redução de dimensionalidade**: simplificar dados com muitas variáveis em poucas dimensões, mantendo a informação relevante (ex.: PCA — Aula 7).

### 2.3 Aprendizado Semi-Supervisionado

É uma solução intermediária, muito comum na prática: você tem **poucos dados rotulados** (rotular dados costuma ser caro/manual — ex.: um analista revisando cada contrato) e **muitos dados não rotulados** (fáceis de coletar).

O algoritmo usa a pequena parte rotulada para "guiar" o aprendizado, e a grande massa de dados não rotulados para entender melhor a estrutura geral do problema, geralmente resultando em modelos melhores do que usar apenas os poucos dados rotulados isoladamente.

- **Exemplo:** uma empresa tem milhares de e-mails de clientes, mas só 200 foram manualmente classificados como "reclamação", "elogio" ou "dúvida". Um modelo semi-supervisionado usa esses 200 rótulos + a estrutura dos milhares de e-mails não rotulados para classificar todo o restante com melhor qualidade do que usando só os 200 exemplos.

<div align="center">
  <img width="586" height="523" alt="tres_tipos_aprendizado" src="https://github.com/user-attachments/assets/bbd701a0-24f1-450a-a25a-7f9c3ee16f5d" />
</div>

### Resumo comparativo

| Tipo | Dados rotulados? | Objetivo típico | Exemplos de algoritmos |
|---|---|---|---|
| Supervisionado | Sim, todos | Prever uma saída conhecida | Regressão linear/logística, k-NN, árvore de decisão, SVM |
| Não supervisionado | Não | Encontrar estrutura/padrões | K-means, DBSCAN, PCA |
| Semi-supervisionado | Parcialmente | Aproveitar poucos rótulos + muitos dados | Self-training, label propagation |

> Existe ainda o **aprendizado por reforço**, em que um agente aprende por tentativa e erro recebendo recompensas (usado em robótica, jogos, otimização) — não é foco deste curso, mas vale saber que existe.

---

## 3. Como decidir qual tipo de aprendizado usar?

Perguntas-guia:

1. **Eu tenho um "rótulo" histórico que quero prever no futuro?** → Supervisionado.
2. **Eu não tenho rótulo, só quero entender/organizar os dados?** → Não supervisionado.
3. **Eu tenho poucos rótulos e uma massa grande de dados sem rótulo?** → Semi-supervisionado.

---

## 4. O fio condutor do curso

A partir da Aula 2, usaremos um **dataset fictício de clientes de uma empresa de assinatura** (`dados/clientes.csv`), com informações como idade, renda, plano contratado, tempo de casa, uso do produto e se o cliente cancelou (`churn`). Esse mesmo dataset será reaproveitado em quase todas as aulas — assim, você constrói um raciocínio contínuo, do pré-processamento até modelos de deep learning.

---

## Para refletir

Pense em um processo do seu trabalho que envolve prever algo (ex.: quem vai comprar, quem vai atrasar pagamento, qual equipamento vai falhar) ou agrupar algo (ex.: perfis de clientes, perfis de fornecedores). Em qual dos três tipos de aprendizado esse problema se encaixaria? Vamos retomar essa reflexão nos exercícios.

**Próxima aula:** [Aula 02 — Pré-processamento de dados](../aula-02-pre-processamento-de-dados/README.md)
