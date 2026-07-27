# Aula 03 — Teorema de Bayes e Aplicações em Ciência de Dados

## Objetivos da aula

- Entender probabilidade condicional e o Teorema de Bayes de forma intuitiva.
- Compreender como o Teorema de Bayes fundamenta o algoritmo **Naive Bayes**, muito usado em classificação de texto (ex.: spam, sentimento).
- Aplicar Bayes em um problema de negócio prático.

---

## 1. Probabilidade condicional

Probabilidade condicional é a probabilidade de um evento **A** acontecer, **dado que já sabemos** que outro evento **B** aconteceu. Notação: `P(A | B)`.

**Exemplo:** `P(cliente cancelar | cliente teve atraso de pagamento)` é diferente de `P(cliente cancelar)` no geral — saber que houve atraso muda nossa expectativa sobre o cancelamento.

---

## 2. O Teorema de Bayes

O Teorema de Bayes nos permite **inverter** uma probabilidade condicional: a partir de `P(B | A)`, conseguimos calcular `P(A | B)`.

```
P(A | B) = [ P(B | A) × P(A) ] / P(B)
```

Em palavras:

```
P(hipótese | evidência) = [ P(evidência | hipótese) × P(hipótese) ] / P(evidência)
```

- **P(A)** — *probabilidade a priori*: o que sabíamos sobre A antes de ver a evidência.
- **P(B | A)** — *verossimilhança*: quão provável é observar a evidência B, se A for verdade.
- **P(A | B)** — *probabilidade a posteriori*: nossa crença atualizada sobre A, depois de observar B.
- **P(B)** — probabilidade total da evidência (normalizador).

### 2.1 Exemplo clássico (exame médico)

Um exame para uma doença rara tem 95% de sensibilidade (detecta corretamente 95% dos doentes) e 90% de especificidade (acerta 90% dos saudáveis). A doença afeta 1% da população. Se uma pessoa testar positivo, qual a chance real de ela estar doente?

Muita gente responde intuitivamente "95%", mas isso está errado — a resposta correta considera **quão rara é a doença**:

```
P(doente) = 0,01
P(positivo | doente) = 0,95
P(positivo | saudável) = 0,10  (1 - especificidade)
P(saudável) = 0,99

P(positivo) = P(positivo|doente)×P(doente) + P(positivo|saudável)×P(saudável)
            = 0,95 × 0,01 + 0,10 × 0,99
            = 0,0095 + 0,099
            = 0,1085

P(doente | positivo) = (0,95 × 0,01) / 0,1085 ≈ 0,0876 → cerca de 8,8%
```

Ou seja: mesmo com um teste positivo em um exame razoavelmente bom, a chance real de estar doente é de apenas ~8,8%, porque a doença é rara. Esse é o tipo de raciocínio que o Teorema de Bayes torna possível — e é um erro de interpretação muito comum em decisões de negócio (ex.: interpretar um "alerta de fraude" sem considerar quão rara é a fraude).

Veremos esse cálculo no exemplo prático desta aula.

---

## 3. Naive Bayes: usando o Teorema de Bayes para classificação

O **Naive Bayes** é um algoritmo de classificação supervisionada que aplica o Teorema de Bayes para prever a classe mais provável de um exemplo, com base em suas características (features).

```
P(classe | features) ∝ P(features | classe) × P(classe)
```

O termo "naive" (ingênuo) vem de uma simplificação forte: o algoritmo assume que **todas as features são independentes entre si**, dado a classe. Isso raramente é verdade na prática, mas, curiosamente, o algoritmo funciona muito bem em diversos problemas reais, principalmente em:

- **Classificação de texto**: spam vs. não-spam, análise de sentimento, categorização de tickets de suporte.
- **Diagnóstico simples** com poucas variáveis.
- Como um **baseline rápido** antes de partir para modelos mais complexos.

### 3.1 Como funciona (intuição)

Para cada classe possível (ex.: "spam" e "não spam"), o modelo calcula a probabilidade de observar aquele conjunto de features, assumindo independência, e multiplica pela probabilidade a priori da classe. A classe com maior probabilidade resultante é a predição final.

### 3.2 Variantes mais comuns

| Variante | Quando usar |
|---|---|
| `GaussianNB` | Features numéricas contínuas (assume distribuição normal) |
| `MultinomialNB` | Contagens (ex.: frequência de palavras em um texto) |
| `BernoulliNB` | Features binárias (0/1), como presença/ausência de uma palavra |

### 3.3 Vantagens e limitações

**Vantagens:**
- Extremamente rápido de treinar, mesmo com muitos dados.
- Funciona bem mesmo com poucos dados de treino.
- Fácil de interpretar.

**Limitações:**
- Assume independência entre features (raramente verdadeiro), o que pode limitar a acurácia em problemas mais complexos.
- Estimativas de probabilidade tendem a ser menos confiáveis (embora a classificação final costume ser boa).

---

## 4. Aplicações em ciência de dados no dia a dia

- **Marketing:** classificar leads como "quente" ou "frio" com base em algumas poucas características.
- **Suporte ao cliente:** categorizar automaticamente tickets como "financeiro", "técnico", "elogio".
- **Compliance/Risco:** interpretar corretamente alertas de sistemas (ex.: "esse alerta de fraude realmente indica alta chance de fraude, dado quão raro é o evento?").
- **Saúde:** apoio a triagem inicial com poucos sintomas.

No exemplo prático, aplicaremos Naive Bayes para classificar o risco de cancelamento (`churn`) de clientes, e também refaremos manualmente o cálculo de Bayes do exemplo do exame médico.

**Próxima aula:** [Aula 04 — Regressão Linear e Logística](../aula-04-regressao-linear-e-logistica/README.md)
