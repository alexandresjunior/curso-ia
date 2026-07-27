# Aula 02 — Pré-processamento de Dados

## Objetivos da aula

- Entender por que os algoritmos de ML quase sempre exigem dados "tratados".
- Diferenciar **normalização** e **padronização**, e saber quando usar cada uma.
- Aprender técnicas de **codificação de variáveis categóricas** (One-Hot Encoding, Label/Ordinal Encoding).
- Lidar com valores ausentes de forma simples e responsável.

---

## 1. Por que pré-processar os dados?

Dados do mundo real raramente estão prontos para alimentar um modelo:

- Variáveis numéricas podem estar em **escalas muito diferentes** (ex.: idade de 18 a 75, renda de 1.200 a 20.000). Muitos algoritmos (k-NN, SVM, redes neurais, regressão com regularização) são sensíveis a essa diferença de escala.
- Variáveis categóricas (texto, categorias) como "cidade" ou "plano" **não podem ser interpretadas diretamente** pela maioria dos algoritmos, que trabalham com números.
- Podem existir **valores ausentes** (campos vazios) que precisam de tratamento antes de treinar o modelo.

> Regra geral: **"Garbage in, garbage out"** — um modelo é tão bom quanto os dados usados para treiná-lo.

---

## 2. Normalização vs. Padronização

Ambas são formas de **reescalar variáveis numéricas** para que fiquem em intervalos comparáveis, mas com propósitos ligeiramente diferentes.

### 2.1 Normalização (Min-Max Scaling)

Reescala os valores para um intervalo fixo, tipicamente **[0, 1]**:

```
x_norm = (x - x_min) / (x_max - x_min)
```

- **Quando usar:** quando você sabe (ou espera) que os dados têm limites bem definidos, e quer preservar a forma da distribuição original. Muito usado em redes neurais e em algoritmos baseados em distância como k-NN.
- **Cuidado:** é sensível a *outliers* (valores extremos) — um único valor muito alto "espreme" todos os outros para perto de 0.

### 2.2 Padronização (Standardization / Z-score)

Reescala os valores para que tenham **média 0 e desvio padrão 1**:

```
x_padr = (x - média) / desvio_padrão
```

- **Quando usar:** é a escolha mais comum e "segura" em ML, especialmente quando os dados seguem (ou se aproximam de) uma distribuição normal, ou quando o algoritmo assume isso (ex.: regressão linear/logística com regularização, SVM, PCA).
- **Vantagem:** menos sensível a outliers do que a normalização min-max (embora ainda seja afetada por eles).

### 2.3 Qual escolher?

| Situação | Recomendação |
|---|---|
| Dados com distribuição aproximadamente normal | Padronização |
| Algoritmos baseados em distância (k-NN, k-means, SVM) | Padronização (ou normalização) |
| Redes neurais (entrada limitada, ex.: pixels 0-255) | Normalização |
| Árvores de decisão, Random Forest | Geralmente **não é necessário** escalar — esses modelos são baseados em divisões/regras, não em distância |
| Presença de outliers fortes | Padronização, ou técnicas robustas (ex.: RobustScaler) |

**Importante:** o "ajuste" da escala (calcular média, desvio, mínimo, máximo) deve ser feito **apenas com os dados de treino**, e depois aplicado (não recalculado) nos dados de teste — isso evita "vazamento de informação" (*data leakage*) do conjunto de teste para o treino.

---

## 3. Codificação de variáveis categóricas

Variáveis categóricas representam categorias/rótulos (ex.: cidade, plano, sexo, categoria de produto). A maioria dos algoritmos exige que tudo seja numérico, então é preciso **codificar** essas variáveis.

### 3.1 One-Hot Encoding

Cria uma coluna binária (0/1) para cada categoria possível.

Exemplo: a coluna `plano` com valores `Básico`, `Padrão`, `Premium` vira três colunas:

| plano_Básico | plano_Padrão | plano_Premium |
|---|---|---|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

- **Quando usar:** quando as categorias **não têm ordem natural** (ex.: cidade, cor, tipo de produto). É a técnica mais usada e mais segura na maioria dos casos.
- **Cuidado:** se a variável tem muitas categorias distintas (alta cardinalidade, ex.: "CEP"), o One-Hot pode gerar um número enorme de colunas.

### 3.2 Label Encoding / Ordinal Encoding

Atribui um número inteiro a cada categoria (ex.: `Básico`→0, `Padrão`→1, `Premium`→2).

- **Quando usar:** quando existe uma **ordem natural** entre as categorias (ex.: "Básico < Padrão < Premium", "Baixo < Médio < Alto"). Nesse caso, chamamos de **Ordinal Encoding**.
- **Cuidado:** se não há ordem real (ex.: cidades), usar Label Encoding pode enganar o algoritmo, fazendo-o "achar" que uma categoria é maior/menor que outra sem sentido algum — nesse caso, prefira One-Hot.

### 3.3 Resumo

| Tipo de variável | Técnica recomendada |
|---|---|
| Categórica sem ordem (nominal) | One-Hot Encoding |
| Categórica com ordem (ordinal) | Ordinal/Label Encoding |
| Alta cardinalidade (muitas categorias) | Técnicas avançadas (target encoding, frequency encoding) — fora do escopo deste curso introdutório |

---

## 4. Valores ausentes (missing values)

Estratégias simples e comuns:

- **Remover linhas/colunas** com muitos valores ausentes (se forem poucas linhas afetadas e a base for grande).
- **Imputação pela média/mediana** (variáveis numéricas) — a mediana é mais robusta a outliers.
- **Imputação pela moda** (variáveis categóricas) — o valor mais frequente.
- Técnicas mais avançadas (ex.: imputação por modelos preditivos) existem, mas fogem do escopo introdutório deste curso.

---

## 5. Fluxo típico de pré-processamento

1. Explorar os dados (`.info()`, `.describe()`, verificar `.isna().sum()`).
2. Tratar valores ausentes.
3. Codificar variáveis categóricas.
4. Separar treino/teste **antes de** escalar os dados (evitar vazamento de dados).
5. Ajustar o "scaler" (normalizador/padronizador) apenas nos dados de treino, e aplicá-lo em treino e teste.

Veremos esse fluxo completo no exemplo prático desta aula, usando o `scikit-learn` (`StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `SimpleImputer`).

**Próxima aula:** [Aula 03 — Teorema de Bayes](../aula-03-teorema-de-bayes/README.md)
