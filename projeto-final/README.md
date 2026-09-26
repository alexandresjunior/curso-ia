# Projeto #01: Modelagem Preditiva de Indicadores Socioeconômicos Globais

## 📋 Descrição do Projeto

Este projeto consolida os conhecimentos adquiridos ao longo das seis primeiras aulas do curso. O objetivo é desenvolver um pipeline completo de Machine Learning — desde a coleta e pré-processamento dos dados até a modelagem e avaliação — utilizando dados reais de **Indicadores Socioeconômicos dos Países**, disponibilizados pela API pública do IBGE.

Os alunos deverão extrair dados como PIB, esperança de vida, densidade demográfica, entre outros, e formular dois problemas de negócio: um focado em **Classificação** e outro em **Regressão**.

---

## 📚 Mapeamento de Competências

O projeto deve obrigatoriamente demonstrar a aplicação prática dos conceitos vistos nas seguintes aulas:

* **[Aula 01] Fundamentos de IA/ML:** Definição clara de qual será o "target" (variável alvo) e justificar o uso de **aprendizado supervisionado** para as tarefas escolhidas.
* **[Aula 02] Pré-processamento:** Limpeza de dados nulos da API, aplicação de **Normalização/Padronização** (ex: `StandardScaler` ou `MinMaxScaler`) e codificação de variáveis categóricas (ex: transformar o continente do país usando `OneHotEncoder`).
* **[Aula 03] Teorema de Bayes:** Utilização do algoritmo **Naive Bayes** como uma de suas abordagens (baseline) para o problema de classificação.
* **[Aula 04] Regressão Linear e Logística:** Implementação de **Regressão Linear** (para prever um valor contínuo, ex: PIB per capita) e **Regressão Logística** (para classificação binária ou multiclasse).
* **[Aula 05] Modelos Supervisionados II:** Ampliação da experimentação utilizando **k-NN, Árvores de Decisão e/ou SVM** para comparar a performance com os modelos da Aula 04.
* **[Aula 06] Avaliação de Modelos:** Uso obrigatório de **Validação Cruzada (Cross-validation)**. Avaliação do modelo de Regressão via **MSE** e **MAE**, e do modelo de Classificação via **Matriz de Confusão, Acurácia e AUC-ROC**.

---

## 🛠️ Requisitos Técnicos

### 1. Fonte de Dados
Os dados devem ser consumidos via requisição HTTP (`requests` ou integração direta via Pandas) a partir da **API do IBGE**:
* **Endpoint / Documentação:** [https://servicodados.ibge.gov.br/api/docs/paises](https://servicodados.ibge.gov.br/api/docs/paises)

### 2. Stack Tecnológica
* **Pandas:** Para consumo, manipulação, agregação e estruturação do DataFrame a partir do JSON da API.
* **Numpy:** Para operações matriciais e transformações numéricas.
* **Scikit-Learn (Sklearn):** Para todo o pipeline de pré-processamento, instanciação dos algoritmos, validação e cálculo das métricas.
* **Matplotlib:** Para a visualização de dados exploratória e apresentação dos resultados preditivos (ex: Gráfico de dispersão `Valor Real vs Predito`, Curva ROC, ou limites de decisão).

### 3. Entregáveis Obrigatórios
1. **Notebook Google Colab (.ipynb):** 
   * Código comentado e estruturado de forma lógica.
   * Textos explicativos (Markdown) detalhando as escolhas e conclusões de cada etapa.
2. **Modelo de Regressão:**
   * Exemplo de proposta: *Prever a expectativa de vida de um país com base em seus indicadores de saúde e economia.*
3. **Modelo de Classificação:**
   * Exemplo de proposta: *Classificar se um país possui Índice de Desenvolvimento (Alto/Baixo) com base nas suas características numéricas, ou classificar a qual continente pertence.*

---

## 🚀 Atividade Extra (Bônus)

Como forma de enriquecer o projeto para portfólio e desenvolver habilidades analíticas de ponta a ponta, sugere-se a exportação do DataFrame processado (em `.csv` ou `.xlsx`) e a construção de um **Dashboard de Business Intelligence (BI)**.

Ferramentas sugeridas: **Power BI, Metabase ou Google Looker Studio**.
* **Objetivo do BI:** Criar um painel interativo visualizando a distribuição global dos indicadores socioeconômicos (usando mapas) e segmentando as métricas exploratórias antes mesmo de entrarem no modelo preditivo do Sklearn.

---

# Projeto #02: Previsão de Séries Temporais de Indicadores Socioeconômicos com Deep Learning (LSTM)

## 📋 Descrição do Projeto

Dando continuidade à nossa jornada de Machine Learning e avançando para o universo de **Deep Learning**, este projeto foca na construção de um pipeline de **Séries Temporais**. O objetivo é prever a evolução de um indicador socioeconômico ao longo do tempo (como o crescimento do PIB, evolução populacional ou emissões de CO2 de um país) utilizando dados históricos da API pública do IBGE.

Os alunos deverão extrair o histórico de anos de um indicador específico, tratar esses dados no formato de "janelas deslizantes" (sliding windows) e treinar uma **Rede Neural Recorrente (RNN/LSTM)** para prever os valores futuros, compreendendo as nuances de se trabalhar com dados sequenciais.

---

## 📚 Mapeamento de Competências

O projeto deve obrigatoriamente demonstrar a aplicação prática dos conceitos, especialmente os focados na modelagem sequencial:

* **Manipulação de Dados Sequenciais:** Criação de janelas deslizantes (*sliding windows*) para transformar uma sequência temporal em pares de entradas ($X$) e saídas ($y$) preditivas.
* **Divisão Treino/Teste em Séries Temporais:** Aplicação de fatiamento cronológico respeitando a ordem do tempo (sem embaralhamento/shuffle), garantindo que o conjunto de teste seja sempre composto pelos anos mais recentes.
* **Pré-processamento:** Uso obrigatório do `MinMaxScaler` para normalizar a série temporal, passo fundamental para a convergência e estabilidade no treinamento de redes neurais.
* **Redes Neurais Recorrentes:** Instanciação, compilação e treinamento de uma arquitetura **LSTM** (Long Short-Term Memory) combinada com camadas Densas (`Dense`), definindo o número de neurônios, funções de ativação e épocas de treinamento.
* **Avaliação e Reversão de Escala:** Avaliação do modelo de Regressão contínua via **MSE** (Erro Quadrático Médio) e **MAE** (Erro Absoluto Médio). Os alunos devem reverter a normalização (desnormalizar) para interpretar o MAE na escala original do indicador (ex: "erro médio de 2 bilhões de dólares").

---

## 🛠️ Requisitos Técnicos

### 1. Fonte de Dados

Os dados devem ser consumidos via requisição HTTP a partir da **API do IBGE**, explorando agora os arrays históricos (valores ao longo dos anos) de um indicador:

* **Endpoint / Documentação:** [https://servicodados.ibge.gov.br/api/docs/paises](https://servicodados.ibge.gov.br/api/docs/paises?utm_source=gemini) (Focar na extração da série histórica de um país específico, ex: Brasil ou EUA).

### 2. Stack Tecnológica

* **Pandas & Numpy:** Para extração do JSON, ordenação cronológica e construção da função de janelas deslizantes (`X` e `y` em formatos multidimensionais).
* **Scikit-Learn (Sklearn):** Para o pré-processamento escalar (`MinMaxScaler`) e cálculo de métricas.
* **TensorFlow / Keras:** Para a construção da arquitetura sequencial da rede neural (`layers.LSTM`, `layers.Dense`).
* **Matplotlib:** Para a visualização de dados exploratória (tendência histórica) e o plot final comparando a linha da **Série Real vs. Série Prevista** no conjunto de teste.

### 3. Entregáveis Obrigatórios

1. **Notebook Google Colab (.ipynb):**
* Código comentado e estruturado de forma didática.
* Textos explicativos (Markdown) detalhando o que é o indicador escolhido, o tamanho da janela adotada (ex: usar os últimos 5 anos para prever o próximo) e as decisões da arquitetura da rede.

2. **Preparação da Série Temporal:**
* Evidenciar no código a criação da função que gera as janelas temporais (adaptação da função `criar_janelas`) e o reshape (formato 3D) exigido pela LSTM.

3. **Modelo de Previsão (LSTM):**
* Treinamento do modelo avaliando a perda (loss) ao longo das épocas de validação e o print final da métrica MAE na escala original de grandeza do dado.

---

## 🚀 Atividade Extra (Bônus)

Como forma de enriquecer o projeto e comprovar a eficácia do uso de Deep Learning, sugere-se a implementação de um **Modelo Baseline Comparativo**.

* **Objetivo do Bônus:** Antes de aplicar a LSTM, construa uma previsão ingênua (*Naive Forecast* - onde o valor de amanhã é igual ao valor de hoje) ou uma Regressão Linear Simples para a mesma série temporal. Plote os resultados e compare o MAE do baseline com o MAE da LSTM para justificar se a complexidade da rede neural realmente trouxe ganhos de performance para o indicador escolhido.

---

Instagram • Telegram • Youtube - @profalexandrejr
