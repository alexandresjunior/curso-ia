# 🌍 Projeto Final: Modelagem Preditiva de Indicadores Socioeconômicos Globais

**Instituição:** Treina Recife - Formação de Desenvolvedores e Cientistas de Dados  
**Módulo:** Fundamentos de Machine Learning (Aulas 01 a 06)  
**Formato de Entrega:** Notebook Google Colab e Repositório GitHub  

---

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

## 📅 Critérios de Avaliação

| Critério | Peso | Descrição |
|---|---|---|
| **Consumo e Limpeza (Pandas)** | 20% | Extração correta da API, tratamento de inconsistências e dados faltantes. |
| **Pré-processamento (Sklearn)** | 20% | Correto escalonamento Numérico e Encoding Categórico. |
| **Modelagem Supervisionada** | 30% | Correto treinamento de pelo menos 1 modelo de Regressão e 1 de Classificação. |
| **Validação e Métricas** | 20% | Aplicação de Validação Cruzada, MSE/MAE (Regressão) e AUC/Acurácia (Classificação). |
| **Visualização (Matplotlib)** | 10% | Gráficos claros, com títulos e rótulos bem definidos ilustrando os resultados preditivos. |

