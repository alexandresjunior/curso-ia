# Curso de Inteligência Artificial Aplicada

Curso introdutório e prático de Inteligência Artificial e Machine Learning, pensado para **profissionais de diferentes áreas** (não apenas programadores/dados) que querem entender os conceitos e aplicá-los no seu dia a dia — em marketing, finanças, RH, saúde, operações, vendas, etc.

Cada aula traz **teoria explicada em linguagem simples**, **exemplos práticos comentados em Python** e **exercícios com solução comentada**, usando cenários de negócio reais (previsão de vendas, churn de clientes, segmentação de público, detecção de fraude, diagnóstico, etc.).

## Pré-requisitos

- Noções básicas de lógica e matemática (nível ensino médio).
- Conhecimento básico de Python é desejável, mas cada aula reforça o necessário.
- Não é necessário conhecimento prévio de estatística avançada ou álgebra linear — os conceitos são construídos do zero.

## Como usar este repositório

1. Clone ou baixe o repositório.
2. Crie um ambiente virtual e instale as dependências:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Cada pasta `aula-XX-...` contém:
   - `README.md` → teoria da aula
   - `exemplos/` → scripts Python comentados, prontos para rodar (`python exemplo.py`) ou explorar célula a célula em um Jupyter/VS Code
   - `exercicios/` → um exercício proposto (`exercicio.py`) e uma solução comentada (`solucao.py`)
4. Recomenda-se seguir a ordem das aulas, pois os conceitos são cumulativos.

## Ementa (ordem didática)

| Aula | Tema | Conceitos-chave |
|---|---|---|
| [01](aula-01-fundamentos-tipos-de-aprendizado/README.md) | Fundamentos de IA/ML e tipos de aprendizado | O que é IA/ML, aprendizado supervisionado, não supervisionado e semi-supervisionado |
| [02](aula-02-pre-processamento-de-dados/README.md) | Pré-processamento de dados | Normalização, padronização, codificação de variáveis categóricas |
| [03](aula-03-teorema-de-bayes/README.md) | Teorema de Bayes | Probabilidade condicional, Bayes, Naive Bayes aplicado |
| [04](aula-04-regressao-linear-e-logistica/README.md) | Modelos supervisionados I | Regressão linear e regressão logística |
| [05](aula-05-knn-arvores-svm/README.md) | Modelos supervisionados II | k-NN, árvore de decisão, SVM |
| [06](aula-06-avaliacao-de-modelos/README.md) | Avaliação de modelos | Validação cruzada, MSE, MAE, AUC e outras métricas |
| [07](aula-07-aprendizado-nao-supervisionado/README.md) | Aprendizado não supervisionado | K-means, DBSCAN, PCA e redução de dimensionalidade |
| [08](aula-08-introducao-deep-learning/README.md) | Introdução a Deep Learning | Redes neurais (MLP), CNNs e RNNs |

## Estrutura do repositório

```
curso-ia/
├── README.md
├── requirements.txt
├── LICENSE
├── dados/                              <- datasets sintéticos usados nas aulas
├── aula-01-fundamentos-tipos-de-aprendizado/
│   ├── README.md
│   ├── exemplos/
│   └── exercicios/
├── aula-02-pre-processamento-de-dados/
├── aula-03-teorema-de-bayes/
├── aula-04-regressao-linear-e-logistica/
├── aula-05-knn-arvores-svm/
├── aula-06-avaliacao-de-modelos/
├── aula-07-aprendizado-nao-supervisionado/
└── aula-08-introducao-deep-learning/
```

## Licença

Este material é distribuído sob a licença MIT (veja [LICENSE](LICENSE)) — use, adapte e compartilhe livremente, inclusive para fins de ensino.
