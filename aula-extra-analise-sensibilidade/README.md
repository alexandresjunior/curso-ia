\# Aula Extra: Análise de Sensibilidade e Explicabilidade de Modelos



Nesta aula extra, vamos explorar como interpretar as decisões dos nossos modelos de Machine Learning, saindo da abordagem de "caixa preta". Utilizaremos os dados da base `clientes.csv` para entender quais características (features) têm maior impacto nas predições e como o modelo se comporta em diferentes cenários.



\## Conceitos-chave

1\. \*\*Importância de Atributos (Feature Importance):\*\* Como algoritmos baseados em árvores avaliam o ganho de informação de cada variável.

2\. \*\*Permutation Importance:\*\* Uma técnica de análise de sensibilidade que avalia a queda de performance do modelo quando embaralhamos aleatoriamente uma variável específica.

3\. \*\*Partial Dependence Plots (PDP):\*\* Visualização do efeito marginal que uma ou duas features têm sobre o resultado previsto pelo modelo.



\## Estrutura

\- `/exemplos`: Scripts práticos demonstrando a análise de sensibilidade com o Scikit-Learn e Matplotlib.

\- `/exercicios`: Desafios para aplicar os conceitos, testando a sensibilidade de diferentes algoritmos (ex: SVM vs Árvores).

