"""
Aula 08 - Exercício

Parte A (conceitual):

1. Por que, para um problema de classificação binária (como churn) com
   dados tabulares, muitas vezes um MLP não traz ganho relevante sobre uma
   regressão logística? Em que tipo de dado o Deep Learning tende a se
   destacar mais claramente?

2. Explique, com suas palavras, por que uma CNN usa "camadas convolucionais"
   em vez de conectar cada neurônio diretamente a cada pixel da imagem
   (como um MLP tradicional faria).

3. Por que, ao dividir uma série temporal em treino/teste, NÃO devemos
   embaralhar (shuffle) os dados antes da divisão?

Parte B (prática):

1. Usando `dados/clientes.csv`, treine um MLP com uma arquitetura diferente
   da usada no exemplo desta aula (ex.: 3 camadas ocultas com 32, 16 e 8
   neurônios). Treine por 60 épocas e compare a acurácia final no teste com
   a do exemplo (2 camadas: 16 e 8 neurônios). Aumentar a complexidade da
   rede melhorou o resultado?

2. Usando o dataset `load_digits()` do scikit-learn (o mesmo do exemplo),
   treine uma CNN removendo a segunda camada convolucional (deixe apenas uma
   camada `Conv2D` antes do `Flatten`). Compare a acurácia no teste com a do
   modelo original (duas camadas convolucionais). O que isso sugere sobre a
   relação entre profundidade da rede e capacidade de aprender padrões mais
   complexos?

3. Usando a série temporal sintética do exemplo desta aula, teste um
   `TAMANHO_JANELA` diferente (ex.: 6 meses em vez de 12) e compare o MAE
   resultante no teste. Uma janela menor ajudou ou prejudicou a previsão?
   Por quê, na sua avaliação, isso pode ter acontecido (pense na
   sazonalidade anual da série)?
"""

# Escreva seu código abaixo
