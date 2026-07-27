"""
Aula 07 - Exercício

Parte A (conceitual):

1. Você quer segmentar uma base de 50.000 clientes em grupos de perfil de
   consumo, mas não faz ideia de quantos grupos "naturais" existem, e
   suspeita que alguns clientes são bem atípicos (outliers). Você usaria
   K-means ou DBSCAN? Justifique.

2. Por que é indispensável padronizar (Aula 02) as variáveis antes de
   aplicar K-means, DBSCAN ou PCA?

3. Um colega diz: "Rodei o PCA e o Componente Principal 1 representa a
   variável 'renda do cliente'". O que há de conceitualmente incorreto nessa
   afirmação?

Parte B (prática):

Usando `dados/clientes.csv`:

1. Selecione as variáveis `renda_mensal`, `valor_mensalidade`,
   `tempo_de_casa_meses` e `qtd_acessos_mes`, padronize-as, e rode o K-means
   testando `k` de 2 a 6.
2. Use o coeficiente de silhueta para escolher o melhor `k`.
3. Para o melhor `k` encontrado, imprima o perfil médio de cada cluster
   (média das variáveis usadas + média de `churn`) e dê um "nome de negócio"
   para cada cluster (ex.: "clientes premium fiéis", "clientes novos de
   baixo engajamento" etc.), baseado nos valores observados.
4. Aplique PCA (2 componentes) nessas mesmas variáveis e reporte a variância
   total explicada pelos 2 componentes.
5. Desafio extra: rode o DBSCAN nas mesmas variáveis com `eps=1.0` e
   `min_samples=5`, e responda: quantos clusters e quantos outliers foram
   encontrados? Os outliers do DBSCAN parecem coincidir com algum cluster
   específico do K-means?
"""

# Escreva seu código abaixo
