"""
Aula 02 - Solução comentada do exercício
"""

# ---------------------------------------------------------------------------
# PARTE A — Respostas conceituais
# ---------------------------------------------------------------------------
#
# 1. O k-NN decide a "proximidade" entre exemplos calculando distância entre
#    seus valores. Se salário varia na casa dos milhares e idade varia entre
#    18-75, a distância será dominada quase totalmente pela diferença de
#    salário, tornando a idade praticamente irrelevante para o cálculo --
#    mesmo que ela seja importante para o problema. Escalando ambas as
#    variáveis para faixas comparáveis, cada uma passa a contribuir de forma
#    proporcional e justa para a distância calculada.
#
# 2. ORDINAL ENCODING. Existe uma ordem natural e clara entre as categorias
#    (Fundamental < Médio < Superior < Pós-graduação), então mapear para
#    números crescentes (0, 1, 2, 3) preserva essa relação de ordem, o que é
#    desejável e interpretável pelo modelo.
#
# 3. ONE-HOT ENCODING. Não há ordem natural entre cidades -- usar números
#    sequenciais (Label Encoding) faria o modelo assumir erroneamente que
#    "Curitiba > São Paulo", por exemplo, o que não tem significado algum.

# ---------------------------------------------------------------------------
# PARTE B — Prática
# ---------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 1) Carregar e identificar valores ausentes
df = pd.read_csv("../../dados/clientes.csv")
print("Valores ausentes por coluna:")
print(df.isna().sum()[df.isna().sum() > 0], "\n")

# 2) Tratamento de valores ausentes com a MEDIANA
# A mediana é preferida à média quando há outliers, pois não é "puxada" por
# valores extremos (ex.: uma renda muito alta distorceria a média, mas quase
# não afeta a mediana).
X = df.drop(columns=["cliente_id", "churn"])
y = df["churn"]

# 3) Separar treino/teste ANTES de ajustar imputador/scaler
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")
colunas_para_imputar = ["renda_mensal", "qtd_acessos_mes"]

X_treino_imputado = X_treino.copy()
X_teste_imputado = X_teste.copy()
X_treino_imputado[colunas_para_imputar] = imputer.fit_transform(X_treino[colunas_para_imputar])
X_teste_imputado[colunas_para_imputar] = imputer.transform(X_teste[colunas_para_imputar])

print("Valores ausentes após imputação (treino):")
print(X_treino_imputado[colunas_para_imputar].isna().sum(), "\n")

# 4) Padronização (Z-score) ajustada apenas no treino
colunas_para_padronizar = ["idade", "renda_mensal", "tempo_de_casa_meses"]
scaler = StandardScaler()
X_treino_padr = X_treino_imputado.copy()
X_teste_padr = X_teste_imputado.copy()
X_treino_padr[colunas_para_padronizar] = scaler.fit_transform(X_treino_imputado[colunas_para_padronizar])
X_teste_padr[colunas_para_padronizar] = scaler.transform(X_teste_imputado[colunas_para_padronizar])

# 5) One-Hot Encoding na coluna 'plano'
onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
plano_treino_onehot = onehot.fit_transform(X_treino_padr[["plano"]])
plano_teste_onehot = onehot.transform(X_teste_padr[["plano"]])

colunas_onehot = onehot.get_feature_names_out(["plano"])
df_plano_treino = pd.DataFrame(plano_treino_onehot, columns=colunas_onehot, index=X_treino_padr.index)

X_treino_final = pd.concat(
    [X_treino_padr.drop(columns=["plano", "cidade"]), df_plano_treino], axis=1
)

print("Formato do conjunto de treino processado:", X_treino_final.shape)
print(X_treino_final.head())

# Desafio extra: comparação MinMax vs StandardScaler
minmax = MinMaxScaler()
X_treino_minmax = minmax.fit_transform(X_treino_imputado[colunas_para_padronizar])

print("\nComparação para os 5 primeiros clientes do treino (coluna 'renda_mensal'):")
print("Padronizado (Z-score): ", X_treino_padr["renda_mensal"].values[:5].round(3))
print("Normalizado (Min-Max): ", X_treino_minmax[:5, 1].round(3))
# Observação: a padronização pode gerar valores negativos (a variável passa a
# ter média 0), enquanto a normalização sempre resulta em valores entre 0 e 1.
