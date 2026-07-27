"""
Aula 02 - Exemplo prático
Objetivo: aplicar normalização, padronização, tratamento de valores ausentes
e codificação de variáveis categóricas no dataset de clientes, seguindo o
fluxo correto (ajustar apenas no treino, aplicar em treino e teste).

Como rodar:
    python exemplo_preprocessamento.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("../../dados/clientes.csv")

print("Valores ausentes por coluna:")
print(df.isna().sum(), "\n")

# ---------------------------------------------------------------------------
# 1) Separar variáveis explicativas (X) e alvo (y) ANTES de qualquer ajuste
#    de escaler/encoder, e dividir treino/teste antes de "aprender" as
#    estatísticas de normalização (evita vazamento de dados / data leakage).
# ---------------------------------------------------------------------------
colunas_numericas = [
    "idade", "renda_mensal", "tempo_de_casa_meses",
    "qtd_acessos_mes", "qtd_chamados_suporte",
]
colunas_categoricas = ["cidade", "plano"]

X = df[colunas_numericas + colunas_categoricas]
y = df["churn"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Tamanho treino: {len(X_treino)} | Tamanho teste: {len(X_teste)}\n")

# ---------------------------------------------------------------------------
# 2) Demonstração manual: normalização (Min-Max) vs. padronização (Z-score)
#    na coluna 'renda_mensal', ajustando os parâmetros SOMENTE no treino.
# ---------------------------------------------------------------------------
imputer_num = SimpleImputer(strategy="median")
renda_treino_imputada = imputer_num.fit_transform(X_treino[["renda_mensal"]])
renda_teste_imputada = imputer_num.transform(X_teste[["renda_mensal"]])

# Normalização (Min-Max)
minmax = MinMaxScaler()
renda_treino_norm = minmax.fit_transform(renda_treino_imputada)
renda_teste_norm = minmax.transform(renda_teste_imputada)

# Padronização (Z-score)
zscore = StandardScaler()
renda_treino_padr = zscore.fit_transform(renda_treino_imputada)
renda_teste_padr = zscore.transform(renda_teste_imputada)

print("Renda mensal (treino) -- 5 primeiros valores:")
print("Original:      ", renda_treino_imputada[:5].ravel().round(2))
print("Normalizada:   ", renda_treino_norm[:5].ravel().round(3), " (intervalo [0,1])")
print("Padronizada:   ", renda_treino_padr[:5].ravel().round(3), " (média 0, desvio 1)\n")

# ---------------------------------------------------------------------------
# 3) Pipeline completo de pré-processamento com ColumnTransformer:
#    - imputação + padronização para colunas numéricas
#    - imputação + one-hot encoding para colunas categóricas
# ---------------------------------------------------------------------------
pipeline_numerico = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

pipeline_categorico = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessador = ColumnTransformer(transformers=[
    ("num", pipeline_numerico, colunas_numericas),
    ("cat", pipeline_categorico, colunas_categoricas),
])

X_treino_processado = preprocessador.fit_transform(X_treino)
X_teste_processado = preprocessador.transform(X_teste)

print("Formato de X_treino ANTES do pré-processamento:", X_treino.shape)
print("Formato de X_treino DEPOIS do pré-processamento:", X_treino_processado.shape)

nomes_colunas_onehot = preprocessador.named_transformers_["cat"]["onehot"].get_feature_names_out(colunas_categoricas)
nomes_finais = colunas_numericas + list(nomes_colunas_onehot)
print("\nColunas finais após o pré-processamento:")
print(nomes_finais)

# Este 'preprocessador' pronto pode agora alimentar diretamente qualquer
# modelo de scikit-learn (veremos isso nas Aulas 4, 5 e 6), dentro de um
# Pipeline único -- técnica muito usada no dia a dia de projetos de ML.
