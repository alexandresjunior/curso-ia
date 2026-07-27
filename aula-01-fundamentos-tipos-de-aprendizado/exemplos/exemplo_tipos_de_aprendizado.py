"""
Aula 01 - Exemplo prático
Objetivo: visualizar, de forma simples, a diferença entre um problema
supervisionado (com rótulo) e um problema não supervisionado (sem rótulo),
usando o mesmo tipo de dado de negócio: clientes de uma empresa.

Como rodar:
    python exemplo_tipos_de_aprendizado.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# Carrega o dataset de clientes usado ao longo de todo o curso
df = pd.read_csv("../../dados/clientes.csv")

print("Primeiras linhas do dataset:")
print(df.head(), "\n")

print("Dimensões do dataset:", df.shape)
print("\nColunas disponíveis:", list(df.columns))

# ---------------------------------------------------------------------------
# 1) Visão SUPERVISIONADA: usamos a coluna 'churn' (rótulo conhecido) para
# colorir os pontos. Isso é o que teríamos disponível em um problema
# supervisionado: sabemos, para cada cliente do passado, se ele cancelou ou não.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cores = df["churn"].map({0: "tab:blue", 1: "tab:red"})
axes[0].scatter(df["tempo_de_casa_meses"], df["qtd_acessos_mes"], c=cores, alpha=0.6)
axes[0].set_title("Visão SUPERVISIONADA\n(cor = rótulo 'churn' conhecido)")
axes[0].set_xlabel("Tempo de casa (meses)")
axes[0].set_ylabel("Acessos por mês")

# ---------------------------------------------------------------------------
# 2) Visão NÃO SUPERVISIONADA: os mesmos pontos, mas SEM usar o rótulo.
# Em um cenário real de aprendizado não supervisionado, nós não teríamos a
# coluna 'churn' disponível -- apenas os dados brutos. O que o algoritmo "vê"
# é isto:
# ---------------------------------------------------------------------------
axes[1].scatter(df["tempo_de_casa_meses"], df["qtd_acessos_mes"], c="gray", alpha=0.6)
axes[1].set_title("Visão NÃO SUPERVISIONADA\n(sem rótulo -- só os dados brutos)")
axes[1].set_xlabel("Tempo de casa (meses)")
axes[1].set_ylabel("Acessos por mês")

plt.tight_layout()
plt.savefig("comparacao_supervisionado_vs_nao_supervisionado.png", dpi=120)
print("\nGráfico salvo em 'comparacao_supervisionado_vs_nao_supervisionado.png'")

# ---------------------------------------------------------------------------
# 3) Simulando um cenário SEMI-SUPERVISIONADO
# Imagine que só temos o rótulo 'churn' para 10% dos clientes (os outros 90%
# ainda não foram "revisados" -- situação comum: rotular dados custa tempo/
# dinheiro). Vamos simular isso escondendo o rótulo de 90% das linhas.
# ---------------------------------------------------------------------------
import numpy as np

rng = np.random.default_rng(0)
df_semi = df.copy()
mask_desconhecido = rng.uniform(0, 1, len(df_semi)) < 0.90
df_semi.loc[mask_desconhecido, "churn"] = np.nan

qtd_rotulados = df_semi["churn"].notna().sum()
qtd_nao_rotulados = df_semi["churn"].isna().sum()

print(f"\nCenário semi-supervisionado simulado:")
print(f"  Clientes com rótulo conhecido:   {qtd_rotulados} ({qtd_rotulados/len(df_semi):.1%})")
print(f"  Clientes SEM rótulo (a inferir): {qtd_nao_rotulados} ({qtd_nao_rotulados/len(df_semi):.1%})")
print("\nEm um projeto real, algoritmos semi-supervisionados (ex.: self-training)")
print("usariam esses poucos exemplos rotulados + a estrutura dos não rotulados")
print("para tentar rotular o restante da base com boa qualidade.")
