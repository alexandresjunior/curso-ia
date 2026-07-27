"""
Aula 01 - Solução comentada do exercício
"""

# ---------------------------------------------------------------------------
# PARTE A — Respostas conceituais
# ---------------------------------------------------------------------------
#
# 1. NÃO SUPERVISIONADO
#    Não há rótulos/categorias pré-definidas; o objetivo é descobrir grupos
#    (clusters) de clientes parecidos a partir da estrutura dos dados.
#
# 2. SUPERVISIONADO (classificação)
#    Todas as transações já têm o rótulo "fraude"/"não fraude"; o modelo
#    aprende a mapear as características da transação para esse rótulo.
#
# 3. SEMI-SUPERVISIONADO
#    Existe uma pequena parte rotulada (300 currículos) e uma massa grande
#    não rotulada (9.700); o ideal é aproveitar as duas para triar melhor.
#
# 4. SUPERVISIONADO (classificação)
#    Há diagnóstico confirmado (rótulo) em todos os exames históricos, e o
#    objetivo é prever esse rótulo (risco de diabetes) para pacientes novos.

# ---------------------------------------------------------------------------
# PARTE B — Prática
# ---------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# 1) Carregar o dataset
df = pd.read_csv("../../dados/clientes.csv")

# 2) Contagem geral e por classe de churn
total_clientes = len(df)
qtd_churn_1 = (df["churn"] == 1).sum()
qtd_churn_0 = (df["churn"] == 0).sum()

print(f"Total de clientes: {total_clientes}")
print(f"Cancelaram (churn=1): {qtd_churn_1} ({qtd_churn_1/total_clientes:.1%})")
print(f"Não cancelaram (churn=0): {qtd_churn_0} ({qtd_churn_0/total_clientes:.1%})")

# 3) Gráficos comparativos
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# a) Sem usar o rótulo
axes[0].scatter(df["renda_mensal"], df["qtd_chamados_suporte"], color="gray", alpha=0.6)
axes[0].set_title("Sem rótulo (não supervisionado)")
axes[0].set_xlabel("Renda mensal")
axes[0].set_ylabel("Qtd. chamados de suporte")

# b) Colorindo pelo rótulo churn
cores = df["churn"].map({0: "tab:blue", 1: "tab:red"})
axes[1].scatter(df["renda_mensal"], df["qtd_chamados_suporte"], c=cores, alpha=0.6)
axes[1].set_title("Colorido por 'churn' (supervisionado)")
axes[1].set_xlabel("Renda mensal")
axes[1].set_ylabel("Qtd. chamados de suporte")

plt.tight_layout()
plt.savefig("solucao_exercicio_scatter.png", dpi=120)
print("\nGráfico salvo em 'solucao_exercicio_scatter.png'")

# 4) Observação (comentário-resposta):
# Ao colorir por 'churn', é possível notar que clientes com MAIS chamados de
# suporte tendem a concentrar mais pontos vermelhos (churn=1), sugerindo uma
# relação entre insatisfação (medida indiretamente por chamados de suporte) e
# cancelamento. Essa relação não é visível na versão sem rótulo -- é
# justamente essa "pista extra" que o aprendizado supervisionado explora.
