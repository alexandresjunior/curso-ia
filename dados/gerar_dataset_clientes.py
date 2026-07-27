"""
Gerador do dataset sintético "clientes.csv", usado como fio condutor em várias
aulas do curso (pré-processamento, regressão, classificação, avaliação,
clusterização).

Contexto de negócio: uma base fictícia de clientes de uma empresa de assinatura
(ex.: streaming, academia, software B2B), com dados demográficos, de uso do
produto e um rótulo de "cancelamento" (churn) — problema muito comum em
qualquer área de negócio.

Para reproduzir: basta rodar `python gerar_dataset_clientes.py` dentro da
pasta `dados/`. O arquivo `clientes.csv` já vai gerado neste repositório,
então você não precisa rodar este script — ele existe apenas para
transparência/reprodutibilidade.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CLIENTES = 800


def gerar_dataset(n=N_CLIENTES, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    idade = rng.normal(38, 12, n).clip(18, 75).round().astype(int)
    renda_mensal = (rng.normal(4500, 1800, n) + (idade - 38) * 30).clip(1200, 20000).round(2)

    cidade = rng.choice(
        ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Recife"],
        size=n,
        p=[0.35, 0.25, 0.15, 0.15, 0.10],
    )

    plano = rng.choice(["Básico", "Padrão", "Premium"], size=n, p=[0.5, 0.35, 0.15])
    plano_map_valor = {"Básico": 29.9, "Padrão": 59.9, "Premium": 119.9}
    valor_mensalidade = np.array([plano_map_valor[p] for p in plano])

    tempo_de_casa_meses = rng.integers(1, 60, n)
    qtd_acessos_mes = rng.poisson(lam=(renda_mensal / 800) + 2, size=n).clip(0, 60)
    qtd_chamados_suporte = rng.poisson(lam=1.2, size=n).clip(0, 15)
    atraso_pagamento = rng.choice([0, 1], size=n, p=[0.82, 0.18])

    # Probabilidade "real" de churn (regra latente, não observável pelo aluno),
    # usada só para gerar rótulos realistas e coerentes com as features.
    # Os coeficientes abaixo foram calibrados para que os padrões sejam
    # aprendíveis pelos modelos das aulas (sinal claro, mas com ruído
    # realista) -- ou seja, o dataset foi desenhado para fins didáticos.
    logit = (
        -1.6
        + 1.8 * atraso_pagamento
        - 0.09 * qtd_acessos_mes
        + 0.55 * qtd_chamados_suporte
        - 0.045 * tempo_de_casa_meses
        + 0.0004 * (renda_mensal - 4500)
        + rng.normal(0, 0.5, n)
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, n) < prob_churn).astype(int)

    df = pd.DataFrame(
        {
            "cliente_id": np.arange(1, n + 1),
            "idade": idade,
            "cidade": cidade,
            "renda_mensal": renda_mensal,
            "plano": plano,
            "valor_mensalidade": valor_mensalidade,
            "tempo_de_casa_meses": tempo_de_casa_meses,
            "qtd_acessos_mes": qtd_acessos_mes,
            "qtd_chamados_suporte": qtd_chamados_suporte,
            "atraso_pagamento": atraso_pagamento,
            "churn": churn,
        }
    )

    # Introduzindo alguns valores ausentes de propósito (comum na vida real)
    for col in ["renda_mensal", "qtd_acessos_mes"]:
        idx_na = rng.choice(df.index, size=int(0.03 * n), replace=False)
        df.loc[idx_na, col] = np.nan

    return df


if __name__ == "__main__":
    df = gerar_dataset()
    df.to_csv("clientes.csv", index=False)
    print(f"Arquivo clientes.csv gerado com {len(df)} linhas.")
    print(df.head())
