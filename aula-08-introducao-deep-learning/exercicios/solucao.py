"""
Aula 08 - Solução comentada do exercício
"""

# ---------------------------------------------------------------------------
# PARTE A — Respostas conceituais
# ---------------------------------------------------------------------------
#
# 1. Dados tabulares costumam ter relativamente poucas variáveis e relações
#    mais simples/diretas entre elas e o alvo -- exatamente o tipo de padrão
#    que a regressão logística já captura bem. Redes neurais precisam de
#    volumes maiores de dados para aproveitar sua capacidade de aprender
#    relações não lineares complexas, e em dados tabulares "modestos" essa
#    capacidade extra frequentemente não é aproveitada (podendo até piorar
#    por overfitting). Deep Learning se destaca mais em dados não
#    estruturados e volumosos: imagens, áudio, texto, vídeo.
#
# 2. Conectar cada neurônio a cada pixel geraria um número gigantesco de
#    parâmetros (para uma imagem de 224x224, por exemplo, isso explode
#    rapidamente), tornando o treinamento caro e propenso a overfitting.
#    Além disso, perderia a noção de ESTRUTURA ESPACIAL: um MLP tradicional
#    trata cada pixel de forma independente, sem "saber" que pixels vizinhos
#    estão relacionados. As camadas convolucionais usam pequenos filtros que
#    "deslizam" pela imagem, reaproveitando os mesmos poucos parâmetros para
#    detectar o mesmo padrão (ex.: uma borda) em qualquer posição da imagem
#    -- muito mais eficiente e adequado à natureza dos dados de imagem.
#
# 3. Em uma série temporal, existe uma dependência natural de ORDEM: o valor
#    de um mês depende do histórico anterior. Se embaralharmos antes de
#    dividir treino/teste, o modelo poderia "ver" dados do futuro durante o
#    treino (vazamento de informação temporal) e teria uma avaliação
#    artificialmente otimista, que não reflete como o modelo se sairia numa
#    previsão real (onde só temos acesso ao passado).

# ---------------------------------------------------------------------------
# PARTE B — Prática
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_digits

tf.random.set_seed(42)
np.random.seed(42)

# --- 1) MLP com arquitetura maior (32 -> 16 -> 8) ---
df = pd.read_csv("../../dados/clientes.csv").dropna()
features = ["idade", "renda_mensal", "tempo_de_casa_meses", "qtd_acessos_mes",
            "qtd_chamados_suporte", "atraso_pagamento"]
X = df[features].values
y = df["churn"].values

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_treino_esc = scaler.fit_transform(X_treino)
X_teste_esc = scaler.transform(X_teste)

peso_0 = len(y_treino) / (2 * np.sum(y_treino == 0))
peso_1 = len(y_treino) / (2 * np.sum(y_treino == 1))

modelo_mlp_grande = keras.Sequential([
    layers.Input(shape=(X_treino_esc.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
modelo_mlp_grande.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
modelo_mlp_grande.fit(
    X_treino_esc, y_treino, validation_split=0.2, epochs=60, batch_size=16,
    class_weight={0: peso_0, 1: peso_1}, verbose=0,
)
_, acc_grande = modelo_mlp_grande.evaluate(X_teste_esc, y_teste, verbose=0)
print(f"[1] MLP maior (32-16-8): acurácia no teste = {acc_grande:.2%}")
print("    Compare com o exemplo (16-8), que obteve ~69-70%. Frequentemente o")
print("    ganho é pequeno ou inexistente -- reforçando o conceito da Parte A.\n")

# --- 2) CNN com apenas 1 camada convolucional ---
digits = load_digits()
X_img = (digits.images / 16.0)[..., np.newaxis]
y_img = digits.target

Xi_treino, Xi_teste, yi_treino, yi_teste = train_test_split(
    X_img, y_img, test_size=0.2, random_state=42, stratify=y_img
)

modelo_cnn_simples = keras.Sequential([
    layers.Input(shape=(8, 8, 1)),
    layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
modelo_cnn_simples.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
modelo_cnn_simples.fit(Xi_treino, yi_treino, validation_split=0.2, epochs=15, batch_size=16, verbose=0)
_, acc_cnn_simples = modelo_cnn_simples.evaluate(Xi_teste, yi_teste, verbose=0)
print(f"[2] CNN com 1 camada convolucional: acurácia no teste = {acc_cnn_simples:.2%}")
print("    Compare com o exemplo (2 camadas), que obteve ~95-96%. Em geral, mais")
print("    camadas permitem aprender padrões mais abstratos/hierárquicos, até um")
print("    certo ponto (depois do qual o ganho tende a diminuir ou exigir mais dados).\n")

# --- 3) RNN com janela menor (6 meses) ---
n_meses = 120
t = np.arange(n_meses)
serie = 100 + 1.5 * t + 15 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 5, n_meses)
minmax = MinMaxScaler()
serie_norm = minmax.fit_transform(serie.reshape(-1, 1)).flatten()

def criar_janelas(serie, tamanho_janela):
    X, y = [], []
    for i in range(len(serie) - tamanho_janela):
        X.append(serie[i:i + tamanho_janela])
        y.append(serie[i + tamanho_janela])
    return np.array(X), np.array(y)

for tamanho_janela in [6, 12]:
    X_serie, y_serie = criar_janelas(serie_norm, tamanho_janela)
    X_serie = X_serie[..., np.newaxis]
    corte = int(len(X_serie) * 0.8)
    Xs_treino, Xs_teste = X_serie[:corte], X_serie[corte:]
    ys_treino, ys_teste = y_serie[:corte], y_serie[corte:]

    modelo_rnn = keras.Sequential([
        layers.Input(shape=(tamanho_janela, 1)),
        layers.LSTM(32, activation="tanh"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    modelo_rnn.compile(optimizer="adam", loss="mse", metrics=["mae"])
    modelo_rnn.fit(Xs_treino, ys_treino, validation_split=0.2, epochs=60, batch_size=8, verbose=0)
    _, mae = modelo_rnn.evaluate(Xs_teste, ys_teste, verbose=0)
    print(f"[3] Janela={tamanho_janela} meses: MAE no teste (normalizado) = {mae:.4f}")

print(
    "\n    Como a série tem sazonalidade ANUAL (período de 12 meses), uma "
    "janela de 6 meses captura apenas 'meio ciclo' sazonal, dificultando a "
    "identificação do padrão completo -- normalmente a janela de 12 meses "
    "tende a produzir erro igual ou menor, por conseguir 'ver' o ciclo "
    "sazonal inteiro em cada exemplo de treino."
)
