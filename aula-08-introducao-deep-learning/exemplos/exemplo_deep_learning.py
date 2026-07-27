"""
Aula 08 - Exemplo prático
Objetivo: construir, treinar e avaliar, de forma simples e comentada:
 1) Um MLP (rede neural densa) para prever churn -- comparando com a
    regressão logística da Aula 04.
 2) Uma CNN simples para classificar imagens de dígitos escritos à mão
    (dataset "digits" do próprio scikit-learn, 8x8 pixels -- não requer
    download da internet).
 3) Uma RNN (LSTM) simples para prever o próximo valor de uma série
    temporal sintética (ex.: receita mensal com tendência e sazonalidade).

Como rodar:
    python exemplo_deep_learning.py

Observação: os modelos aqui são propositalmente pequenos e rápidos de
treinar, priorizando clareza didática sobre desempenho máximo.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

tf.random.set_seed(42)
np.random.seed(42)

# ===========================================================================
# 1) MLP para prever CHURN (dados tabulares -- mesmo problema das Aulas 04/05)
# ===========================================================================
print("=" * 70)
print("1) MLP (Multi-Layer Perceptron) para previsão de churn")
print("=" * 70)

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

# Arquitetura: entrada -> camada oculta (16 neurônios, ReLU) ->
# camada oculta (8 neurônios, ReLU) -> saída (1 neurônio, sigmoide)
modelo_mlp = keras.Sequential([
    layers.Input(shape=(X_treino_esc.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

modelo_mlp.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# class_weight para lidar com o desbalanceamento de classes (poucos casos de
# churn), como discutido nas Aulas 04-06.
peso_classe_0 = len(y_treino) / (2 * np.sum(y_treino == 0))
peso_classe_1 = len(y_treino) / (2 * np.sum(y_treino == 1))

historico = modelo_mlp.fit(
    X_treino_esc, y_treino,
    validation_split=0.2,
    epochs=40,
    batch_size=16,
    class_weight={0: peso_classe_0, 1: peso_classe_1},
    verbose=0,
)

perda_teste, acuracia_teste = modelo_mlp.evaluate(X_teste_esc, y_teste, verbose=0)
print(f"Acurácia do MLP no conjunto de teste: {acuracia_teste:.2%}")
print(f"Última acurácia de treino: {historico.history['accuracy'][-1]:.2%}")
print(f"Última acurácia de validação: {historico.history['val_accuracy'][-1]:.2%}")
print(
    "\nCompare esse resultado com a Regressão Logística da Aula 04/06: em "
    "problemas tabulares como este, o MLP raramente supera modelos mais "
    "simples por uma margem grande -- mas serve bem para ilustrar a mecânica "
    "de uma rede neural.\n"
)

# ===========================================================================
# 2) CNN para classificar imagens de dígitos (0 a 9), 8x8 pixels
# ===========================================================================
print("=" * 70)
print("2) CNN (Rede Neural Convolucional) para classificar dígitos manuscritos")
print("=" * 70)

digits = load_digits()
X_img = digits.images  # formato (n_amostras, 8, 8), valores de 0 a 16
y_img = digits.target  # dígitos de 0 a 9

# Normalizando os pixels para o intervalo [0, 1] (Aula 02: normalização)
X_img = X_img / 16.0
X_img = X_img[..., np.newaxis]  # adiciona o "canal" (CNN espera formato H x W x canais)

Xi_treino, Xi_teste, yi_treino, yi_teste = train_test_split(
    X_img, y_img, test_size=0.2, random_state=42, stratify=y_img
)

modelo_cnn = keras.Sequential([
    layers.Input(shape=(8, 8, 1)),
    layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation="softmax"),  # 10 classes (dígitos 0-9)
])

modelo_cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

modelo_cnn.fit(Xi_treino, yi_treino, validation_split=0.2, epochs=15, batch_size=16, verbose=0)

perda_cnn, acuracia_cnn = modelo_cnn.evaluate(Xi_teste, yi_teste, verbose=0)
print(f"Acurácia da CNN no conjunto de teste: {acuracia_cnn:.2%}")
print(
    "\nRepare como a CNN consegue uma acurácia bem alta em um problema de "
    "imagem -- essa é a força das camadas convolucionais em capturar padrões "
    "espaciais (bordas, formas) que compõem cada dígito.\n"
)

# ===========================================================================
# 3) RNN (LSTM) para prever série temporal sintética (ex.: receita mensal)
# ===========================================================================
print("=" * 70)
print("3) RNN/LSTM para previsão de série temporal (receita mensal simulada)")
print("=" * 70)

# Gerando uma série temporal sintética: tendência de crescimento +
# sazonalidade anual + ruído -- um cenário comum de previsão de vendas/receita.
n_meses = 120
t = np.arange(n_meses)
tendencia = 100 + 1.5 * t
sazonalidade = 15 * np.sin(2 * np.pi * t / 12)
ruido = np.random.normal(0, 5, n_meses)
serie = tendencia + sazonalidade + ruido

# Normalizando a série (Aula 02) para ajudar o treinamento da rede
minmax = MinMaxScaler()
serie_norm = minmax.fit_transform(serie.reshape(-1, 1)).flatten()

def criar_janelas(serie, tamanho_janela=12):
    """Transforma a série em pares (janela de entrada -> próximo valor)."""
    X, y = [], []
    for i in range(len(serie) - tamanho_janela):
        X.append(serie[i:i + tamanho_janela])
        y.append(serie[i + tamanho_janela])
    return np.array(X), np.array(y)

TAMANHO_JANELA = 12  # usamos os últimos 12 meses para prever o próximo mês
X_serie, y_serie = criar_janelas(serie_norm, TAMANHO_JANELA)
X_serie = X_serie[..., np.newaxis]  # formato (amostras, passos_de_tempo, features=1)

# Como é uma série temporal, dividimos treino/teste RESPEITANDO A ORDEM
# (não embaralhamos!) -- os últimos meses viram o conjunto de teste.
corte = int(len(X_serie) * 0.8)
Xs_treino, Xs_teste = X_serie[:corte], X_serie[corte:]
ys_treino, ys_teste = y_serie[:corte], y_serie[corte:]

modelo_rnn = keras.Sequential([
    layers.Input(shape=(TAMANHO_JANELA, 1)),
    layers.LSTM(32, activation="tanh"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1),  # saída numérica contínua (regressão)
])

modelo_rnn.compile(optimizer="adam", loss="mse", metrics=["mae"])

modelo_rnn.fit(Xs_treino, ys_treino, validation_split=0.2, epochs=60, batch_size=8, verbose=0)

perda_rnn, mae_rnn = modelo_rnn.evaluate(Xs_teste, ys_teste, verbose=0)

# Convertendo o MAE de volta para a escala original (desnormalizando)
amplitude_original = minmax.data_max_[0] - minmax.data_min_[0]
mae_rnn_escala_original = mae_rnn * amplitude_original

print(f"MAE da LSTM no teste (escala normalizada): {mae_rnn:.4f}")
print(f"MAE da LSTM no teste (escala original, aprox.): {mae_rnn_escala_original:.2f} unidades de receita")
print(
    "\nNote como preparamos a série em 'janelas deslizantes' (últimos 12 "
    "meses -> próximo mês) -- essa é a abordagem clássica para adaptar dados "
    "sequenciais ao formato que uma RNN/LSTM espera."
)
