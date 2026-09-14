"""
LSTM multivariado para previsão de pH e CE.
Entradas: pH, CE, temperatura, umidade (Seção 4.6 e H2 do documento).

Estrategia de treinamento: o modelo preve o DELTA
    delta = y[t+h] - y[t-1]
em vez do valor absoluto. Isto e uma tecnica padrao para series suaves
com forte tendencia local: elimina o vies "regressao a media" e permite
que o LSTM aprenda a dinamica (mudanca) alem do nivel corrente.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import LSTM_WINDOW, FORECAST_HORIZON_STEPS, MODELS_DIR, RANDOM_SEED
from src.data_loader import load_timeseries, train_test_split_series

FEATURES = ["pH", "EC", "temperature", "humidity"]


def _lazy_tf():
    """Import tardio de TF para permitir uso de outros módulos sem TF instalado."""
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    return tf


def build_supervised_delta(values: np.ndarray, target_raw: np.ndarray,
                           window: int, target_idx: int,
                           horizon: int = FORECAST_HORIZON_STEPS):
    """
    X = janela escalada [i, i+window)
    delta = target_raw[i+window+horizon-1] - target_raw[i+window-1]
    y_ref = target_raw[i+window-1]   (valor "atual" para reconstruir)
    y_final = target_raw[i+window+horizon-1]  (valor real a prever)
    """
    X, delta, y_ref, y_final = [], [], [], []
    for i in range(len(values) - window - horizon + 1):
        X.append(values[i: i + window])
        cur = float(target_raw[i + window - 1])
        fut = float(target_raw[i + window + horizon - 1])
        delta.append(fut - cur)
        y_ref.append(cur)
        y_final.append(fut)
    return (np.array(X), np.array(delta), np.array(y_ref), np.array(y_final))


def build_model(window: int, n_features: int):
    tf = _lazy_tf()
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.20),
        layers.LSTM(32),
        layers.Dropout(0.20),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),   # saida = delta
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def run(target: str = "pH", epochs: int = 200, batch_size: int = 64) -> dict:
    tf = _lazy_tf()

    df = load_timeseries()
    train_df, test_df = train_test_split_series(df, test_frac=0.20)

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES].values)
    train_scaled = scaler.transform(train_df[FEATURES].values)
    test_scaled  = scaler.transform(test_df[FEATURES].values)

    target_idx = FEATURES.index(target)
    target_train_raw = train_df[target].values
    target_test_raw  = test_df[target].values

    X_train, d_train, _, _ = build_supervised_delta(
        train_scaled, target_train_raw, LSTM_WINDOW, target_idx)

    # concatena o fim de treino para gerar as janelas iniciais de teste
    combined_scaled = np.concatenate([train_scaled[-LSTM_WINDOW:], test_scaled], axis=0)
    combined_raw    = np.concatenate([target_train_raw[-LSTM_WINDOW:], target_test_raw])
    X_test, d_test, y_ref_test, y_true = build_supervised_delta(
        combined_scaled, combined_raw, LSTM_WINDOW, target_idx)

    model = build_model(LSTM_WINDOW, len(FEATURES))
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5
    )
    model.fit(
        X_train, d_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        shuffle=False,
        callbacks=[early, reduce_lr],
    )
    delta_pred = model.predict(X_test, verbose=0).flatten()
    y_pred = y_ref_test + delta_pred

    out = {
        "target":      target,
        "features":    FEATURES,
        "window":      LSTM_WINDOW,
        "horizon":     FORECAST_HORIZON_STEPS,
        "train_size":  len(X_train),
        "test_size":   len(X_test),
        "y_true":      y_true.tolist(),
        "y_pred":      y_pred.tolist(),
        "timestamps":  test_df["timestamp"].astype(str).tolist()[-len(y_pred):],
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"lstm_{target}.keras")
    joblib.dump(scaler, MODELS_DIR / f"lstm_{target}_scaler.joblib")
    with open(MODELS_DIR / f"lstm_{target}_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


if __name__ == "__main__":
    from src.metrics import rmse, mae, mape

    for tgt in ["pH", "EC"]:
        r = run(target=tgt)
        print(f"[LSTM {tgt}] RMSE={rmse(r['y_true'], r['y_pred']):.4f} "
              f"MAE={mae(r['y_true'], r['y_pred']):.4f} "
              f"MAPE={mape(r['y_true'], r['y_pred']):.2f}%")
