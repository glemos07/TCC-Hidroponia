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

from src.config import (
    ARTIFACT_SUFFIX,
    MAX_GAP_FACTOR,
    MODELS_DIR,
    RANDOM_SEED,
    features_de,
    intervalo_amostragem,
    passos_horizonte,
    passos_janela,
)
from src.data_loader import load_timeseries, train_test_split_series

#: Vetor de entrada da fonte padrao (Kaggle). Para a telemetria do ESP32 o
#: conjunto e outro - ver `config.FEATURE_SETS` e o parametro `fonte` de `run`.
FEATURES = features_de("kaggle")


def _lazy_tf():
    """Import tardio de TF para permitir uso de outros módulos sem TF instalado."""
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    return tf


def build_supervised_delta(values: np.ndarray, target_raw: np.ndarray,
                           window: int, target_idx: int,
                           horizon: int = 1,
                           timestamps: np.ndarray | None = None,
                           max_gap_min: float | None = None):
    """
    X = janela escalada [i, i+window)
    delta = target_raw[i+window+horizon-1] - target_raw[i+window-1]
    y_ref = target_raw[i+window-1]   (valor "atual" para reconstruir)
    y_final = target_raw[i+window+horizon-1]  (valor real a prever)

    `timestamps` + `max_gap_min` descartam janelas que atravessam uma
    interrupcao da coleta: a telemetria do ESP32 tem lacunas de horas ou dias
    (queda de energia, rede fora), e uma janela que as cruze ensinaria o
    modelo a interpretar como variacao de minutos o que levou dias.
    Series ja reamostradas em grade regular dispensam os dois argumentos.
    """
    sem_lacuna = None
    if timestamps is not None and max_gap_min:
        deltas_min = np.diff(timestamps).astype("timedelta64[s]").astype(float) / 60.0
        # prefixo acumulado permite testar cada janela em tempo constante
        prefixo = np.concatenate([[0], np.cumsum(deltas_min > max_gap_min)])
        sem_lacuna = prefixo

    X, delta, y_ref, y_final = [], [], [], []
    for i in range(len(values) - window - horizon + 1):
        fim = i + window + horizon - 1
        if sem_lacuna is not None and sem_lacuna[fim] - sem_lacuna[i] > 0:
            continue
        X.append(values[i: i + window])
        cur = float(target_raw[i + window - 1])
        fut = float(target_raw[fim])
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


def run(target: str = "pH", epochs: int = 200, batch_size: int = 64,
        fonte: str = "kaggle") -> dict:
    """
    Treina e avalia o LSTM da fonte indicada.

    fonte="kaggle" grava `lstm_<alvo>.keras` (artefatos do Quadro 6);
    fonte="api" grava `lstm_<alvo>_api.keras`, treinado sobre a telemetria
    real do ESP32. Os dois convivem sem se sobrescrever.
    """
    tf = _lazy_tf()
    features = features_de(fonte)
    sufixo = ARTIFACT_SUFFIX[fonte]
    janela = passos_janela(fonte)
    horizonte = passos_horizonte(fonte)

    df = load_timeseries(fonte=fonte)
    minimo = janela + horizonte
    if len(df) < minimo * 2:
        raise ValueError(
            f"Serie da fonte '{fonte}' tem apenas {len(df)} pontos; "
            f"o LSTM precisa de pelo menos {minimo * 2} "
            f"(janela {janela} + horizonte {horizonte}, treino e teste)."
        )
    train_df, test_df = train_test_split_series(df, test_frac=0.20)

    scaler = StandardScaler()
    scaler.fit(train_df[features].values)
    train_scaled = scaler.transform(train_df[features].values)
    test_scaled  = scaler.transform(test_df[features].values)

    target_idx = features.index(target)
    target_train_raw = train_df[target].values
    target_test_raw  = test_df[target].values

    # A serie do Kaggle ja vem reamostrada em grade de 1 min; a da API chega
    # como o ESP32 conseguiu enviar, com lacunas. So nesse caso ha janelas a
    # descartar - manter o Kaggle intocado preserva os numeros do Quadro 6.
    ts_train = ts_test = max_gap = None
    if fonte == "api":
        ts_train = train_df["timestamp"].values
        ts_test = test_df["timestamp"].values
        max_gap = MAX_GAP_FACTOR * intervalo_amostragem(fonte)

    X_train, d_train, _, _ = build_supervised_delta(
        train_scaled, target_train_raw, janela, target_idx, horizon=horizonte,
        timestamps=ts_train, max_gap_min=max_gap)

    # concatena o fim de treino para gerar as janelas iniciais de teste
    combined_scaled = np.concatenate([train_scaled[-janela:], test_scaled], axis=0)
    combined_raw    = np.concatenate([target_train_raw[-janela:], target_test_raw])
    combined_ts = (np.concatenate([ts_train[-janela:], ts_test])
                   if ts_train is not None else None)
    X_test, d_test, y_ref_test, y_true = build_supervised_delta(
        combined_scaled, combined_raw, janela, target_idx, horizon=horizonte,
        timestamps=combined_ts, max_gap_min=max_gap)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Nenhuma janela continua de {janela}+{horizonte} passos "
            f"sobrou na fonte '{fonte}' apos descartar as que cruzam lacunas de coleta. "
            "Colete um historico mais continuo ou reduza LSTM_WINDOW_MIN."
        )

    model = build_model(janela, len(features))
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
        "fonte":       fonte,
        "features":    features,
        "window":      janela,
        "horizon":     horizonte,
        "train_size":  len(X_train),
        "test_size":   len(X_test),
        "y_true":      y_true.tolist(),
        "y_pred":      y_pred.tolist(),
        "timestamps":  test_df["timestamp"].astype(str).tolist()[-len(y_pred):],
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"lstm_{target}{sufixo}.keras")
    joblib.dump(scaler, MODELS_DIR / f"lstm_{target}{sufixo}_scaler.joblib")
    with open(MODELS_DIR / f"lstm_{target}{sufixo}_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


if __name__ == "__main__":
    import sys

    from src.metrics import rmse, mae, mape

    fonte_cli = sys.argv[1] if len(sys.argv) > 1 else "kaggle"
    for tgt in ["pH", "EC"]:
        r = run(target=tgt, fonte=fonte_cli)
        print(f"[LSTM {tgt} | {fonte_cli}] RMSE={rmse(r['y_true'], r['y_pred']):.4f} "
              f"MAE={mae(r['y_true'], r['y_pred']):.4f} "
              f"MAPE={mape(r['y_true'], r['y_pred']):.2f}%")
