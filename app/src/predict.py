"""
Modulo de inferencia: carrega os LSTMs treinados e faz previsao
de pH e CE a partir do historico de uma planta especifica.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import LSTM_WINDOW, FORECAST_HORIZON_STEPS, MODELS_DIR
from src.database import fetch_df

FEATURES = ["pH", "EC", "temperature", "humidity"]


def _lazy_tf():
    import tensorflow as tf
    return tf


def load_plant_history(id_plant: int) -> pd.DataFrame:
    """
    Retorna o historico da planta como DataFrame com colunas
    ['timestamp', 'pH', 'EC', 'temperature', 'humidity'], ordenado.
    """
    sql = """
    SELECT m.dt_register AS timestamp,
           s.sensor_type AS sensor_type,
           sd.value      AS value
    FROM   monitoring m
    JOIN   sensor_data sd ON sd.id_monitoring = m.id
    JOIN   sensor s       ON s.id = sd.id_sensor
    WHERE  m.id_plant = ?
      AND  s.sensor_type IN ('pH', 'EC', 'Temperatura', 'Umidade')
    ORDER  BY m.dt_register
    """
    df = fetch_df(sql, params=(int(id_plant),))
    if df.empty:
        return pd.DataFrame(columns=["timestamp"] + FEATURES)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sensor_type"] = df["sensor_type"].map({
        "pH": "pH", "EC": "EC", "Temperatura": "temperature", "Umidade": "humidity"
    })
    wide = (df.pivot_table(index="timestamp",
                           columns="sensor_type",
                           values="value",
                           aggfunc="mean")
              .sort_index()
              .reset_index())
    for col in FEATURES:
        if col not in wide.columns:
            wide[col] = float("nan")
    return wide[["timestamp"] + FEATURES].dropna().reset_index(drop=True)


def _model_paths(target: str) -> tuple[Path, Path]:
    return (MODELS_DIR / f"lstm_{target}.keras",
            MODELS_DIR / f"lstm_{target}_scaler.joblib")


def models_available() -> dict[str, bool]:
    return {t: _model_paths(t)[0].exists() and _model_paths(t)[1].exists()
            for t in ("pH", "EC")}


def predict_next(id_plant: int, target: str = "pH") -> dict | None:
    """
    Faz previsao de UM ponto (horizonte = FORECAST_HORIZON_STEPS a frente)
    usando as ultimas LSTM_WINDOW leituras da planta.
    Retorna dict com timestamp previsto, valor previsto e ultimo observado.
    """
    tf = _lazy_tf()
    model_path, scaler_path = _model_paths(target)
    if not model_path.exists() or not scaler_path.exists():
        return None

    hist = load_plant_history(id_plant)
    if len(hist) < LSTM_WINDOW:
        return None

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    window = hist.iloc[-LSTM_WINDOW:][FEATURES].values
    window_scaled = scaler.transform(window)
    last_value = float(hist[target].iloc[-1])
    last_ts = pd.to_datetime(hist["timestamp"].iloc[-1])

    delta_pred = float(model.predict(window_scaled[np.newaxis, ...], verbose=0).flatten()[0])
    y_pred = last_value + delta_pred

    # estima intervalo entre observacoes
    if len(hist) >= 2:
        dt = (hist["timestamp"].iloc[-1] - hist["timestamp"].iloc[-2]).total_seconds() / 60
    else:
        dt = 1.0
    horizon_min = dt * FORECAST_HORIZON_STEPS
    future_ts = last_ts + pd.Timedelta(minutes=horizon_min)

    return {
        "target":         target,
        "id_plant":       id_plant,
        "last_timestamp": last_ts,
        "future_timestamp": future_ts,
        "horizon_min":    float(horizon_min),
        "last_value":     last_value,
        "delta":          delta_pred,
        "predicted":      y_pred,
    }


def predict_trajectory(id_plant: int, target: str = "pH",
                       steps_ahead: int = 12) -> pd.DataFrame | None:
    """
    Previsao recursiva: preve N horizontes rolando a janela.
    Cada previsao ocupa o slot correspondente da variavel alvo na janela.
    As demais variaveis (nao-alvo) sao mantidas com o ultimo valor observado.
    """
    tf = _lazy_tf()
    model_path, scaler_path = _model_paths(target)
    if not model_path.exists() or not scaler_path.exists():
        return None

    hist = load_plant_history(id_plant)
    if len(hist) < LSTM_WINDOW:
        return None

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    target_idx = FEATURES.index(target)
    window = hist.iloc[-LSTM_WINDOW:][FEATURES].values.copy()
    window_scaled = scaler.transform(window)

    last_value_raw = float(hist[target].iloc[-1])
    if len(hist) >= 2:
        dt = (hist["timestamp"].iloc[-1] - hist["timestamp"].iloc[-2]).total_seconds() / 60
    else:
        dt = 1.0
    last_ts = pd.to_datetime(hist["timestamp"].iloc[-1])

    predictions = []
    current_value = last_value_raw
    for k in range(1, steps_ahead + 1):
        delta = float(model.predict(window_scaled[np.newaxis, ...], verbose=0).flatten()[0])
        y_hat = current_value + delta
        future_ts = last_ts + pd.Timedelta(minutes=dt * FORECAST_HORIZON_STEPS * k)
        predictions.append({"timestamp": future_ts, target: y_hat, "delta": delta})

        # atualiza janela: descarta o primeiro e insere novo passo com y_hat
        new_row_raw = window[-1].copy()
        new_row_raw[target_idx] = y_hat
        new_row_scaled = scaler.transform(new_row_raw[np.newaxis, :]).flatten()
        window = np.vstack([window[1:], new_row_raw])
        window_scaled = np.vstack([window_scaled[1:], new_row_scaled])
        current_value = y_hat

    return pd.DataFrame(predictions)
