"""
Modulo de inferencia: carrega os LSTMs treinados e faz previsao
de pH e CE a partir do historico de uma planta especifica.

O parametro `fonte` escolhe o par (vetor de entrada, artefatos treinados):

    fonte="kaggle"  features pH, EC, temperature, humidity
                    -> lstm_<alvo>.keras          (modelos do Quadro 6)
    fonte="api"     features pH, EC, temperature, air_temp
                    -> lstm_<alvo>_api.keras      (telemetria real do ESP32)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACT_SUFFIX,
    FORECAST_HORIZON_STEPS,
    LSTM_WINDOW,
    MODELS_DIR,
    features_de,
)
from src.data_loader import load_plant_features

#: Vetor de entrada da fonte padrao (mantido por compatibilidade).
FEATURES = features_de("kaggle")


def _lazy_tf():
    import tensorflow as tf
    return tf


#: Margem sobre a janela pedida ao banco: leituras com algum sensor fora de
#: faixa somem no dropna, entao buscar exatamente LSTM_WINDOW linhas pode
#: devolver menos do que o modelo precisa.
MARGEM_JANELA = 3


def load_plant_history(id_plant: int, fonte: str = "kaggle",
                       janela: int | None = None) -> pd.DataFrame:
    """
    Historico da planta como DataFrame ['timestamp', *features da fonte],
    ordenado cronologicamente.

    `janela` pede so as ultimas N observacoes completas - o suficiente para
    inferir, sem varrer todo o historico. Se a margem nao bastar, recai para
    a leitura completa.
    """
    features = features_de(fonte)
    if janela is None:
        return load_plant_features(int(id_plant), features)

    hist = load_plant_features(int(id_plant), features, limite=janela * MARGEM_JANELA)
    if len(hist) < janela:
        completo = load_plant_features(int(id_plant), features)
        if len(completo) > len(hist):
            return completo
    return hist


def _model_paths(target: str, fonte: str = "kaggle") -> tuple[Path, Path]:
    sufixo = ARTIFACT_SUFFIX[fonte]
    return (MODELS_DIR / f"lstm_{target}{sufixo}.keras",
            MODELS_DIR / f"lstm_{target}{sufixo}_scaler.joblib")


def models_available(fonte: str = "kaggle") -> dict[str, bool]:
    return {t: all(p.exists() for p in _model_paths(t, fonte))
            for t in ("pH", "EC")}


@lru_cache(maxsize=8)
def _carregar_cached(model_path: Path, scaler_path: Path, _assinatura: tuple):
    """
    Carrega modelo + scaler do disco. Chave inclui a assinatura (mtime/tamanho)
    dos arquivos, entao retreinar invalida o cache sozinho.
    """
    tf = _lazy_tf()
    return tf.keras.models.load_model(model_path), joblib.load(scaler_path)


def _carregar(target: str, fonte: str):
    """
    Modelo + scaler da fonte, ou (None, None) se ainda nao foram treinados.

    O resultado fica em cache no processo: carregar o .keras do disco custa
    ~8 s, e sem cache cada rerun do Streamlit pagaria isso de novo, o que
    inviabiliza acompanhar os sensores ao vivo.
    """
    model_path, scaler_path = _model_paths(target, fonte)
    if not model_path.exists() or not scaler_path.exists():
        return None, None

    assinatura = tuple((p.stat().st_mtime_ns, p.stat().st_size)
                       for p in (model_path, scaler_path))
    return _carregar_cached(model_path, scaler_path, assinatura)


def _intervalo_min(hist: pd.DataFrame) -> float:
    """Minutos entre observacoes consecutivas (mediana, robusta a lacunas)."""
    if len(hist) < 2:
        return 1.0
    deltas = hist["timestamp"].diff().dropna().dt.total_seconds() / 60
    mediana = float(deltas.median())
    return mediana if mediana > 0 else 1.0


def _maior_lacuna(janela: pd.DataFrame) -> float:
    """Maior intervalo entre observacoes consecutivas da janela, em minutos."""
    if len(janela) < 2:
        return 0.0
    return float(janela["timestamp"].diff().dropna().dt.total_seconds().max() / 60)


def predict_next(id_plant: int, target: str = "pH", fonte: str = "kaggle") -> dict | None:
    """
    Faz previsao de UM ponto (horizonte = FORECAST_HORIZON_STEPS a frente)
    usando as ultimas LSTM_WINDOW leituras da planta.
    Retorna dict com timestamp previsto, valor previsto e ultimo observado.
    """
    model, scaler = _carregar(target, fonte)
    if model is None:
        return None

    features = features_de(fonte)
    hist = load_plant_history(id_plant, fonte=fonte, janela=LSTM_WINDOW)
    if len(hist) < LSTM_WINDOW:
        return None

    window = hist.iloc[-LSTM_WINDOW:][features].values
    window_scaled = scaler.transform(window)
    last_value = float(hist[target].iloc[-1])
    last_ts = pd.to_datetime(hist["timestamp"].iloc[-1])

    delta_pred = float(model.predict(window_scaled[np.newaxis, ...], verbose=0).flatten()[0])
    y_pred = last_value + delta_pred

    intervalo = _intervalo_min(hist)
    horizon_min = intervalo * FORECAST_HORIZON_STEPS
    future_ts = last_ts + pd.Timedelta(minutes=horizon_min)

    return {
        "target":         target,
        "fonte":          fonte,
        "id_plant":       id_plant,
        "last_timestamp": last_ts,
        "future_timestamp": future_ts,
        "horizon_min":    float(horizon_min),
        "last_value":     last_value,
        "delta":          delta_pred,
        "predicted":      y_pred,
        # maior interrupcao dentro da janela usada: se for muito maior que o
        # intervalo tipico, a janela cruza uma parada da coleta e a previsao
        # merece ressalva na tela
        "maior_lacuna_min": _maior_lacuna(hist.iloc[-LSTM_WINDOW:]),
        "intervalo_min":    float(intervalo),
    }


def backtest_recente(id_plant: int, target: str = "pH", fonte: str = "api",
                     n_pontos: int = 30) -> pd.DataFrame | None:
    """
    Confronta previsao com realidade nos ultimos `n_pontos` horizontes que ja
    venceram: para cada instante, roda o modelo com a janela que terminava um
    horizonte antes e compara com o valor que o sensor de fato registrou.

    Serve para mostrar a acuracia ao vivo sem esperar o horizonte correr - com
    96 min a frente, aguardar a previsao vencer nao cabe numa apresentacao.

    Todas as janelas vao ao modelo num unico `predict` em lote; chamar uma por
    vez custaria ~130 ms cada.
    """
    model, scaler = _carregar(target, fonte)
    if model is None:
        return None

    features = features_de(fonte)
    horizonte = FORECAST_HORIZON_STEPS
    necessario = LSTM_WINDOW + horizonte + n_pontos
    hist = load_plant_history(id_plant, fonte=fonte, janela=necessario)
    if len(hist) < necessario:
        return None

    valores = hist[features].values
    escalado = scaler.transform(valores)
    alvo = hist[target].values
    n = len(hist)

    janelas, refs, reais, instantes = [], [], [], []
    for k in range(n_pontos):
        idx_real = n - n_pontos + k
        fim_janela = idx_real - (horizonte - 1)
        inicio = fim_janela - LSTM_WINDOW + 1
        if inicio < 0:
            continue
        janelas.append(escalado[inicio: fim_janela + 1])
        refs.append(alvo[fim_janela])
        reais.append(alvo[idx_real])
        instantes.append(hist["timestamp"].iloc[idx_real])

    if not janelas:
        return None

    deltas = model.predict(np.stack(janelas), verbose=0).flatten()
    referencia = np.asarray(refs)
    previsto = referencia + deltas
    real = np.asarray(reais)

    # Baseline de persistencia: supor que o valor nao muda no horizonte. E o
    # que um ARIMA(p,1,q) faz em serie muito autocorrelacionada, e o piso que
    # o LSTM precisa superar para estar agregando alguma coisa.
    return pd.DataFrame({
        "timestamp":   instantes,
        "previsto":    previsto,
        "real":        real,
        "erro":        previsto - real,
        "persistente": referencia,
        "erro_persistente": referencia - real,
    })


def predict_trajectory(id_plant: int, target: str = "pH",
                       steps_ahead: int = 12, fonte: str = "kaggle") -> pd.DataFrame | None:
    """
    Previsao recursiva: preve N horizontes rolando a janela.
    Cada previsao ocupa o slot correspondente da variavel alvo na janela.
    As demais variaveis (nao-alvo) sao mantidas com o ultimo valor observado.
    """
    model, scaler = _carregar(target, fonte)
    if model is None:
        return None

    features = features_de(fonte)
    hist = load_plant_history(id_plant, fonte=fonte, janela=LSTM_WINDOW)
    if len(hist) < LSTM_WINDOW:
        return None

    target_idx = features.index(target)
    window = hist.iloc[-LSTM_WINDOW:][features].values.copy()
    window_scaled = scaler.transform(window)

    current_value = float(hist[target].iloc[-1])
    dt = _intervalo_min(hist)
    last_ts = pd.to_datetime(hist["timestamp"].iloc[-1])

    predictions = []
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
