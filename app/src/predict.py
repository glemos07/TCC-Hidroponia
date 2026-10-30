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
    API_RESAMPLE_MIN,
    ARTIFACT_SUFFIX,
    MODELS_DIR,
    features_de,
    intervalo_amostragem,
    passos_horizonte,
    passos_janela,
)
from src.data_loader import load_plant_features, reamostrar

#: Vetor de entrada da fonte padrao (mantido por compatibilidade).
FEATURES = features_de("kaggle")


def _lazy_tf():
    import tensorflow as tf
    return tf


#: Margem sobre a janela pedida ao banco: leituras com algum sensor fora de
#: faixa somem no dropna, entao buscar exatamente a janela pode devolver
#: menos pontos do que o modelo precisa.
MARGEM_JANELA = 3


def load_plant_history(id_plant: int, fonte: str = "kaggle",
                       janela: int | None = None) -> pd.DataFrame:
    """
    Historico da planta na MESMA grade usada no treino da fonte.

    `janela` pede so os ultimos N pontos da grade - o suficiente para inferir,
    sem varrer todo o historico. O limite e convertido para linhas cruas
    levando em conta a reamostragem; se nao bastar, recai para a leitura
    completa.
    """
    features = features_de(fonte)
    minutos = API_RESAMPLE_MIN if fonte == "api" else None

    if janela is None:
        bruto = load_plant_features(int(id_plant), features)
        return reamostrar(bruto, minutos, features) if minutos else bruto

    # cada ponto da grade consome ~(minutos / cadencia do sensor) linhas cruas
    por_ponto = max(1, round(minutos / 1.6)) if minutos else 1
    limite = janela * por_ponto * MARGEM_JANELA
    bruto = load_plant_features(int(id_plant), features, limite=limite)
    hist = reamostrar(bruto, minutos, features) if minutos else bruto

    if len(hist) < janela:
        completo = load_plant_features(int(id_plant), features)
        completo = reamostrar(completo, minutos, features) if minutos else completo
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
    Faz previsao de UM ponto (um horizonte a frente) usando a ultima janela
    de contexto da planta, ambos definidos em minutos na config.
    Retorna dict com timestamp previsto, valor previsto e ultimo observado.
    """
    model, scaler = _carregar(target, fonte)
    if model is None:
        return None

    features = features_de(fonte)
    janela = passos_janela(fonte)
    hist = load_plant_history(id_plant, fonte=fonte, janela=janela)
    if len(hist) < janela:
        return None

    window = hist.iloc[-janela:][features].values
    window_scaled = scaler.transform(window)
    last_value = float(hist[target].iloc[-1])
    last_ts = pd.to_datetime(hist["timestamp"].iloc[-1])

    delta_pred = float(model.predict(window_scaled[np.newaxis, ...], verbose=0).flatten()[0])
    y_pred = last_value + delta_pred

    intervalo = _intervalo_min(hist)
    horizon_min = intervalo_amostragem(fonte) * passos_horizonte(fonte)
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
        "maior_lacuna_min": _maior_lacuna(hist.iloc[-janela:]),
        "intervalo_min":    float(intervalo),
    }


def backtest_recente(id_plant: int, target: str = "pH", fonte: str = "api",
                     n_pontos: int = 30) -> pd.DataFrame | None:
    """
    Confronta previsao com realidade nos ultimos `n_pontos` horizontes que ja
    venceram: para cada instante, roda o modelo com a janela que terminava um
    horizonte antes e compara com o valor que o sensor de fato registrou.

    Serve para mostrar a acuracia ao vivo sem esperar o horizonte correr -
    aguardar a previsao atual vencer nao cabe numa apresentacao.

    Todas as janelas vao ao modelo num unico `predict` em lote; chamar uma por
    vez custaria ~130 ms cada.
    """
    model, scaler = _carregar(target, fonte)
    if model is None:
        return None

    features = features_de(fonte)
    horizonte = passos_horizonte(fonte)
    janela = passos_janela(fonte)
    necessario = janela + horizonte + n_pontos
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
        inicio = fim_janela - janela + 1
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


# `predict_trajectory` (previsao recursiva de N horizontes) foi removida.
# Ela realimentava a propria saida na janela, e com o horizonte de 24 h cada
# passo avancava um dia inteiro: o vies de +0,14 por passo se acumulava ate
# projetar pH 8,87 em 12 dias, fisicamente impossivel em hidroponia. Para
# varios horizontes, treine um modelo por horizonte em vez de realimentar.
