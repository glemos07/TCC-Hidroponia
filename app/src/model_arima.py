"""
Modelo ARIMA (baseline univariado) para previsão de pH e CE.
Seção 3.3.1 e 4.6 do documento.

Estratégia de avaliação: para cada ponto do teste, refita a partir da série ate t-h
e faz forecast(h). Isso reflete o cenário "prever 30 min à frente com os dados
disponíveis agora", padrão em benchmarks de séries temporais.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from src.config import ARTIFACT_SUFFIX, MODELS_DIR, passos_horizonte
from src.data_loader import load_timeseries, train_test_split_series

warnings.filterwarnings("ignore")


def fit_arima(series: np.ndarray, order: tuple[int, int, int] = (2, 1, 2)):
    model = ARIMA(series, order=order, enforce_stationarity=False, enforce_invertibility=False)
    return model.fit()


def multistep_forecast(train: np.ndarray, test: np.ndarray, order=(2, 1, 2),
                       horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Para cada ponto t do teste, forecast(h) usando dados até t-h.
    Retorna (y_true, y_pred) alinhados.
    """
    fit = fit_arima(train, order=order)
    y_true, y_pred = [], []
    for i in range(len(test)):
        # forecast h passos à frente a partir do estado atual do fit
        yhat = float(fit.forecast(steps=horizon)[-1])
        # atualiza estado incorporando a observação real (sem refit)
        fit = fit.append([test[i]], refit=False)
        # o valor a prever é o de horizon passos à frente
        if i + horizon - 1 < len(test):
            y_true.append(float(test[i + horizon - 1]))
            y_pred.append(yhat)
    return np.array(y_true), np.array(y_pred)


def run(target: str = "pH", order=(2, 1, 2), fonte: str = "kaggle") -> dict:
    """Baseline univariado sobre a fonte indicada ("kaggle" ou "api")."""
    horizonte = passos_horizonte(fonte)
    df = load_timeseries(fonte=fonte)
    if len(df) < horizonte * 3:
        raise ValueError(
            f"Serie da fonte '{fonte}' tem apenas {len(df)} pontos; "
            f"o ARIMA precisa de pelo menos {horizonte * 3}."
        )
    train_df, test_df = train_test_split_series(df, test_frac=0.20)
    train, test = train_df[target].values, test_df[target].values

    y_true, y_pred = multistep_forecast(train, test, order=order, horizon=horizonte)

    out = {
        "target":      target,
        "fonte":       fonte,
        "order":       order,
        "horizon":     horizonte,
        "train_size":  len(train),
        "test_size":   len(y_true),
        "y_true":      y_true.tolist(),
        "y_pred":      y_pred.tolist(),
        "timestamps":  test_df["timestamp"].astype(str).tolist()[:len(y_true)],
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, MODELS_DIR / f"arima_{target}{ARTIFACT_SUFFIX[fonte]}.joblib")
    return out


if __name__ == "__main__":
    import sys

    from src.metrics import rmse, mae, mape

    fonte_cli = sys.argv[1] if len(sys.argv) > 1 else "kaggle"
    for tgt in ["pH", "EC"]:
        r = run(target=tgt, fonte=fonte_cli)
        print(f"[ARIMA {tgt} | {fonte_cli}] h={r['horizon']}  "
              f"RMSE={rmse(r['y_true'], r['y_pred']):.4f} "
              f"MAE={mae(r['y_true'], r['y_pred']):.4f} "
              f"MAPE={mape(r['y_true'], r['y_pred']):.2f}%")
