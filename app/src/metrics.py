"""Métricas de avaliação: RMSE, MAE, MAPE e teste de Diebold-Mariano."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def diebold_mariano(y_true, y_pred_a, y_pred_b, h: int = 1, loss: str = "squared"):
    """
    Teste DM: H0 = ambos modelos têm a mesma acurácia preditiva.
    Retorna (estatística DM, p-valor bilateral).
    Referência: Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997).
    """
    from scipy import stats

    y_true = np.asarray(y_true, dtype=float)
    e_a = y_true - np.asarray(y_pred_a, dtype=float)
    e_b = y_true - np.asarray(y_pred_b, dtype=float)

    if loss == "squared":
        d = e_a ** 2 - e_b ** 2
    elif loss == "abs":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError("loss deve ser 'squared' ou 'abs'")

    n = len(d)
    d_bar = d.mean()

    # variância de longo prazo com correção HLN para pequenas amostras
    gamma0 = np.var(d, ddof=0)
    gammas = []
    for k in range(1, h):
        cov = np.cov(d[:-k], d[k:], ddof=0)[0, 1]
        gammas.append(cov)
    var_d = (gamma0 + 2 * sum(gammas)) / n
    if var_d <= 0:
        return float("nan"), float("nan")

    dm_stat = d_bar / np.sqrt(var_d)
    # correção Harvey-Leybourne-Newbold
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat *= hln

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_value)
