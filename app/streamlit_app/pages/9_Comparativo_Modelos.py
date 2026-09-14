"""
Pagina extra - Quadro 6 do documento (ARIMA vs LSTM sobre o hold-out).
Metricas globais comparativas.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_utils import header, sidebar_info
from src.config import REPORTS_DIR

sidebar_info()
header("Previsao de pH e CE (Quadro 6)", "ARIMA (baseline) vs LSTM (multivariado)")

metrics_path = REPORTS_DIR / "metrics.json"
if not metrics_path.exists():
    st.warning(
        "Metricas ainda nao geradas. Execute:\n\n"
        "    python -m src.evaluate\n\n"
        "no diretorio `app/`."
    )
    st.stop()

with open(metrics_path, encoding="utf-8") as f:
    results = json.load(f)

target = st.selectbox("Variavel", [r["target"] for r in results])
res = next(r for r in results if r["target"] == target)

col1, col2, col3 = st.columns(3)
col1.metric("RMSE ARIMA", f"{res['ARIMA']['RMSE']:.4f}")
col1.metric("RMSE LSTM",  f"{res['LSTM']['RMSE']:.4f}",
            delta=f"{(res['LSTM']['RMSE'] - res['ARIMA']['RMSE']) / res['ARIMA']['RMSE'] * 100:.1f}%")
col2.metric("MAE ARIMA",  f"{res['ARIMA']['MAE']:.4f}")
col2.metric("MAE LSTM",   f"{res['LSTM']['MAE']:.4f}",
            delta=f"{(res['LSTM']['MAE'] - res['ARIMA']['MAE']) / res['ARIMA']['MAE'] * 100:.1f}%")
col3.metric("MAPE ARIMA (%)", f"{res['ARIMA']['MAPE']:.2f}")
col3.metric("MAPE LSTM (%)",  f"{res['LSTM']['MAPE']:.2f}",
            delta=f"{(res['LSTM']['MAPE'] - res['ARIMA']['MAPE']) / res['ARIMA']['MAPE'] * 100:.1f}%")

dm = res["diebold_mariano"]
sig = "significativa (p < 0,05)" if dm["p_value"] < 0.05 else "nao significativa"
st.info(f"Diebold-Mariano: stat={dm['stat']:.3f}, p-value={dm['p_value']:.4f} - diferenca {sig}.")

img_path = REPORTS_DIR / f"grafico3_{target}.png"
if img_path.exists():
    st.image(str(img_path), caption=f"Grafico 3 - Previsao de {target}: observado x previsto (LSTM e ARIMA)")
