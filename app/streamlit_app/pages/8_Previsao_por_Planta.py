"""
Previsao de pH e CE para uma planta especifica.
Usa os LSTMs treinados em src.model_lstm sobre o historico da planta.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_utils import header, sidebar_info, PH_MIN, PH_MAX, EC_MIN, EC_MAX
from src.config import LSTM_WINDOW, FORECAST_HORIZON_STEPS
from src.database import fetch_df
from src.predict import (
    load_plant_history, models_available,
    predict_next, predict_trajectory,
)

sidebar_info()
header("Previsao de pH e CE por planta",
       "Selecione uma planta cadastrada para ver a previsao com o LSTM treinado")

plants = fetch_df("SELECT p.id, p.name_plant, p.diversity, fs.status_name "
                  "FROM plants p JOIN farm_status fs ON fs.id=p.status_id "
                  "ORDER BY p.name_plant")
if plants.empty:
    st.warning("Nenhuma planta cadastrada. Execute o seed em RF07.")
    st.stop()

avail = models_available()
if not (avail.get("pH") and avail.get("EC")):
    st.error(
        "Modelos LSTM ainda nao treinados.\n\n"
        "No PowerShell, do diretorio `app/`, rode:\n\n"
        "    python -m src.evaluate"
    )
    st.stop()

col_l, col_r = st.columns([2, 3])
with col_l:
    id_plant = st.selectbox(
        "Planta cadastrada",
        plants["id"],
        format_func=lambda i: (f"{plants.loc[plants['id']==i,'name_plant'].values[0]} "
                               f"[{plants.loc[plants['id']==i,'status_name'].values[0]}]"),
    )
    steps_ahead = st.slider("Passos de previsao (horizontes)", 1, 24, 12)

with col_r:
    st.info(
        f"Janela de contexto do LSTM: **{LSTM_WINDOW}** leituras\n\n"
        f"Cada passo previsto corresponde a **{FORECAST_HORIZON_STEPS} intervalos** de coleta a frente."
    )

hist = load_plant_history(int(id_plant))
if len(hist) < LSTM_WINDOW:
    st.warning(
        f"Planta possui apenas {len(hist)} leituras. "
        f"O LSTM precisa de pelo menos {LSTM_WINDOW}."
    )
    st.stop()

st.markdown("---")
st.markdown("### Situacao atual")

last = hist.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("pH atual",          f"{last['pH']:.2f}",
          "OK" if PH_MIN <= last["pH"] <= PH_MAX else "Fora da faixa")
c2.metric("CE atual (mS/cm)",  f"{last['EC']:.2f}",
          "OK" if EC_MIN <= last["EC"] <= EC_MAX else "Fora da faixa")
c3.metric("Temperatura (°C)",  f"{last['temperature']:.2f}")
c4.metric("Umidade (%)",       f"{last['humidity']:.2f}")

st.markdown("### Previsao proxima observacao")
next_pH = predict_next(int(id_plant), target="pH")
next_EC = predict_next(int(id_plant), target="EC")

pc, ec = st.columns(2)
if next_pH:
    delta_pH = next_pH["predicted"] - next_pH["last_value"]
    pc.metric(
        f"pH previsto (+{next_pH['horizon_min']:.0f} min)",
        f"{next_pH['predicted']:.3f}",
        delta=f"{delta_pH:+.3f}",
    )
    if not (PH_MIN <= next_pH["predicted"] <= PH_MAX):
        pc.warning(f"Previsao aponta pH fora da faixa ideal ({PH_MIN}-{PH_MAX})")
if next_EC:
    delta_EC = next_EC["predicted"] - next_EC["last_value"]
    ec.metric(
        f"CE prevista (+{next_EC['horizon_min']:.0f} min)",
        f"{next_EC['predicted']:.3f}",
        delta=f"{delta_EC:+.3f}",
    )
    if not (EC_MIN <= next_EC["predicted"] <= EC_MAX):
        ec.warning(f"Previsao aponta CE fora da faixa ideal ({EC_MIN}-{EC_MAX})")

st.markdown("### Trajetoria prevista")
traj_pH = predict_trajectory(int(id_plant), target="pH", steps_ahead=steps_ahead)
traj_EC = predict_trajectory(int(id_plant), target="EC", steps_ahead=steps_ahead)

hist_tail = hist.tail(LSTM_WINDOW).copy()


def _plot(hist_df, traj_df, target, ideal_lo, ideal_hi, cor):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["timestamp"], y=hist_df[target],
        mode="lines", name=f"{target} observado", line=dict(color=cor, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=traj_df["timestamp"], y=traj_df[target],
        mode="lines+markers", name=f"{target} previsto (LSTM)",
        line=dict(color=cor, width=2, dash="dash"),
    ))
    fig.add_hrect(y0=ideal_lo, y1=ideal_hi,
                  fillcolor="green", opacity=0.08, line_width=0,
                  annotation_text="Faixa ideal", annotation_position="top left")
    fig.update_layout(
        title=f"Trajetoria prevista de {target}",
        xaxis_title="Tempo", yaxis_title=target,
        margin=dict(t=40, b=40),
    )
    return fig


if traj_pH is not None:
    st.plotly_chart(_plot(hist_tail, traj_pH, "pH", PH_MIN, PH_MAX, "#1f4e79"),
                    use_container_width=True)
if traj_EC is not None:
    st.plotly_chart(_plot(hist_tail, traj_EC, "EC", EC_MIN, EC_MAX, "#c00000"),
                    use_container_width=True)

with st.expander("Dados da trajetoria prevista"):
    if traj_pH is not None:
        combined = traj_pH.rename(columns={"delta": "delta_pH"}).merge(
            traj_EC.rename(columns={"delta": "delta_EC", "EC": "EC"}),
            on="timestamp", how="outer",
        )
        st.dataframe(combined, use_container_width=True)
