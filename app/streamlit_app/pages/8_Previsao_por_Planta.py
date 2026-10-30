"""
Previsao de pH e CE para uma planta especifica.

Usa os LSTMs treinados sobre a telemetria do ESP32 (`python -m src.evaluate api`)
aplicados ao historico da planta no banco. A comparacao ARIMA x LSTM sobre o
dataset publico esta na tela 9.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_utils import header, sidebar_info, PH_MIN, PH_MAX, EC_MIN, EC_MAX
from src.config import (
    MAX_GAP_FACTOR,
    features_de,
    horizonte_min,
    intervalo_amostragem,
    janela_min,
    passos_janela,
)
from src.data_loader import plantas_api
from src.database import fetch_df
from src.predict import (
    backtest_recente, load_plant_history, models_available,
    predict_next, predict_trajectory,
)

#: Rotulo amigavel de cada feature, para os indicadores de situacao atual.
ROTULO_FEATURE = {
    "pH":          ("pH atual", ""),
    "EC":          ("CE atual (mS/cm)", ""),
    "temperature": ("Temp. da solucao (C)", ""),
    "humidity":    ("Umidade do ar (%)", ""),
    "air_temp":    ("Temp. do ambiente (C)", ""),
}

sidebar_info()
header("Previsao de pH e CE por planta",
       "Selecione uma planta cadastrada para ver a previsao com o LSTM treinado")

# A previsao por planta le o historico do banco, e o banco guarda apenas o que
# os sensores mediram - por isso a fonte aqui e sempre "api". A comparacao com
# o dataset do Kaggle vive na tela 9, sobre o conjunto de teste daquela base.
fonte = "api"
features = features_de(fonte)
st.caption("Fonte: sensores do ESP32 | vetor de entrada: " + ", ".join(features))
JANELA = passos_janela(fonte)

ingeridas = plantas_api()
if ingeridas.empty or int(ingeridas["leituras"].max() or 0) == 0:
    st.warning(
        "Nenhuma leitura dos sensores foi importada ainda. "
        "Abra a tela **10_Servidor_API** e sincronize o servidor."
    )
    st.stop()

plants = fetch_df(
    "SELECT p.id, p.name_plant, p.diversity, fs.status_name "
    "FROM plants p JOIN farm_status fs ON fs.id = p.status_id "
    f"WHERE p.id IN ({','.join(str(int(i)) for i in ingeridas['id_plant'])}) "
    "ORDER BY p.name_plant"
)
if plants.empty:
    st.warning("Nenhuma planta com leituras de sensor. Sincronize na tela 10_Servidor_API.")
    st.stop()

avail = models_available(fonte)
if not (avail.get("pH") and avail.get("EC")):
    st.error(
        "Modelos LSTM dos sensores ainda nao treinados.\n\n"
        "No PowerShell, do diretorio `app/`, rode:\n\n"
        "    python -m src.evaluate api"
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
        f"Contexto do LSTM: **{janela_min(fonte) / 60:.0f} h** "
        f"({JANELA} pontos de {intervalo_amostragem(fonte)} min)\n\n"
        f"Cada passo previsto avanca **{horizonte_min(fonte) / 60:.0f} h**."
    )

# so a janela do modelo: a tela plota os ultimos JANELA pontos, e varrer o
# historico inteiro custaria varias vezes mais a cada interacao
hist = load_plant_history(int(id_plant), fonte=fonte, janela=JANELA)
if len(hist) < JANELA:
    st.warning(
        f"Planta possui apenas {len(hist)} pontos na grade de "
        f"{intervalo_amostragem(fonte)} min. O LSTM precisa de {JANELA}."
    )
    st.stop()

st.markdown("---")
st.markdown("### Situacao atual")

last = hist.iloc[-1]
colunas = st.columns(len(features))
for coluna, feature in zip(colunas, features):
    rotulo = ROTULO_FEATURE.get(feature, (feature, ""))[0]
    if feature == "pH":
        delta = "OK" if PH_MIN <= last["pH"] <= PH_MAX else "Fora da faixa"
    elif feature == "EC":
        delta = "OK" if EC_MIN <= last["EC"] <= EC_MAX else "Fora da faixa"
    else:
        delta = None
    coluna.metric(rotulo, f"{last[feature]:.2f}", delta)

st.caption(f"Ultima leitura: {pd.Timestamp(last['timestamp']):%d/%m/%Y %H:%M:%S}")

lacuna = hist.tail(JANELA)["timestamp"].diff().dropna().dt.total_seconds().max() / 60
intervalo = hist["timestamp"].diff().dropna().dt.total_seconds().median() / 60
if pd.notna(lacuna) and lacuna > MAX_GAP_FACTOR * max(intervalo, 1e-6):
    st.warning(
        f"A janela usada na previsao contem uma interrupcao de "
        f"{lacuna / 60:.1f} h, bem acima do intervalo tipico de "
        f"{intervalo:.1f} min entre leituras. A previsao sai, mas tratando "
        "como consecutivas leituras separadas por essa parada."
    )

st.markdown("### Previsao proxima observacao")
next_pH = predict_next(int(id_plant), target="pH", fonte=fonte)
next_EC = predict_next(int(id_plant), target="EC", fonte=fonte)

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

st.markdown("### Acuracia nos horizontes que ja venceram")
st.caption(
    "Para cada instante recente, o modelo foi rodado com a janela que terminava "
    "um horizonte antes, e o resultado comparado com o que o sensor registrou. "
    "Mede o acerto sem precisar esperar a previsao atual vencer."
)

col_bt = st.columns(2)
for coluna, (alvo, cor) in zip(col_bt, [("pH", "#1f4e79"), ("EC", "#c00000")]):
    bt = backtest_recente(int(id_plant), target=alvo, fonte=fonte, n_pontos=40)
    if bt is None or bt.empty:
        coluna.info(f"{alvo}: historico insuficiente para avaliar.")
        continue

    mae = bt["erro"].abs().mean()
    rmse = float((bt["erro"] ** 2).mean() ** 0.5)
    mae_persistente = bt["erro_persistente"].abs().mean()
    ganho = (mae_persistente - mae) / mae_persistente * 100 if mae_persistente else 0.0

    coluna.metric(f"{alvo} - MAE do LSTM ({len(bt)} horizontes)", f"{mae:.4f}",
                  delta=f"{ganho:+.0f}% vs supor que nada muda")

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt["timestamp"], y=bt["real"], name="real",
                                line=dict(color=cor, width=2)))
    fig_bt.add_trace(go.Scatter(x=bt["timestamp"], y=bt["previsto"], name="previsto (LSTM)",
                                line=dict(color=cor, width=2, dash="dash")))
    fig_bt.add_trace(go.Scatter(x=bt["timestamp"], y=bt["persistente"],
                                name="persistencia", line=dict(color="#7f7f7f", width=1.5,
                                                               dash="dot")))
    fig_bt.update_layout(title=f"{alvo}: previsto x real (RMSE {rmse:.4f})",
                         xaxis_title="Tempo", yaxis_title=alvo, height=280,
                         margin=dict(t=40, b=30),
                         legend=dict(orientation="h", y=1.2))
    coluna.plotly_chart(fig_bt, use_container_width=True)

    if mae > mae_persistente:
        coluna.warning(
            f"Nesta faixa o LSTM de {alvo} erra mais (MAE {mae:.4f}) do que supor "
            f"que o valor nao muda (MAE {mae_persistente:.4f})."
        )

st.markdown("### Trajetoria prevista")
traj_pH = predict_trajectory(int(id_plant), target="pH", steps_ahead=steps_ahead, fonte=fonte)
traj_EC = predict_trajectory(int(id_plant), target="EC", steps_ahead=steps_ahead, fonte=fonte)

hist_tail = hist.tail(JANELA).copy()


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
    if traj_pH is not None and traj_EC is not None:
        combined = traj_pH.rename(columns={"delta": "delta_pH"}).merge(
            traj_EC.rename(columns={"delta": "delta_EC"}),
            on="timestamp", how="outer",
        )
        st.dataframe(combined, use_container_width=True)
