"""
Previsao de pH e CE para uma planta especifica.

Usa os LSTMs treinados sobre a telemetria do ESP32 (`python -m src.evaluate api`)
aplicados ao historico da planta no banco. A comparacao ARIMA x LSTM sobre o
dataset publico esta na tela 9.
"""
from datetime import datetime

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
    backtest_recente, load_plant_history, models_available, predict_next,
)

st.set_page_config(page_title="Previsao por planta", page_icon="🌱", layout="wide")

#: Rotulo amigavel de cada feature, para os indicadores de situacao atual.
ROTULO_FEATURE = {
    "pH":          "pH atual",
    "EC":          "CE atual (mS/cm)",
    "temperature": "Temp. da solucao (C)",
    "humidity":    "Umidade do ar (%)",
    "air_temp":    "Temp. do ambiente (C)",
}

#: Janelas de exibicao oferecidas. O padrao iguala o horizonte, para que o
#: trecho observado e o previsto ocupem metades comparaveis do grafico.
JANELAS_H = {"6 h": 6, "12 h": 12, "24 h": 24, "3 dias": 72, "7 dias": 168}

sidebar_info()
header("Previsao de pH e CE por planta",
       "LSTM treinado sobre as leituras dos sensores desta planta")

# A previsao por planta le o historico do banco, e o banco guarda apenas o que
# os sensores mediram - por isso a fonte aqui e sempre "api". A comparacao com
# o dataset do Kaggle vive na tela 9, sobre o conjunto de teste daquela base.
fonte = "api"
features = features_de(fonte)
GRADE = intervalo_amostragem(fonte)
CONTEXTO = passos_janela(fonte)
HORIZONTE_H = horizonte_min(fonte) / 60

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

if not all(models_available(fonte).values()):
    st.error(
        "Modelos LSTM dos sensores ainda nao treinados.\n\n"
        "No PowerShell, do diretorio `app/`, rode:\n\n"
        "    python -m src.evaluate api"
    )
    st.stop()

# ----------------------------------------------------------------- controles

c_planta, c_janela, c_vivo, c_int = st.columns([3, 2, 1, 2])
with c_planta:
    id_plant = int(st.selectbox(
        "Planta",
        plants["id"],
        format_func=lambda i: (f"{plants.loc[plants['id']==i,'name_plant'].values[0]} "
                               f"[{plants.loc[plants['id']==i,'status_name'].values[0]}]"),
    ))
with c_janela:
    rotulo_janela = st.selectbox("Janela exibida", list(JANELAS_H), index=2)
with c_vivo:
    ao_vivo = st.toggle("Ao vivo", value=False,
                        help="Recalcula sozinho. O ESP32 publica a cada 90 s.")
with c_int:
    intervalo = st.select_slider("Atualizar a cada", options=[10, 30, 60, 120],
                                 value=30, format_func=lambda s: f"{s} s",
                                 disabled=not ao_vivo)

PONTOS_JANELA = max(CONTEXTO, JANELAS_H[rotulo_janela] * 60 // GRADE)

st.caption(
    f"Sensores do ESP32 · grade de {GRADE} min · entradas: {', '.join(features)} · "
    f"o modelo olha {janela_min(fonte) / 60:.0f} h para tras e preve "
    f"{HORIZONTE_H:.0f} h a frente"
)


# ------------------------------------------------------------------ previsao

def painel(id_plant: int, pontos: int):
    """Situacao atual, previsao e grafico - tudo recalculado a cada ciclo."""
    hist = load_plant_history(id_plant, fonte=fonte, janela=pontos)
    if len(hist) < CONTEXTO:
        st.warning(
            f"Planta tem {len(hist)} pontos na grade de {GRADE} min; "
            f"o modelo precisa de {CONTEXTO}."
        )
        return

    ultima = hist.iloc[-1]
    ts_ultima = pd.Timestamp(ultima["timestamp"])
    idade = (datetime.now() - ts_ultima.to_pydatetime()).total_seconds() / 60

    previsoes = {alvo: predict_next(id_plant, target=alvo, fonte=fonte)
                 for alvo in ("pH", "EC")}

    # ---- a previsao em destaque, antes de tudo
    st.markdown(f"### Previsao para {_quando(previsoes)}")
    cols = st.columns(2)
    for coluna, (alvo, unidade, lo, hi) in zip(
            cols, [("pH", "", PH_MIN, PH_MAX), ("EC", " mS/cm", EC_MIN, EC_MAX)]):
        r = previsoes[alvo]
        if r is None:
            coluna.info(f"{alvo}: historico insuficiente.")
            continue
        variacao = r["predicted"] - r["last_value"]
        coluna.metric(
            f"{alvo} daqui a {HORIZONTE_H:.0f} h",
            f"{r['predicted']:.2f}{unidade}",
            delta=f"{variacao:+.2f} em relacao a agora ({r['last_value']:.2f})",
            delta_color="off",
        )
        if not (lo <= r["predicted"] <= hi):
            coluna.warning(
                f"Fora da faixa ideal ({lo}–{hi}{unidade}). "
                f"{'Tende a subir' if variacao > 0 else 'Tende a cair'} a partir de "
                f"{r['last_value']:.2f}."
            )

    _grafico(hist, previsoes, rotulo_janela)

    # ---- situacao medida agora
    st.markdown("### Medido agora")
    medidas = st.columns(len(features))
    for coluna, feature in zip(medidas, features):
        if feature == "pH":
            estado = "OK" if PH_MIN <= ultima["pH"] <= PH_MAX else "Fora da faixa"
        elif feature == "EC":
            estado = "OK" if EC_MIN <= ultima["EC"] <= EC_MAX else "Fora da faixa"
        else:
            estado = None
        coluna.metric(ROTULO_FEATURE.get(feature, feature),
                      f"{ultima[feature]:.2f}", estado)

    # o rotulo do ponto e o inicio do bin de GRADE minutos, entao ate uma
    # grade inteira de idade e normal mesmo com a coleta perfeita em dia
    if idade <= GRADE * 2:
        st.success(
            f"Dados em dia — ultimo ponto da grade as {ts_ultima:%d/%m %H:%M}, "
            f"ha {idade:.0f} min (a grade e de {GRADE} min)."
        )
    else:
        st.warning(
            f"Ultimo ponto da grade as {ts_ultima:%d/%m %H:%M} — ha {idade / 60:.1f} h. "
            "Sincronize na tela 10_Servidor_API para trazer o que falta."
        )

    lacuna = hist.tail(CONTEXTO)["timestamp"].diff().dropna().dt.total_seconds().max() / 60
    if pd.notna(lacuna) and lacuna > MAX_GAP_FACTOR * GRADE:
        st.warning(
            f"A janela usada na previsao contem uma interrupcao de {lacuna / 60:.1f} h. "
            "A previsao sai, mas tratando como consecutivas leituras separadas por essa parada."
        )


def _quando(previsoes: dict) -> str:
    for r in previsoes.values():
        if r:
            return f"{pd.Timestamp(r['future_timestamp']):%d/%m as %Hh}"
    return "-"


def _grafico(hist: pd.DataFrame, previsoes: dict, rotulo: str):
    """Observado na janela escolhida, com a previsao como continuacao tracejada."""
    for alvo, cor, lo, hi, titulo in [
        ("pH", "#1f4e79", PH_MIN, PH_MAX, "pH"),
        ("EC", "#c00000", EC_MIN, EC_MAX, "CE (mS/cm)"),
    ]:
        r = previsoes[alvo]
        fig = go.Figure()
        fig.add_hrect(y0=lo, y1=hi, fillcolor="green", opacity=0.07, line_width=0,
                      annotation_text="faixa ideal", annotation_position="top left")
        fig.add_trace(go.Scatter(
            x=hist["timestamp"], y=hist[alvo], mode="lines", name=f"{titulo} medido",
            line=dict(color=cor, width=2),
        ))
        if r is not None:
            fig.add_trace(go.Scatter(
                x=[hist["timestamp"].iloc[-1], r["future_timestamp"]],
                y=[r["last_value"], r["predicted"]],
                mode="lines+markers", name=f"previsto (+{HORIZONTE_H:.0f} h)",
                line=dict(color=cor, width=2, dash="dash"),
                marker=dict(size=[0, 13], symbol="diamond"),
            ))
            fig.add_annotation(
                x=r["future_timestamp"], y=r["predicted"],
                text=f"<b>{r['predicted']:.2f}</b>", showarrow=False,
                yshift=20, font=dict(color=cor, size=14),
            )
        fig.update_layout(
            title=f"{titulo} — ultimas {rotulo} medidas e previsao de {HORIZONTE_H:.0f} h",
            xaxis_title="Tempo", yaxis_title=titulo, height=320,
            margin=dict(t=48, b=36), legend=dict(orientation="h", y=1.14),
        )
        st.plotly_chart(fig, use_container_width=True)


# `run_every` e fixado na criacao do fragment, entao ele e montado aqui ja com
# o intervalo escolhido (None = so recalcula quando algo e tocado na tela).
st.experimental_fragment(run_every=intervalo if ao_vivo else None)(painel)(
    id_plant, PONTOS_JANELA
)

# ------------------------------------------------------- acuracia do modelo

st.markdown("---")
st.markdown("### Acuracia nos horizontes que ja venceram")
st.caption(
    "Para cada instante recente, o modelo foi rodado com a janela que terminava "
    f"{HORIZONTE_H:.0f} h antes, e o resultado comparado com o que o sensor registrou. "
    "A linha cinza e a baseline: supor que o valor nao muda no horizonte."
)

col_bt = st.columns(2)
for coluna, (alvo, cor) in zip(col_bt, [("pH", "#1f4e79"), ("EC", "#c00000")]):
    bt = backtest_recente(id_plant, target=alvo, fonte=fonte, n_pontos=40)
    if bt is None or bt.empty:
        coluna.info(f"{alvo}: historico insuficiente para avaliar.")
        continue

    mae = bt["erro"].abs().mean()
    rmse = float((bt["erro"] ** 2).mean() ** 0.5)
    mae_persistente = bt["erro_persistente"].abs().mean()
    ganho = (mae_persistente - mae) / mae_persistente * 100 if mae_persistente else 0.0

    coluna.metric(f"{alvo} — MAE do LSTM ({len(bt)} horizontes)", f"{mae:.4f}",
                  delta=f"{ganho:+.0f}% vs supor que nada muda", delta_color="normal")

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt["timestamp"], y=bt["real"], name="real",
                                line=dict(color=cor, width=2)))
    fig_bt.add_trace(go.Scatter(x=bt["timestamp"], y=bt["previsto"], name="previsto (LSTM)",
                                line=dict(color=cor, width=2, dash="dash")))
    fig_bt.add_trace(go.Scatter(x=bt["timestamp"], y=bt["persistente"], name="persistencia",
                                line=dict(color="#7f7f7f", width=1.5, dash="dot")))
    fig_bt.update_layout(title=f"{alvo}: previsto x real (RMSE {rmse:.4f})",
                         xaxis_title="Tempo", yaxis_title=alvo, height=280,
                         margin=dict(t=40, b=30), legend=dict(orientation="h", y=1.2))
    coluna.plotly_chart(fig_bt, use_container_width=True)

    if mae > mae_persistente:
        coluna.warning(
            f"Nesta faixa o LSTM de {alvo} erra mais (MAE {mae:.4f}) do que supor "
            f"que o valor nao muda (MAE {mae_persistente:.4f})."
        )
