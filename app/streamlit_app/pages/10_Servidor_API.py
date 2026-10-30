"""
Servidor de sensores (API FastAPI + ESP32).

Tela de operacao da fonte de dados real: confere se o servidor esta no ar,
mostra a telemetria que esta chegando e importa as leituras para o modelo
relacional do TCC, de onde os modelos preditivos as consomem.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_utils import header, sidebar_info, PH_MIN, PH_MAX, EC_MIN, EC_MAX
from src.api_client import ApiIndisponivel, ServidorSensores, resumo_simulados
from src.api_ingest import resumo_ingestao, sincronizar, ultima_leitura_importada
from src.config import API_BASE_URL, API_ID_FAZENDA

st.set_page_config(page_title="Servidor de sensores", page_icon="📡", layout="wide")


def _fmt(valor, casas: int = 2) -> str:
    """Formata um sensor que pode ter vindo nulo (falha de leitura no ESP32)."""
    return "-" if pd.isna(valor) else f"{float(valor):.{casas}f}"


def _faixa(valor, minimo: float, maximo: float) -> str:
    if pd.isna(valor):
        return "sem leitura"
    return "OK" if minimo <= float(valor) <= maximo else "Fora da faixa"


def _tabela_estatisticas(est: dict) -> pd.DataFrame:
    """Converte o dict plano de /v1/estatisticas numa tabela por sensor."""
    rotulos = {
        "temp_agua":     "Temperatura da solucao (C)",
        "temp_ambiente": "Temperatura ambiente (C)",
        "ph":            "pH",
        "ec":            "CE (mS/cm)",
    }
    linhas = [
        {
            "sensor": rotulo,
            "media":  _fmt(est.get(f"media_{campo}")),
            "minima": _fmt(est.get(f"min_{campo}")),
            "maxima": _fmt(est.get(f"max_{campo}")),
        }
        for campo, rotulo in rotulos.items()
    ]
    return pd.DataFrame(linhas)


sidebar_info()
header("Servidor de sensores (API + ESP32)",
       "Telemetria recebida pelo servidor FastAPI e importacao para o banco do TCC")

# ----------------------------------------------------------------- conexao

with st.sidebar:
    st.markdown("---")
    st.markdown("### Conexao com a API")
    url = st.text_input("URL do servidor", value=API_BASE_URL,
                        help="Ex.: http://localhost:8000 ou http://192.168.0.10:8000")
    fazenda_filtro = st.text_input("Filtrar por id_fazenda", value=API_ID_FAZENDA,
                                   help="Deixe em branco para todas as fazendas")

api = ServidorSensores(base_url=url)

try:
    status = api.status()
except ApiIndisponivel as e:
    st.error(
        f"Nao foi possivel falar com a API em `{url}`.\n\n"
        f"Detalhe: {e}\n\n"
        "Suba o servidor com:\n\n"
        "    python servidor/servidor_api.py"
    )
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Servidor", status.get("STATUS", "?").upper())
c2.metric("Hora do servidor", status.get("hora_servidor", "-"))
c3.metric("Leituras no servidor", f"{status.get('total_leituras', 0):,}".replace(",", "."))

# ------------------------------------------------------------- escopo/planta

fazenda = fazenda_filtro.strip() or None

if not fazenda:
    # sem filtro informado, descobre a fazenda ativa pela leitura mais recente
    # em vez de exigir que o operador saiba o id de cor
    recente = api.ultima_leitura()
    if recente and recente.get("id_fazenda"):
        fazenda = str(recente["id_fazenda"])
        st.caption(f"Fazenda detectada pela ultima leitura recebida: `{fazenda}`")

plantas_api = pd.DataFrame()
if fazenda:
    try:
        plantas_api = api.plantas(fazenda)
    except ApiIndisponivel as e:
        st.warning(f"Falha ao listar plantas da fazenda: {e}")

id_planta = None
if not plantas_api.empty:
    st.markdown("### Plantas publicando nesta fazenda")
    st.dataframe(plantas_api, use_container_width=True, hide_index=True)
    rotulos = {
        f"{r.nome_planta or r.id_planta} ({r.total_leituras} leituras)": r.id_planta
        for r in plantas_api.itertuples(index=False)
    }
    escolha = st.selectbox("Planta em foco", ["(todas)"] + list(rotulos))
    id_planta = rotulos.get(escolha)

# ------------------------------------------------------------ telemetria ao vivo

st.markdown("---")
st.markdown("### Telemetria recente")

n_linhas = st.slider("Leituras exibidas", 20, 1000, 200, step=20)
try:
    recentes = api.leituras(id_planta=id_planta, limite=n_linhas)
except ApiIndisponivel as e:
    st.error(f"Falha ao consultar leituras: {e}")
    st.stop()

if recentes.empty:
    st.info("O servidor ainda nao recebeu nenhuma leitura do ESP32.")
else:
    ultima = recentes.iloc[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("pH", _fmt(ultima["ph"]),
              _faixa(ultima["ph"], PH_MIN, PH_MAX))
    m2.metric("CE (mS/cm)", _fmt(ultima["ec"]),
              _faixa(ultima["ec"], EC_MIN, EC_MAX))
    m3.metric("Temp. solucao (C)", _fmt(ultima["temp_agua"]))
    m4.metric("Temp. ambiente (C)", _fmt(ultima["temp_ambiente"]))
    m5.metric("Ultima leitura", pd.Timestamp(ultima["timestamp"]).strftime("%d/%m %H:%M:%S"))

    simulados = resumo_simulados(recentes["sensores_simulados"])
    if simulados != "nenhum":
        st.warning(f"Sensores em modo simulado no periodo exibido: {simulados}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recentes["timestamp"], y=recentes["ph"],
                             name="pH", line=dict(color="#1f4e79", width=2)))
    fig.add_trace(go.Scatter(x=recentes["timestamp"], y=recentes["ec"],
                             name="CE (mS/cm)", line=dict(color="#c00000", width=2),
                             yaxis="y2"))
    fig.update_layout(
        title="pH e CE recebidos pela API",
        xaxis_title="Tempo",
        yaxis=dict(title="pH"),
        yaxis2=dict(title="CE (mS/cm)", overlaying="y", side="right"),
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=recentes["timestamp"], y=recentes["temp_agua"],
                               name="Solucao", line=dict(color="#1f77b4", width=2)))
    fig_t.add_trace(go.Scatter(x=recentes["timestamp"], y=recentes["temp_ambiente"],
                               name="Ambiente", line=dict(color="#ff7f0e", width=2)))
    fig_t.update_layout(title="Temperaturas (C)", xaxis_title="Tempo",
                        yaxis_title="C", margin=dict(t=50, b=40),
                        legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig_t, use_container_width=True)

    with st.expander("Tabela bruta recebida da API"):
        st.dataframe(recentes.sort_values("timestamp", ascending=False),
                     use_container_width=True, hide_index=True)

# ------------------------------------------------------------- estatisticas

with st.expander("Estatisticas agregadas no servidor"):
    janela = st.selectbox("Janela", ["Tudo", "Ultimas 24h", "Ultimos 7 dias"])
    desde = None
    if janela == "Ultimas 24h":
        desde = datetime.now() - timedelta(hours=24)
    elif janela == "Ultimos 7 dias":
        desde = datetime.now() - timedelta(days=7)
    try:
        est = api.estatisticas(id_fazenda=fazenda, id_planta=id_planta, desde=desde)
        st.dataframe(_tabela_estatisticas(est), use_container_width=True, hide_index=True)
    except ApiIndisponivel as e:
        st.warning(f"Falha ao obter estatisticas: {e}")

# -------------------------------------------------------------- importacao

st.markdown("---")
st.markdown("### Importar para o banco do TCC")
st.caption(
    "A importacao converte a tabela plana da API nas entidades do modelo "
    "relacional (PLANTS, AMBIENT, MONITORING, SENSOR_DATA). E idempotente: "
    "leituras ja importadas sao ignoradas."
)

marcador = ultima_leitura_importada(id_planta)
st.info(
    f"Ultima leitura ja importada: **{marcador:%d/%m/%Y %H:%M:%S}**"
    if marcador else "Nenhuma leitura importada ainda."
)

col_a, col_b = st.columns([1, 2])
with col_a:
    completo = st.checkbox("Reler todo o historico", value=marcador is None,
                           help="Desmarcado, baixa apenas o que chegou depois da ultima importacao")
with col_b:
    executar = st.button("Sincronizar agora", type="primary")

if executar:
    with st.spinner("Baixando e gravando leituras..."):
        try:
            rel = sincronizar(api, id_fazenda=fazenda, id_planta=id_planta,
                              incremental=not completo)
        except ApiIndisponivel as e:
            st.error(f"Sincronizacao falhou: {e}")
            st.stop()
    st.success(f"{rel.novas} leitura(s) nova(s) importada(s).")
    st.json(rel.as_dict())

ingerido = resumo_ingestao()
if not ingerido.empty:
    st.markdown("#### Ja importado no banco")
    st.dataframe(ingerido, use_container_width=True, hide_index=True)
    st.caption(
        "Treine os modelos sobre esses dados com `python -m src.evaluate api` "
        "e depois escolha a fonte **api** na tela 8 (Previsao por planta)."
    )
