"""
Página inicial do protótipo Streamlit.
Executar:
    streamlit run streamlit_app/Home.py

TCC UNIP 2026 - Elber Jose Silva Nascimento e Guilherme Lemos Freitas
Orientador: Prof. Marco Gomes
"""
from __future__ import annotations

import streamlit as st

from app_utils import header, sidebar_info


st.set_page_config(
    page_title="Hidroponia Inteligente - TCC UNIP",
    page_icon="🌱",
    layout="wide",
)

sidebar_info()

header(
    "Sistema de monitoramento hidroponico inteligente",
    "Previsao de pH, CE e nutrientes com series temporais (ARIMA e LSTM)",
)

st.markdown("Use o menu lateral para navegar entre as telas.")

st.markdown("### Requisitos funcionais implementados")
st.table({
    "Codigo":   ["RF01", "RF02", "RF03", "RF04", "RF05", "RF06", "RF07"],
    "Requisito": [
        "Cadastrar planta",
        "Cadastrar ambiente",
        "Registrar monitoramento",
        "Consultar historico",
        "Registrar producao",
        "Visualizar indicadores",
        "Inicializar banco (schema + catalogos)",
    ],
    "Tela":     [
        "1_Cadastrar_Planta",
        "2_Cadastrar_Ambiente",
        "3_Registrar_Monitoramento",
        "4_Consultar_Historico",
        "5_Registrar_Producao",
        "6_Painel_de_Indicadores",
        "7_Importar_Dados",
    ],
    "Status":   ["Implementado"] * 7,
})

st.markdown("### Telas adicionais")
st.markdown(
    "- **8_Previsao_por_Planta** - selecione uma planta e veja a previsao de "
    "pH e CE pelo LSTM treinado sobre as leituras dos sensores dela.\n"
    "- **9_Comparativo_Modelos** - Quadro 6 do documento: metricas ARIMA vs LSTM "
    "sobre o conjunto de teste.\n"
    "- **10_Servidor_API** - estado do servidor FastAPI, telemetria que o ESP32 "
    "esta enviando e importacao dessas leituras para o banco."
)
