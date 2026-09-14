"""Utilitários compartilhados entre as páginas Streamlit."""
from __future__ import annotations

import sys
from pathlib import Path

# garante import de src.* independente de onde o Streamlit for executado
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import pandas as pd
import streamlit as st

from src.database import fetch_df, connect  # noqa: E402


PH_MIN, PH_MAX = 5.5, 6.5
EC_MIN, EC_MAX = 1.0, 3.5


def header(titulo: str, subtitulo: str | None = None):
    st.title(titulo)
    if subtitulo:
        st.caption(subtitulo)


def sidebar_info():
    with st.sidebar:
        st.markdown("### TCC UNIP 2026")
        st.markdown("**Hidroponia inteligente**")
        st.markdown("Previsao de pH e CE com séries temporais.")
        st.markdown("---")
        st.markdown("**Faixas ideais**")
        st.markdown(f"- pH: {PH_MIN}–{PH_MAX}")
        st.markdown(f"- CE: {EC_MIN}–{EC_MAX} mS/cm")


def dropdown_from_table(label: str, table: str, id_col: str = "id", name_col: str = "name"):
    df = fetch_df(f"SELECT {id_col}, {name_col} FROM {table} ORDER BY {name_col}")
    if df.empty:
        st.warning(f"Nenhum registro em '{table}'. Execute o seed.")
        return None
    mapa = {f"{r[name_col]} (id={r[id_col]})": int(r[id_col]) for _, r in df.iterrows()}
    escolha = st.selectbox(label, list(mapa.keys()))
    return mapa[escolha]
