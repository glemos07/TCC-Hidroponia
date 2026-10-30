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


def sincronizar_ao_abrir():
    """
    Na primeira tela aberta da sessao, traz do servidor tudo que chegou
    enquanto o app esteve fora do ar.

    O app nao fica rodando o tempo todo, mas o ESP32 nao para de publicar.
    Sem isto, reabrir o sistema mostraria o historico parado no dia em que
    foi fechado ate alguem lembrar de sincronizar a mao.

    Nunca derruba a tela: sem rede, o app segue com o que ja esta no banco.
    """
    if st.session_state.get("sync_inicial_feito"):
        return
    st.session_state["sync_inicial_feito"] = True

    try:
        from src.api_ingest import sincronizar
        with st.spinner("Buscando no servidor as leituras recebidas enquanto o app esteve fora..."):
            rel = sincronizar()
        st.session_state["sync_inicial_novas"] = rel.novas
        st.session_state["sync_inicial_ate"] = rel.ultimo_ts
    except Exception as e:                       # noqa: BLE001
        st.session_state["sync_inicial_erro"] = str(e)


def _resumo_sync_na_sidebar():
    """Mostra o resultado da recuperacao inicial, sem ocupar a area principal."""
    erro = st.session_state.get("sync_inicial_erro")
    if erro:
        st.caption("⚠️ Servidor fora de alcance — exibindo o que ja esta no banco.")
        st.caption(f"`{erro[:90]}`")
        return
    novas = st.session_state.get("sync_inicial_novas")
    if novas is None:
        return
    ate = st.session_state.get("sync_inicial_ate")
    if novas:
        st.caption(f"✅ {novas} leitura(s) recuperada(s) do servidor ao abrir"
                   + (f", ate {ate:%d/%m %H:%M}." if ate else "."))
    else:
        st.caption("✅ Banco ja estava em dia com o servidor.")


def sidebar_info():
    sincronizar_ao_abrir()
    with st.sidebar:
        st.markdown("### TCC UNIP 2026")
        st.markdown("**Hidroponia inteligente**")
        st.markdown("Previsao de pH e CE com séries temporais.")
        st.markdown("---")
        st.markdown("**Faixas ideais**")
        st.markdown(f"- pH: {PH_MIN}–{PH_MAX}")
        st.markdown(f"- CE: {EC_MIN}–{EC_MAX} mS/cm")
        st.markdown("---")
        _resumo_sync_na_sidebar()


def dropdown_from_table(label: str, table: str, id_col: str = "id", name_col: str = "name"):
    df = fetch_df(f"SELECT {id_col}, {name_col} FROM {table} ORDER BY {name_col}")
    if df.empty:
        st.warning(f"Nenhum registro em '{table}'. Execute o seed.")
        return None
    mapa = {f"{r[name_col]} (id={r[id_col]})": int(r[id_col]) for _, r in df.iterrows()}
    escolha = st.selectbox(label, list(mapa.keys()))
    return mapa[escolha]
