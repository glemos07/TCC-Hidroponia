"""Cadastro de planta."""
from datetime import date

import streamlit as st

from app_utils import header, sidebar_info, dropdown_from_table
from src.database import connect, fetch_df

sidebar_info()
header("Cadastrar planta", "Registra uma nova cultura no sistema")

with st.form("form_planta", clear_on_submit=True):
    nome = st.text_input("Nome da planta", placeholder="Ex.: Alface Crespa #121")
    diversidade = st.text_input("Diversidade / variedade", placeholder="Ex.: Alface Crespa")
    dt_plantio = st.date_input("Data de plantio", value=date.today())
    st.write("**Status do cultivo**")
    df_status = fetch_df("SELECT id, status_name FROM farm_status ORDER BY id")
    mapa = {r["status_name"]: int(r["id"]) for _, r in df_status.iterrows()}
    status_label = st.selectbox("Selecione", list(mapa.keys()))
    submit = st.form_submit_button("Cadastrar planta", type="primary")

if submit:
    if not (nome and diversidade):
        st.error("Preencha nome e diversidade.")
    else:
        with connect() as conn:
            cur = conn.execute(
                "INSERT INTO plants (name_plant, diversity, dt_planting, status_id) "
                "VALUES (?, ?, ?, ?)",
                (nome, diversidade, dt_plantio.isoformat(), mapa[status_label]),
            )
            new_id = cur.lastrowid
        st.success(f"Planta cadastrada com id={new_id}.")

st.markdown("---")
st.markdown("### Plantas cadastradas")
st.dataframe(
    fetch_df(
        "SELECT p.id, p.name_plant, p.diversity, p.dt_planting, fs.status_name "
        "FROM plants p JOIN farm_status fs ON fs.id = p.status_id "
        "ORDER BY p.id DESC LIMIT 20"
    ),
    use_container_width=True,
)
