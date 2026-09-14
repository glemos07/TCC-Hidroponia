"""Registro de producao (colheita)."""
from datetime import date

import streamlit as st

from app_utils import header, sidebar_info
from src.database import connect, fetch_df

sidebar_info()
header("Registrar producao", "Registra o resultado final da colheita")

plants = fetch_df("SELECT id, name_plant FROM plants ORDER BY name_plant")
ambient = fetch_df("SELECT id, nm_ambient FROM ambient ORDER BY nm_ambient")

if plants.empty or ambient.empty:
    st.error("Necessario cadastrar plantas e ambientes primeiro.")
    st.stop()

with st.form("form_prod", clear_on_submit=True):
    id_plant = st.selectbox(
        "Planta colhida",
        plants["id"],
        format_func=lambda i: plants.loc[plants["id"] == i, "name_plant"].values[0],
    )
    id_amb = st.selectbox(
        "Ambiente",
        ambient["id"],
        format_func=lambda i: ambient.loc[ambient["id"] == i, "nm_ambient"].values[0],
    )
    dt = st.date_input("Data da colheita", value=date.today())
    peso = st.number_input("Peso colhido (kg)", min_value=0.0, value=12.0, step=0.1)
    tempo = st.number_input("Tempo de crescimento (dias)", min_value=0.0, value=35.0, step=0.5)
    obs = st.text_input("Observacao (opcional)")
    submit = st.form_submit_button("Registrar producao", type="primary")

if submit:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO farm (dt_harvest, weight_harvest, time_growth, obs, id_plant, id_ambient) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (dt.isoformat(), float(peso), float(tempo), obs or None, int(id_plant), int(id_amb)),
        )
    st.success(f"Producao registrada com id={cur.lastrowid}.")

st.markdown("---")
st.markdown("### Ultimas colheitas registradas")
st.dataframe(
    fetch_df("""
        SELECT f.id, f.dt_harvest, p.name_plant, a.nm_ambient,
               f.weight_harvest AS peso_kg, f.time_growth AS ciclo_dias, f.obs
        FROM farm f
        JOIN plants p  ON p.id = f.id_plant
        JOIN ambient a ON a.id = f.id_ambient
        ORDER BY f.id DESC LIMIT 20
    """),
    use_container_width=True,
)
