"""Registro de monitoramento e leituras dos sensores."""
from datetime import datetime

import streamlit as st

from app_utils import header, sidebar_info, PH_MIN, PH_MAX, EC_MIN, EC_MAX
from src.database import connect, fetch_df

sidebar_info()
header("Registrar monitoramento", "Insere um evento de monitoramento com as leituras dos sensores")

plants = fetch_df("SELECT id, name_plant FROM plants ORDER BY name_plant")
ambient = fetch_df("SELECT id, nm_ambient FROM ambient ORDER BY nm_ambient")
users = fetch_df("SELECT id, name FROM users ORDER BY name")
nutrients = fetch_df("SELECT id, name FROM nutrient ORDER BY name")
sensors = fetch_df("SELECT id, sensor_type FROM sensor ORDER BY id")

if plants.empty or ambient.empty:
    st.error("Cadastre plantas e ambientes antes de registrar monitoramento.")
    st.stop()

with st.form("form_mon", clear_on_submit=True):
    dt = st.date_input("Data")
    hora = st.time_input("Hora")
    id_plant = st.selectbox(
        "Planta",
        plants["id"],
        format_func=lambda i: plants.loc[plants["id"] == i, "name_plant"].values[0],
    )
    id_amb = st.selectbox(
        "Ambiente",
        ambient["id"],
        format_func=lambda i: ambient.loc[ambient["id"] == i, "nm_ambient"].values[0],
    )
    id_user = st.selectbox(
        "Responsavel",
        users["id"],
        format_func=lambda i: users.loc[users["id"] == i, "name"].values[0],
    )
    st.markdown("**Leituras dos sensores**")
    c1, c2, c3, c4 = st.columns(4)
    ph  = c1.number_input("pH", value=6.0, step=0.01, min_value=0.0, max_value=14.0)
    ec  = c2.number_input("CE (mS/cm)", value=1.8, step=0.01)
    tmp = c3.number_input("Temperatura (°C)", value=22.0, step=0.1)
    hum = c4.number_input("Umidade (%)", value=68.0, step=0.1)

    nut_sel = st.multiselect(
        "Nutrientes presentes",
        nutrients["id"],
        format_func=lambda i: nutrients.loc[nutrients["id"] == i, "name"].values[0],
    )
    obs = st.text_input("Observacao (opcional)")
    submit = st.form_submit_button("Registrar monitoramento", type="primary")

if submit:
    dt_reg = datetime.combine(dt, hora).strftime("%Y-%m-%d %H:%M:%S")
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO monitoring (dt_register, id_plant, id_ambient, id_user, obs) "
            "VALUES (?, ?, ?, ?, ?)",
            (dt_reg, int(id_plant), int(id_amb), int(id_user), obs or None),
        )
        mid = cur.lastrowid
        sensor_values = {1: ph, 2: ec, 3: tmp, 4: hum}
        for sid, val in sensor_values.items():
            conn.execute(
                "INSERT INTO sensor_data (id_monitoring, id_sensor, value) VALUES (?, ?, ?)",
                (mid, sid, float(val)),
            )
        for nid in nut_sel:
            conn.execute(
                "INSERT INTO monitoring_nutrient (id_monitoring, id_nutrient) VALUES (?, ?)",
                (mid, int(nid)),
            )
    st.success(f"Monitoramento id={mid} registrado com sucesso.")

    alertas = []
    if not (PH_MIN <= ph <= PH_MAX):
        alertas.append(f"pH {ph} fora da faixa ideal ({PH_MIN}-{PH_MAX}).")
    if not (EC_MIN <= ec <= EC_MAX):
        alertas.append(f"CE {ec} fora da faixa ideal ({EC_MIN}-{EC_MAX} mS/cm).")
    for a in alertas:
        st.warning(a)

st.markdown("---")
st.markdown("### Ultimos monitoramentos")
st.dataframe(
    fetch_df("""
        SELECT m.id, m.dt_register, p.name_plant, a.nm_ambient, u.name AS responsavel, m.obs
        FROM monitoring m
        JOIN plants  p ON p.id = m.id_plant
        JOIN ambient a ON a.id = m.id_ambient
        LEFT JOIN users u ON u.id = m.id_user
        ORDER BY m.id DESC LIMIT 20
    """),
    use_container_width=True,
)
